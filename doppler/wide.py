# search over an area too big for one flat map, or a whole day of flights
# 1. tiles, each with its own local frame, rank their flight combinations (find._rank)
# 2. the best of all tiles are searched again as small areas with the start nearly known (find.search)
# 3. one verdict over everything found
import os
import time
from multiprocessing import get_context

import numpy as np

from . import find
from .model import from_enu

_W = {}


# flights from the local ads-b index; picklable, so each worker loads its own tile
class Index:

    def __init__(self, store=None, sources=find.TONAL):
        self.store, self.sources = store, sources

    # everyone: jets and the rest too, for the broadband check
    def __call__(self, bbox, t_lo, t_hi, everyone=False):
        from . import adsb, adsbindex
        tracks, info = adsbindex.query(t_lo, t_hi, bbox, self.store or adsbindex.DEFAULT_STORE)
        kind = {k: adsb.classify(t, c) for k, (t, c) in info.items()}
        if not everyone:
            tracks = {k: a for k, a in tracks.items() if kind[k] in self.sources}
        return adsb.segments(tracks, kind=kind), info


# flights from a csv of tracks, for areas small enough to have one
class Tracks:

    def __init__(self, path, sources=find.TONAL):
        import csv
        from . import adsb
        self.tracks, self.kind = adsb.load_csv(path, with_kind=True)
        self.sources, self.types = sources, {}
        with open(path) as fh:
            for r in csv.DictReader(fh):
                self.types.setdefault(r["icao"], (r.get("type") or "", r.get("cat") or ""))

    # bbox None: everything in the file
    def __call__(self, bbox, t_lo, t_hi, everyone=False):
        from . import adsb
        if bbox is None:
            return adsb.segments(self.tracks, kind=self.kind), self.types
        sub = {}
        for k, a in self.tracks.items():
            m = (a[:, 0] >= t_lo) & (a[:, 0] <= t_hi) & (a[:, 1] >= bbox[0]) & (a[:, 1] <= bbox[1]) & \
                (a[:, 2] >= bbox[2]) & (a[:, 2] <= bbox[3])
            if m.sum() >= 2 and (everyone or self.kind.get(k, "unknown") in self.sources):
                sub[k] = a[m]
        return adsb.segments(sub, kind=self.kind), self.types


# bbox [lat0, lat1, lon0, lon1] cut into tiles about km wide
def tiles(bbox, km=100.0):
    out = []
    n_la = max(1, int(np.ceil((bbox[1] - bbox[0]) * 111.32 / km)))
    for la in np.linspace(bbox[0], bbox[1], n_la + 1)[:-1]:
        la1 = la + (bbox[1] - bbox[0]) / n_la
        w = 111.32 * np.cos(np.radians((la + la1) / 2))
        n_lo = max(1, int(np.ceil((bbox[3] - bbox[2]) * w / km)))
        for lo in np.linspace(bbox[2], bbox[3], n_lo + 1)[:-1]:
            out.append([float(la), float(la1), float(lo), float(lo + (bbox[3] - bbox[2]) / n_lo)])
    return out


def _grow(bbox, km):
    d = km / 111.32
    dn = d / np.cos(np.radians((bbox[0] + bbox[1]) / 2))
    return [bbox[0] - d, bbox[1] + d, bbox[2] - dn, bbox[3] + dn]


def _init(W):
    _W.update(W)


# one tile's best flight combinations: where, when, and how well the ridges fit
# a tile too crowded to look at in full (an event fits tens of millions of places) is done as four smaller ones
def _tile(bbox, depth=0):
    W, t0 = _W, time.time()
    segs, _ = W["load"](_grow(bbox, W["pad"]), *W["window"])
    if not segs:
        return [], (time.time() - t0, 0, bbox)
    A = find._setup(None, W["sr"], W["S"], W["dur"], segs, bbox, W["events"], W["tol"], W["temp_c"], W["sources"],
                    W["elev"], W["spec"], W["dem_step"], lapse=W["lapse"])
    over = []
    R = find._rank(A, n_fine=W["n_tile"], lone=200000, over=over, log=lambda m: None) if A.flights else None  # fewer single passes per tile
    if over and depth < 2:
        la, lo = (bbox[0] + bbox[1]) / 2, (bbox[2] + bbox[3]) / 2
        del A, R, segs
        parts = [_tile(b, depth + 1) for b in ([bbox[0], la, bbox[2], lo], [bbox[0], la, lo, bbox[3]],
                                               [la, bbox[1], bbox[2], lo], [la, bbox[1], lo, bbox[3]])]
        return [g for out, _ in parts for g in out], (time.time() - t0, sum(st[1] for _, st in parts), bbox)
    out = []
    for sig, ix in (R["ranked"] if R else []):
        i = ix[0]
        la, lo = from_enu(R["gx"][i], R["gy"][i], A.lat0, A.lon0)
        out.append(dict(pre=float(R["pre"][i]), lat=float(la), lon=float(lo), offset=float(R["go"][i]), n=len(sig),
                        icaos=[A.flights[f]["icao"] for _, f in sig]))
    return out, (time.time() - t0, len(A.flights), bbox)


# a small area around one tile result, start known to a few seconds
def _local(job):
    W = _W
    la, lo, o = job
    S, tol, dur = W["S"], W["tol"], W["dur"]
    a, b = o - W["slack"], o + W["slack"]
    if tol is not None:
        a, b = max(a, -tol), min(b, tol)
    S2, tol2 = S + (a + b) / 2, max((b - a) / 2, 1.0)
    bbox = _grow([la, la, lo, lo], W["local_km"])
    t_lo, t_hi = S2 - tol2 - 900, S2 + dur + tol2 + 900
    if tol is None:  # only the window's flights are there to load
        t_lo, t_hi = max(t_lo, W["window"][0]), min(t_hi, W["window"][1])
    segs, types = W["load"](_grow(bbox, W["pad"]), t_lo, t_hi, True)
    res, info = find.search(W["x"], W["sr"], S2, dur, segs, bbox, W["events"], tol=tol2, temp_c=W["temp_c"],
                            sources=W["sources"], elev=W["elev"], spec=W["spec"], rspec=W["rspec"], lapse=W["lapse"],
                            grid_s=20.0, log=lambda m: None)  # many of these: the grid search gets little time
    for r in res:
        r["offset"] += S2 - S
        r["local"] = dict(bbox=bbox, start=S2, tol=tol2)
    return res


# best places anywhere in bbox; load(bbox, t_lo, t_hi) returns (segments, types)
# tol=None: start unknown, window is the (t_lo, t_hi) of flights to try
def search(x, sr, S, dur, load, bbox, events, tol=30.0, window=None, temp_c=15.0, sources=find.TONAL, elev=None,
           tile_km=100.0, local_km=8.0, n_local=None, workers=None, lapse=6.5, log=print):
    from .events import ridge_spectrogram, spectrogram
    t_start, say = time.time(), log
    log = lambda m: say(f"{m}  [{time.time() - t_start:.0f} s]")
    if window is None:
        window = (S - tol - 900, S + dur + tol + 900)
    T = tiles(bbox, tile_km)
    n_local = n_local or int(np.clip(len(T) // 4, 48, 400))  # more ground, more chance pairs ahead of the right one
    span = max((bbox[1] - bbox[0]) * 111.32, (bbox[3] - bbox[2]) * 111.32 * np.cos(np.radians((bbox[0] + bbox[1]) / 2)))
    workers = workers or max(1, min(8, (os.cpu_count() or 2) // 2, len(T)))
    W = dict(x=x, sr=sr, S=S, dur=dur, tol=tol, window=window, temp_c=temp_c, sources=sources, elev=elev, load=load,
             events=events, spec=spectrogram(x, sr, fmax=1500, cents=8, win_s=1.0, hop_s=0.5), pad=12.0,
             dem_step=500.0 if span > 1000 else 250.0, n_tile=24, local_km=local_km, slack=6.0,
             rspec=ridge_spectrogram(x, sr), lapse=lapse)
    log(f"{len(T)} tiles, {workers} workers")
    old = os.environ.get("OMP_NUM_THREADS")
    os.environ["OMP_NUM_THREADS"] = str(max(1, (os.cpu_count() or 2) // workers))  # kernels share the cores
    try:
        # spawn: a forked child hangs in openmp once the parent has used it
        with get_context("spawn").Pool(workers, _init, (W,)) as pool:
            found, done, stats = [], 0, []
            for out, stat in pool.imap_unordered(_tile, T):
                found += out
                done += 1
                stats.append(stat)
                if done % max(1, len(T) // 10) == 0:
                    log(f"  {done} of {len(T)} tiles, {len(found)} combinations")
            found.sort(key=lambda g: -g["pre"])
            jobs = []

            def take(g):
                near = lambda j: abs(j[2] - g["offset"]) <= W["slack"] and \
                    find._dist(dict(lat=j[0], lon=j[1]), g) < 1000 * local_km / 2
                if not any(near(j) for j in jobs):
                    jobs.append((g["lat"], g["lon"], g["offset"]))
            for g in found:
                if len(jobs) >= n_local:
                    break
                take(g)
            if tol is None:  # single passes would lose every slot to chance pairs
                for g in [g for g in found if g["n"] == 1][:n_local // 2]:
                    take(g)
            log(f"{len(found)} combinations, {len(jobs)} places to look at closely")
            res = [r for rs in pool.imap_unordered(_local, jobs) for r in rs]
    finally:
        if old is None:
            del os.environ["OMP_NUM_THREADS"]
        else:
            os.environ["OMP_NUM_THREADS"] = old
    res = find._verdict(res, tail=True)
    log(f"{len(res)} answers")
    return res, dict(found=found, tiles=stats)
