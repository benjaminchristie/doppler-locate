# per-day low-altitude adsb store: one pass over a day keeps points below max_alt in 1-degree tiles
# queries then read a few small files instead of re-streaming 3.5 GB
import gzip
import io
import json
import os
import tarfile
import time
import urllib.request
from datetime import date, datetime, timedelta, timezone
from multiprocessing import Pool

import numpy as np

from .adsb import FPM, FT, KT, _Cat, adsblol_assets
from .progress import report

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_STORE = os.path.join(ROOT, "data", "adsb")
FIELDS = ("t", "lat", "lon", "alt", "gs", "trk", "vr")
_MAX_ALT = 8000.0


def _parse(data):
    try:
        d = json.loads(gzip.decompress(data) if data[:2] == b"\x1f\x8b" else data)
    except Exception:
        return None
    base = d.get("timestamp", 0)
    rows = []
    for p in d.get("trace", []):
        alt = p[10] if len(p) > 10 and p[10] is not None else p[3]
        if alt is None or isinstance(alt, str) or alt * FT > _MAX_ALT:
            continue
        rows.append((base + p[0], p[1], p[2], alt * FT,
                     np.nan if p[4] is None else p[4] * KT, np.nan if p[5] is None else p[5],
                     np.nan if p[7] is None else p[7] * FPM))
    if not rows:
        return None
    cat = next((p[8].get("category") for p in d.get("trace", []) if len(p) > 8 and isinstance(p[8], dict)
                and p[8].get("category")), "") or ""
    return d.get("icao", ""), d.get("t") or "", cat, np.array(rows)


def build(day, store=DEFAULT_STORE, tar_paths=None, workers=8, log=print):
    out = os.path.join(store, day.isoformat())
    if os.path.exists(os.path.join(out, "aircraft.json")):
        log(f"{out} exists")
        return out
    os.makedirs(out, exist_ok=True)
    if tar_paths:
        opener, srcs, total = (lambda p: open(p, "rb")), tar_paths, sum(os.path.getsize(p) for p in tar_paths)
    else:
        assets = adsblol_assets(day)
        opener, srcs, total = urllib.request.urlopen, [u for u, _ in assets], sum(n for _, n in assets)
    got, t0 = [0], time.time()

    def on_read(n):
        got[0] += n
        rate = got[0] / max(time.time() - t0, 1e-3)
        report("index", got[0] / total, f"{got[0] / 1e9:.1f} of {total / 1e9:.1f} GB, {rate / 1e6:.0f} MB/s")

    tf = tarfile.open(fileobj=io.BufferedReader(_Cat(opener, srcs, on_read), 1 << 20), mode="r|")

    def blobs():
        try:
            for m in tf:
                if m.isfile() and "trace_full" in m.name:
                    yield tf.extractfile(m).read()
        except (tarfile.ReadError, EOFError) as e:
            log(f"tar stream ended early: {e}")

    aircraft, tiles = [], {}
    with Pool(workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_parse, blobs(), chunksize=64)):
            if r is None:
                continue
            icao, typ, cat, a = r
            aid = len(aircraft)
            aircraft.append([icao, typ, cat])
            key = (np.floor(a[:, 1]).astype(int) + 90) * 360 + (np.floor(a[:, 2]).astype(int) + 180)
            for k in np.unique(key):
                tiles.setdefault(int(k), []).append((aid, a[key == k]))
    for k, parts in tiles.items():
        aid = np.concatenate([np.full(len(a), i, np.int32) for i, a in parts])
        a = np.concatenate([a for _, a in parts])
        cols = {f: a[:, j].astype(np.float64 if f == "t" else np.float32) for j, f in enumerate(FIELDS)}
        np.savez(os.path.join(out, _tile_name(k // 360 - 90, k % 360 - 180)), aid=aid, **cols)
    json.dump(aircraft, open(os.path.join(out, "aircraft.json"), "w"))
    report("index", 1.0, "done")
    log(f"{len(aircraft)} aircraft below {_MAX_ALT:.0f} m in {len(tiles)} tiles, {time.time() - t0:.0f} s -> {out}")
    return out


def _tile_name(la, lo):
    return f"{la}_{lo}.npz"


# returns tracks {icao: array(t, lat, lon, alt, gs, trk, vr)} and {icao: (type, cat)}; bbox=None means everywhere
def query(t_start, t_end, bbox=None, store=DEFAULT_STORE):
    tracks, kinds = {}, {}
    d = datetime.fromtimestamp(t_start, timezone.utc).date()
    while d <= datetime.fromtimestamp(t_end, timezone.utc).date():
        out = os.path.join(store, d.isoformat())
        if not os.path.exists(os.path.join(out, "aircraft.json")):
            raise FileNotFoundError(f"no ADS-B index for {d}; run: python -m doppler adsb-index --date {d}")
        aircraft = json.load(open(os.path.join(out, "aircraft.json")))
        if bbox is None:
            files = [f for f in os.listdir(out) if f.endswith(".npz")]
        else:
            files = [_tile_name(la, lo) for la in range(int(np.floor(bbox[0])), int(np.floor(bbox[1])) + 1)
                     for lo in range(int(np.floor(bbox[2])), int(np.floor(bbox[3])) + 1)]
        parts = {}
        for f in files:
            p = os.path.join(out, f)
            if not os.path.exists(p):
                continue
            z = np.load(p)
            m = (z["t"] >= t_start) & (z["t"] <= t_end)
            if bbox is not None:
                m &= (z["lat"] >= bbox[0]) & (z["lat"] <= bbox[1]) & (z["lon"] >= bbox[2]) & (z["lon"] <= bbox[3])
            if not m.any():
                continue
            a = np.stack([z[f][m].astype(np.float64) for f in FIELDS], 1)
            aid = z["aid"][m]
            for i in np.unique(aid):
                parts.setdefault(int(i), []).append(a[aid == i])
        for i, ps in parts.items():
            icao, typ, cat = aircraft[i]
            a = np.concatenate(ps)
            a = a[np.argsort(a[:, 0])]
            key = icao if icao not in tracks else f"{icao}@{d}"
            tracks[key] = a
            kinds[key] = (typ, cat)
        d += timedelta(days=1)
    return tracks, kinds


def days_between(t_start, t_end):
    d, out = datetime.fromtimestamp(t_start, timezone.utc).date(), []
    while d <= datetime.fromtimestamp(t_end, timezone.utc).date():
        out.append(d)
        d += timedelta(days=1)
    return out


def has_days(t_start, t_end, store=DEFAULT_STORE):
    return all(os.path.exists(os.path.join(store, d.isoformat(), "aircraft.json")) for d in days_between(t_start, t_end))


def write_csv(path, tracks, info):
    import csv
    from .adsb import COLS
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(COLS)
        for k, a in tracks.items():
            typ, cat = info[k]
            for row in a:
                w.writerow([k, round(row[0], 2), round(row[1], 6), round(row[2], 6), round(row[3], 1),
                            "" if np.isnan(row[4]) else round(row[4], 2), "" if np.isnan(row[5]) else round(row[5], 1),
                            "" if np.isnan(row[6]) else round(row[6], 2), typ, cat])
