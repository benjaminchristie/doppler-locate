# adsb.lol daily archives to csv to resampled track segments
import csv
import gzip
import io
import json
import tarfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from multiprocessing import Pool

import numpy as np
from scipy.interpolate import CubicHermiteSpline

from .model import from_enu, to_enu

FT, KT, FPM = 0.3048, 0.514444, 0.3048 / 60
COLS = ["icao", "t", "lat", "lon", "alt_m", "gs_ms", "track_deg", "vr_ms", "type", "cat"]
NUM = COLS[1:8]
# common prop types; light aircraft (A1) and rotorcraft (A7) are caught by category
TURBOPROPS = set("""AT43 AT44 AT45 AT46 AT72 AT73 AT75 AT76 DH8A DH8B DH8C DH8D SF34 SB20 B190 BE20 BE30 BE35
BE36 BE55 BE58 BE60 BE99 BE9L BE9T C208 C210 C206 C182 C172 C152 C150 C177 C180 C185 C25A P28A P28B
P28R P28T P32R PA31 PA32 PA34 PA44 PA46 SR20 SR22 DA40 DA42 DA62 M20P M20T PC12 PC6T PC7 PC9 PC21 TBM7 TBM8 TBM9
JS31 JS32 JS41 SW4 D228 D328 F27 F50 AN24 AN26 AN28 AN32 L410 C130 C30J A400 E110 E120 SH33 SH36 BN2P BN2T
DHC2 DHC3 DHC6 DHC7 CN35 C295 P180 P68 G58 KODI GA8 C404 C421 C414 C340 C310 AC90 AC95 MU2 P46T PAY2 PAY3
SC7 Y12 AN2 AN3 C46 DC3 S2P C170 C140 C120 PA12 PA18 PA20 PA22 PA24 PA30 J3 HUSK CH7A CH7B BL8 M5 M6 M7 G21 G44""".split())


# light jets that report the light-aircraft category (A1)
LIGHT_JETS = set("HDJT C510 C25M E50P E55P SF50 EA50 PRM1 C525 C25A C25B C25C FA10 D140 PC24".split())
JET_CATS = {"A2", "A3", "A4", "A5", "A6"}  # 15.5k lb and up, high performance


# returns heli, prop, jet or unknown from icao type code and emitter category
def classify(typ, cat):
    if cat == "A7":
        return "heli"
    if typ in LIGHT_JETS:
        return "jet"
    if typ in TURBOPROPS or cat == "A1":
        return "prop"
    if cat in JET_CATS:
        return "jet"
    return "unknown"  # e.g. a Super Cub or Goose with no category; scored as tonal


# reads split tar parts back to back
class _Cat(io.RawIOBase):

    def __init__(self, opener, sources, on_read=None):
        self.opener, self.sources, self.cur, self.on_read = opener, list(sources), None, on_read

    def readable(self):
        return True

    def readinto(self, b):
        while True:
            if self.cur is None:
                if not self.sources:
                    return 0
                src = self.sources.pop(0)
                print(f"  reading {src}", flush=True)
                self.cur = self.opener(src)
            n = self.cur.readinto(b)
            if n:
                if self.on_read:
                    self.on_read(n)
                return n
            self.cur.close()
            self.cur = None


# [(url, bytes)] for a day's archive parts, in order
def adsblol_assets(day, kind="prod"):
    repo = f"adsblol/globe_history_{day.year}"
    found = []
    # some days only exist as "-0tmp" (e.g. 2025-06-01..12), some as both; take the bigger one
    for suffix in ("0", "0tmp"):
        tag = f"v{day:%Y.%m.%d}-planes-readsb-{kind}-{suffix}"
        try:
            with urllib.request.urlopen(f"https://api.github.com/repos/{repo}/releases/tags/{tag}") as r:
                assets = json.load(r)["assets"]
        except urllib.error.HTTPError as e:
            if e.code == 404:
                continue
            raise
        found.append(sorted((a["browser_download_url"], a["size"]) for a in assets if ".tar" in a["name"]))
    if not found:
        raise RuntimeError(f"adsb.lol has no {kind} archive for {day} (github.com/{repo}/releases)")
    return max(found, key=lambda a: sum(n for _, n in a))


def adsblol_urls(day, kind="prod"):
    return [u for u, _ in adsblol_assets(day, kind)]


def _parse(args):
    data, t0, t1, bbox = args
    try:
        d = json.loads(gzip.decompress(data) if data[:2] == b"\x1f\x8b" else data)
    except Exception:
        return []
    base, icao = d.get("timestamp", 0), d.get("icao", "")
    typ = d.get("t") or ""
    cat = next((p[8].get("category") for p in d.get("trace", []) if len(p) > 8 and isinstance(p[8], dict)
                and p[8].get("category")), "") or ""
    la0, la1, lo0, lo1 = bbox
    rows = []
    for p in d.get("trace", []):
        t = base + p[0]
        if not (t0 <= t <= t1) or not (la0 <= p[1] <= la1 and lo0 <= p[2] <= lo1):
            continue
        alt = p[10] if len(p) > 10 and p[10] is not None else p[3]
        if alt is None or isinstance(alt, str):  # "ground"
            continue
        gs, trk, vr = p[4], p[5], p[7]
        rows.append([icao, round(t, 2), p[1], p[2], round(alt * FT, 1),
                     None if gs is None else round(gs * KT, 2), trk,
                     None if vr is None else round(vr * FPM, 2), typ, cat])
    return rows


# stream archives (or local tar parts), keep points in the time window and bbox
def fetch(t_start, t_end, bbox, out_csv, tar_paths=None, workers=8, mlat=False):
    t0, t1 = t_start.timestamp(), t_end.timestamp()
    if tar_paths:
        groups = [(lambda p: open(p, "rb"), tar_paths)]
    else:
        groups = []
        day = t_start.date()
        while day <= t_end.date():
            for kind in ("prod", "mlatonly") if mlat else ("prod",):
                groups.append((lambda u: urllib.request.urlopen(u), adsblol_urls(day, kind)))
            day += timedelta(days=1)

    n_tr = 0
    with open(out_csv, "w", newline="") as fh, Pool(workers) as pool:
        w = csv.writer(fh)
        w.writerow(COLS)
        for opener, srcs in groups:
            tf = tarfile.open(fileobj=io.BufferedReader(_Cat(opener, srcs), 1 << 20), mode="r|")

            def blobs():
                try:
                    for m in tf:
                        if m.isfile() and "trace_full" in m.name:
                            yield tf.extractfile(m).read(), t0, t1, bbox
                except (tarfile.ReadError, EOFError) as e:
                    print(f"  tar stream ended early: {e}")

            for rows in pool.imap_unordered(_parse, blobs(), chunksize=64):
                if rows:
                    n_tr += 1
                    w.writerows(rows)
    print(f"{n_tr} aircraft -> {out_csv}")


LOL = "https://adsb.lol"
RECENT_DAYS = 2  # adsb.lol serves replay files and per-aircraft traces for about this many past days


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "doppler-tracking"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read()
    return gzip.decompress(data) if data[:2] == b"\x1f\x8b" else data


# aircraft inside bbox between t0 and t1 from the 10 s replay files; hex has a ~ prefix if not icao
def _seen(t0, t1, bbox):
    MAGIC = 0x0E7F7C9D
    la0, la1, lo0, lo1 = bbox
    out = set()
    t = int(t0 // 1800 * 1800)
    while t <= t1:
        d = datetime.fromtimestamp(t, timezone.utc)
        if t + 1800 > time.time():  # this half hour's file isn't written yet: whoever is there now
            out |= _now(bbox)
            break
        raw = _get(f"{LOL}/globe_history/{d:%Y/%m/%d}/heatmap/{d.hour * 2 + d.minute // 30:02d}.bin.ttf")
        a = np.frombuffer(raw[:len(raw) // 16 * 16], "<i4").reshape(-1, 4)
        mark = np.flatnonzero(a[:, 0] == MAGIC)
        ts = ((a[mark, 1].astype(np.int64) << 32) + (a[mark, 2].astype(np.int64) & 0xFFFFFFFF)) / 1000
        slot = np.searchsorted(mark, np.arange(len(a)), "right") - 1  # slice each record belongs to
        ok = slot >= 0
        ok[ok] &= (ts[slot[ok]] >= t0) & (ts[slot[ok]] <= t1)
        lat, lon = a[:, 1] / 1e6, a[:, 2] / 1e6
        ok &= (a[:, 0] != MAGIC) & (lat >= la0) & (lat <= la1) & (lon >= lo0) & (lon <= lo1)
        for v in np.unique(a[ok, 0]):
            out.add(("~" if v & 0x1000000 else "") + f"{v & 0xFFFFFF:06x}")
        t += 1800
    return out


# aircraft adsb.lol sees inside bbox right now
def _now(bbox):
    la0, la1, lo0, lo1 = bbox
    lat, lon = (la0 + la1) / 2, (lo0 + lo1) / 2
    nm = min(250, 60 * np.hypot(la1 - la0, (lo1 - lo0) * np.cos(np.radians(lat))) / 2 + 1)
    ac = json.loads(_get(f"https://api.adsb.lol/v2/point/{lat:.4f}/{lon:.4f}/{nm:.0f}")).get("ac", [])
    return {a["hex"] for a in ac if a.get("lat") is not None and la0 <= a["lat"] <= la1 and lo0 <= a["lon"] <= lo1}


# tracks for the last few days, before the daily archive exists
# find aircraft in the replay files, then pull each trace (a few MB instead of the whole day)
def fetch_recent(t_start, t_end, bbox, out_csv, workers=8, log=print):
    from concurrent.futures import ThreadPoolExecutor
    t0, t1 = t_start.timestamp(), t_end.timestamp()
    hexes = sorted(_seen(t0, t1, bbox))
    log(f"{len(hexes)} aircraft in the area")
    today = datetime.now(timezone.utc).date()
    days, d = [], t_start.date()
    while d <= t_end.date():
        days.append(d)
        d += timedelta(days=1)

    def one(h):
        rows = []
        for day in days:
            base = f"{LOL}/data/traces" if day == today else f"{LOL}/globe_history/{day:%Y/%m/%d}/traces"
            try:
                rows += _parse((_get(f"{base}/{h[-2:]}/trace_full_{h}.json"), t0, t1, bbox))
            except urllib.error.HTTPError:
                pass
        return rows

    n = 0
    with open(out_csv, "w", newline="") as fh, ThreadPoolExecutor(workers) as ex:
        w = csv.writer(fh)
        w.writerow(COLS)
        for rows in ex.map(one, hexes):
            if rows:
                n += 1
                w.writerows(rows)
    log(f"{n} aircraft -> {out_csv}")


# true if adsb.lol still serves replay files for unix time t
def recent(t):
    return datetime.fromtimestamp(t, timezone.utc).date() >= datetime.now(timezone.utc).date() - timedelta(days=RECENT_DAYS)


# {icao: array of NUM columns}; with_kind also returns {icao: prop|heli|jet|unknown}
def load_csv(path, with_kind=False):
    by, kind = {}, {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            by.setdefault(r["icao"], []).append(
                [float(r[k]) if r[k] not in ("", "None") else np.nan for k in NUM])
            if r["icao"] not in kind:
                kind[r["icao"]] = classify(r.get("type") or "", r.get("cat") or "")
    tracks = {k: np.array(sorted(v)) for k, v in by.items()}
    return (tracks, kind) if with_kind else tracks


# altitude and climb rate from quantized, sometimes jumping (baro vs geometric) reports
# drops points far from the running median, then a local-linear gaussian fit at each report time
def smooth_alt(t, alt, sigma=10.0, jump=60.0):
    def band(half):  # padded neighbour indices within +-half seconds
        lo, hi = np.searchsorted(t, t - half), np.searchsorted(t, t + half, "right")
        j = lo[:, None] + np.arange((hi - lo).max())
        return np.minimum(j, len(t) - 1), j < hi[:, None]

    j, m = band(20.0)
    med = np.nanmedian(np.where(m, alt[j], np.nan), 1)
    good = np.abs(alt - med) < jump
    if good.sum() < 2:
        good[:] = True
    j, m = band(4 * sigma)
    m &= good[j]
    x, y = t[j] - t[:, None], alt[j]
    w = np.exp(-0.5 * (x / sigma) ** 2) * m
    s0, s1, s2 = w.sum(1), (w * x).sum(1), (w * x * x).sum(1)
    b0, b1 = (w * y).sum(1), (w * x * y).sum(1)
    det = s0 * s2 - s1 * s1
    ok = det > 1e-9 * s0 * s0 + 1e-12
    some = s0 > 1e-6
    u = np.where(ok, (s2 * b0 - s1 * b1) / np.where(ok, det, 1), np.where(some, b0 / np.where(some, s0, 1), alt))
    vu = np.where(ok, (s0 * b1 - s1 * b0) / np.where(ok, det, 1), 0.0)
    return u, vu


# split on gaps, resample each piece at dt with a hermite spline (pos + velocity)
def segments(tracks, max_gap=120.0, dt=1.0, kind=None, smooth=True):
    out = []
    for icao, a in tracks.items():
        t = a[:, 0]
        if len(t) < 2:
            continue
        a = a[np.concatenate([[True], np.diff(t) > 0.05])]
        breaks = np.where(np.diff(a[:, 0]) > max_gap)[0] + 1
        for s in np.split(a, breaks):
            if len(s) < 2 or s[-1, 0] - s[0, 0] < 30:
                continue
            seg = _resample(s, dt, smooth)
            if seg is not None:
                seg["icao"] = icao
                seg["kind"] = (kind or {}).get(icao, "unknown")
                out.append(seg)
    return out


def _resample(s, dt, smooth=True):
    t, lat, lon, alt, gs, trk, vr = s.T
    if smooth and np.isfinite(alt).all() and len(t) >= 3:
        alt, vr = smooth_alt(t, alt)
    lat0, lon0 = lat[0], lon[0]
    e, n, u = to_enu(lat, lon, alt, lat0, lon0)
    # velocities: reported where available, finite differences otherwise
    fd = lambda x: np.gradient(x, t)
    ve = np.where(np.isfinite(gs + trk), gs * np.sin(np.radians(trk)), fd(e))
    vn = np.where(np.isfinite(gs + trk), gs * np.cos(np.radians(trk)), fd(n))
    vu = np.where(np.isfinite(vr), vr, fd(u))
    if not np.all(np.isfinite(e + n + u + ve + vn + vu)):
        return None
    tt = np.arange(t[0], t[-1], dt)
    sp = [CubicHermiteSpline(t, x, v) for x, v in ((e, ve), (n, vn), (u, vu))]
    E, N, U = (p(tt) for p in sp)
    la, lo = from_enu(E, N, lat0, lon0)
    VE, VN, VU = (p.derivative()(tt) for p in sp)
    return dict(t=tt, lat=la, lon=lo, alt=U, ve=VE, vn=VN, vu=VU)


def parse_time(s):
    from .timeutil import parse_time as _p
    return _p(s)
