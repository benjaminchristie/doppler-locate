# grid search without ridge tracing
# scores each location by spectrogram energy along the doppler curves nearby aircraft would produce
import ctypes
import functools
import os
import subprocess

import numpy as np
from scipy.ndimage import maximum_filter1d, median_filter
from scipy.spatial import cKDTree

from . import audio
from .model import CENTS, from_enu, to_enu

_DIR = os.path.dirname(os.path.abspath(__file__))
INVALID = np.iinfo(np.int16).min


@functools.lru_cache(None)
def _kernel():
    src, so = os.path.join(_DIR, "mf.cpp"), os.path.join(_DIR, "_mf.so")
    if not os.path.exists(so) or os.path.getmtime(so) < os.path.getmtime(src):
        tmp = f"{so}.{os.getpid()}"
        subprocess.run(["g++", "-O3", "-march=native", "-fopenmp", "-shared", "-fPIC", src, "-o", tmp],
                       check=True)
        os.replace(tmp, so)  # atomic, so parallel runs don't load a half-written file
    lib = ctypes.CDLL(so)
    P = np.ctypeslib.ndpointer
    f32, f64, i32 = (P(t, flags="C") for t in (np.float32, np.float64, np.int32))
    I, D = ctypes.c_int, ctypes.c_double
    lib.mf_seg.argtypes = [f32, I, I, f64, I, i32, I, f64, f64, f64, f64, f64, f64, I, D, D, f64, f64, I,
                           f64, D, D, D, I, i32, I, D, D, f32, i32, i32, f32, i32]
    lib.fringe_table.argtypes = [f32, I, I, D, D, D, D, I, I, f32, i32]
    lib.sweep_bank.argtypes = [f32, I, I, i32, i32, I, I, i32, I, I, i32, f32, i32, i32]
    lib.fringe_seg.argtypes = [f32, i32, I, I, D, I, f64, I, i32, I, f64, f64, f64, f64, f64, f64, I, D, D,
                               f64, f64, I, f64, D, D, D, D, f64, I, f32, i32, f32]
    lib.mf_each.argtypes = [f32, I, I, D, D, i32, I, f64, f64, f64, f64, f64, f64, I, D, D, f64, f64, f64, f64, I,
                            D, D, D, I, f32, i32]
    lib.sub_median.argtypes = [f32, I, I, I]
    lib.link_ridges.argtypes = [i32, I, f64, f32, D, D, I, I, i32, i32]
    lib.link_ridges.restype = I
    lib.line_fit.argtypes = [f64, f64, I, D, f64, I, f64, f64, D, f64]
    lib.ridge_fit.argtypes = [f64, f64, f64, f64, f64, f64, I, D, D, f64, f64, I, f64, f64, f64, I, f64, I, D, D, D,
                              f64, f64]
    return lib


# adds the per-frame comb table (see mf.cpp fringe_table) to a fringe dict(spec, hs)
def fringe_prep(fringe, Q=4):
    if "table" in fringe:
        return fringe
    ft, ff, FL, fsig = fringe["spec"]
    bpl = CENTS / (1200 * np.log2(ff[1] / ff[0]))
    blo, bhi = np.log(1.05) * bpl, np.log(ff[-1] * 0.95 / ff[0]) * bpl
    smin = np.log(300.0 / (2 * max(fringe["hs"])) / ff[0]) * bpl  # widest comb: c = 300, overhead
    NS = int(np.ceil((bhi - smin) * Q)) + 1
    R = np.empty((len(ft), NS), np.float32)
    N = np.empty((len(ft), NS), np.int32)
    _kernel().fringe_table(FL, FL.shape[0], FL.shape[1], bpl, blo, bhi, smin, NS, Q, R, N)
    return dict(fringe, table=(R, N, float(smin), Q))


# whitened dB on a log-frequency axis, mean-removed; returns times, freqs, L[t, k], sigma
# split: keep tones, drop bursts (wind, handling) with a soft mask from a median along time
# (tones are horizontal) against one along frequency (bursts are vertical), then remove each frame's level
def logspec(x, sr, fmin=50, fmax=600, cents=8, win_s=1.0, hop_s=0.5, dilate=True, split=False):
    times, fr, P = audio.stft_band(x, sr, win_s, hop_s, fmin * 0.9, fmax * 1.1)
    if split:
        Ht = median_filter(P, size=(1, max(3, int(round(4.5 / hop_s)) | 1)))
        Pf = median_filter(P, size=(max(3, int(round(8.0 / (fr[1] - fr[0]))) | 1), 1))
        P = P * Ht ** 2 / (Ht ** 2 + Pf ** 2 + 1e-30)
        dB = 10 * np.log10(P + 1e-20)
        dB -= np.median(dB, 0, keepdims=True)
        W = np.clip(dB - np.median(dB, 1, keepdims=True), -5, 25)
    else:
        W = np.clip(audio.whiten_db(P), -5, 25)
    freqs = fmin * 2 ** (np.arange(int(1200 * np.log2(fmax / fmin) / cents)) * cents / 1200)
    pos = np.clip((freqs - fr[0]) / (fr[1] - fr[0]), 0, len(fr) - 1.001)
    i = pos.astype(int)
    w = (pos - i)[:, None]
    L = ((1 - w) * W[i] + w * W[i + 1]).T  # (F, K)
    if dilate:
        L = maximum_filter1d(L, 3, axis=1)  # tolerate a little pitch drift
    L = L - L.mean()
    return times, freqs, np.ascontiguousarray(L, np.float32), float(L.std())


# fraction of independent frames: 1 / (1 + 2 sum of lag correlations), median over frequency bins
def _indep(L, lags=10):
    A = L - L.mean(0)
    v = (A * A).sum(0) + 1e-12
    rho = [np.median((A[k:] * A[:-k]).sum(0) / v) for k in range(1, min(lags, len(L) - 1) + 1)]
    return 1 / (1 + 2 * sum(max(r, 0) for r in rho))


# _indep of the plain front end on white noise, which z thresholds were tuned on
@functools.lru_cache(None)
def _indep_ref(sr, fmin, fmax, cents, win_s, hop_s):
    x = np.random.default_rng(0).normal(size=int(sr * max(120, 60 * win_s)))
    return _indep(logspec(x, sr, fmin, fmax, cents, win_s, hop_s)[2])


class Grid:
    def __init__(self, bbox, step):
        self.lat0, self.lon0 = (bbox[0] + bbox[1]) / 2, (bbox[2] + bbox[3]) / 2
        e0, n0, _ = to_enu(bbox[0], bbox[2], 0, self.lat0, self.lon0)
        e1, n1, _ = to_enu(bbox[1], bbox[3], 0, self.lat0, self.lon0)
        self.xs = np.arange(e0, e1 + 1e-6, step)
        self.ys = np.arange(n0, n1 + 1e-6, step)
        X, Y = np.meshgrid(self.xs, self.ys)
        self.x, self.y = X.ravel(), Y.ravel()
        self.shape = X.shape

    def latlon(self, x, y):
        return from_enu(x, y, self.lat0, self.lon0)


def _seg_enu(seg, lat0, lon0):
    e, n, u = to_enu(seg["lat"], seg["lon"], seg["alt"], lat0, lon0)
    f = lambda a: np.ascontiguousarray(a, np.float64)
    return dict(t0=seg["t"][0], dt=seg["t"][1] - seg["t"][0], e=f(e), n=f(n), u=f(u),
                ve=f(seg["ve"]), vn=f(seg["vn"]), vu=f(seg["vu"]), icao=seg["icao"], kind=seg.get("kind", "unknown"))


# segments in local enu, optionally only some kinds; pass to score(se=...) to skip redoing it
def prepare(segs, lat0, lon0, sources=None):
    if sources is not None:  # e.g. only aircraft that make clean tones
        segs = [s for s in segs if s.get("kind", "unknown") in sources]
    return [_seg_enu(s, lat0, lon0) for s in segs]


def _interp(s, tau):
    pos = (tau - s["t0"]) / s["dt"]
    bad = (pos < 0) | (pos > len(s["e"]) - 1.001)
    pos = np.clip(pos, 0, len(s["e"]) - 1.001)
    i = pos.astype(int)
    w = pos - i
    f = lambda a: a[i] * (1 - w) + a[i + 1] * w
    return f(s["e"]), f(s["n"]), f(s["u"]), (f(s["ve"]), f(s["vn"]), f(s["vu"])), bad


# predicted doppler shift (log-freq bins) for grid points x frames; INVALID where silent
def shifts(s, gx, gy, gz, T, c, r_max, cents):
    T = T[None, :]
    e, n, u, _, bad = _interp(s, np.broadcast_to(T, (len(gx), T.shape[1])))
    tau = T - np.sqrt((e - gx[:, None]) ** 2 + (n - gy[:, None]) ** 2 + (u - gz) ** 2) / c
    for _ in range(4):  # retarded time, converges as (v/c)^k
        e, n, u, V, bad = _interp(s, tau)
        dx, dy, dz = e - gx[:, None], n - gy[:, None], u - gz
        r = np.sqrt(dx * dx + dy * dy + dz * dz)
        tau = T - r / c
    rdot = (dx * V[0] + dy * V[1] + dz * V[2]) / r
    sh = np.rint(-np.log1p(rdot / c) * CENTS / cents)
    sh[bad | (r > r_max)] = INVALID
    return sh.astype(np.int16)


# sorted unique reception times over all clock offsets, and each (offset, frame)'s index into them
def _times(start, dts, frames):
    T = np.round(start + np.asarray(dts, float)[:, None] + frames[None, :], 6)
    U, inv = np.unique(T, return_inverse=True)
    return np.ascontiguousarray(U), np.ascontiguousarray(inv.reshape(T.shape), np.int32)


# indices of points within r_max (scalar or per path point) of any (e, n); a Grid or a cKDTree
def _near(grid_pts, e, n, r_max):
    rr = np.broadcast_to(np.asarray(r_max, float), e.shape)
    r_max = float(rr.max())
    if isinstance(grid_pts, Grid):
        g = grid_pts
        step = g.xs[1] - g.xs[0] if len(g.xs) > 1 else 1.0
        ix0 = max(0, int(np.floor((e.min() - r_max - g.xs[0]) / step)))
        ix1 = min(len(g.xs), int(np.ceil((e.max() + r_max - g.xs[0]) / step)) + 1)
        iy0 = max(0, int(np.floor((n.min() - r_max - g.ys[0]) / step)))
        iy1 = min(len(g.ys), int(np.ceil((n.max() + r_max - g.ys[0]) / step)) + 1)
        if ix0 >= ix1 or iy0 >= iy1:
            return np.empty(0, int)
        IY, IX = np.meshgrid(np.arange(iy0, iy1), np.arange(ix0, ix1), indexing="ij")
        cand = (IY * len(g.xs) + IX).ravel()
    else:
        hits = grid_pts.query_ball_point(np.c_[e, n], rr)
        return np.unique(np.concatenate(hits).astype(int)) if len(hits) else np.empty(0, int)
    gx, gy = _xy(grid_pts)
    keep = np.zeros(len(cand), bool)
    for a in range(0, len(cand), 20000):
        c = cand[a:a + 20000]
        d2 = (gx[c, None] - e) ** 2 + (gy[c, None] - n) ** 2
        keep[a:a + 20000] = (d2 < rr ** 2).any(1)
    return cand[keep]


def _xy(grid_pts):
    return (grid_pts.x, grid_pts.y) if isinstance(grid_pts, Grid) else grid_pts


# indices of grid points within r_max of this aircraft at some point in the clip
def _heard(s, where, start, dts, dur, elev, r_max, c):
    tt = np.arange(start + min(dts) - r_max / c, start + max(dts) + dur, 5.0)
    e, n, u, _, bad = _interp(s, tt)
    h = u - elev  # only heard within sqrt(r_max^2 - h^2) horizontally
    ok = ~bad & (h < r_max)
    if not ok.any():
        return np.empty(0, int)
    return _near(where, e[ok], n[ok], np.sqrt(np.maximum(r_max ** 2 - np.maximum(h[ok], 0) ** 2, 0)) + 200)


# mf_seg for one aircraft: best, arg, n frames, closest range, pitch spread (bins), each [dts, points]
# frames weigh min(1, r_ref / range) (loudness) times min(1, pitch rate / sweep_ref cents/s); 0 turns either off
def _tone(L, U, IU, s, gx, gy, elev, c, r_max, cents, harmonics, hoff, r_ref=1000.0, sweep_ref=4.0):
    ND, m = IU.shape[0], len(gx)
    out = [np.empty((ND, m), t) for t in (np.float32, np.int32, np.int32, np.float32, np.int32)]
    _kernel().mf_seg(L, L.shape[0], L.shape[1], U, len(U), IU, ND, *(s[k] for k in ("e", "n", "u", "ve", "vn", "vu")),
                     len(s["e"]), s["t0"], s["dt"], np.ascontiguousarray(gx, np.float64),
                     np.ascontiguousarray(gy, np.float64), m,
                     np.ascontiguousarray(np.broadcast_to(np.asarray(elev, np.float64), (m,))), c, r_max, CENTS / cents,
                     harmonics, hoff,
                     10, r_ref, sweep_ref / cents, *out)
    return out


# (a, b) so that a * t + b puts raw tone scores t on the scale the thresholds were set on
# null: predicted curves against the time-reversed spectrogram (keeps its noise, breaks real alignments),
# compared with the same curves on white noise; only discounts, a busy or windy clip has a higher null
def calibrate(spec, se, grid, start, c, sr, elev=0.0, r_max=8000.0, fmin=50, fmax=600, cents=8, win_s=1.0,
              hop_s=0.5, harmonics=1, min_sweep=50.0, r_ref=1000.0, sweep_ref=4.0, n=3000, seed=0, **_):
    times, freqs, L, sigma = spec
    F = len(times)
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=int(((F - 1) * hop_s + win_s) * sr) + int(hop_s * sr))
    Lw = np.ascontiguousarray(logspec(noise, sr, fmin, fmax, cents, win_s, hop_s)[2][:F])
    if len(Lw) < F:
        return 1.0, 0.0
    sw = float(Lw.std())
    Lr = np.ascontiguousarray(L[::-1])
    hoff = np.rint(CENTS * np.log2(np.arange(1, harmonics + 1)) / cents).astype(np.int32)
    U, IU = _times(start, (0.0,), times)
    dur = times[-1] + win_s
    k = max(20, n // max(len(se), 1))
    tr, tw = [], []
    for s in se:
        idx = _heard(s, grid, start, (0.0,), dur, elev, r_max, c)
        if len(idx) == 0:
            continue
        idx = rng.choice(idx, min(k, len(idx)), replace=False)
        for LL, sg, out in ((Lr, sigma, tr), (Lw, sw, tw)):
            best, _, nv, _, spread = _tone(LL, U, IU, s, grid.x[idx], grid.y[idx], elev, c, r_max, cents, harmonics,
                                           hoff, r_ref, sweep_ref)
            ok = (nv >= 10) & (spread * cents >= min_sweep) & (best > -1e29)
            out.append((best * np.sqrt(hop_s / win_s) / sg)[ok])
    tr, tw = (np.concatenate(v) if v else np.empty(0) for v in (tr, tw))
    if min(len(tr), len(tw)) < 200:
        return 1.0, 0.0
    spread_ = lambda v: max(np.subtract(*np.percentile(v, [75, 25])) / 1.349, 1e-6)
    a = min(1.0, spread_(tw) / spread_(tr))
    b = float(np.median(tw) - a * np.median(tr))
    if a == 1.0:
        b = min(b, 0.0)
    return a, b


# evidence at grid points for each clock offset; returns Z[dt, g] and best contributor info
# cal: (a, b) from calibrate(); r_ref, sweep_ref: frame weights, see _tone
# min_sweep: aircraft whose predicted pitch moves less than this (cents) add no evidence, e.g. parked
# info["n"] counts aircraft above z_min at each point
def score(x, sr, start, segs, grid_pts, lat0, lon0, c, elev=0.0, dts=(0.0,), fmin=50, fmax=600,
          cents=8, win_s=1.0, hop_s=0.5, r_max=8000.0, harmonics=1, spec=None, chunk=20000,
          z_min=3.0, loud_r=2000.0, absent_penalty=0.0, track=True, sources=None, progress=None,
          fringe=None, fringe_kinds=("jet",), fringe_rmax=6000.0, se=None, cal=(1.0, 0.0), min_sweep=50.0,
          r_ref=1000.0, sweep_ref=4.0):
    times, freqs, L, sigma = spec or logspec(x, sr, fmin, fmax, cents, win_s, hop_s)
    F, K = L.shape
    hoff = np.rint(CENTS * np.log2(np.arange(1, harmonics + 1)) / cents).astype(np.int32)
    n_eff = hop_s / win_s  # frames overlap
    bonus = np.sqrt(2 * np.log(K))  # expected max of K noise draws
    lib = _kernel()
    if fringe:
        fringe = fringe_prep(fringe)
        ft, ff, FL, fsig = fringe["spec"]
        FR, FN, smin, Q = fringe["table"]
        hs = np.ascontiguousarray(fringe["hs"], np.float64)
        f_bonus = np.sqrt(2 * np.log(len(hs)))
        fcents = 1200 * np.log2(ff[1] / ff[0])
        UF, IF = _times(start, dts, ft)
    gx, gy = _xy(grid_pts)
    G = len(gx)
    where = grid_pts if isinstance(grid_pts, Grid) else cKDTree(np.c_[gx, gy])
    ev = np.broadcast_to(np.asarray(elev, np.float64), (G,))  # ground height, one or per point
    dur = times[-1] + win_s
    if se is None:
        se = prepare(segs, lat0, lon0, sources)
    se = [s for s in se if s["t0"] < start + max(dts) + dur + 60
          and s["t0"] + s["dt"] * (len(s["e"]) - 1) > start + min(dts) - r_max / c - 60]
    Z = np.zeros((len(dts), G), np.float32)
    if track:
        top = np.zeros((len(dts), G), np.float32)
        who = np.full((len(dts), G), -1, np.int32)
        f0 = np.zeros((len(dts), G), np.float32)
        count = np.zeros((len(dts), G), np.int16)
    U, IU = _times(start, dts, times)
    ND = len(dts)
    for j, s in enumerate(se):
        if progress:
            progress(j / max(len(se), 1))
        idx = _heard(s, where, start, dts, dur, float(ev.min()) if G else 0.0, r_max, c)
        if len(idx) == 0:
            continue
        ez = np.ascontiguousarray(ev[idx])
        arrs = [s[k] for k in ("e", "n", "u", "ve", "vn", "vu")]
        gxi, gyi = np.ascontiguousarray(gx[idx], np.float64), np.ascontiguousarray(gy[idx], np.float64)
        for a in range(0, len(idx), chunk):
            ii = idx[a:a + chunk]
            m = len(ii)
            best, arg, nv, rmin, spread = _tone(L, U, IU, s, gxi[a:a + chunk], gyi[a:a + chunk], ez[a:a + chunk], c, r_max,
                                                cents, harmonics, hoff, r_ref, sweep_ref)
            z = cal[0] * best * np.sqrt(n_eff) / sigma + cal[1] - bonus
            z[(nv < 10) | (spread * cents < min_sweep) | (best < -1e29)] = 0
            gain = np.maximum(z - z_min, 0)  # noise matches add nothing
            if fringe and s.get("kind") in fringe_kinds:
                fb = np.empty((ND, m), np.float32)
                fn = np.empty((ND, m), np.int32)
                frm = np.empty((ND, m), np.float32)
                lib.fringe_seg(FR, FN, FR.shape[0], FR.shape[1], smin, Q, UF, len(UF), IF, ND, *arrs, len(arrs[0]),
                               s["t0"], s["dt"], gxi[a:a + chunk], gyi[a:a + chunk], m, ez[a:a + chunk], c,
                               fringe_rmax, float(ff[0]), CENTS / fcents, hs, len(hs), fb, fn, frm)
                zf = np.where(fn > 0, fb * np.sqrt(np.maximum(fn, 1)) / (np.sqrt(1.5) * fsig) - f_bonus, 0)
                gain = gain + np.maximum(zf - z_min, 0)
                z = np.maximum(z, zf)
            Z[:, ii] += gain
            if absent_penalty:  # a tonal source passing this close inside the clip should be heard
                Z[:, ii] -= absent_penalty * ((rmin < loud_r) & (nv >= 0.3 * F) & (z < 1))
            if track:
                count[:, ii] += z >= z_min
                cur = top[:, ii]
                better = z > cur
                cur[better] = z[better]
                top[:, ii] = cur
                w = who[:, ii]
                w[better] = j
                who[:, ii] = w
                fo = f0[:, ii]
                fo[better] = freqs[np.maximum(arg[better], 0)]
                f0[:, ii] = fo
    info = dict(icao=[s["icao"] for s in se], spec=(times, freqs, L, sigma))
    if track:
        info.update(top=top, who=who, f0=f0, n=count)
    return Z, info


def _km(a, b):
    e, n, _ = to_enu(a["lat"], a["lon"], 0, b["lat"], b["lon"])
    return float(np.hypot(e, n)) / 1000


# coarse grid over bbox, then fine grids around the best peaks (and any extra lat/lons)
# jets: also score ground-reflection fringes for jets; mic_step: spacing of mic heights tried (m)
# split: suppress bursts in the tone spectrogram, see logspec; "auto" only when the noise is bursty
# (frames much less independent than white noise), on steady noise it costs more than it gains
def search(x, sr, start, segs, bbox, c, elev=0.0, clock_tol=2.0, step=250.0, fine_step=40.0,
           n_peaks=24, log=print, extra=(), coarse_zmin=1.5, progress=None, jets=False, mic_step=0.05,
           split="auto", sources=None, **kw):
    if jets:
        kw["fringe"] = fringe_prep(dict(spec=logspec(x, sr, 100, min(2000, 0.45 * sr), 16, 0.25, 0.25,
                                                     dilate=False), hs=np.arange(0.8, 2.0 + 1e-6, mic_step)))
    grid = Grid(bbox, step)
    se = prepare(segs, grid.lat0, grid.lon0, sources)
    # a clock error shifts the match along the track by ~speed*dt, so the coarse grid only needs
    # offsets about one cell apart (~step/80 m/s); the fine stage sweeps the clock properly
    dt_step = max(1.0, step / 80.0)
    k = int(np.floor(clock_tol / dt_step + 0.5))
    dts = np.unique(np.clip(np.arange(-k, k + 1) * dt_step, -clock_tol, clock_tol)) if clock_tol > 0 \
        else np.array([0.0])
    sp = (kw.get("fmin", 50), kw.get("fmax", 600), kw.get("cents", 8), kw.get("win_s", 1.0), kw.get("hop_s", 0.5))
    spec = logspec(x, sr, *sp, split=split is True)
    if split == "auto":
        indep = _indep(spec[2]) / _indep_ref(sr, *sp)
        if indep < 0.6:
            spec = logspec(x, sr, *sp, split=True)
        log(f"noise: {indep:.2f} of white-noise frame independence, burst suppression "
            f"{'on' if indep < 0.6 else 'off'}")
    a, b = calibrate(spec, se, grid, start, c, sr, elev, **kw)
    kw["cal"] = (a, b)
    log(f"calibration: tone scores x {a:.2f} {b:+.2f}")
    log(f"coarse: {grid.shape[1]}x{grid.shape[0]} points, {len(dts)} clock offsets")
    ckw = dict(kw, z_min=coarse_zmin) if coarse_zmin is not None else kw
    prog = progress or (lambda f, m: None)
    Z, _ = score(x, sr, start, None, grid, grid.lat0, grid.lon0, c, elev, dts, spec=spec, track=False,
                 progress=lambda f: prog(0.75 * f, "coarse grid"), se=se, **ckw)
    zmax = Z.max(0)

    # peaks at least 1.5 km apart
    order = np.argsort(-zmax)
    peaks = []
    for g in order[:5000]:
        if zmax[g] <= 0:
            break
        if all((grid.x[g] - grid.x[p]) ** 2 + (grid.y[g] - grid.y[p]) ** 2 > 1500 ** 2 for p in peaks):
            peaks.append(g)
        if len(peaks) >= n_peaks:
            break

    def clock_range(center, half, stepsz):
        if clock_tol <= 0:
            return np.array([0.0])
        return np.unique(np.clip(np.arange(center - half, center + half + 1e-9, stepsz), -clock_tol, clock_tol))

    # best (x, y, clock) on a square grid around each center; one kernel pass for all centers
    def best_on(centers, half, spacing, dt_sets, msg=""):
        off = np.arange(-half, half + 1e-6, spacing)
        OX, OY = (a.ravel() for a in np.meshgrid(off, off))
        m = len(OX)
        X = (np.array([p[0] for p in centers])[:, None] + OX).ravel()
        Y = (np.array([p[1] for p in centers])[:, None] + OY).ravel()
        fdts = np.unique(np.concatenate(dt_sets))
        prog(0.8, msg)
        Zf, inf = score(x, sr, start, None, (X, Y), grid.lat0, grid.lon0, c, elev, fdts, spec=spec, se=se, **kw)
        out = []
        for i in range(len(centers)):
            Zi = Zf[:, i * m:(i + 1) * m].copy()
            Zi[~np.isin(fdts, dt_sets[i])] = -np.inf  # each center keeps its own clock range
            di, k = np.unravel_index(np.argmax(Zi), Zi.shape)
            j = i * m + k
            w = inf["who"][di, j]
            out.append(dict(x=X[j], y=Y[j], dt=fdts[di], score=float(Zi[di, k]), support=int(inf["n"][di, j]),
                            icao=inf["icao"][w] if w >= 0 else None, f0=float(inf["f0"][di, j])))
        return out

    # extras (e.g. the true spot) are refined exactly like a peak, starting from their nearest cell
    starts = list(peaks)
    for la, lo in extra:
        ex, ny, _ = to_enu(la, lo, 0, grid.lat0, grid.lon0)
        starts.append(int(np.argmin((grid.x - ex) ** 2 + (grid.y - ny) ** 2)))
    found = []
    if starts:
        # two levels: ~60 m and 1 s around the coarse cell, then fine_step and 0.25 s
        sp1 = max(fine_step, step / 4)
        l1 = best_on([(grid.x[g], grid.y[g]) for g in starts], step * 3, sp1,
                     [clock_range(dts[np.argmax(Z[:, g])], max(2.0, dt_step / 2 + 1), 1.0) for g in starts],
                     msg=f"refining {len(starts)} peaks")
        l2 = best_on([(r["x"], r["y"]) for r in l1], 1.5 * sp1, fine_step,
                     [clock_range(r["dt"], 1.0, 0.25) for r in l1], msg="refining peaks, fine")
        for r in l2:
            la, lo = grid.latlon(r["x"], r["y"])
            found.append(dict(lat=float(la), lon=float(lo), score=r["score"], clock_offset=float(r["dt"]),
                              icao=r["icao"], f0=r["f0"], support=r["support"]))
    refined, extras = found[:len(peaks)], found[len(peaks):]
    results = []
    for r in sorted(refined, key=lambda r: -r["score"]):
        # coarse peaks can refine onto the same spot
        if all(_km(r, q) > 0.15 for q in results):
            results.append(r)
    for r in results:  # how much better than the best spot more than 1 km away; near 1 means ambiguous
        other = [q["score"] for q in results if _km(r, q) > 1.0]
        r["margin"] = r["score"] / max(max(other, default=0.0), 0.5)
    lat, lon = grid.latlon(grid.x, grid.y)
    return results, dict(Z=zmax.reshape(grid.shape), lat=lat.reshape(grid.shape),
                         lon=lon.reshape(grid.shape), bbox=bbox, step=step, spec=spec,
                         dts=dts, extra=extras)
