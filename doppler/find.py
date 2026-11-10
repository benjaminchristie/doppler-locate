# event-based search for where and when a recording was made
# 1. events.track_events measures each pass: closest-approach time, pitch, speed, slant distance
# 2. candidates: strips beside each matching flight's track, as 200 m cells and 3 s clock-offset bins
# 3. z screen (mf.cpp), ridge fit, then refine the best (both sides of a straight pass)
# 4. each place is scored by everything it explains: tracked events, faint passes found along their
#    predicted curves (heard), minus close passes that are missing
# 5. start known and the events don't settle it: add the peaks of the grid search (heatmap.search)
import time
import warnings
from types import SimpleNamespace

import numpy as np
from scipy.spatial import cKDTree

from . import dem as demmod
from . import heatmap
from .model import CENTS, from_enu, sound_speed

TONAL = ("prop", "heli", "unknown")


# distances [min, max] a flight at speeds sp could pass at to make event e, and pass time
def _reach(e, sp, spread=3.0, r_max=8000.0):
    prof = np.array(e.get("profile") or [(e["v"], e["d"], e["t"])], float)
    k = np.abs(np.log(np.maximum(sp, 1.0))[:, None] - np.log(prof[:, 0])[None, :]).argmin(1)
    d0 = prof[k, 1] * (sp / prof[k, 0]) ** 2
    return np.maximum(d0 / spread, 100.0), np.minimum(d0 * spread, r_max), prof[k, 2]


# cells and offset bins from which flight s could be event e's pass; returns (ix, iy, ob)
# distance is scanned by a factor spread either way for turning or climbing planes
def _strips(e, s, S, tol, dgrid, cell, obin, c, r_max, box=None, spread=3.0):
    n = len(s["e"])
    tt = s["t0"] + s["dt"] * np.arange(n)
    sp = np.sqrt(s["ve"] ** 2 + s["vn"] ** 2 + s["vu"] ** 2)
    dlo, dhi, T0 = _reach(e, sp, spread, r_max)
    T0 = S + T0
    sel = (sp >= e["v_lo"]) & (sp <= e["v_hi"]) & (dhi > dlo)
    if tol is not None:  # start unknown: anywhere on the track
        sel &= (tt >= T0 - tol - 10 - dhi / c) & (tt <= T0 + tol + 10 - dlo / c)
    if not sel.any():
        return None
    idx = np.flatnonzero(sel)
    step = max(1, int(cell / 2 / max(np.median(sp[idx]), 1.0)))
    idx = idx[::step]
    x, y, u = s["e"][idx], s["n"][idx], s["u"][idx]
    vh = np.hypot(s["ve"][idx], s["vn"][idx]) + 1e-6
    nx, ny = -s["vn"][idx] / vh, s["ve"][idx] / vh
    dlo, dhi, T0 = dlo[idx][:, None], dhi[idx][:, None], T0[idx][:, None]
    lat = np.arange(0, dhi.max() + cell / 2, cell / 2)
    out = []
    for side in (-1, 1):
        L = side * lat[None, :]
        px, py = x[:, None] + L * nx[:, None], y[:, None] + L * ny[:, None]
        r = np.sqrt(L ** 2 + (u[:, None] - dgrid(px, py)) ** 2)
        ok = (r >= dlo) & (r <= dhi)
        if side == -1:
            ok[:, 0] = False  # lateral 0 only once
        o = tt[idx][:, None] + r / c - T0
        if tol is not None:
            ok &= np.abs(o) <= tol + obin
        if box is not None:
            ok &= (px >= box[0]) & (px <= box[1]) & (py >= box[2]) & (py <= box[3])
        out.append((px[ok], py[ok], o[ok]))
    px, py, o = (np.concatenate(v) for v in zip(*out))
    if not len(px):
        return None
    return np.floor(px / cell).astype(np.int64), np.floor(py / cell).astype(np.int64), np.round(o / obin).astype(np.int64)


# air cools with height, so sound from a high aircraft is slower than at the ground
# each track point gets the sound speed at the mean temperature of the air below it. that one speed sets
# the delay and, for level flight, the doppler shift too: refraction keeps the horizontal slowness of a ray
def _air(s, temp_c, lapse, ground):
    h = np.maximum(s["u"] - ground, 0.0) / 1000.0
    s["cbar"] = sound_speed(temp_c - lapse * h / 2)


# sound speed for flight s's pass around unix time T; c if it has no air model
def _c(s, T, c):
    if "cbar" not in s:
        return c
    return float(s["cbar"][int(np.clip((T - s["t0"]) / s["dt"], 0, len(s["cbar"]) - 1))])


# start known: cells near flight s where its exact curve fits event e's track, and the clock that fits
# a straight-pass fit of a short or bent track puts its strip in the wrong place a third of the time;
# this tries every cell within reach of where the plane was. returns (ix, iy, ob) or None
def _sweep(e, s, S, tol, dgrid, cell, obin, c, r_max, box=None, min_score=2.0, max_pts=40):
    ts = np.asarray(_seed(e)[0])
    n = len(s["e"])
    i = int(round((S + ts[len(ts) // 2] - s["t0"]) / s["dt"]))
    if i < -30 or i > n + 30:
        return None
    i = int(np.clip(i, 0, n - 1))
    sp = np.sqrt(s["ve"][i] ** 2 + s["vn"][i] ** 2 + s["vu"][i] ** 2)
    if not e["v_lo"] <= sp <= e["v_hi"]:
        return None
    prof = e.get("profile") or [(e["v"], e["d"], e["t"])]
    reach = float(np.clip(3.0 * max(p[1] for p in prof), 1500.0, r_max))
    x0, y0 = s["e"][i], s["n"][i]
    off = np.arange(-reach, reach + 1, cell)
    X, Y = (a.ravel() for a in np.meshgrid(np.floor(x0 / cell) * cell + cell / 2 + off, np.floor(y0 / cell) * cell + cell / 2 + off))
    ok = (X - x0) ** 2 + (Y - y0) ** 2 <= reach ** 2
    if box is not None:
        ok &= (X >= box[0]) & (X <= box[1]) & (Y >= box[2]) & (Y <= box[3])
    if not ok.any():
        return None
    X, Y = X[ok], Y[ok]
    dts = np.arange(-tol, tol + 1e-6, 1.0 if tol <= 12 else 2.0)
    sc = _ridge_score(_seed(e), s, X, Y, dgrid(X, Y), dts, S, c, max_pts=max_pts)
    best = sc.max(0)
    keep = best >= max(min_score, 0.6 * best.max())
    if not keep.any():
        return None
    return (np.floor(X[keep] / cell).astype(np.int64), np.floor(Y[keep] / cell).astype(np.int64),
            np.round(dts[sc.argmax(0)[keep]] / obin).astype(np.int64))


def _key(ix, iy, ob):
    return ((ix + 2 ** 20) << 42) | ((iy + 2 ** 20) << 21) | (ob + 2 ** 20)


# sorted unique (cell, offset) keys event e could be heard from, and one flight for each
def _event_keys(e, flights, S, tol, dgrid, cell, obin, c, r_max, budget, box=None, sweep=True):
    ks, fs, total = [], [], 0
    for fi, s in enumerate(flights):
        got = [r for r in (_strips(e, s, S, tol, dgrid, cell, obin, c, r_max, box),
                           _sweep(e, s, S, tol, dgrid, cell, obin, c, r_max, box) if sweep and tol is not None else None)
               if r is not None]
        if not got:
            continue
        k = np.unique(np.concatenate([_key(*r) for r in got]))
        ks.append(k)
        fs.append(np.full(len(k), fi, np.int32))
        total += len(k)
        if total > budget:
            raise RuntimeError(f"event at {e['t']:.0f} s fits {total:,}+ places")
    if not ks:
        return np.empty(0, np.int64), np.empty(0, np.int32)
    k, f = np.concatenate(ks), np.concatenate(fs)
    k, first = np.unique(k, return_index=True)
    return k, f[first]


# flight of key q in sorted K, also trying neighbouring offset bins; -1 if none
def _lookup(K, F, q):
    out = np.full(len(q), -1, np.int32)
    if not len(K):
        return out
    for d in (0, -1, 1):
        i = np.minimum(np.searchsorted(K, q + d), len(K) - 1)
        hit = (K[i] == q + d) & (out < 0)
        out[hit] = F[i[hit]]
    return out


# how good a seed an event is; with the start unknown, an event any speed fits lands everywhere
def _quality(e, loose_ok=True):
    q = e.get("strength", 1.0) / max(e.get("rms", 3.0), 3.0)
    return q if loose_ok else q / np.sqrt(len(e.get("profile") or [0]))


# (cell, offset) keys that two or more of the strongest events agree on, and the flight for each event
# only the seed events generate places
def candidates(events, flights, S, tol, dgrid, cell=200.0, obin=3.0, c=340.0, r_max=8000.0, n_seed=None,
               budget=3e7, v_max=180.0, box=None, keep=400000, lone=0, sweep=True, over=None, log=print):
    n_seed = n_seed or 6
    seeds, E = [], {}  # cleanest fits first
    for i in sorted(range(len(events)), key=lambda i: -_quality(events[i], tol is not None)):
        e = events[i]
        if e["v_hi"] < 30 or e["v_lo"] > v_max:
            continue
        try:
            E[i] = _event_keys(e, flights, S, tol, dgrid, cell, obin, c, r_max, budget, box, sweep)
        except RuntimeError as err:  # too loose to pin down, try the next
            log(f"  skipped: {err}")
            if over is not None:  # the caller may want a smaller area instead
                over.append(i)
            continue
        log(f"  event at {e['t']:.0f} s: {len(E[i][0]):,} places")
        seeds.append(i)
        if len(seeds) >= n_seed:
            break
    cand = []
    for a in range(len(seeds)):
        for b in range(a + 1, len(seeds)):
            Ka, Fa = E[seeds[a]]
            fb = _lookup(*E[seeds[b]], Ka)
            # one flight can't explain two events less than 60 s apart
            apart = abs(events[seeds[a]]["t"] - events[seeds[b]]["t"]) >= 60
            hit = (fb >= 0) & ((fb != Fa) | apart)
            cand.append(Ka[hit])
    if seeds and sum(len(E[i][0]) for i in seeds) <= keep:
        # few enough to keep all
        C = np.unique(np.concatenate([E[i][0] for i in seeds]))
    else:
        C = np.unique(np.concatenate(cand)) if cand else np.empty(0, np.int64)
        if not len(C) and seeds:  # no two agree: first seed's places
            C = E[seeds[0]][0]
        # two events only agree if both strips have the place and the clock right, which fails a third of the
        # time each; a clean event's own places go in too while they fit, to be checked against the rest later
        room, own = lone, []
        for i in seeds:
            if 0 < len(E[i][0]) <= room:
                own.append(E[i][0])
                room -= len(E[i][0])
        if own:
            C = np.unique(np.concatenate([C] + own))
    if not len(C):
        return None
    Fm = np.stack([_lookup(*E[ei], C) for ei in seeds], 1)
    return dict(keys=C, n=(Fm >= 0).sum(1), flight=Fm, seeds=seeds, cell=cell, obin=obin)


def _dist(a, b):
    from .model import to_enu
    e, n, _ = to_enu(a["lat"], a["lon"], 0, b["lat"], b["lon"])
    return float(np.hypot(e, n))


def _decode(k, cell, obin):
    ix = (k >> 42) - 2 ** 20
    iy = ((k >> 21) & (2 ** 21 - 1)) - 2 ** 20
    ob = (k & (2 ** 21 - 1)) - 2 ** 20
    return (ix + 0.5) * cell, (iy + 0.5) * cell, ob * obin


# track points of flights loud enough to hear, in a KD-tree
def _loud(flights, min_speed=25.0):
    pts, owner, times, up = [], [], [], []
    for fi, s in enumerate(flights):
        if s.get("kind", "unknown") not in TONAL:
            continue
        ok = np.hypot(s["ve"], s["vn"]) >= min_speed
        if not ok.any():
            continue
        pts.append(np.c_[s["e"][ok], s["n"][ok]])
        owner.append(np.full(ok.sum(), fi))
        times.append(s["t0"] + s["dt"] * np.flatnonzero(ok))
        up.append(s["u"][ok])
    if not pts:
        return None
    P = np.concatenate(pts)
    return cKDTree(P), P, np.concatenate(owner), np.concatenate(times), np.concatenate(up)


# for each candidate (x, y, offset): close, fast passes it would hear that no event explains
def silence(cands, events, flights, S, dur, dgrid, c=340.0, r_loud=1500.0, edge=15.0, loud=None):
    loud = loud or _loud(flights)
    if loud is None:
        return [[] for _ in cands]
    tree, P, Ow, Tm, U = loud
    et = np.array([e["t"] for e in events])
    C = np.asarray(cands, float).reshape(-1, 3)
    Z = np.broadcast_to(dgrid(C[:, 0], C[:, 1]), (len(C),))
    out = []
    for (x, y, o), z, hits in zip(C, Z, tree.query_ball_point(C[:, :2], r_loud)):
        missed = []
        hits = np.asarray(hits, np.int64)
        if len(hits):  # the tree may hold a whole day; keep track points from around the clip
            hits = hits[(Tm[hits] >= S + o - 60) & (Tm[hits] <= S + o + dur + 60)]
        if len(hits):
            r = np.sqrt((P[hits, 0] - x) ** 2 + (P[hits, 1] - y) ** 2 + (U[hits] - z) ** 2)
            for fi in np.unique(Ow[hits]):
                m = Ow[hits] == fi
                j = np.argmin(r[m])
                rc = r[m][j]
                if rc > r_loud:
                    continue
                T = Tm[hits][m][j] + rc / c - S - o  # time in the recording of closest approach
                if not edge <= T <= dur - edge:
                    continue
                if len(et) and np.min(np.abs(et - T)) <= 8 + rc / 200:
                    continue
                missed.append((int(fi), round(float(T), 1), round(float(rc))))
        out.append(missed)
    return out


# spectrogram around one event: its pass window and a band around its pitch
def _event_spec(spec, e, c):
    times, freqs, L, sigma = spec
    w = max(30.0, 2.5 * e["d"] / e["v"])
    ts = (times >= e["t"] - w) & (times <= e["t"] + w)
    swing = 1.25 * max(e["v_hi"], 60.0) / c
    fs = (freqs >= e["f0"] * np.exp(-swing) / 1.05) & (freqs <= e["f0"] * np.exp(swing) * 1.05)
    if ts.sum() < 12 or fs.sum() < 8:
        return None
    sub = np.ascontiguousarray(L[np.ix_(ts, fs)])
    return times[ts], freqs[fs], sub, sigma


# z of flight s's exact emission-time curve on event spectrogram es at points (X, Y, Z); returns [dts, points]
def _z(es, s, X, Y, Z, dts, S, c, hop_s=0.5, win_s=1.0):
    times, freqs, L, sigma = es
    cents = 1200 * np.log2(freqs[1] / freqs[0])
    U, IU = heatmap._times(S, dts, times)
    c = _c(s, S + np.mean(dts) + times[len(times) // 2], c)
    best, _, nv, _, _ = heatmap._tone(L, U, IU, s, X, Y, Z, c, 12000.0, cents, 1, np.zeros(1, np.int32), 0.0, 0.0)
    z = best * np.sqrt(hop_s / win_s) / sigma - np.sqrt(2 * np.log(L.shape[1]))
    return np.where((nv >= 10) & (best > -1e29), z, 0.0)


# log(f heard / f source) at ground points (X, Y, Z) and reception times T, nan off the track
# radial velocity at emission time, solved by Newton on tau + r(tau)/c = T
def predict(s, X, Y, Z, T, c):
    c = _c(s, float(np.mean(T)), c)
    tau = T - 3000.0 / c
    for _ in range(6):
        e, n, u, V, _ = heatmap._interp(s, tau)
        dx, dy, dz = e - X, n - Y, u - Z
        r = np.sqrt(dx * dx + dy * dy + dz * dz)
        tau = tau - (tau - T + r / c) / (1 + (dx * V[0] + dy * V[1] + dz * V[2]) / r / c)
    e, n, u, V, bad = heatmap._interp(s, tau)
    dx, dy, dz = e - X, n - Y, u - Z
    r = np.sqrt(dx * dx + dy * dy + dz * dz)
    with np.errstate(invalid="ignore"):  # glitched tracks report speeds past sound
        return np.where(bad, np.nan, -np.log1p((dx * V[0] + dy * V[1] + dz * V[2]) / r / c))


# the track an event was measured on; its points are that track re-traced along the straight-pass fit
def _seed(e):
    return e.get("seed") or e["points"]


# best of the re-traced ridge and the seed track it grew from
def ridge_score(e, s, X, Y, Z, dts, S, c, sigma=10.0, max_pts=120):
    a = _ridge_score(e["points"], s, X, Y, Z, dts, S, c, sigma, max_pts)
    if "seed" in e and len(e["seed"][0]) < len(e["points"][0]):
        a = np.maximum(a, _ridge_score(e["seed"], s, X, Y, Z, dts, S, c, sigma, max_pts))
    return a


# ridge fit of flight s's curve at (X, Y, Z) with clock offsets dts: sqrt(N / 4) * exp(-(rms / sigma)^2 / 2)
# source pitch is free; the true geometry fits to a few cents, look-alikes score 10-20; returns [dts, points]
def _ridge_score(pts, s, X, Y, Z, dts, S, c, sigma=10.0, max_pts=120):
    t, f = (np.asarray(v, np.float64) for v in pts)  # ridge points, harmonics scaled to the fundamental
    n_all = len(t)
    if n_all > max_pts:
        k = np.linspace(0, n_all - 1, max_pts).astype(int)
        t, f = t[k], f[k]
    X, Y, Z = (np.ascontiguousarray(np.broadcast_to(v, np.shape(X)), np.float64) for v in (X, Y, Z))
    dts = np.ascontiguousarray(np.atleast_1d(dts), np.float64)
    c = _c(s, S + dts.mean() + t[len(t) // 2], c)
    rms, frac = np.empty((len(dts), len(X))), np.empty((len(dts), len(X)))
    heatmap._kernel().ridge_fit(*(s[k] for k in ("e", "n", "u", "ve", "vn", "vu")), len(s["e"]), s["t0"], s["dt"],
                                np.ascontiguousarray(t), np.log(f), len(t), X, Y, Z, len(X), dts, len(dts), S, c,
                                60 / CENTS, rms, frac)
    return np.where((frac >= 0.8) & np.isfinite(rms), np.sqrt(n_all / 4) * np.exp(-0.5 * (rms * CENTS / sigma) ** 2), 0.0)


# frames near event e where the spectrogram has a tone on flight s's exact curve, beyond the seed track
# pitch is anchored on the seed; a steady hum fits any flat tail, so only the sweeping part counts
def support(e, s, x, y, z0, o, S, c, rspec, band=2, min_db=7.0, min_run=8, pad=60.0, min_rate=1.0):
    from scipy.ndimage import median_filter
    times, freqs, L = rspec
    cents = 1200 * np.log2(freqs[1] / freqs[0])
    ts, fs = (np.asarray(v, float) for v in _seed(e))
    with np.errstate(all="ignore"):
        lf0 = np.nanmedian(np.log(fs) - predict(s, x, y, z0, S + o + ts, c))
    sel = np.flatnonzero((times >= e["t_span"][0] - pad) & (times <= e["t_span"][1] + pad))
    if not np.isfinite(lf0) or len(sel) < min_run:
        return 0
    t = times[sel]
    p = predict(s, x, y, z0, S + o + t, c)
    kc = (lf0 + p - np.log(freqs[0])) * CENTS / cents  # curve in bins
    ok = np.isfinite(kc) & (kc >= band) & (kc <= L.shape[1] - 1 - band)
    idx = np.rint(np.where(ok, kc, band)).astype(int)[:, None] + np.arange(-band, band + 1)[None, :]
    val = median_filter(L[sel[:, None], idx].max(1), 5, mode="nearest")
    rate = np.abs(np.gradient(np.where(ok, p, 0.0), t)) * CENTS
    on = ok & (val >= min_db) & (rate >= min_rate)
    i = np.clip(np.searchsorted(ts, t), 1, len(ts) - 1)
    new = np.minimum(np.abs(t - ts[i - 1]), np.abs(t - ts[i])) > 0.3  # the seed already counts its own frames
    edges = np.flatnonzero(np.diff(np.r_[0, on.astype(int), 0]))
    return int(sum(new[a:b].sum() for a, b in zip(edges[::2], edges[1::2]) if b - a >= min_run))


# per-event ridge scores of each matched flight, {event: [dts, points]}
def _ridges(S, flights, match, X, Y, dts, dgrid, c, events):
    Z = dgrid(X, Y)
    return {ei: ridge_score(events[ei], flights[fi], X, Y, Z, dts, S, c) for ei, fi in match.items()}


# per-event z of each matched flight's exact curve, scored around that event only, {event: [dts, points]}
def _curves(sr, S, flights, match, X, Y, dts, lat0, lon0, dgrid, spec, c, events):
    Z = dgrid(X, Y)
    per = {}
    for ei, fi in match.items():
        es = _event_spec(spec, events[ei], c)
        per[ei] = _z(es, flights[fi], X, Y, Z, dts, S, c) if es is not None else np.zeros((len(dts), len(X)))
    return per


# ridge fit of each candidate's event flights at its cell and offset (+- one bin)
def rescreen(S, flights, events, seeds, Fm, gx, gy, go, dgrid, c, obin):
    Z = dgrid(gx, gy)
    per = np.zeros((len(gx), len(seeds)))
    for j, ei in enumerate(seeds):
        has = np.flatnonzero(Fm[:, j] >= 0)
        if not len(has):
            continue
        key = Fm[has, j].astype(np.int64) * (1 << 32) + (np.round(go[has] / obin).astype(np.int64) + (1 << 31))
        uk, inv = np.unique(key, return_inverse=True)
        for g in range(len(uk)):
            idx = has[inv == g]
            dts = go[idx[0]] + np.arange(-obin, obin + 1e-6, 0.5)
            per[idx, j] = ridge_score(events[ei], flights[int(uk[g] >> 32)], gx[idx], gy[idx], Z[idx], dts, S,
                                      c).max(0)
    tot = per.sum(1)
    for a in range(len(seeds)):  # one flight can't be two passes at once
        for b in range(a + 1, len(seeds)):
            if abs(events[seeds[a]]["t"] - events[seeds[b]]["t"]) < 60:
                same = (Fm[:, a] == Fm[:, b]) & (Fm[:, a] >= 0)
                tot[same] -= np.minimum(per[same, a], per[same, b])
    return tot


# z of each candidate's event curves at its cell and offset (+- one bin); returns [candidates, seeds]
def screen(S, flights, events, seeds, Fm, gx, gy, go, dgrid, spec, c, obin):
    Z = np.asarray(dgrid(gx, gy), np.float64)
    out = np.zeros((len(gx), len(seeds)))
    for j, ei in enumerate(seeds):
        es = _event_spec(spec, events[ei], c)
        has = np.flatnonzero(Fm[:, j] >= 0)
        if es is None or not len(has):
            continue
        order = has[np.argsort(Fm[has, j], kind="stable")]
        fl, start = np.unique(Fm[order, j], return_index=True)
        for fi, a, b in zip(fl, start, np.r_[start[1:], len(order)]):  # one kernel call per flight
            idx = order[a:b]
            out[idx, j] = _z_each(es, flights[int(fi)], gx[idx], gy[idx], Z[idx], go[idx], obin, S, c).max(0)
    return out


# _z with a clock offset per point: z at offsets off[g] + (-half .. half) in 1 s steps; returns [shifts, points]
def _z_each(es, s, X, Y, Z, off, half, S, c, win_s=1.0):
    times, freqs, L, sigma = es
    cents = 1200 * np.log2(freqs[1] / freqs[0])
    hop = times[1] - times[0]
    m = np.round(np.arange(-half, half + 1e-6, 1.0) / hop).astype(np.int32)  # shifts in frames
    best, nv = np.empty((len(m), len(X)), np.float32), np.empty((len(m), len(X)), np.int32)
    f64 = lambda a: np.ascontiguousarray(a, np.float64)
    c = _c(s, S + float(np.median(off)) + times[len(times) // 2], c)
    heatmap._kernel().mf_each(L, L.shape[0], L.shape[1], S + times[0], hop, m, len(m),
                              *(s[k] for k in ("e", "n", "u", "ve", "vn", "vu")), len(s["e"]), s["t0"], s["dt"],
                              f64(X), f64(Y), f64(Z), f64(off), len(X), c, 12000.0, CENTS / cents, 10, best, nv)
    z = best * np.sqrt(hop / win_s) / sigma - np.sqrt(2 * np.log(L.shape[1]))
    return np.where((nv >= 10) & (best > -1e29), z, 0.0)


# sum over events; events of one flight within 60 s count once (the best)
def _total(per, match, events):
    tot, done = 0, set()
    for ei, fi in match.items():
        if ei in done:
            continue
        same = [ej for ej, fj in match.items() if fj == fi and abs(events[ej]["t"] - events[ei]["t"]) < 60]
        done.update(same)
        tot = tot + (np.max([per[ej] for ej in same], 0) if len(same) > 1 else per[ei])
    return tot


# score of a faint pass with z excess ex over its null, on the scale of ridge fits (about sqrt(points) / 2)
def _credit(ex):
    return min(0.5 * ex, 15.0)


# clock shifts (s) for the null of heard: same curve, wrong part of the audio
SHIFTS = np.array([s * k for k in (20, 30, 40, 50, 60, 75, 90) for s in (-1, 1)], float)


# matched filter along the exact curve of every flight a place should hear
# expl is {event: (flight, score)}; returns [(flight, z, z - best null, range m, time s, f0, own)]
# own: the flight already explains an event then; the others are passes no event tracked
def heard(S, dur, events, flights, expl, x, y, o, dgrid, spec, c, r_max=4000.0, min_speed=25.0, min_sweep=50.0,
          half=150.0):
    times, freqs, L, sigma = spec
    cents = 1200 * np.log2(freqs[1] / freqs[0])
    hop = times[1] - times[0]
    z0 = float(dgrid(np.array([x]), np.array([y]))[0])
    t_a = np.array([s["t0"] for s in flights])
    t_b = np.array([s["t0"] + s["dt"] * len(s["e"]) for s in flights])
    box = np.array([(s["e"].min(), s["e"].max(), s["n"].min(), s["n"].max()) for s in flights]).reshape(-1, 4)
    near = (box[:, 0] - r_max <= x) & (box[:, 1] + r_max >= x) & (box[:, 2] - r_max <= y) & (box[:, 3] + r_max >= y)
    dts = o + np.r_[np.arange(-1, 1.01, 0.5), SHIFTS]
    out = []
    for fi in np.flatnonzero(near & (t_a <= S + o + dur) & (t_b >= S + o - r_max / c)):
        s = flights[fi]
        i0 = max(0, int((S + o - r_max / c - s["t0"]) / s["dt"]))
        i1 = min(len(s["e"]), int((S + o + dur - s["t0"]) / s["dt"]) + 1)
        if i1 - i0 < 2:
            continue
        r = np.sqrt((s["e"][i0:i1] - x) ** 2 + (s["n"][i0:i1] - y) ** 2 + (s["u"][i0:i1] - z0) ** 2)
        T = s["t0"] + s["dt"] * np.arange(i0, i1) + r / c - S - o  # recording time each track point is heard
        m = (T >= 0) & (T <= dur)
        if not m.any() or r[m].min() > r_max:
            continue
        i = np.flatnonzero(m)[r[m].argmin()]
        if np.hypot(s["ve"][i0 + i], s["vn"][i0 + i]) < min_speed:
            continue
        tc = float(T[i])
        own = any(f == fi and abs(events[ei]["t"] - tc) < 60 for ei, (f, _) in expl.items())
        w = (times >= tc - half) & (times <= tc + half)
        if w.sum() < 20:
            continue
        U, IU = heatmap._times(S, dts, times[w])
        best, arg, nv, _, spread = heatmap._tone(np.ascontiguousarray(L[w]), U, IU, s, np.array([x]), np.array([y]), z0,
                                               _c(s, S + o + tc, c),
                                               12000.0, cents, 1, np.zeros(1, np.int32), 1000.0, 4.0)
        if spread[2, 0] * cents < min_sweep:
            continue
        z = np.where((nv >= 10) & (best > -1e29), best * np.sqrt(hop / 1.0) / sigma - np.sqrt(2 * np.log(L.shape[1])),
                     0.0)[:, 0]
        k = int(z[:5].argmax())
        f0 = float(freqs[max(arg[k, 0], 0)])
        if not own and z[k] >= 5 and _tracked(events, set(expl), s, x, y, z0, dts[k], S, c, f0, tc, half):
            continue
        out.append((int(fi), float(z[k]), float(z[k] - z[5:].max()), float(r[i]), tc, f0, own))
    return out


# broadband level of the recording in dB: median over 80-1000 Hz (so tones don't count), 5 s average
def envelope(x, sr):
    from . import audio
    t, _, P = audio.stft_band(x, sr, 1.0, 0.5, 80, min(1000, 0.45 * sr), pad=1)
    e = 10 * np.log10(np.median(P, 0) + 1e-20)
    return t, np.convolve(e, np.ones(11) / 11, "same")


# aircraft of any kind heard as a rise in the broadband level when they pass (a jet has no tone to track)
# every flight passing within r_max of (x, y) should raise the level at its own moment; the whole pattern of
# moments is slid against the recording, and z says how far the true alignment stands above the others
# skip: aircraft already counted by their tone. returns ([(flight, range m, time s, rise dB, rise / spread)], z)
def roar(S, dur, flights, x, y, o, dgrid, env, c, skip=(), r_max=4000.0, min_speed=25.0):
    t, e = env
    hop = t[1] - t[0]
    z0 = float(dgrid(np.array([x]), np.array([y]))[0])
    found, combs = [], []
    for fi, s in enumerate(flights):
        if s["icao"] in skip:
            continue
        i0 = max(0, int((S + o - r_max / c - s["t0"]) / s["dt"]))
        i1 = min(len(s["e"]), int((S + o + dur - s["t0"]) / s["dt"]) + 1)
        if i1 - i0 < 2:
            continue
        r = np.sqrt((s["e"][i0:i1] - x) ** 2 + (s["n"][i0:i1] - y) ** 2 + (s["u"][i0:i1] - z0) ** 2)
        if r.min() > r_max:
            continue
        T = s["t0"] + s["dt"] * np.arange(i0, i1) + r / c - S - o
        m = (T >= 0) & (T <= dur)
        if not m.any() or r[m].min() > r_max:
            continue
        i = np.flatnonzero(m)[r[m].argmin()]
        v = np.hypot(s["ve"][i0 + i], s["vn"][i0 + i])
        if v < min_speed:
            continue
        tc, rc = float(T[i]), float(r[i])
        w = float(np.clip(rc / v, 8.0, 40.0))  # the rise lasts about as long as the plane takes to fly its range
        tp = tc + 0.3 * rc / v  # engines are loudest behind the aircraft
        if tp - w - 3 < t[0] or tp + w + 3 > t[-1]:
            continue
        k0, kw = int((tp - t[0]) / hop), int(w / hop)
        combs.append((k0, kw))
        found.append((fi, rc, tc))
    if not found:
        return [], 0.0
    # level at the moment over the mean of the levels one width before and after: sharp in time
    shifts = (np.r_[0, [k for k in np.arange(-600, 601, 5) if abs(k) >= 20]] / hop).astype(int)
    rise = np.full((len(shifts), len(combs)), np.nan)
    for j, (k0, kw) in enumerate(combs):
        k = k0 + shifts
        ok = (k - kw >= 0) & (k + kw < len(e))
        rise[ok, j] = e[k[ok]] - 0.5 * (e[k[ok] - kw] + e[k[ok] + kw])
    # a far pass has a wide, noisy rise: each flight counts in units of its own spread
    with warnings.catch_warnings():  # a shift can push every flight off the recording
        warnings.simplefilter("ignore")
        sd = np.maximum(np.nanstd(rise[1:], 0), 0.5)
        tot = np.nanmean(rise / sd, 1) * len(combs)
    null = tot[1:][np.isfinite(tot[1:])]
    z = float((tot[0] - null.mean()) / max(null.std(), 1e-6)) if len(null) >= 20 and np.isfinite(tot[0]) else 0.0
    return [(fi, rc, tc, float(rise[0, j]), float(rise[0, j] / sd[j])) for j, (fi, rc, tc) in enumerate(found)], z


# pitch ratios between a tone and the harmonic another detector may lock onto
_HARM = np.array([1, 2, 3, 4, 1 / 2, 1 / 3, 1 / 4, 3 / 2, 2 / 3])


# true if flight s's curve (source pitch f0) runs along a tracked event's ridge and can't claim it:
# the event is already explained here (one sound, one aircraft), or the curve doesn't fit the ridge
# the matched filter is loose (tens of cents); a tracked sound has to fit within max_rms cents plus the
# track's own jitter (multilateration tracks predict a jagged curve)
# judged on the event that shares the most points with the curve
def _tracked(events, taken, s, x, y, z0, o, S, c, f0, tc, half, tol=40.0, share=0.3, max_rms=20.0):
    from scipy.ndimage import median_filter
    best, n_best, rough = None, 0.0, 0.0
    for ei, e in enumerate(events):
        if e["t_span"][1] < tc - half or e["t_span"][0] > tc + half:
            continue
        t, f = (np.asarray(v, float) for v in e["points"])
        pr = predict(s, x, y, z0, S + o + t, c)
        ok = np.isfinite(pr)
        if ok.sum() < 8:
            continue
        d = np.log(f[ok])[:, None] - np.log(f0) - pr[ok][:, None] - np.log(_HARM)[None, :]
        sh = (np.abs(d) * CENTS < tol).mean(0).max()
        if sh >= share and sh * len(t) > n_best:
            jit = pr[ok] - median_filter(pr[ok], 21, mode="nearest")
            best, n_best, rough = ei, sh * len(t), 1.4826 * np.median(np.abs(jit)) * CENTS
    if best is None:
        return False
    if best in taken:
        return True
    e = events[best]
    sc = ridge_score(e, s, np.array([x]), np.array([y]), np.array([z0]), o + np.arange(-1.5, 1.6, 0.5), S, c).max()
    lim = np.hypot(max_rms, 1.5 * rough)
    return sc < np.sqrt(len(e["points"][0]) / 4) * np.exp(-0.5 * (lim / 10.0) ** 2)


# flight that best explains each event at one place and offset; returns {event: (flight, score)}
# rspec: the fit grows by the tone found further along the flight's own curve (support)
def explain(S, events, flights, x, y, o, dgrid, c, rspec=None, min_score=2.0, slack=1.5, r_max=12000.0):
    z0 = dgrid(np.array([x]), np.array([y]))
    out = {}
    # flights in the air then and within reach, from time span and bounding box
    t_a = np.array([s["t0"] for s in flights])
    t_b = np.array([s["t0"] + s["dt"] * len(s["e"]) for s in flights])
    box = np.array([(s["e"].min(), s["e"].max(), s["n"].min(), s["n"].max()) for s in flights]).reshape(-1, 4)
    near = (box[:, 0] - r_max <= x) & (box[:, 1] + r_max >= x) & (box[:, 2] - r_max <= y) & (box[:, 3] + r_max >= y)
    for ei, e in enumerate(events):
        T = S + o + e["t"]
        best = (0.0, -1)
        for fi in np.flatnonzero(near & (t_a <= T + 60) & (t_b >= T - 90)):
            s = flights[fi]
            n = len(s["e"])
            i = int((T - s["t0"]) / s["dt"])
            if i < -60 or i > n + 60:
                continue
            j = np.clip(np.arange(i - 60, i + 31), 0, n - 1)  # sound takes a while to arrive
            r = np.sqrt((s["e"][j] - x) ** 2 + (s["n"][j] - y) ** 2 + (s["u"][j] - z0[0]) ** 2)
            k = j[r.argmin()]
            sp = np.sqrt(s["ve"][k] ** 2 + s["vn"][k] ** 2 + s["vu"][k] ** 2)
            # speed its fit allows, at a distance that sweeps that fast
            if not e["v_lo"] <= sp <= e["v_hi"] or r.min() > 1.5 * _reach(e, np.array([sp]), r_max=r_max)[1][0]:
                continue
            dts = o + np.arange(-slack, slack + 1e-6, 0.5)
            v = float(ridge_score(e, s, np.array([x]), np.array([y]), z0, dts, S, c).max())
            if rspec is not None:  # or the seed track alone, plus the tone found further along this flight's curve
                sc = _ridge_score(_seed(e), s, np.array([x]), np.array([y]), z0, dts, S, c)[:, 0]
                if sc.max() >= min_score:
                    more = support(e, s, x, y, float(z0[0]), dts[sc.argmax()], S, c, rspec)
                    v = max(v, float(sc.max() * np.sqrt(1 + more / len(_seed(e)[0]))))
            if v > best[0]:
                best = (v, int(fi))
        if best[0] >= min_score:
            out[ei] = (best[1], best[0])
    # one flight, one pass: keep its best event within a minute
    for ei in sorted(out, key=lambda k: -out[k][1]):
        for ej in list(out):
            if ej != ei and ei in out and out[ej][0] == out[ei][0] and abs(events[ej]["t"] - events[ei]["t"]) < 60:
                del out[ej]
    return out


# score each flight combination's cells with one shared offset, then refine the best on a 25 m / 0.25 s grid
# groups: (match {event: flight}, xs, ys, offsets)
def fine(sr, S, flights, groups, lat0, lon0, dgrid, spec, c, events, obin=3.0, tol=None):
    out = []
    for g, (match, xs, ys, os) in enumerate(groups):
        lo, hi = os.min() - obin, os.max() + obin
        if tol is not None:  # clock can't be off by more than tol
            lo, hi = max(lo, -tol), min(hi, tol)
        dts = np.arange(lo, hi + 1e-6, 1.0) if hi >= lo else np.array([np.clip(os.mean(), -tol, tol)])
        per = _ridges(S, flights, match, xs, ys, dts, dgrid, c, events)
        tot = _total(per, match, events)
        starts = [np.unravel_index(np.argmax(tot), tot.shape)]
        # a straight pass sounds the same from both sides of the track; keep the far side too
        far = np.hypot(xs - xs[starts[0][1]], ys - ys[starts[0][1]]) > 800
        if far.any():
            alt = np.unravel_index(np.argmax(np.where(far[None, :], tot, -np.inf)), tot.shape)
            if tot[alt] >= 0.8 * tot[starts[0]]:
                starts.append(alt)
        for di, gi in starts:
            cx, cy, co = xs[gi], ys[gi], dts[di]
            lo_o = -tol if tol is not None else co - 6 * obin
            hi_o = tol if tol is not None else co + 6 * obin
            for stp in (100.0, 50.0, 25.0, 10.0):  # pattern search: re-centre on the best point, then go finer
                off = np.arange(-3, 4) * stp
                OX, OY = (a.ravel() for a in np.meshgrid(off, off))
                X, Y = cx + OX, cy + OY
                dts2 = np.arange(max(lo_o, co - 4), min(hi_o, co + 4) + 1e-6, 0.5 if stp > 25 else 0.25)
                per2 = _ridges(S, flights, match, X, Y, dts2, dgrid, c, events)
                tot2 = _total(per2, match, events)
                dj, gj = np.unravel_index(np.argmax(tot2), tot2.shape)
                cx, cy, co = X[gj], Y[gj], dts2[dj]
            zz = _curves(sr, S, flights, match, np.array([cx]), np.array([cy]), np.array([co]), lat0, lon0, dgrid,
                         spec, c, events)
            out.append(dict(score=float(tot2[dj, gj]), x=float(cx), y=float(cy), offset=float(co),
                            fit={int(ei): round(float(v[dj, gj]), 2) for ei, v in per2.items()},
                            z={int(ei): round(float(v[0, 0]), 2) for ei, v in zz.items()},
                            match=match, g=g, i=int(gi)))
    return out


# uncertainty of an answer: places within 1.5 km that fit within frac of the best at any allowed clock
# one pass trades position along the track against clock error; returns [n, 2] m
def region(S, flights, match, x, y, o, dgrid, c, events, tol=None, frac=0.85):
    off = np.arange(-1500, 1501, 50.0)
    RX, RY = (a.ravel() for a in np.meshgrid(off, off))
    lo, hi = (o - 6, o + 6) if tol is None else (max(-tol, o - 6), min(tol, o + 6))
    dts = np.unique(np.r_[np.arange(lo, hi + 1e-6, 1.0), o])
    rt = _total(_ridges(S, flights, match, x + RX, y + RY, dts, dgrid, c, events), match, events).max(0)
    keep = rt >= frac * rt.max()
    return np.c_[x + RX[keep], y + RY[keep]]


# same for an answer that rests on faint passes; summed z of their curves
def curve_region(S, flights, hs, x, y, o, dgrid, spec, c, tol=None, frac=0.85, half=150.0):
    times, freqs, L, sigma = spec
    cents = 1200 * np.log2(freqs[1] / freqs[0])
    off = np.arange(-1500, 1501, 100.0)
    RX, RY = (a.ravel() for a in np.meshgrid(off, off))
    X, Y = x + RX, y + RY
    Z = np.asarray(dgrid(X, Y), np.float64)
    lo, hi = (o - 6, o + 6) if tol is None else (max(-tol, o - 6), min(tol, o + 6))
    dts = np.unique(np.r_[np.arange(lo, hi + 1e-6, 1.0), o])
    tot = np.zeros((len(dts), len(X)))
    for h in hs:
        w = (times >= h["t"] - half) & (times <= h["t"] + half)
        U, IU = heatmap._times(S, dts, times[w])
        best, _, nv, _, _ = heatmap._tone(np.ascontiguousarray(L[w]), U, IU, flights[h["fi"]], X, Y, Z,
                                          _c(flights[h["fi"]], S + o + h["t"], c), 12000.0, cents,
                                          1, np.zeros(1, np.int32), 1000.0, 4.0)
        tot += np.where((nv >= 10) & (best > -1e29),
                        best * np.sqrt(times[1] - times[0]) / sigma - np.sqrt(2 * np.log(L.shape[1])), 0.0)
    rt = tot.max(0)
    keep = rt >= frac * rt.max()
    return np.c_[X[keep], Y[keep]]


# flights in local enu around the middle of bbox, terrain and spectrogram: what searching one area needs
# temp_c is the air temperature at the ground, falling by lapse K per km of height
def _setup(x, sr, S, dur, segs, bbox, events, tol=30.0, temp_c=15.0, sources=TONAL, elev=None, spec=None,
           dem_step=None, rspec=None, trace=True, lapse=6.5, roar=True):
    from .events import ridge_spectrogram, spectrogram
    from .model import to_enu
    A = SimpleNamespace(x=x, sr=sr, S=S, dur=dur, tol=tol, bbox=bbox, sources=sources, elev=elev, segs=segs)
    A.events = [e for e in events if e["v"] >= 20 and e["d"] >= 120]  # boats, cars, people: nothing flies that close
    A.c = sound_speed(temp_c)
    A.lat0, A.lon0 = (bbox[0] + bbox[1]) / 2, (bbox[2] + bbox[3]) / 2
    A.flights = heatmap.prepare(segs, A.lat0, A.lon0, sources)
    for s in A.flights:
        s["type"] = s.get("type", "")
    A.dgrid = demmod.Grid(bbox, A.lat0, A.lon0, dem_step) if elev is None else \
        (lambda X, Y: np.full(np.shape(X), float(elev)))
    A.v_max = 260.0 if "jet" in sources else 180.0
    x0, y0, _ = to_enu(bbox[0], bbox[2], 0, A.lat0, A.lon0)
    x1, y1, _ = to_enu(bbox[1], bbox[3], 0, A.lat0, A.lon0)
    A.box = (x0, x1, y0, y1)
    if lapse:
        ground = float(elev) if elev is not None else float(np.median(A.dgrid(*np.meshgrid(np.linspace(x0, x1, 30),
                                                                                              np.linspace(y0, y1, 30)))))
        for s in A.flights:
            _air(s, temp_c, lapse, ground)
    A.spec = spec if spec is not None else spectrogram(x, sr, fmax=1500, cents=8, win_s=1.0, hop_s=0.5)
    A.loud = _loud(A.flights)
    A.trace, held = trace, [rspec, None]

    def get():  # only the final scoring needs it
        if held[0] is None:
            held[0] = ridge_spectrogram(x, sr)
        return held[0]

    def env():
        if held[1] is None:
            held[1] = envelope(x, sr)
        return held[1]
    A.rspec, A.env = get, env
    # every aircraft, tonal or not, for the broadband check
    A.everyone = heatmap.prepare(segs, A.lat0, A.lon0) if roar and x is not None else []
    return A


# candidates, z screen, ridge fits; returns the flight combinations, best first, and their cells
# over: a list that gets an entry when the area held too much to look at all of it
def _rank(A, cell=200.0, obin=3.0, n_fine=40, lone=600000, sweep=True, over=None, log=print):
    S, tol, flights, events = A.S, A.tol, A.flights, A.events
    # single passes get their own places when nothing else can vouch for them: start unknown, or few events
    lone = lone if tol is None or len(events) <= 3 else 0
    C = candidates(events, flights, S, tol, A.dgrid, cell, obin, A.c, v_max=A.v_max, box=A.box,
                   lone=lone, sweep=sweep, over=over, log=log)
    if C is None:
        return None
    n = C["n"]
    log(f"{len(C['keys'])} cell-offsets, {int((n >= 2).sum())} where 2+ events agree")
    good = np.argsort(-n, kind="stable")[:400000 + lone]
    if over is not None and len(good) < len(n) and n[good[-1]] >= 2:  # places two events agree on were cut
        over.append(-1)
    gx, gy, go = _decode(C["keys"][good], cell, obin)
    Fm = C["flight"][good]
    zs = screen(S, flights, events, C["seeds"], Fm, gx, gy, go, A.dgrid, A.spec, A.c, obin)
    pre = zs.sum(1)
    top = np.argsort(-pre)[:4000]  # ridge fit only the best
    if lone:  # and the best by one event alone
        top = np.unique(np.r_[top[:3000], np.argsort(-zs.max(1))[:3000]])
    log(f"screened {len(good)} candidates, best {pre[top[0]]:.1f}")
    cla, clo = from_enu(gx, gy, A.lat0, A.lon0)
    cloud = dict(lat=cla, lon=clo, offset=go, z=zs, flight=Fm, seeds=C["seeds"])
    good, gx, gy, go, Fm, pre = good[top], gx[top], gy[top], go[top], Fm[top], pre[top]
    pre = rescreen(S, flights, events, C["seeds"], Fm, gx, gy, go, A.dgrid, A.c, obin)
    log("ridge fits")
    match = [{C["seeds"][j]: int(f) for j, f in enumerate(row) if f >= 0} for row in Fm]
    # group by which flight explains each event
    groups = {}
    for i in np.argsort(-pre, kind="stable"):
        sig = tuple(sorted(match[i].items()))
        groups.setdefault(sig, []).append(i)
    ranked = sorted(groups.items(), key=lambda kv: (-pre[kv[1][0]], -len(kv[1])))
    if lone:  # single-event groups would lose every slot to chance pairs
        one = [kv for kv in ranked if len(kv[0]) == 1][:n_fine]
        ranked = ranked[:n_fine] + [kv for kv in one if kv not in ranked[:n_fine]]
    else:
        ranked = ranked[:n_fine]
    return dict(ranked=ranked, gx=gx, gy=gy, go=go, pre=pre, n=n[good], cloud=cloud, n_keys=len(C["keys"]))


# score of one place: tracked events + faint passes - missing passes
def _place(A, f, r_loud=1500.0, guided=True):
    S, dur, events, flights, dgrid, c = A.S, A.dur, A.events, A.flights, A.dgrid, A.c
    f["explained"] = explain(S, events, flights, f["x"], f["y"], f["offset"], dgrid, c, A.rspec() if A.trace else None)
    rl = r_loud
    if f["explained"]:  # a plane heard at d would also be heard at d / 2
        ebest = max(f["explained"], key=lambda k: f["explained"][k][1])
        rl = float(np.clip(0.5 * events[ebest]["d"], r_loud, 4000.0))
    ms = silence([(f["x"], f["y"], f["offset"])], events, flights, S, dur, dgrid, c, rl, loud=A.loud)[0]
    hd = heard(S, dur, events, flights, f["explained"], f["x"], f["y"], f["offset"], dgrid, A.spec, c) if guided else []
    # ridge fits tie between look-alike flights and the two sides of a straight pass; the matched filter
    # sees the whole curve, so a little of it goes in as a tie-break
    tie = 0.2 * sum(min(max(h[2], 0.0), 30.0) for h in hd if h[6])
    hd = [h[:6] for h in hd if not h[6]]
    # a faint pass counts, and then it isn't missing
    f["heard"] = [h for h in hd if h[1] >= 6 and h[2] >= 4]
    faint = {h[0] for h in hd if h[1] >= 5 and h[2] >= 2}
    ms = [m for m in ms if m[0] not in faint]
    f["extra"] = sum(_credit(h[2]) for h in f["heard"])
    # aircraft heard as a rise in level, if no tone of theirs already counts
    known = {flights[fi]["icao"] for fi, _ in f["explained"].values()} | {flights[h[0]]["icao"] for h in f["heard"]}
    rr, zr = roar(S, dur, A.everyone, f["x"], f["y"], f["offset"], dgrid, A.env(), c, known) if A.everyone else ([], 0.0)
    rumble = float(np.clip(zr - 3.0, 0.0, 15.0))  # on the scale of a tracked event; nothing below z = 3
    rr = [(A.everyone[fi], rc, tc, h) for fi, rc, tc, h, zi in rr if zi >= 2.0] if rumble > 0 else []
    f["extra"] += rumble
    la, lo = from_enu(f["x"], f["y"], A.lat0, A.lon0)
    return dict(lat=float(la), lon=float(lo),
                score=sum(v for _, v in f["explained"].values()) + f["extra"] + tie - 3.0 * len(ms),
                match_score=f["score"], extra=f["extra"], tie=round(tie, 2), offset=f["offset"], n_events=f["n_events"], grid=f.get("grid"),
                heard=[dict(icao=flights[fi]["icao"], kind=flights[fi]["kind"], fi=fi, z=round(z, 1),
                            excess=round(ex, 1), r=round(r), t=round(t, 1), f0=round(f0, 1))
                       for fi, z, ex, r, t, f0 in f["heard"]],
                roar=[dict(icao=q[0]["icao"], kind=q[0]["kind"], r=round(q[1]), t=round(q[2], 1), db=round(q[3], 1))
                      for q in rr], rumble=round(rumble, 2),
                missed=[dict(icao=flights[fi]["icao"], t=t, r=r) for fi, t, r in ms],
                x=f["x"], y=f["y"], match=f["match"],
                events={ei: dict(icao=flights[fi]["icao"], kind=flights[fi]["kind"], fit=round(v, 2),
                                 z=f["z"].get(ei), fi=fi) for ei, (fi, v) in f["explained"].items()},
                missed_fi=[fi for fi, _, _ in ms])


# chance places expected to score this high, from an exponential tail fitted to the k best rivals
# a wide search has many rivals bunched under what two look-alike passes can reach; None if too few to tell
def _tail(score, other, k=8):
    o = sorted(other, reverse=True)
    k = min(k, len(o) - 1)
    if k < 4:
        return None
    beta = max(float(np.mean(o[:k])) - o[k], 0.5)
    return float(k * np.exp(-(score - o[k]) / beta))


# best first, one answer per spot, each with its margin and whether it is confident
# tail: a wide search, where the best may also stand out from the rivals' spread rather than by 1.4x
def _verdict(out, tail=False):
    out = sorted(out, key=lambda r: -r["score"])
    merged = []  # same spot is one answer
    for r in out:
        if all(_dist(r, q) > 1000 for q in merged):
            merged.append(r)
    for r in merged:
        r["tonal"] = {m["icao"] for m in r["events"].values()} | {h["icao"] for h in r["heard"]}
        r["icaos"] = sorted(r["tonal"] | {q["icao"] for q in r.get("roar", [])})
    for r in merged:
        # margin: score over the best rival; near 1 is ambiguous. a place within 3 km that hears only some of
        # the same aircraft is this answer a little off, not a rival (the far side of a straight pass hears all)
        # (by their tones: the broadband rise is the same for kilometres around)
        other = [q["score"] for q in merged if q is not r and not (_dist(r, q) < 3000 and q["tonal"] < r["tonal"])]
        r["margin"] = max(r["score"], 0.0) / max(max(other, default=0.0), 0.5)
        r["tail"] = _tail(r["score"], other) if tail else None
        clear = r["margin"] >= 1.4 or (r["tail"] is not None and r["tail"] <= 0.01 and r["margin"] >= 1.2)
        # confident: 2+ aircraft (tracked or faint), enough evidence beyond the strongest one (any single
        # pass fits several places), few loud passes missing, no rival
        items = [m["fit"] for m in r["events"].values()] + [_credit(h["excess"]) for h in r["heard"]] + \
            [r.get("rumble", 0.0)] * bool(r.get("rumble"))
        r["aircraft"] = len(r["icaos"])
        r["confident"] = bool(r["aircraft"] >= 2 and sum(items) - max(items) >= 4 and r["score"] - r["tie"] >= 10
                              and 2 * len(r["missed"]) < r["aircraft"] and clear)
        # one aircraft is enough when its pass is long, nothing is missing and nowhere else comes close
        # (both sides of a straight pass tie, so this takes a pass that bends)
        fits = [m["fit"] for m in r["events"].values()]
        if not r["confident"] and fits and max(fits) >= 8 and not r["missed"] and r["margin"] >= 1.6:
            r["confident"] = True
    return [r for r in merged if r["events"] or r["heard"]]  # a place that explains nothing is no answer


# ranked candidate places and offsets for a recording starting near unix time S (+- tol s)
# tol=None: any time. elev: constant ground height in m, None reads terrain tiles
def search(x, sr, S, dur, segs, bbox, events, tol=30.0, temp_c=15.0, cell=200.0, obin=3.0, r_loud=1500.0,
           n_fine=40, sources=TONAL, elev=None, guided=True, grid=True, grid_s=180.0, n_region=6, lone=600000,
           spec=None, rspec=None, trace=True, lapse=6.5, sweep=True, roar=True, log=print):
    from .model import to_enu
    t_start, say = time.time(), log
    log = lambda m: say(f"{m}  [{time.time() - t_start:.0f} s]")
    A = _setup(x, sr, S, dur, segs, bbox, events, tol, temp_c, sources, elev, spec, None, rspec, trace, lapse, roar)
    flights, events, dgrid, c, lat0, lon0 = A.flights, A.events, A.dgrid, A.c, A.lat0, A.lon0
    log(f"{len(flights)} flights, {len(events)} events")
    fr, n_keys = [], 0
    cloud = dict(lat=np.zeros(0), lon=np.zeros(0), offset=np.zeros(0), z=np.zeros((0, 0)), flight=np.zeros((0, 0), int),
                 seeds=[])
    R = _rank(A, cell, obin, n_fine, lone, sweep, log=log)
    if R is not None:
        ranked, gx, gy, go, cloud, n_keys = R["ranked"], R["gx"], R["gy"], R["go"], R["cloud"], R["n_keys"]
        log(f"fine-scoring {len(ranked)} flight combinations")
        fr = fine(sr, S, flights, [(dict(sig), gx[ix], gy[ix], go[ix]) for sig, ix in ((s_, np.array(v)) for s_, v in ranked)],
                  lat0, lon0, dgrid, A.spec, c, events, obin=obin, tol=tol)
        for f in fr:
            f["n_events"] = int(R["n"][ranked[f["g"]][1][f["i"]]])
        log("refined")
    out = [_place(A, f, r_loud, guided) for f in fr]
    # a place resting on one aircraft: the same pass heard from the other side of its path gets scored too
    for r in sorted(out, key=lambda r: -r["score"])[:10]:
        if len({m["fi"] for m in r["events"].values()}) != 1:
            continue
        ei, m = max(r["events"].items(), key=lambda kv: kv[1]["fit"])
        s = flights[m["fi"]]
        k = np.clip(int((S + r["offset"] + events[ei]["t"] - s["t0"]) / s["dt"]) + np.arange(-90, 31), 0, len(s["e"]) - 1)
        j = k[np.hypot(s["e"][k] - r["x"], s["n"][k] - r["y"]).argmin()]  # where the plane was closest
        mx, my = 2 * s["e"][j] - r["x"], 2 * s["n"][j] - r["y"]
        if np.hypot(mx - r["x"], my - r["y"]) < 300 or any(np.hypot(mx - q["x"], my - q["y"]) < 300 for q in out):
            continue
        for f in fine(sr, S, flights, [({ei: m["fi"]}, np.array([mx]), np.array([my]), np.array([r["offset"]]))],
                      lat0, lon0, dgrid, A.spec, c, events, obin=obin, tol=tol):
            f["n_events"] = 1
            out.append(_place(A, f, r_loud, guided))
    first = _verdict([dict(r) for r in out])
    sure = bool(first and first[0]["confident"])
    # grid search hears passes too faint to track but is slow on long clips in busy airspace
    # only if the events alone give no confident answer and the start is known; given up if it would run past grid_s
    if grid and tol is not None and not sure:
        x0, x1, y0, y1 = A.box
        el = float(elev) if elev is not None else float(np.median(dgrid(*np.meshgrid(np.linspace(x0, x1, 30),
                                                                                     np.linspace(y0, y1, 30)))))
        t0 = time.time()

        def pace(frac, msg):  # the first few percent say little about the rest: wait for a tenth, or grid_s / 6
            dt = time.time() - t0
            if 0.02 < frac < 0.6 and dt / frac > grid_s and (frac > 0.1 or dt > grid_s / 6):
                raise TimeoutError(dt / frac)
        try:
            wide = max(x1 - x0, y1 - y0) > 45000
            peaks, _ = heatmap.search(x, sr, S, segs, bbox, c, elev=el, clock_tol=tol, step=400.0 if wide else 250.0,
                                      n_peaks=40 if wide else 24, log=lambda *_: None,
                                      sources=tuple(k for k in sources if k != "jet"), fmax=600, progress=pace)
        except TimeoutError as err:
            peaks = []
            log(f"grid search given up: would take about {float(str(err)):.0f} s")
        for h in peaks:
            hx, hy, _ = to_enu(h["lat"], h["lon"], 0, lat0, lon0)
            out.append(_place(A, dict(score=0.0, x=float(hx), y=float(hy), offset=float(h["clock_offset"]), fit={}, z={},
                                      match={}, n_events=0, grid=round(h["score"], 1)), r_loud, guided))
        log(f"grid search: {len(peaks)} places")
    log("scored")
    out = _verdict(out)
    for i, r in enumerate(out):
        if i < n_region and r["match"]:
            pts = region(S, flights, r["match"], r["x"], r["y"], r["offset"], dgrid, c, events, tol)
        elif i < n_region and r["heard"]:
            pts = curve_region(S, flights, r["heard"], r["x"], r["y"], r["offset"], dgrid, A.spec, c, tol)
        else:
            pts = np.array([[r["x"], r["y"]]])
        r["region"] = np.c_[from_enu(pts[:, 0], pts[:, 1], lat0, lon0)].round(5).tolist()
    log("regions")
    return out, dict(flights=flights, lat0=lat0, lon0=lon0, n_keys=n_keys, cloud=cloud, events=events,
                     dgrid=dgrid, c=c, everyone=A.everyone)


# ground temperature (C) that fits answer r best: the search repeated in a small area around it
# sound speed moves a distant pass's place by about 100 m per 3 C; two or more aircraft at different
# heights have to agree, which the right temperature lets them do. returns (temp, {temp: score})
def fit_temp(x, sr, S, dur, segs, events, r, km=3.0, slack=4.0, temps=np.arange(-10.0, 40.1, 5.0), **kw):
    la, lo = r["lat"], r["lon"]
    d = (km + 12) / 111.32
    dn = d / np.cos(np.radians(la))
    near = [s for s in segs if ((s["lat"] > la - d) & (s["lat"] < la + d) & (s["lon"] > lo - dn) & (s["lon"] < lo + dn)).any()]
    bbox = [la - km / 111.32, la + km / 111.32, lo - km / 111.32 / np.cos(np.radians(la)), lo + km / 111.32 / np.cos(np.radians(la))]
    got = {}

    def run(t):
        if t not in got:
            res, _ = search(x, sr, S + r["offset"], dur, near, bbox, events, tol=slack, temp_c=float(t), grid=False,
                            n_region=0, log=lambda m: None, **kw)
            got[t] = res[0]["score"] if res else 0.0
        return got[t]
    best = max(temps, key=run)
    best = max((best - 2.5, best, best + 2.5), key=run)
    return float(best), got


# everything the viewer shows, as json-able data
def summary(x, sr, S, dur, results, info, tol, bbox, types=None, truth=None, n=6):
    import base64
    from .events import ridge_spectrogram
    flights, events, dgrid, c = info["flights"], info["events"], info["dgrid"], info["c"]
    lat0, lon0 = info["lat0"], info["lon0"]
    types = types or {}
    iso = lambda t: __import__("datetime").datetime.fromtimestamp(t, __import__("datetime").timezone.utc) \
        .strftime("%Y-%m-%dT%H:%M:%S.%f")[:-5] + "Z"

    def thin(a, k):
        a = np.asarray(a)
        return a[np.linspace(0, len(a) - 1, k).astype(int)] if len(a) > k else a

    ev = []
    for e in events:
        t, f = thin(e["points"][0], 400), thin(e["points"][1], 400)
        ev.append(dict(t=e["t"], f0=e["f0"], v=e["v"], d=e["d"], rms=e["rms"], v_lo=e["v_lo"], v_hi=e["v_hi"],
                       d_lo=e["d_lo"], d_hi=e["d_hi"], span=e["t_span"], pts=[np.round(t, 2).tolist(), np.round(f, 1).tolist()]))
    used, res, loud = set(), [], set()
    for r in results[:n]:
        z0 = float(dgrid(np.array([r["x"]]), np.array([r["y"]]))[0])
        evs = []
        for ei, m in sorted(r["events"].items()):
            e, s = events[ei], flights[m["fi"]]
            used.add(m["fi"])
            tt = np.arange(e["t_span"][0] - 20, e["t_span"][1] + 20, 0.5)
            p = predict(s, r["x"], r["y"], z0, S + r["offset"] + tt, c)
            pr = predict(s, r["x"], r["y"], z0, S + r["offset"] + np.asarray(e["points"][0]), c)
            lf0 = np.nanmedian(np.log(e["points"][1]) - pr)
            ok = np.isfinite(p)
            evs.append(dict(event=int(ei), icao=m["icao"], type=types.get(m["icao"], ("", ""))[0], kind=m["kind"],
                            fit=m["fit"], curve=[tt[ok].round(2).tolist(), np.exp(lf0 + p[ok]).round(2).tolist()]))
        used.update(r["missed_fi"])
        faint = []  # passes found along their predicted curve only
        for h in r.get("heard", []):
            tt = np.arange(h["t"] - 60, h["t"] + 60, 0.5)
            pr = predict(flights[h["fi"]], r["x"], r["y"], z0, S + r["offset"] + tt, c)
            ok = np.isfinite(pr)
            used.add(h["fi"])
            faint.append(dict(icao=h["icao"], type=types.get(h["icao"], ("", ""))[0], kind=h["kind"], t=h["t"], r=h["r"],
                              z=h["z"], curve=[tt[ok].round(2).tolist(), (h["f0"] * np.exp(pr[ok])).round(2).tolist()]))
        loud.update(q["icao"] for q in r.get("roar", []))
        res.append(dict(lat=r["lat"], lon=r["lon"], score=round(r["score"], 2), margin=round(r["margin"], 2),
                        confident=r.get("confident", False),
                        rumble=[dict(q, type=types.get(q["icao"], ("", ""))[0]) for q in r.get("roar", [])],
                        start=iso(S + r["offset"]), offset=round(r["offset"], 2), events=evs, heard=faint,
                        missed=[dict(m, type=types.get(m["icao"], ("", ""))[0]) for m in r["missed"]],
                        region=thin(np.asarray(r["region"]), 300).tolist()))
    o0 = results[0]["offset"] if results else 0.0
    tracks = {}
    for s in [flights[fi] for fi in used] + [s for s in info.get("everyone", []) if s["icao"] in loud]:
        tt = s["t0"] + s["dt"] * np.arange(len(s["e"]))
        m = (tt > S + o0 - 600) & (tt < S + o0 + dur + 600)
        k = np.flatnonzero(m)[::5]
        la, lo = from_enu(s["e"][k], s["n"][k], lat0, lon0)
        tracks[s["icao"]] = dict(t=(tt[k] - S - o0).round(1).tolist(), lat=la.round(5).tolist(), lon=lo.round(5).tolist(),
                                 alt=s["u"][k].round(0).tolist(), type=types.get(s["icao"], ("", ""))[0])
    cl = info["cloud"]
    tot = cl["z"].sum(1)
    k = np.argsort(-tot)[:4000]
    cloud = [np.round(cl["lat"][k], 5).tolist(), np.round(cl["lon"][k], 5).tolist(), np.round(tot[k], 1).tolist()]
    # display spectrogram: 1 s x 12 cents, 0.25 s x 6 cents around events the answers use
    times, freqs, L = ridge_spectrogram(x, sr)
    q = lambda A: np.clip((A + 4) / 22 * 255, 0, 255).astype(np.uint8)
    F4 = len(times) // 4 * 4
    lo = np.maximum(L[:F4:4], np.maximum(L[1:F4:4], np.maximum(L[2:F4:4], L[3:F4:4])))
    lo = np.maximum(lo[:, : lo.shape[1] // 2 * 2:2], lo[:, 1: lo.shape[1] // 2 * 2:2])
    spec = dict(t0=float(times[0]), hop=float(times[4] - times[0]), fmin=float(freqs[0]),
                cents=float(1200 * np.log2(freqs[2] / freqs[0])), F=lo.shape[0], K=lo.shape[1],
                q=base64.b64encode(q(lo).tobytes()).decode())
    patches = {}
    for ei in sorted({e["event"] for r in res[:3] for e in r["events"]}):
        e = events[ei]
        a, b = e["t_span"][0] - 30, e["t_span"][1] + 30
        m = (times >= a) & (times <= b)
        P = L[m]
        patches[ei] = dict(t0=float(times[m][0]), hop=float(times[1] - times[0]), fmin=float(freqs[0]),
                           cents=float(1200 * np.log2(freqs[1] / freqs[0])), F=P.shape[0], K=P.shape[1],
                           q=base64.b64encode(q(P).tobytes()).decode())
    return dict(start=iso(S), tol=tol, dur=dur, bbox=bbox, truth=truth, events=ev, results=res, tracks=tracks,
                cloud=cloud, spec=spec, patches=patches)
