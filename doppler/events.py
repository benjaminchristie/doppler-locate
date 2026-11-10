# finds doppler events (aircraft passes) in a recording without flight data
# a straight pass at speed v, closest slant distance d has a pitch curve set by (v, d); templates of these are
# swept over a log-frequency spectrogram (C++), and the time-reversed spectrogram gives each template's null
import numpy as np
from scipy.ndimage import maximum_filter1d

from . import audio
from .heatmap import _kernel
from .model import CENTS, line_logratio, sound_speed

V_GRID = np.geomspace(20, 140, 10)  # m/s
D_GRID = np.geomspace(60, 6000, 14)  # m, slant distance at closest approach


# whitened dB on a log-frequency axis, broadband level of each frame removed; returns times, freqs, L[t, k], sigma
def spectrogram(x, sr, fmin=50, fmax=1500, cents=12, win_s=0.5, hop_s=0.25):
    fmax = min(fmax, 0.45 * sr)
    times, fr, P = audio.stft_band(x, sr, win_s, hop_s, fmin * 0.9, fmax * 1.1)
    dB = 10 * np.log10(P + 1e-20)
    dB -= np.median(dB, 0, keepdims=True)
    W = np.clip(dB - np.median(dB, 1, keepdims=True), -5, 25)
    freqs = fmin * 2 ** (np.arange(int(1200 * np.log2(fmax / fmin) / cents)) * cents / 1200)
    pos = np.clip((freqs - fr[0]) / (fr[1] - fr[0]), 0, len(fr) - 1.001)
    i = pos.astype(int)
    w = (pos - i)[:, None]
    L = maximum_filter1d(((1 - w) * W[i] + w * W[i + 1]).T, 3, axis=1)
    L = L - L.mean()
    return times, freqs, np.ascontiguousarray(L, np.float32), float(L.std())


# pitch offsets (bins) of each (v, d) template around closest approach, and window half-lengths
def templates(hop_s, cents, c, v_grid=V_GRID, d_grid=D_GRID, max_half_s=30.0, min_half_s=3.0):
    V, D = np.meshgrid(v_grid, d_grid, indexing="ij")
    V, D = V.ravel(), D.ravel()
    half = np.clip(np.ceil(2.5 * D / V / hop_s), min_half_s / hop_s, max_half_s / hop_s).astype(np.int32)
    J = int(half.max())
    t = np.arange(-J, J + 1) * hop_s
    sh = np.stack([line_logratio(t, 0.0, d, v, c) for v, d in zip(V, D)]) * CENTS / cents
    return V, D, np.ascontiguousarray(np.rint(sh), np.int32), half, J


def _bank(L, shifts, half, J, centers, H, hoff):
    S, NC = len(half), len(centers)
    best = np.empty(NC * S, np.float32)
    arg = np.empty(NC * S, np.int32)
    nfr = np.empty(NC * S, np.int32)
    lib = _kernel()
    lib.sweep_bank(L, L.shape[0], L.shape[1], shifts, half, S, J, np.ascontiguousarray(centers, np.int32), NC, H,
                   hoff, best, arg, nfr)
    return best.reshape(NC, S), arg.reshape(NC, S), nfr.reshape(NC, S)


# is (t, f) in stronger event e's time window and swept pitch range (or a harmonic)
def _inside(t, f, e, c, min_gap_s):
    if abs(t - e["t"]) > max(min_gap_s, 2.5 * e["d"] / e["v"]):
        return False
    lo, hi = -np.log1p(e["v"] / c) - 0.05, -np.log1p(-e["v"] / c) + 0.05  # log pitch range of the pass
    return any(lo < np.log(f / (e["f0"] * r)) < hi for r in (1 / 2, 1, 2, 3))


# doppler events, strongest first: dict(t closest approach in s, f0 in Hz, z, v, d of best template,
# v_lo/v_hi and d_lo/d_hi of templates within 1.5 of best); also returns spectrogram and detection trace
def detect(x, sr, temp_c=15.0, harmonics=3, step=2, min_gap_s=6.0, spec=None, log=print):
    c = sound_speed(temp_c)
    times, freqs, L, sigma = spec or spectrogram(x, sr)
    cents = 1200 * np.log2(freqs[1] / freqs[0])
    hop = times[1] - times[0]
    V, D, shifts, half, J = templates(hop, cents, c)
    hoff = np.rint(CENTS * np.log2(np.arange(1, harmonics + 1)) / cents).astype(np.int32)
    centers = np.arange(0, len(times), step)
    best, arg, nfr = _bank(L, shifts, half, J, centers, harmonics, hoff)
    nbest, _, _ = _bank(np.ascontiguousarray(L[::-1]), shifts, half, J, centers, harmonics, hoff)
    ok = nfr >= 8
    # per template: centre and spread of the null
    mu = np.array([np.median(nbest[ok[:, s], s]) if ok[:, s].any() else 0 for s in range(len(V))])
    sd = np.array([max(np.subtract(*np.percentile(nbest[ok[:, s], s], [75, 25])) / 1.349, 1e-6)
                   if ok[:, s].any() else 1 for s in range(len(V))])
    z = np.where(ok, (best - mu) / sd, -np.inf)
    zn = np.where(ok, (nbest - mu) / sd, -np.inf)
    trace, null = z.max(1), zn.max(1)
    thr = max(4.0, float(np.quantile(null, 1 - 2.0 / len(null))))  # ~2 false events per clip
    log(f"events: {len(V)} templates, threshold z {thr:.1f}")
    events = []
    for ci in np.argsort(-trace):
        if trace[ci] < thr:
            break
        t = float(times[centers[ci]])
        s = int(np.argmax(z[ci]))
        f = float(freqs[arg[ci, s]])
        if any(_inside(t, f, e, c, min_gap_s) for e in events):  # piece of a stronger pass
            continue
        near = z[max(0, ci - 1):ci + 2].max(0) >= trace[ci] - 1.5
        events.append(dict(t=round(t, 2), f0=round(float(freqs[arg[ci, s]]), 1), z=round(float(trace[ci]), 1),
                           v=round(float(V[s]), 1), d=round(float(D[s]), 1),
                           v_lo=round(float(V[near].min() / 1.1), 1), v_hi=round(float(V[near].max() * 1.1), 1),
                           d_lo=round(float(D[near].min() / 1.2), 1), d_hi=round(float(D[near].max() * 1.2), 1)))
    events.sort(key=lambda e: e["t"])
    return events, dict(spec=(times, freqs, L, sigma), trace_t=times[centers], trace=trace, null=null, thr=thr)


# finer pitch, no dilation, for following ridges
def ridge_spectrogram(x, sr, fmin=40, fmax=1500, cents=6, win_s=1.0, hop_s=0.25):
    times, fr, P = audio.stft_band(x, sr, win_s, hop_s, fmin * 0.9, min(fmax, 0.45 * sr) * 1.1)
    W = np.ascontiguousarray(10 * np.log10(P + 1e-20), np.float32)
    k = _kernel()
    k.sub_median(W, *W.shape, 0)  # each frame's level, then each bin's
    k.sub_median(W, *W.shape, 1)
    freqs = fmin * 2 ** (np.arange(int(1200 * np.log2(min(fmax, 0.45 * sr) / fmin) / cents)) * cents / 1200)
    pos = np.clip((freqs - fr[0]) / (fr[1] - fr[0]), 0, len(fr) - 1.001)
    i = pos.astype(int)
    w = (pos - i)[:, None]
    return times, freqs, np.ascontiguousarray(((1 - w) * W[i] + w * W[i + 1]).T, np.float32)


# link per-frame spectral peaks into tracks; returns list of (frames, sub-bin pitch, level)
def ridges(L, peak_db=6.0, tol=2.5, max_gap=6, min_len=40, smooth=True):
    from scipy.ndimage import gaussian_filter1d
    F, K = L.shape
    if smooth:  # smoothing keeps slow tones but erases steep sweeps
        L = gaussian_filter1d(L, 1.0, axis=0)
    c = L[:, 2:-2]
    pk = (c >= peak_db) & (c >= L[:, 1:-3]) & (c >= L[:, 3:-1]) & (c >= L[:, :-4]) & (c >= L[:, 4:])
    fi, ki = np.nonzero(pk)
    ki = ki + 2
    y0, y1, y2 = L[fi, ki - 1], L[fi, ki], L[fi, ki + 1]
    den = y0 - 2 * y1 + y2
    kk = ki + np.clip(np.where(np.abs(den) > 1e-6, 0.5 * (y0 - y2) / np.where(np.abs(den) > 1e-6, den, 1), 0), -.5, .5)
    starts = np.searchsorted(fi, np.arange(F + 1)).astype(np.int32)
    kk, y1 = np.ascontiguousarray(kk, np.float64), np.ascontiguousarray(y1, np.float32)
    track, order = np.empty(len(kk), np.int32), np.empty(len(kk), np.int32)
    n = _kernel().link_ridges(starts, F, kk, y1, peak_db, tol, max_gap, min_len, track, order)
    by = np.argsort(track, kind="stable")
    lo = np.searchsorted(track[by], order[:n])
    hi = np.searchsorted(track[by], order[:n], side="right")
    return [(fi[by[a:b]], kk[by[a:b]], y1[by[a:b]]) for a, b in zip(lo, hi)]


# spectrogram cells on steady tones (hum, machinery): tracks of 30 s or more that stay within 15 cents
# a re-traced ridge must not run along them, or a flat tail of any pass fits
def _steady(tr, shape, min_frames=120, max_drift=2.5):
    M = np.zeros(shape, bool)
    for fr, k, _ in tr:
        if len(fr) >= min_frames and np.percentile(k, 95) - np.percentile(k, 5) <= max_drift:
            kk = np.rint(k).astype(int)
            for d in range(-2, 3):
                M[fr, np.clip(kk + d, 0, shape[1] - 1)] = True
    return M


# speeds that fit the track almost as well as the best: [(v, d, t0)]; short tracks get a long list
def _profile(t, f, c, fit, v_grid=np.geomspace(18, 250, 24)):
    from .model import fit_line_at
    out = [(fit["v"], fit["d"], fit["t0"])]
    for v in v_grid:
        r = fit_line_at(t, f, c, v, fit)
        if r["rms_cents"] <= 1.5 * fit["rms_cents"] + 2.0:  # ridge errors wander slowly
            out.append((float(v), r["d"], r["t0"]))
    return sorted(out)


_RATIOS = sorted({n / m for n in range(1, 6) for m in range(1, 6)})


# doppler events at two window lengths (long for slow distant passes, short for close fast ones)
# the same pass found twice counts once
def track_events(x, sr, temp_c=15.0, min_drop=0.03, max_rms=15.0, v_range=(18.0, 280.0), fmax=3000.0, log=print,
                 windows=(1.0, 0.5)):
    out = []
    same = lambda e, g: abs(e["t"] - g["t"]) < max(10.0, 0.3 * g["d"] / g["v"]) and \
        min(abs(np.log(e["f0"] / g["f0"] / r)) for r in _RATIOS) < 0.06
    specs = {w: ridge_spectrogram(x, sr, fmax=fmax, win_s=w) for w in windows}
    hum = {}
    for w in windows:
        for e in _track_events(specs[w], temp_c, min_drop, max_rms, v_range, w, log, hum):
            if not any(same(e, g) for g in out):
                out.append(e)
    # templates catch steep passes that break ridge tracking, then the ridge is re-traced
    from .model import fit_line
    c = sound_speed(temp_c)
    times, freqs, L = specs[0.5] if 0.5 in specs else ridge_spectrogram(x, sr, fmax=fmax, win_s=0.5)
    cents = 1200 * np.log2(freqs[1] / freqs[0])
    tev, _ = detect(x, sr, temp_c, log=lambda *_: None)
    n0 = len(out)
    mask = hum[0.5] if 0.5 in hum else _steady(ridges(L, smooth=False), L.shape)
    for te in tev:
        if any(same(te, g) for g in out):
            continue
        ext = _extend(dict(t0=te["t"], f0=te["f0"], v=te["v"], d=te["d"]), L, times, freqs, cents, c, mask=mask)
        if ext is None:
            continue
        t, f = _trim(*ext, c)
        if len(t) < 16:
            continue
        fit = fit_line(t, f, c)
        if fit["rms_cents"] > max_rms or not v_range[0] <= fit["v"] <= v_range[1]:
            continue
        e = _event(t, f, fit, c, te["z"] * np.sqrt(len(t)) / 2, 1, [(t.round(2).tolist(), f.round(2).tolist(), 1.0)],
                   (t, f))
        if not any(same(e, g) for g in out):  # template pitch can be a harmonic
            out.append(e)
    log(f"template events: {len(tev)}, {len(out) - n0} new")
    return sorted(out, key=lambda e: e["t"])


# points that lie on their own straight-pass fit
def _trim(t, f, c, tol=15.0):
    from .model import fit_line
    for _ in range(3):
        fit = fit_line(t, f, c)
        res = (np.log(f) - np.log(fit["f0"]) - line_logratio(t, fit["t0"], fit["d"], fit["v"], c)) * CENTS
        keep = np.abs(res) < tol
        if keep.all() or keep.sum() < 20:
            break
        t, f = t[keep], f[keep]
    return t, f


def _event(t, f, fit, c, strength, n_tracks, curves, seed):
    prof = _profile(t, f, c, fit)
    vs, ds = [p[0] for p in prof], [p[1] for p in prof]
    return dict(t=round(fit["t0"], 2), f0=round(fit["f0"], 1), v=round(fit["v"], 1), d=round(fit["d"]),
                rms=round(fit["rms_cents"], 1), strength=round(float(strength), 1), n_tracks=n_tracks,
                # leave room for planes that turn, climb or slow
                v_lo=round(min(vs) / 1.4, 1), v_hi=round(max(vs) * 1.4, 1), d_lo=round(min(ds) / 2), d_hi=round(max(ds) * 2),
                profile=[(round(v, 1), round(d), round(t0, 2)) for v, d, t0 in prof],
                t_span=(float(t[0]), float(t[-1])), points=(np.asarray(t).round(2).tolist(), np.asarray(f).round(2).tolist()),
                seed=(np.asarray(seed[0]).round(2).tolist(), np.asarray(seed[1]).round(2).tolist()), curves=curves)


# doppler events from ridge tracks that fit a straight pass; harmonics of a pass become one event
def _track_events(spec, temp_c, min_drop, max_rms, v_range, win_s, log, hum=None):
    from .model import fit_line
    c = sound_speed(temp_c)
    times, freqs, L = spec
    cents = 1200 * np.log2(freqs[1] / freqs[0])
    tr = ridges(L, smooth=win_s >= 1.0)
    mask = _steady(tr, L.shape)
    if hum is not None:
        hum[win_s] = mask
    pts = [(times[fr], freqs[0] * 2 ** (k * cents / 1200), lv) for fr, k, lv in tr]
    found = []
    for ti, (t, f, lv) in enumerate(pts):
        n = max(3, len(t) // 10)
        if np.median(f[:n]) / np.median(f[-n:]) < 1 + min_drop:  # not falling: hum, engines, music
            continue
        fit = fit_line(t, f, c)
        if fit["rms_cents"] > max_rms or not v_range[0] <= fit["v"] <= v_range[1]:
            continue
        if not t[0] - 0.5 * (t[-1] - t[0]) <= fit["t0"] <= t[-1] + 0.5 * (t[-1] - t[0]):
            continue
        found.append((float(np.mean(lv) * np.sqrt(len(lv))), ti, fit))
    found.sort(key=lambda a: -a[0])
    used, out = set(), []
    for strength, ti, fit in found:
        if ti in used:
            continue
        # pieces of the same pass and its harmonics join the event
        members = [(ti, 1.0)] + _absorb(fit, pts, used | {ti}, c)
        used.update(m for m, _ in members)
        t = np.concatenate([pts[m][0] for m, _ in members])
        f = np.concatenate([pts[m][1] / r for m, r in members])
        o = np.argsort(t)
        t, f = t[o], f[o]
        if len(members) > 1:
            refit = fit_line(t, f, c)
            if refit["rms_cents"] <= 1.6 * fit["rms_cents"] + 2:
                fit = refit
            else:  # pieces disagree: keep the track alone
                members = members[:1]
                t, f = pts[ti][0], pts[ti][1]
        t_seed, f_seed = t, f  # extension must still explain the seed ridge
        for _ in range(2):  # follow the curve further, and on its harmonics
            ext = _extend(fit, L, times, freqs, cents, c, mask=mask)
            if ext is None:
                break
            te, fe = _trim(*ext, c)  # band catches other lines: keep points on the curve
            refit = fit_line(te, fe, c)
            seed = (np.log(f_seed) - np.log(refit["f0"]) -
                    line_logratio(t_seed, refit["t0"], refit["d"], refit["v"], c)) * CENTS
            if len(te) < 1.2 * len(t) or refit["rms_cents"] > 1.5 * fit["rms_cents"] + 2 or \
                    np.mean(np.abs(seed) < 15) < 0.9:
                break
            t, f, fit = te, fe, refit
        st = float(sum(np.mean(pts[m][2]) * np.sqrt(len(pts[m][2])) for m, _ in members))
        out.append(_event(t, f, fit, c, st, len(members),
                          [(pts[m][0].round(2).tolist(), pts[m][1].round(2).tolist(), r) for m, r in members],
                          (t_seed, f_seed)))
    kept = []  # tracks of one pass that didn't join (e.g. near-octave): keep the stronger
    for e in sorted(out, key=lambda e: -e["strength"]):
        g = next((g for g in kept if abs(e["t"] - g["t"]) < max(10.0, 0.3 * g["d"] / g["v"]) and
                  min(abs(np.log(e["f0"] / g["f0"] / r)) for r in _RATIOS) < 0.06), None)
        if g is not None:
            g["n_tracks"] += e["n_tracks"]
        else:
            kept.append(e)
    out = sorted(kept, key=lambda e: e["t"])
    log(f"ridge events ({win_s:g} s window): {len(tr)} tracks, {len(found)} fit a pass, {len(out)} events")
    return out


# re-trace an event along its fitted curve and harmonics (Viterbi in a band around the model pitch)
# returns (t, f scaled to the fundamental) or None
# mask: cells on steady tones, which the path may not use
def _extend(fit, L, times, freqs, cents, c, band=12, min_db=5.0, min_run=8, ratios=(1.0, 2.0, 3.0, 0.5), mask=None):
    from scipy.ndimage import median_filter
    w = min(150.0, max(40.0, 3.5 * fit["d"] / fit["v"]))
    sel = np.flatnonzero((times >= fit["t0"] - w) & (times <= fit["t0"] + w))
    if len(sel) < min_run:
        return None
    t = times[sel]
    lf = np.log(fit["f0"]) + line_logratio(t, fit["t0"], fit["d"], fit["v"], c)
    K = L.shape[1]
    T, Fq = [], []
    for r in ratios:
        kc = (lf + np.log(r) - np.log(freqs[0])) * CENTS / cents  # model pitch in bins
        if kc.min() < band or kc.max() > K - 1 - band:
            continue
        off = np.arange(-band, band + 1)
        idx = np.rint(kc).astype(int)[:, None] + off[None, :]
        B = L[sel[:, None], idx]  # frames x band
        if mask is not None:
            B = np.where(mask[sel[:, None], idx], -10.0, B)
        path = audio.viterbi(B.T, 1, 0.5)
        val = median_filter(B[np.arange(len(sel)), path], 5, mode="nearest")
        on = val >= min_db
        edges = np.flatnonzero(np.diff(np.r_[0, on.astype(int), 0]))
        for a, b in zip(edges[::2], edges[1::2]):
            if b - a >= min_run:
                k = idx[np.arange(a, b), path[a:b]]
                T.append(t[a:b])
                Fq.append(freqs[0] * 2 ** (k * cents / 1200) / r)
    if not T:
        return None
    t, f = np.concatenate(T), np.concatenate(Fq)
    o = np.argsort(t)
    return t[o], f[o]


# tracks (index, harmonic ratio) on fit's curve or a harmonic of it for most of their points
def _absorb(fit, pts, used, c, tol_cents=12.0, min_frac=0.75, min_pts=8,
            ratios=(1 / 3, 1 / 2, 2 / 3, 3 / 4, 4 / 3, 3 / 2, 2, 5 / 2, 3, 4)):
    w = 4 * fit["d"] / fit["v"] + 30
    lf0 = np.log(fit["f0"])
    got = []
    for ti, (t, f, _) in enumerate(pts):
        if ti in used or len(t) < min_pts or t[-1] < fit["t0"] - w or t[0] > fit["t0"] + w:
            continue
        model = lf0 + line_logratio(t, fit["t0"], fit["d"], fit["v"], c)
        drop = (model[0] - model[-1]) * CENTS
        n = max(2, len(t) // 8)
        own = (np.median(np.log(f[:n])) - np.median(np.log(f[-n:]))) * CENTS
        if drop < 15 or own < 0.5 * drop:  # flat tails match any steady line
            continue
        for r in (1.0,) + tuple(ratios):
            dev = np.abs(np.log(f / r) - model) * CENTS
            if np.mean(dev < tol_cents) >= min_frac:
                got.append((ti, r))
                break
    return got
