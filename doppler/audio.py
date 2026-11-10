import json
import re
import subprocess

import numpy as np
from scipy.ndimage import median_filter

from .timeutil import parse_time


# start time (utc) and location tags from container metadata
def probe(path):
    out = subprocess.run(["ffprobe", "-v", "quiet", "-print_format", "json",
                          "-show_format", "-show_streams", path],
                         capture_output=True, text=True, check=True).stdout
    info = json.loads(out)
    tags = dict(info["format"].get("tags", {}))
    for s in info.get("streams", []):
        for k, v in s.get("tags", {}).items():
            tags.setdefault(k, v)
    tags = {k.lower(): v for k, v in tags.items()}

    start = None
    # apple's tag carries the local offset; creation_time is utc
    for key in ("com.apple.quicktime.creationdate", "creation_time"):
        if key in tags:
            try:
                start = parse_time(tags[key])
                break
            except ValueError:
                continue
    loc = tags.get("com.apple.quicktime.location.iso6709") or tags.get("location")
    return dict(start=start, duration=float(info["format"].get("duration", 0)),
                location=loc, tags=tags)


# '+51.5007-000.1246+011.000/' to (lat, lon, alt or None); None if not decimal degrees
def parse_iso6709(s):
    m = re.match(r"^([+-]\d{1,2}(?:\.\d+)?)([+-]\d{1,3}(?:\.\d+)?)([+-]\d+(?:\.\d+)?)?", (s or "").strip())
    if not m:
        return None
    lat, lon = float(m[1]), float(m[2])
    if abs(lat) > 90 or abs(lon) > 180:
        return None
    return lat, lon, float(m[3]) if m[3] else None


# mono float32 at sr via ffmpeg
def load_audio(path, sr=8000):
    raw = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-ac", "1", "-ar", str(sr),
                          "-f", "f32le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).copy()


# power spectrogram in [fmin, fmax]; returns (times, freqs, P[f, t])
def stft_band(x, sr, win_s, hop_s, fmin, fmax, pad=4, chunk=256):
    nper = int(win_s * sr)
    hop = max(1, int(hop_s * sr))
    nfft = pad * nper
    freqs = np.fft.rfftfreq(nfft, 1 / sr)
    band = (freqs >= fmin) & (freqs <= fmax)
    frames = np.lib.stride_tricks.sliding_window_view(x, nper)[::hop]
    w = np.hanning(nper).astype(np.float32)
    P = np.empty((band.sum(), len(frames)), np.float32)
    for i in range(0, len(frames), chunk):
        Z = np.fft.rfft(frames[i:i + chunk] * w, n=nfft)[:, band]
        P[:, i:i + chunk] = (Z.real ** 2 + Z.imag ** 2).T
    times = (np.arange(len(frames)) * hop + nper / 2) / sr
    return times, freqs[band], P


# dB above each bin's median over time (kills stationary hum)
def whiten_db(P):
    return 10 * np.log10(P + 1e-20) - 10 * np.log10(np.median(P, axis=1, keepdims=True) + 1e-20)


# best path through L[f, t] with |bin step| <= max_jump per frame
def viterbi(L, max_jump, penalty):
    nf, nt = L.shape
    shifts = np.arange(-max_jump, max_jump + 1)
    acc = L[:, 0].astype(np.float64)
    back = np.zeros((nf, nt), np.int16)
    cand = np.empty((len(shifts), nf))
    for j in range(1, nt):
        cand.fill(-np.inf)
        for i, s in enumerate(shifts):
            if s >= 0:
                cand[i, s:] = acc[:nf - s] - penalty * abs(s)
            else:
                cand[i, :nf + s] = acc[-s:] - penalty * abs(s)
        bi = cand.argmax(0)
        acc = cand[bi, np.arange(nf)] + L[:, j]
        back[:, j] = shifts[bi]
    path = np.empty(nt, int)
    path[-1] = acc.argmax()
    for j in range(nt - 1, 0, -1):
        path[j - 1] = path[j] - back[path[j], j]
    return path


# track the dominant tone in a time/frequency box, optionally stacking harmonics
# returns times (s from file start), freqs (Hz, sub-bin), snr (mean dB above median)
def extract_ridge(x, sr, t_lo, t_hi, f_lo, f_hi, win_s=1.0, hop_s=0.1,
                  max_slope=20.0, penalty=0.5, harmonics=1):
    a, b = int(max(0, t_lo) * sr), min(int(t_hi * sr), len(x))
    times, fr, P = stft_band(x[a:b], sr, win_s, hop_s, f_lo, min(harmonics * f_hi, sr / 2 - 1))
    Lall = np.clip(whiten_db(P), -10, 40)
    df = fr[1] - fr[0]
    freqs = fr[fr <= f_hi]
    idx = [np.clip(np.round((k * freqs - fr[0]) / df).astype(int), 0, len(fr) - 1)
           for k in range(1, harmonics + 1)]
    L = np.mean([Lall[i] for i in idx], axis=0)
    J = max(1, int(np.ceil(max_slope * hop_s / df)))
    path = viterbi(L, J, penalty)
    # parabolic sub-bin refinement on the stacked score
    k = np.clip(path, 1, len(freqs) - 2)
    cols = np.arange(len(k))
    y0, y1, y2 = L[k - 1, cols], L[k, cols], L[k + 1, cols]
    den = y0 - 2 * y1 + y2
    with np.errstate(invalid="ignore", divide="ignore"):
        off = np.where(np.abs(den) > 1e-9, 0.5 * (y0 - y2) / den, 0.0)
    f = freqs[k] + np.clip(off, -1, 1) * df
    snr = median_filter(L[path, cols], size=max(3, int(1.0 / hop_s) | 1), mode="nearest")
    return times + a / sr, f, snr


# frames above min_snr, dropping runs shorter than min_run_s
def keep_mask(t, snr, min_snr, min_run_s=3.0):
    m = snr >= min_snr
    edges = np.flatnonzero(np.diff(np.r_[0, m.astype(int), 0]))
    for s, e in zip(edges[::2], edges[1::2]):
        if t[e - 1] - t[s] < min_run_s:
            m[s:e] = False
    return m
