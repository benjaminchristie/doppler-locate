# audio in: ffmpeg decode, container metadata, spectrograms
import json
import re
import subprocess

import numpy as np

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
