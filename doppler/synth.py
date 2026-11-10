# synthetic test case: fake adsb day plus an audio clip with doppler-shifted engine tones
import csv
import json
import os
import subprocess
from datetime import datetime, timezone

import numpy as np
from scipy.signal import lfilter

from .adsb import COLS, FPM, FT, KT
from .model import from_enu, sound_speed
from .timeutil import parse_time


# constant-speed coordinated turn through p_ref at t_ref, sampled on uniform t
def _fly(t, t_ref, p_ref, heading, speed, turn_dps, vz):
    hd = np.radians(heading + turn_dps * (t - t_ref))
    ve, vn = speed * np.sin(hd), speed * np.cos(hd)
    dt = t[1] - t[0]
    i = np.argmin(np.abs(t - t_ref))
    e = np.cumsum(ve) * dt
    n = np.cumsum(vn) * dt
    e += p_ref[0] - e[i]
    n += p_ref[1] - n[i]
    u = np.maximum(p_ref[2] + vz * (t - t_ref), 50.0)
    return e, n, u, ve, vn, np.full_like(t, vz)


def _adsb_rows(rng, icao, t_abs, lat0, lon0, e, n, u, ve, vn, vz):
    rows, bias = [], rng.normal(0, 15)
    i = 0
    while i < len(t_abs):
        la, lo = from_enu(e[i] + rng.normal(0, 8), n[i] + rng.normal(0, 8), lat0, lon0)
        gs = np.hypot(ve[i], vn[i])
        rows.append([icao, round(t_abs[i], 2), round(la, 6), round(lo, 6),
                     round((u[i] + bias) / (25 * FT)) * 25 * FT,
                     round(gs / KT, 1) * KT, round(np.degrees(np.arctan2(ve[i], vn[i])) % 360, 1),
                     round(vz[i] / (64 * FPM)) * 64 * FPM])
        i += int(rng.uniform(2, 12) / (t_abs[1] - t_abs[0]))
    return rows


# adds a doppler-shifted harmonic source to audio (exact retarded-time phase)
def _render(audio, ts, S, t, e, n, u, f0, amp, c, rng):
    dt = t[1] - t[0]
    r = np.sqrt(e ** 2 + n ** 2 + (u - 1.5) ** 2)
    tau = t - S
    t_rec = tau + r / c
    k = max(3, int(10 / dt))
    wobble = np.convolve(rng.normal(0, 1, len(t)), np.ones(k) / k, "same")
    f_src = f0 * (1 + 0.004 * wobble / (wobble.std() + 1e-9))
    phase = 2 * np.pi * np.cumsum(f_src) * dt
    ok = (ts > t_rec[0]) & (ts < t_rec[-1])
    if not ok.any():
        return r, tau, t_rec
    tau_s = np.interp(ts[ok], t_rec, tau)
    ph = np.interp(tau_s, tau, phase)
    r_s = np.interp(tau_s, tau, r)
    for h, a in ((1, 1.0), (2, 0.6), (3, 0.4)):
        alpha = 4.7 * (h * f0 / 1000) ** 1.6  # rough air absorption, dB/km
        audio[ok] += a * amp * 1000 / r_s * 10 ** (-alpha * r_s / 1000 / 20) \
            * np.sin(h * ph + rng.uniform(0, 2 * np.pi))
    return r, tau, t_rec


def make(out_dir, lat0=51.5007, lon0=-0.1246, start="2026-10-17T15:20:00Z", duration=900,
         n_pass=3, n_bg=250, seed=0, sr=16000, clock_error=3.0, temp_c=15.0, amp=0.006,
         bg_audible=True):
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)
    c = sound_speed(temp_c)
    S = parse_time(start).timestamp()
    hexid = lambda: f"{rng.integers(0, 2**24):06x}"
    rows, passes = [], []
    N = int(duration * sr)
    ts = np.arange(N) / sr
    audio = np.zeros(N)

    # background traffic over a ~80 km box; anything within 15 km is rendered too
    for _ in range(n_bg):
        t0 = S + rng.uniform(-2400, duration + 600)
        t = np.arange(t0, t0 + rng.uniform(600, 3600), 0.5)
        p = [rng.uniform(-40e3, 40e3), rng.uniform(-40e3, 40e3), rng.uniform(300, 12000)]
        trk = _fly(t, t[0], p, rng.uniform(0, 360), rng.uniform(60, 250),
                   rng.choice([0, rng.uniform(-0.5, 0.5)]), rng.uniform(-3, 3))
        rows += _adsb_rows(rng, hexid(), t, lat0, lon0, *trk)
        win = (t > S - 60) & (t < S + duration + 60)
        e, n, u = trk[:3]
        if bg_audible and win.any() and np.min(np.sqrt(e[win] ** 2 + n[win] ** 2 + u[win] ** 2)) < 15e3:
            _render(audio, ts, S, t, e, n, u, rng.uniform(70, 220),
                    amp * rng.lognormal(0, 0.7), c, rng)

    # audible passes, spaced through the clip
    t_cpas = np.sort(rng.uniform(0.12, 0.88, n_pass)) * duration
    while n_pass > 1 and np.min(np.diff(t_cpas)) < min(180, duration / n_pass / 1.5):
        t_cpas = np.sort(rng.uniform(0.12, 0.88, n_pass)) * duration
    for t_cpa in t_cpas:
        hd, v = rng.uniform(0, 360), rng.uniform(60, 110)
        s_ = rng.choice([-1, 1]) * rng.uniform(200, 2500)
        h = rng.uniform(300, 2000)
        p = [s_ * np.cos(np.radians(hd)), -s_ * np.sin(np.radians(hd)), h]
        t = np.arange(S + t_cpa - 700, S + t_cpa + 700, 0.02)
        trk = _fly(t, S + t_cpa, p, hd, v, rng.choice([0, rng.uniform(-0.1, 0.1)]),
                   rng.uniform(-4, 4))
        icao = hexid()
        rows += _adsb_rows(rng, icao, t, lat0, lon0, *trk)
        f0 = rng.uniform(70, 220)
        r, tau, t_rec = _render(audio, ts, S, t, *trk[:3], f0, amp * rng.lognormal(0, 0.3), c, rng)
        near = np.abs(tau - t_cpa) < 300
        i = np.flatnonzero(near)[np.argmin(r[near])]
        d, t_rx = float(r[i]), float(t_rec[i])
        passes.append(dict(icao=icao, t_cpa_rx=round(t_rx, 2), d=round(d, 1), v=round(v, 1),
                           f0=round(f0, 2),
                           t=[round(max(0, t_rx - 75), 1), round(min(duration, t_rx + 75), 1)],
                           f=[round(f0 * 0.72, 1), round(f0 * 1.45, 1)]))

    white = rng.normal(0, 1, N)
    audio += 0.05 * white + 0.02 * lfilter([1], [1, -0.98], rng.normal(0, 1, N)) / 7
    audio += 0.004 * np.sin(2 * np.pi * 60 * ts) + 0.002 * np.sin(2 * np.pi * 120 * ts)
    audio /= np.abs(audio).max() * 1.1

    with open(os.path.join(out_dir, "tracks.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(COLS)
        w.writerows(sorted(rows, key=lambda r: (r[0], r[1])))

    stamp = datetime.fromtimestamp(S + clock_error, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    clip = os.path.join(out_dir, "clip.m4a")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "f32le", "-ar", str(sr), "-ac", "1",
                    "-i", "-", "-c:a", "aac", "-b:a", "96k", "-metadata", f"creation_time={stamp}",
                    clip], input=audio.astype(np.float32).tobytes(), check=True)
    truth = dict(lat=lat0, lon=lon0, start=start, clock_error=clock_error, temp_c=temp_c,
                 bbox=[lat0 - 0.35, lat0 + 0.35, lon0 - 0.45, lon0 + 0.45], passes=passes)
    with open(os.path.join(out_dir, "truth.json"), "w") as fh:
        json.dump(truth, fh, indent=1)
    return truth
