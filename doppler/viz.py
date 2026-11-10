# writes a self-contained html viewer for a heat.npz result
import base64
import io
import json
import os
import shutil

import numpy as np

from . import adsb
from .model import sound_speed


def _png(Z):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = Z / max(Z.max(), 1e-9)
    rgba = plt.get_cmap("inferno")(np.clip(0.25 + 0.75 * z, 0, 1))
    rgba[..., 3] = np.clip(z * 1.6, 0, 1) ** 0.6
    buf = io.BytesIO()
    plt.imsave(buf, rgba[::-1], format="png")  # row 0 = north
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def build(heat_npz, clip, tracks_csv, out_dir, window=300.0, step=2.0):
    h = np.load(heat_npz)
    os.makedirs(out_dir, exist_ok=True)
    S = float(h["start"])
    times, freqs, L = h["times"], h["freqs"], h["L"]
    sigma = float(L.std())
    q = np.clip(L / sigma * 32 + 64, 0, 255).astype(np.uint8)
    dur = float(times[-1] + 0.5)
    cents = float(1200 * np.log2(freqs[1] / freqs[0]))

    segs = []
    tracks, kind = adsb.load_csv(tracks_csv, with_kind=True)
    for s in adsb.segments(tracks, kind=kind):
        m = (s["t"] > S - window) & (s["t"] < S + dur + window)
        if m.sum() < 3:
            continue
        sel = np.flatnonzero(m)[::int(step)]
        segs.append(dict(icao=s["icao"], kind=s["kind"], t0=float(s["t"][sel[0]] - S), dt=step,
                         **{k: np.round(s[k][sel], 6 if k in ("lat", "lon") else 1).tolist()
                            for k in ("lat", "lon", "alt", "ve", "vn", "vu")}))

    ext = os.path.splitext(clip)[1] or ".m4a"
    shutil.copy(clip, os.path.join(out_dir, "clip" + ext))
    Z = h["Z"]
    data = dict(
        start=S, dur=dur, c=sound_speed(float(h["temp"])), elev=float(h["elev"]),
        clockTol=float(h["clock_tol"]) if "clock_tol" in h else 2.0,
        jets=bool(h["jets"]) if "jets" in h else False,
        bbox=[float(v) for v in h["bbox"]], truth=[float(v) for v in h["truth"]],
        results=json.loads(str(h["results"])), audio="clip" + ext,
        heat=dict(png=_png(Z), zmax=float(Z.max())),
        spec=dict(F=int(q.shape[0]), K=int(q.shape[1]), sigma=sigma, fmin=float(freqs[0]),
                  cents=cents, t0=float(times[0]), hop=float(times[1] - times[0]),
                  q=base64.b64encode(q.tobytes()).decode()),
        segs=segs)
    tpl = open(os.path.join(os.path.dirname(__file__), "viz.html")).read()
    out = os.path.join(out_dir, "index.html")
    with open(out, "w") as fh:
        fh.write(tpl.replace("/*DATA*/null", json.dumps(data, separators=(",", ":"))))
    return out
