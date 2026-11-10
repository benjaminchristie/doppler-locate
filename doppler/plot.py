import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .model import CENTS, line_logratio, seg_logratio

plt.rcParams.update({"font.family": "serif", "font.serif": ["Palatino Linotype", "Palatino", "DejaVu Serif"],
                     "font.size": 13})


def spectrogram(times, freqs, L, path, ridge=None, fit_curve=None):
    w = min(40, max(8, (times[-1] - times[0]) / 30))
    fig, ax = plt.subplots(figsize=(w, 6))
    ax.imshow(L, origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=25,
              extent=[times[0], times[-1], freqs[0], freqs[-1]])
    if ridge is not None:
        ax.plot(*ridge, ".", ms=2, color="cyan")
    if fit_curve is not None:
        ax.plot(*fit_curve, "-", lw=1.5, color="#ff9900")
    ax.set_xlabel("time in clip (s)")
    ax.set_ylabel("Hz")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def result(loc, best, tracks_segs, path, truth=None):
    flys = loc.flys
    fig = plt.figure(figsize=(6 + 5 * len(flys), 6))
    ax = fig.add_subplot(1, 1 + len(flys), 1)
    t_lo = min(loc.tau0) - loc.W - 600
    t_hi = max(loc.tau0) + loc.W + 600
    for s in tracks_segs:
        m = (s["t"] > t_lo) & (s["t"] < t_hi)
        if m.any():
            ax.plot(s["lon"][m], s["lat"][m], color="0.8", lw=0.6)
    for seg in best["_segs"]:
        if seg is not None:
            ax.plot(seg["lon"], seg["lat"], color="#ff9900", lw=1.5)
    ax.plot(best["lon"], best["lat"], "k*", ms=14, label="estimate")
    if truth:
        ax.plot(truth[1], truth[0], "o", mfc="none", mec="tab:blue", ms=14, mew=2, label="truth")
    ax.set_aspect(1 / np.cos(np.radians(best["lat"])))
    span = 0.06
    ax.set_xlim(best["lon"] - span, best["lon"] + span)
    ax.set_ylim(best["lat"] - span * 0.8, best["lat"] + span * 0.8)
    ax.legend(loc="upper right")
    ax.set_title(f"{best['rms_cents']:.1f} cents rms")

    for k, fo in enumerate(flys):
        a = fig.add_subplot(1, 1 + len(flys), 2 + k)
        a.plot(fo.t, fo.logf * CENTS, ".", ms=2, color="0.4", label="audio")
        seg = best["_segs"][k]
        if seg is not None:
            lr = seg_logratio(seg, best["lat"], best["lon"], loc.elev,
                              loc.S + best["clock_offset"] + fo.t, loc.c)
            off = np.nanmedian(fo.logf - lr)
            a.plot(fo.t, (lr + off) * CENTS, color="#ff9900", lw=2, label=seg["icao"])
        p = fo.fit
        a.plot(fo.t, (line_logratio(fo.t, p["t0"], p["d"], p["v"], loc.c) + np.log(p["f0"])) * CENTS,
               "--", color="tab:blue", lw=1, label="line fit")
        a.set_xlabel("time in clip (s)")
        a.set_ylabel("cents (rel. 1 Hz)")
        a.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
