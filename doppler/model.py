import numpy as np

R_EARTH = 6371000.0
CENTS = 1200 / np.log(2)  # log-ratio -> cents


def sound_speed(temp_c=15.0):
    return 331.3 * np.sqrt(1 + temp_c / 273.15)


# local flat-earth enu in meters (fine within a few tens of km)
def to_enu(lat, lon, alt, lat0, lon0, alt0=0.0):
    e = np.radians(lon - lon0) * R_EARTH * np.cos(np.radians(lat0))
    n = np.radians(lat - lat0) * R_EARTH
    return e, n, alt - alt0


def from_enu(e, n, lat0, lon0):
    lat = lat0 + np.degrees(n / R_EARTH)
    lon = lon0 + np.degrees(e / (R_EARTH * np.cos(np.radians(lat0))))
    return lat, lon


# log(f_obs/f0) for a straight, constant-speed pass
# t0: reception time of sound emitted at closest approach, d: slant closest-approach distance, v: speed
def line_logratio(t, t0, d, v, c):
    T = t - (t0 - d / c)  # time since the CPA emission
    a = c * c - v * v
    u = (c * c * T - np.sqrt(c * c * v * v * T * T + d * d * a)) / a  # emission time rel. CPA
    r = c * (T - u)
    return -np.log1p(v * v * u / r / c)


def _fit(t, lf, c, starts, lo, hi):
    from .heatmap import _kernel
    out = np.empty(6)
    t, lf = np.ascontiguousarray(t, np.float64), np.ascontiguousarray(lf, np.float64)
    _kernel().line_fit(t, lf, len(t), c, np.ascontiguousarray(starts, np.float64), len(starts),
                       np.array(lo, np.float64), np.array(hi, np.float64), 2e-3, out)
    return dict(t0=float(out[0]), d=float(np.exp(out[1])), v=float(out[2]), f0=float(np.exp(out[3])),
                rms_cents=float(out[5] * CENTS))


# fit the straight-pass model to a ridge; returns dict(t0, d, v, f0, rms_cents)
def fit_line(t, f, c):
    lf = np.log(f)
    n = max(3, len(t) // 10)
    f_hi, f_lo = np.median(f[:n]), np.median(f[-n:])
    ratio = max(f_hi / f_lo, 1.001)
    v0 = np.clip(c * (ratio - 1) / (ratio + 1), 15, 300)
    lf0 = np.log(2 / (1 / f_hi + 1 / f_lo))
    t0 = t[np.argmin(np.abs(lf - lf0))]
    starts = [[t0, np.log(d0), min(v, 299), lf0] for d0 in (150, 400, 1000, 2500, 6000) for v in (v0, 0.7 * v0 + 30)]
    return _fit(t, lf, c, starts, [t[0] - 60, np.log(30), 10, lf0 - 0.5], [t[-1] + 60, np.log(30000), 320, lf0 + 0.5])


# same fit with the speed held at v, started from fit
def fit_line_at(t, f, c, v, fit, pad=15.0):
    lf0 = np.log(fit["f0"])
    return _fit(t, np.log(f), c, [[fit["t0"], np.log(fit["d"]), v, lf0]],
                [t[0] - pad, np.log(30), v, lf0 - 0.5], [t[-1] + pad, np.log(30000), v, lf0 + 0.5])


# predicted log(f_obs/f0) at reception times t_query (unix s) for one track segment
def seg_logratio(seg, lat, lon, elev, t_query, c):
    e, n, u = to_enu(seg["lat"], seg["lon"], seg["alt"], lat, lon, elev)
    r = np.sqrt(e * e + n * n + u * u)
    rdot = (e * seg["ve"] + n * seg["vn"] + u * seg["vu"]) / r
    t_rec = seg["t"] + r / c
    return np.interp(t_query, t_rec, -np.log1p(rdot / c), left=np.nan, right=np.nan)
