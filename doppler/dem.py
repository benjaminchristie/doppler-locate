# ground elevation (m above sea level) from aws terrain tiles (terrarium png), cached under data/dem/
import io
import os
import urllib.request

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STORE = os.path.join(ROOT, "data", "dem")
URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
ZOOM = 11  # ~60-75 m per pixel at mid latitudes


def _frac(lat, lon, z):
    n = 2 ** z
    x = (np.asarray(lon, float) + 180) / 360 * n
    y = (1 - np.arcsinh(np.tan(np.radians(lat))) / np.pi) / 2 * n
    return x, y


def _tile(z, x, y):
    path = os.path.join(STORE, str(z), str(x), f"{y}.npy")
    if os.path.exists(path):
        return np.load(path)
    from PIL import Image
    req = urllib.request.Request(URL.format(z=z, x=x, y=y), headers={"User-Agent": "doppler-tracking"})
    with urllib.request.urlopen(req, timeout=60) as r:
        rgb = np.asarray(Image.open(io.BytesIO(r.read())).convert("RGB"), np.float32)
    elev = (rgb[..., 0] * 256 + rgb[..., 1] + rgb[..., 2] / 256 - 32768).astype(np.float32)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.npy"  # other processes may be reading
    np.save(tmp, elev)
    os.replace(tmp, path)
    return elev


# bilinear elevation at each (lat, lon); sea is ~0
def elevation(lat, lon, z=ZOOM):
    lat, lon = np.broadcast_arrays(np.asarray(lat, float), np.asarray(lon, float))
    fx, fy = _frac(lat, lon, z)
    px, py = fx * 256 - 0.5, fy * 256 - 0.5  # pixel centres
    out = np.empty(lat.shape, np.float32)
    flat = out.reshape(-1)
    px, py = px.reshape(-1), py.reshape(-1)
    x0, y0 = np.floor(px).astype(np.int64), np.floor(py).astype(np.int64)
    wx, wy = (px - x0).astype(np.float32), (py - y0).astype(np.float32)
    acc = np.zeros(len(flat), np.float32)
    for dx, dy, w in ((0, 0, (1 - wx) * (1 - wy)), (1, 0, wx * (1 - wy)), (0, 1, (1 - wx) * wy), (1, 1, wx * wy)):
        X, Y = x0 + dx, y0 + dy
        tx, ty = X // 256, Y // 256
        key = tx * (1 << 24) + ty
        for k in np.unique(key):
            m = key == k
            t = _tile(z, int(k >> 24), int(k & ((1 << 24) - 1)))
            acc[m] += w[m] * t[Y[m] % 256, X[m] % 256]
    flat[:] = acc
    return out


# elevation at local enu points (x east, y north, m) around (lat0, lon0)
def grid_elevation(lat0, lon0, x, y):
    from .model import from_enu
    la, lo = from_enu(np.asarray(x, float), np.asarray(y, float), lat0, lon0)
    return elevation(la, lo)


# elevation on a local enu grid over a bbox, for fast lookups of many points
class Grid:

    def __init__(self, bbox, lat0, lon0, step=None):
        from .model import to_enu
        e0, n0, _ = to_enu(bbox[0], bbox[2], 0, lat0, lon0)
        e1, n1, _ = to_enu(bbox[1], bbox[3], 0, lat0, lon0)
        span = max(e1 - e0, n1 - n0)
        self.step = step or float(np.clip(span / 1500, 60, 500))
        z = int(np.clip(round(np.log2(40075e3 * np.cos(np.radians(lat0)) / 256 / self.step)), 6, 12))
        self.x0, self.y0 = e0 - 2 * self.step, n0 - 2 * self.step
        self.xs = self.x0 + np.arange(int((e1 - e0) / self.step) + 5) * self.step
        self.ys = self.y0 + np.arange(int((n1 - n0) / self.step) + 5) * self.step
        X, Y = np.meshgrid(self.xs, self.ys)
        from .model import from_enu
        la, lo = from_enu(X, Y, lat0, lon0)
        self.z = elevation(la, lo, z).astype(np.float32)

    def __call__(self, x, y):
        fx = np.clip((np.asarray(x) - self.x0) / self.step, 0, len(self.xs) - 1.001)
        fy = np.clip((np.asarray(y) - self.y0) / self.step, 0, len(self.ys) - 1.001)
        i, j = fx.astype(int), fy.astype(int)
        wx, wy = fx - i, fy - j
        Z = self.z
        return ((1 - wy) * ((1 - wx) * Z[j, i] + wx * Z[j, i + 1]) + wy * ((1 - wx) * Z[j + 1, i] + wx * Z[j + 1, i + 1]))
