import numpy as np
from scipy.optimize import minimize

from .model import CENTS, fit_line, from_enu, seg_logratio, to_enu

CAP = 30.0  # cents; worse flyovers count as unexplained, so one bad ridge cannot sink the answer


class Flyover:
    def __init__(self, t, f, c, name=""):
        self.t, self.f, self.logf, self.name = t, f, np.log(f), name
        self.fit = fit_line(t, f, c)


def _state(seg, tau):
    if not (seg["t"][0] <= tau <= seg["t"][-1]):
        return None
    return int(round((tau - seg["t"][0]) / (seg["t"][1] - seg["t"][0])))


class Locator:
    def __init__(self, flys, segs, start, c, elev=0.0, clock_tol=30.0, bbox=None):
        self.flys, self.S, self.c, self.elev, self.W, self.bbox = flys, start, c, elev, clock_tol, bbox
        self.tau0 = [self.S + fo.fit["t0"] - fo.fit["d"] / c for fo in flys]
        span = self.W + 900
        self.near = [[s for s in segs if s["t"][0] < tau + span and s["t"][-1] > tau - span]
                     for tau in self.tau0]

    def candidates(self, step=2.0, v_tol=0.2):
        out = []
        dts = np.arange(-self.W, self.W + 1e-9, step) if self.W > 0 else [0.0]
        for k, fo in enumerate(self.flys):
            v = fo.fit["v"]
            # distance is the least certain fit parameter, so seed a range
            for d, dt in ((fo.fit["d"] * m, dt) for m in (0.7, 1.0, 1.4) for dt in dts):
                for seg in self.near[k]:
                    i = _state(seg, self.tau0[k] + dt)
                    if i is None:
                        continue
                    ve, vn, vu = seg["ve"][i], seg["vn"][i], seg["vu"][i]
                    gs = np.hypot(ve, vn)
                    if gs < 1 or abs(np.sqrt(gs * gs + vu * vu) - v) > v_tol * v + 15:
                        continue
                    h = seg["alt"][i] - self.elev
                    if h <= 0 or h > d + 300:
                        continue
                    s = np.sqrt(max(d * d - h * h, 0.0))
                    for sign in (1, -1):
                        la, lo = from_enu(-sign * s * vn / gs, sign * s * ve / gs,
                                          seg["lat"][i], seg["lon"][i])
                        if self._inside(la, lo):
                            out.append((la, lo, dt))
        return np.array(out).reshape(-1, 3)

    def _inside(self, la, lo):
        if self.bbox is None:
            return True
        return self.bbox[0] <= la <= self.bbox[1] and self.bbox[2] <= lo <= self.bbox[3]

    def fly_cost(self, k, seg, lat, lon, dt):
        fo = self.flys[k]
        lr = seg_logratio(seg, lat, lon, self.elev, self.S + dt + fo.t, self.c)
        ok = np.isfinite(lr)
        if ok.mean() < 0.8:
            return np.inf
        res = fo.logf[ok] - lr[ok]
        res = res - np.median(res)  # unknown source frequency
        a = np.sort(np.abs(res))[: max(3, int(0.9 * len(res)))]
        return float(np.sqrt(np.mean(a * a)) * CENTS)

    def evaluate(self, lat, lon, dt, assign=None):
        costs, ids = [], []
        for k, fo in enumerate(self.flys):
            best, bseg = CAP, None
            if assign is not None:
                pool = [assign[k]] if assign[k] is not None else []
            else:
                pool = []
                radius = 3 * fo.fit["d"] + 3000
                for seg in self.near[k]:
                    i = _state(seg, self.tau0[k] + dt)
                    if i is None:
                        continue
                    e, n, _ = to_enu(seg["lat"][i], seg["lon"][i], 0, lat, lon)
                    if e * e + n * n < radius * radius:
                        pool.append(seg)
            for seg in pool:
                c_ = self.fly_cost(k, seg, lat, lon, dt)
                if c_ < best:
                    best, bseg = c_, seg
            costs.append(best)
            ids.append(bseg)
        return float(np.sqrt(np.mean(np.square(costs)))), costs, ids

    def refine(self, lat, lon, dt, assign):
        def obj(p):
            if abs(dt + p[2]) > self.W:
                return 1e6
            la, lo = from_enu(p[0], p[1], lat, lon)
            return self.evaluate(la, lo, dt + p[2], assign)[0]

        simplex = np.array([[0, 0, 0], [60, 0, 0], [0, 60, 0], [0, 0, 1.0 if self.W else 0]])
        if self.W == 0:
            simplex = simplex[:3, :2]
            f = lambda p: obj([p[0], p[1], 0])
        else:
            f = obj
        r = minimize(f, simplex[0], method="Nelder-Mead",
                     options=dict(initial_simplex=simplex, xatol=1.0, fatol=0.01, maxiter=600))
        p = list(r.x) + [0.0] * (3 - len(r.x))
        la, lo = from_enu(p[0], p[1], lat, lon)
        return la, lo, dt + p[2]

    def run(self, n_refine=25, min_sep=150.0, log=print):
        cands = self.candidates()
        log(f"{len(cands)} candidate positions")
        scored = []
        for la, lo, dt in cands:
            cost, _, ids = self.evaluate(la, lo, dt)
            scored.append((cost, la, lo, dt, ids))
        scored.sort(key=lambda x: x[0])

        seeds = []
        for s in scored:
            if all(_dist(s[1], s[2], q[1], q[2]) > min_sep for q in seeds):
                seeds.append(s)
            if len(seeds) >= n_refine:
                break

        results = []
        for cost0, la, lo, dt, ids in seeds:
            la, lo, dt = self.refine(la, lo, dt, ids)
            cost, costs, ids = self.evaluate(la, lo, dt)
            results.append(dict(lat=float(la), lon=float(lo), clock_offset=float(dt),
                                rms_cents=cost, per_flyover_cents=[round(c, 2) for c in costs],
                                icao=[s["icao"] if s else None for s in ids], _segs=ids))
        results.sort(key=lambda r: r["rms_cents"])
        final = []
        for r in results:
            if all(_dist(r["lat"], r["lon"], q["lat"], q["lon"]) > 30 for q in final):
                final.append(r)
        return final


def _dist(la1, lo1, la2, lo2):
    e, n, _ = to_enu(la2, lo2, 0, la1, lo1)
    return float(np.hypot(e, n))
