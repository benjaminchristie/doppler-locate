// matched filter: predicts each grid point's doppler curve for one aircraft,
// sums a log-frequency spectrogram along it for every source pitch, keeps the best
#include <algorithm>
#include <cmath>
#include <vector>

struct Seg {
    const double *e, *n, *u, *ve, *vn, *vu;
    int N;
    double t0, dt;
};

// position and velocity at time tau; false outside the track (clamped to the ends if clamp)
static inline bool at(const Seg& s, double tau, double* p, double* v, bool clamp = false) {
    double pos = (tau - s.t0) / s.dt;
    bool in = pos >= 0 && pos <= s.N - 1.001;
    if (!in) {
        if (!clamp) return false;
        pos = std::min(std::max(pos, 0.0), s.N - 1.001);
    }
    int i = (int)pos;
    double w = pos - i;
    p[0] = s.e[i] + w * (s.e[i + 1] - s.e[i]);
    p[1] = s.n[i] + w * (s.n[i + 1] - s.n[i]);
    p[2] = s.u[i] + w * (s.u[i + 1] - s.u[i]);
    v[0] = s.ve[i] + w * (s.ve[i + 1] - s.ve[i]);
    v[1] = s.vn[i] + w * (s.vn[i + 1] - s.vn[i]);
    v[2] = s.vu[i] + w * (s.vu[i + 1] - s.vu[i]);
    return in;
}

// emission time for reception time T (Newton on tau + r(tau)/c = T), warm-started from the previous range
// false if outside the track; d = aircraft - listener
static inline bool retarded(const Seg& s, double T, double gx, double gy, double gz, double c,
                            double& rprev, double* d, double* v, double& r) {
    double p[3], tau = T - (rprev > 0 ? rprev : 0) / c;
    int iters = rprev > 0 ? 2 : 6;
    for (int it = 0; it < iters; ++it) {
        at(s, tau, p, v, true);
        d[0] = p[0] - gx; d[1] = p[1] - gy; d[2] = p[2] - gz;
        r = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
        double rdot = (d[0] * v[0] + d[1] * v[1] + d[2] * v[2]) / r;
        tau -= (tau - T + r / c) / (1 + rdot / c);
    }
    bool in = at(s, tau, p, v, true);
    d[0] = p[0] - gx; d[1] = p[1] - gy; d[2] = p[2] - gz;
    r = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
    rprev = r;  // keeps a good warm start just off the track end
    return in;
}

// fastest the range can change per second of reception time
static double range_rate(const Seg& s, double c) {
    double vmax = 0;
    for (int i = 0; i < s.N; ++i)
        vmax = std::max(vmax, std::sqrt(s.ve[i] * s.ve[i] + s.vn[i] * s.vn[i] + s.vu[i] * s.vu[i]));
    return vmax / (1 - std::min(vmax / c, 0.8)) + 1;
}

// geometry at sorted reception times U for one listener: fn(u, d, v, r) for each time heard within r_max, ok[u] = 0 otherwise
// far away it skips ahead by the time the range needs to reach r_max
template <class Fn>
static inline void walk(const Seg& s, const double* U, int NU, double gx, double gy, double gz, double c,
                        double r_max, double rate, char* ok, Fn fn) {
    double tend = s.t0 + (s.N - 1.001) * s.dt, rprev = -1, skip = -1e300;
    for (int u = 0; u < NU; ++u) {
        ok[u] = 0;
        double T = U[u];
        if (T - r_max / c > tend) {  // emission after the track ends, for this and later times
            std::fill(ok + u, ok + NU, 0);
            break;
        }
        if (T < s.t0 || T < skip) { rprev = -1; continue; }
        double d[3], v[3], r;
        if (!retarded(s, T, gx, gy, gz, c, rprev, d, v, r)) continue;
        if (r > r_max) { skip = T + (r - r_max) / rate; rprev = -1; continue; }
        ok[u] = 1;
        fn(u, d, v, r);
    }
}

// clock offsets share geometry: frame f under offset di is heard at U[idx[di * F + f]]
// frames weighted by loudness min(1, r_ref / r) and pitch rate min(1, rate / rate_ref) in bins/s
// best = max over pitch of sum(w L) / (H sqrt(sum w^2)); gz is ground height
// outputs are [ND, G]; spread is how far the predicted pitch moves over the clip, in bins
extern "C" void mf_seg(const float* L, int F, int K, const double* U, int NU, const int* idx, int ND,
                       const double* se, const double* sn, const double* su,
                       const double* sve, const double* svn, const double* svu, int NS,
                       double st0, double sdt, const double* gx, const double* gy, int G,
                       const double* gz, double c, double r_max, double bins_per_lnf, int H,
                       const int* hoff, int min_frames, double r_ref, double rate_ref,
                       float* best, int* arg, int* nvalid, float* rmin, int* spread) {
    Seg s{se, sn, su, sve, svn, svu, NS, st0, sdt};
    double rate = range_rate(s, c);
#pragma omp parallel
    {
        std::vector<float> E(K), rr(NU), w(NU);
        std::vector<double> shc(NU);
        std::vector<int> sh(NU);
        std::vector<char> ok(NU);
#pragma omp for schedule(dynamic, 16)
        for (int g = 0; g < G; ++g) {
            walk(s, U, NU, gx[g], gy[g], gz[g], c, r_max, rate, ok.data(),
                 [&](int u, const double* d, const double* v, double r) {
                     double rdot = (d[0] * v[0] + d[1] * v[1] + d[2] * v[2]) / r;
                     shc[u] = -std::log1p(rdot / c) * bins_per_lnf;
                     sh[u] = (int)std::lround(shc[u]);
                     rr[u] = (float)r;
                 });
            for (int u = 0; u < NU; ++u) {
                if (!ok[u]) continue;
                double wl = r_ref > 0 ? std::min(1.0, r_ref / rr[u]) : 1.0, ws = 1.0;
                int a = u > 0 && ok[u - 1] ? u - 1 : u, b = u + 1 < NU && ok[u + 1] ? u + 1 : u;
                if (rate_ref > 0 && b > a) ws = std::min(1.0, std::fabs(shc[b] - shc[a]) / (U[b] - U[a]) / rate_ref);
                w[u] = (float)(wl * ws);
            }
            for (int di = 0; di < ND; ++di) {
                const int* ix = idx + (size_t)di * F;
                std::fill(E.begin(), E.end(), 0.f);
                int n = 0, s_lo = 1 << 30, s_hi = -(1 << 30);
                float rm = 1e30f;
                double w2 = 0;
                for (int f = 0; f < F; ++f) {
                    int u = ix[f];
                    if (!ok[u]) continue;
                    rm = std::min(rm, rr[u]);
                    s_lo = std::min(s_lo, sh[u]);
                    s_hi = std::max(s_hi, sh[u]);
                    ++n;
                    float wu = w[u];
                    w2 += (double)wu * wu;
                    if (wu <= 0) continue;
                    int lo = std::max(0, -sh[u]), hi = std::min(K, K - sh[u]);
                    const float* row = L + (size_t)f * K + sh[u];
                    for (int k = lo; k < hi; ++k) E[k] += wu * row[k];
                }
                size_t o = (size_t)di * G + g;
                nvalid[o] = n;
                rmin[o] = rm;
                spread[o] = n ? s_hi - s_lo : 0;
                best[o] = -1e30f;
                arg[o] = -1;
                if (n < min_frames || w2 < 1e-6) continue;
                int kmax = K - hoff[H - 1];
                for (int k = 0; k < kmax; ++k) {
                    float acc = 0.f;
                    for (int h = 0; h < H; ++h) acc += E[k + hoff[h]];
                    if (acc > best[o]) { best[o] = acc; arg[o] = k; }
                }
                best[o] /= (float)(H * std::sqrt(w2));
            }
        }
    }
}

// mf_seg with a clock offset per point and no weights or harmonics: point g hears frame f at
// t0 + off[g] + (f + m[d]) * hop for each of the ND frame shifts m. best, nvalid are [ND, G]
extern "C" void mf_each(const float* L, int F, int K, double t0, double hop, const int* m, int ND,
                        const double* se, const double* sn, const double* su,
                        const double* sve, const double* svn, const double* svu, int NS,
                        double st0, double sdt, const double* gx, const double* gy, const double* gz,
                        const double* off, int G, double c, double r_max, double bins_per_lnf, int min_frames,
                        float* best, int* nvalid) {
    Seg s{se, sn, su, sve, svn, svu, NS, st0, sdt};
    double rate = range_rate(s, c);
    int m_lo = *std::min_element(m, m + ND), m_hi = *std::max_element(m, m + ND), NU = F + m_hi - m_lo;
#pragma omp parallel
    {
        std::vector<float> E(K);
        std::vector<double> U(NU);
        std::vector<int> sh(NU);
        std::vector<char> ok(NU);
#pragma omp for schedule(dynamic, 16)
        for (int g = 0; g < G; ++g) {
            for (int u = 0; u < NU; ++u) U[u] = t0 + off[g] + (u + m_lo) * hop;
            walk(s, U.data(), NU, gx[g], gy[g], gz[g], c, r_max, rate, ok.data(),
                 [&](int u, const double* d, const double* v, double r) {
                     double rdot = (d[0] * v[0] + d[1] * v[1] + d[2] * v[2]) / r;
                     sh[u] = (int)std::lround(-std::log1p(rdot / c) * bins_per_lnf);
                 });
            for (int di = 0; di < ND; ++di) {
                std::fill(E.begin(), E.end(), 0.f);
                int n = 0;
                for (int f = 0; f < F; ++f) {
                    int u = f + m[di] - m_lo;
                    if (!ok[u]) continue;
                    ++n;
                    int lo = std::max(0, -sh[u]), hi = std::min(K, K - sh[u]);
                    const float* row = L + (size_t)f * K + sh[u];
                    for (int k = lo; k < hi; ++k) E[k] += row[k];
                }
                size_t o = (size_t)di * G + g;
                nvalid[o] = n;
                best[o] = -1e30f;
                if (n < min_frames) continue;
                best[o] = *std::max_element(E.begin(), E.end()) / (float)std::sqrt((double)n);
            }
        }
    }
}

// ground-reflection fringes (lloyd's mirror) for broadband sources: peaks at k*c/D, notches at (k+-1/2)*c/D
// D = path difference; in bins b = log(f / fmin) * bins_per_lnf the comb depends only on spacing sig = b(c / D)
// fringe_table sums (peak - mean of neighbouring notches) per frame, sig = smin + j / Q; N counts fringes used
// band [blo, bhi] in bins
extern "C" void fringe_table(const float* L, int F, int K, double bins_per_lnf, double blo, double bhi,
                             double smin, int NSIG, int Q, float* R, int* N) {
    const int KM = 1024;
    std::vector<double> lk(KM), la(KM), lb(KM);
    for (int k = 1; k < KM; ++k) {
        lk[k] = std::log((double)k) * bins_per_lnf;
        la[k] = std::log(k - 0.5) * bins_per_lnf;
        lb[k] = std::log(k + 0.5) * bins_per_lnf;
    }
#pragma omp parallel for schedule(static)
    for (int f = 0; f < F; ++f) {
        const float* row = L + (size_t)f * K;
        for (int j = 0; j < NSIG; ++j) {
            double sig = smin + (double)j / Q, acc = 0;
            int n = 0;
            for (int k = 1; k < KM && lb[k] + sig <= bhi; ++k) {
                if (la[k] + sig < blo) continue;
                int bp = (int)std::lround(lk[k] + sig), ba = (int)std::lround(la[k] + sig),
                    bb = (int)std::lround(lb[k] + sig);
                if (ba < 0 || bb >= K || bb - ba < 3) continue;  // fringes too fine to resolve
                acc += row[bp] - 0.5 * (row[ba] + row[bb]);
                ++n;
            }
            R[(size_t)f * NSIG + j] = (float)acc;
            N[(size_t)f * NSIG + j] = n;
        }
    }
}

// score = mean fringe contrast over the clip, best over microphone heights hs; D ~ 4 h dz / (r1 + r2)
// times and outputs as in mf_seg
extern "C" void fringe_seg(const float* R, const int* NR, int F, int NSIG, double smin, int Q,
                           const double* U, int NU, const int* idx, int ND,
                           const double* se, const double* sn, const double* su,
                           const double* sve, const double* svn, const double* svu, int NS,
                           double st0, double sdt, const double* gx, const double* gy, int G,
                           const double* gz, double c, double r_max, double fmin, double bins_per_lnf,
                           const double* hs, int NH, float* best, int* nbest, float* rmin) {
    Seg s{se, sn, su, sve, svn, svu, NS, st0, sdt};
    double rate = range_rate(s, c);
    std::vector<double> lh(NH);
    for (int ih = 0; ih < NH; ++ih) lh[ih] = std::log(hs[ih]) * bins_per_lnf;
#pragma omp parallel
    {
        std::vector<double> acc(NH), base(NU);
        std::vector<int> cnt(NH);
        std::vector<float> rr(NU);
        std::vector<char> ok(NU);
#pragma omp for schedule(dynamic, 16)
        for (int g = 0; g < G; ++g) {
            walk(s, U, NU, gx[g], gy[g], gz[g], c, r_max, rate, ok.data(),
                 [&](int u, const double* d, const double*, double r) {
                     rr[u] = (float)r;  // comb spacing sig for h = 1 m; NaN if not above the listener
                     base[u] = d[2] > 0 ? std::log(c * r / (2 * d[2] * fmin)) * bins_per_lnf : NAN;
                 });
            for (int di = 0; di < ND; ++di) {
                const int* ix = idx + (size_t)di * F;
                std::fill(acc.begin(), acc.end(), 0.0);
                std::fill(cnt.begin(), cnt.end(), 0);
                float rm = 1e30f;
                for (int f = 0; f < F; ++f) {
                    int u = ix[f];
                    if (!ok[u]) continue;
                    rm = std::min(rm, rr[u]);
                    if (std::isnan(base[u])) continue;
                    const float* Rf = R + (size_t)f * NSIG;
                    const int* Nf = NR + (size_t)f * NSIG;
                    for (int ih = 0; ih < NH; ++ih) {
                        int j = (int)std::lround((base[u] - lh[ih] - smin) * Q);
                        if (j < 0 || j >= NSIG) continue;
                        acc[ih] += Rf[j];
                        cnt[ih] += Nf[j];
                    }
                }
                size_t o = (size_t)di * G + g;
                float bv = -1e30f; int bn = 0;
                for (int ih = 0; ih < NH; ++ih)
                    if (cnt[ih] >= 20 && acc[ih] / cnt[ih] > bv) { bv = (float)(acc[ih] / cnt[ih]); bn = cnt[ih]; }
                best[o] = bv;
                nbest[o] = bn;
                rmin[o] = rm;
            }
        }
    }
}

// blind doppler detector: straight-pass templates swept over the spectrogram
// shifts[s * (2J + 1) + J + j] is template s's pitch offset (bins) j frames from closest approach; half[s] limits its window
// best[c * S + s] = max over pitch k of sum over harmonics and frames of L / (H sqrt(n)), centred at frame centers[c]
// arg is that k (pitch at closest approach), nfr the frames used
extern "C" void sweep_bank(const float* L, int F, int K, const int* shifts, const int* half, int S, int J,
                           const int* centers, int NC, int H, const int* hoff,
                           float* best, int* arg, int* nfr) {
#pragma omp parallel
    {
        std::vector<float> E(K);
#pragma omp for schedule(dynamic, 8) collapse(2)
        for (int c = 0; c < NC; ++c)
            for (int s = 0; s < S; ++s) {
                std::fill(E.begin(), E.end(), 0.f);
                int n = 0;
                const int* sh = shifts + (size_t)s * (2 * J + 1) + J;
                for (int j = -half[s]; j <= half[s]; ++j) {
                    int f = centers[c] + j;
                    if (f < 0 || f >= F) continue;
                    int d = sh[j], lo = std::max(0, -d), hi = std::min(K, K - d);
                    const float* row = L + (size_t)f * K + d;
                    for (int k = lo; k < hi; ++k) E[k] += row[k];
                    ++n;
                }
                size_t o = (size_t)c * S + s;
                nfr[o] = n;
                best[o] = -1e30f;
                arg[o] = -1;
                if (n < 8) continue;
                int kmax = K - hoff[H - 1];
                for (int k = 0; k < kmax; ++k) {
                    float a = 0.f;
                    for (int h = 0; h < H; ++h) a += E[k + hoff[h]];
                    if (a > best[o]) { best[o] = a; arg[o] = k; }
                }
                best[o] /= (float)(H * std::sqrt((double)n));
            }
    }
}

// log(f heard / f source) for a straight pass at constant speed; t0 is when closest-approach sound arrives
static inline double line_lr(double t, double t0, double d, double v, double c) {
    double T = t - (t0 - d / c), a = c * c - v * v;
    double u = (c * c * T - std::sqrt(c * c * v * v * T * T + d * d * a)) / a;
    return -std::log1p(v * v * u / (c * (T - u)) / c);
}

// soft-L1 cost of pass p = (t0, ln d, v, ln f0) on ridge (t, lf); res gets the residuals
static double pass_cost(const double* t, const double* lf, int N, const double* p, double c, double fs,
                        double* res) {
    double d = std::exp(p[1]), cost = 0;
    for (int i = 0; i < N; ++i) {
        double r = lf[i] - p[3] - line_lr(t[i], p[0], d, p[2], c);
        if (res) res[i] = r;
        cost += 2 * (std::sqrt(1 + (r / fs) * (r / fs)) - 1);
    }
    return 0.5 * fs * fs * cost;
}

// solve A x = b in place (4x4, partial pivoting); false if singular
static bool solve4(double A[4][4], double* b) {
    for (int k = 0; k < 4; ++k) {
        int piv = k;
        for (int i = k + 1; i < 4; ++i) if (std::fabs(A[i][k]) > std::fabs(A[piv][k])) piv = i;
        if (std::fabs(A[piv][k]) < 1e-300) return false;
        if (piv != k) { for (int j = 0; j < 4; ++j) std::swap(A[k][j], A[piv][j]); std::swap(b[k], b[piv]); }
        for (int i = k + 1; i < 4; ++i) {
            double f = A[i][k] / A[k][k];
            for (int j = k; j < 4; ++j) A[i][j] -= f * A[k][j];
            b[i] -= f * b[k];
        }
    }
    for (int k = 3; k >= 0; --k) {
        for (int j = k + 1; j < 4; ++j) b[k] -= A[k][j] * b[j];
        b[k] /= A[k][k];
    }
    return true;
}

// straight-pass fit to a ridge: levenberg-marquardt on soft-L1 cost from each of M starts [M, 4], inside box lo..hi
// lo = hi holds a parameter; out = t0, ln d, v, ln f0, cost, rms of the best
extern "C" void line_fit(const double* t, const double* lf, int N, double c, const double* starts, int M,
                         const double* lo, const double* hi, double fs, double* out) {
    std::vector<double> best(6, 1e300);
#pragma omp parallel
    {
        std::vector<double> res(N), rp(N), rm(N), J(4 * (size_t)N);
#pragma omp for schedule(dynamic)
        for (int m = 0; m < M; ++m) {
            double p[4], q[4];
            for (int k = 0; k < 4; ++k) p[k] = std::min(std::max(starts[m * 4 + k], lo[k]), hi[k]);
            double cost = pass_cost(t, lf, N, p, c, fs, res.data()), lam = 1e-3;
            for (int it = 0; it < 200; ++it) {
                double g[4] = {0, 0, 0, 0}, A[4][4] = {};
                for (int k = 0; k < 4; ++k) {  // central differences
                    if (hi[k] <= lo[k]) { std::fill(J.begin() + k * (size_t)N, J.begin() + (k + 1) * (size_t)N, 0.0); continue; }
                    double h = 1e-6 * std::max(1.0, std::fabs(p[k]));
                    std::copy(p, p + 4, q);
                    q[k] = p[k] + h; pass_cost(t, lf, N, q, c, fs, rp.data());
                    q[k] = p[k] - h; pass_cost(t, lf, N, q, c, fs, rm.data());
                    for (int i = 0; i < N; ++i) J[k * (size_t)N + i] = (rp[i] - rm[i]) / (2 * h);
                }
                for (int i = 0; i < N; ++i) {  // robust weights
                    double w = 1 / std::sqrt(1 + (res[i] / fs) * (res[i] / fs));
                    for (int a = 0; a < 4; ++a) {
                        double ja = J[a * (size_t)N + i] * w;
                        g[a] += ja * res[i];
                        for (int b = a; b < 4; ++b) A[a][b] += ja * J[b * (size_t)N + i];
                    }
                }
                bool held[4];
                for (int k = 0; k < 4; ++k)  // fixed, or at a bound the step would cross
                    held[k] = hi[k] <= lo[k] || (p[k] <= lo[k] && g[k] > 0) || (p[k] >= hi[k] && g[k] < 0);
                bool ok = false;
                double c2 = cost;
                for (int tries = 0; tries < 12 && !ok; ++tries, lam *= 10) {
                    double B[4][4], x[4];
                    for (int a = 0; a < 4; ++a) {
                        x[a] = held[a] ? 0 : -g[a];
                        for (int b = 0; b < 4; ++b) {
                            double v = a <= b ? A[a][b] : A[b][a];
                            B[a][b] = (held[a] || held[b]) ? (a == b ? 1.0 : 0.0) : v;
                        }
                        if (!held[a]) B[a][a] = A[a][a] * (1 + lam) + 1e-300;
                    }
                    if (!solve4(B, x)) continue;
                    for (int k = 0; k < 4; ++k) q[k] = std::min(std::max(p[k] + x[k], lo[k]), hi[k]);
                    c2 = pass_cost(t, lf, N, q, c, fs, rp.data());
                    ok = std::isfinite(c2) && c2 < cost;
                }
                if (!ok) break;
                double gain = cost - c2;
                std::copy(q, q + 4, p);
                res.swap(rp);
                cost = c2;
                lam = std::max(lam * 0.01, 1e-12);  // one *10 already applied leaving the loop
                if (gain < 1e-10 * cost + 1e-18) break;
            }
#pragma omp critical
            if (cost < best[4]) {
                double ss = 0;
                for (int i = 0; i < N; ++i) ss += res[i] * res[i];
                std::copy(p, p + 4, best.begin());
                best[4] = cost;
                best[5] = std::sqrt(ss / N);
            }
        }
    }
    std::copy(best.begin(), best.end(), out);
}

// rms log-frequency distance between a ridge (t, lf) and the exact curve of one flight heard at each (gx, gy, gz)
// and clock offset: rms, frac [ND, G]; source pitch is free (median removed), residuals clipped at +-clip
// frac is the share of ridge points the track covers
extern "C" void ridge_fit(const double* se, const double* sn, const double* su,
                          const double* sve, const double* svn, const double* svu, int NS, double st0, double sdt,
                          const double* t, const double* lf, int N, const double* gx, const double* gy,
                          const double* gz, int G, const double* dts, int ND, double S, double c, double clip,
                          double* rms, double* frac) {
    Seg s{se, sn, su, sve, svn, svu, NS, st0, sdt};
#pragma omp parallel
    {
        std::vector<double> res(N), tmp(N);
#pragma omp for schedule(dynamic, 16) collapse(2)
        for (int di = 0; di < ND; ++di)
            for (int g = 0; g < G; ++g) {
                int n = 0;
                double rprev = 3000;
                for (int i = 0; i < N; ++i) {
                    double T = S + dts[di] + t[i], tau = T - rprev / c, p[3], v[3], d[3], r = 0;
                    for (int it = 0; it < 5; ++it) {  // emission time: tau + r(tau) / c = T
                        at(s, tau, p, v, true);
                        d[0] = p[0] - gx[g]; d[1] = p[1] - gy[g]; d[2] = p[2] - gz[g];
                        r = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
                        tau -= (tau - T + r / c) / (1 + (d[0] * v[0] + d[1] * v[1] + d[2] * v[2]) / r / c);
                    }
                    if (!at(s, tau, p, v, true)) continue;
                    d[0] = p[0] - gx[g]; d[1] = p[1] - gy[g]; d[2] = p[2] - gz[g];
                    r = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
                    rprev = r;
                    double e = lf[i] + std::log1p((d[0] * v[0] + d[1] * v[1] + d[2] * v[2]) / r / c);
                    if (std::isfinite(e)) res[n++] = e;  // glitched tracks report speeds past sound
                }
                size_t o = (size_t)di * G + g;
                frac[o] = (double)n / N;
                rms[o] = 0;
                if (!n) continue;
                std::copy(res.begin(), res.begin() + n, tmp.begin());
                std::nth_element(tmp.begin(), tmp.begin() + n / 2, tmp.begin() + n);
                double med = tmp[n / 2];
                if (n % 2 == 0) med = 0.5 * (med + *std::max_element(tmp.begin(), tmp.begin() + n / 2));
                double ss = 0;
                for (int i = 0; i < n; ++i) {
                    double e = std::min(std::max(res[i] - med, -clip), clip);
                    ss += e * e;
                }
                rms[o] = std::sqrt(ss / n);
            }
    }
}

// subtract from each row (axis 1) or column (axis 0) of A [R, C] its median
extern "C" void sub_median(float* A, int R, int C, int axis) {
    int n = axis ? C : R, lines = axis ? R : C;
#pragma omp parallel
    {
        std::vector<float> v(n);
#pragma omp for
        for (int l = 0; l < lines; ++l) {
            float* a = axis ? A + (size_t)l * C : A + l;
            size_t step = axis ? 1 : C;
            for (int i = 0; i < n; ++i) v[i] = a[i * step];
            std::nth_element(v.begin(), v.begin() + n / 2, v.end());
            float med = v[n / 2];
            if (n % 2 == 0) med = 0.5f * (med + *std::max_element(v.begin(), v.begin() + n / 2));
            for (int i = 0; i < n; ++i) a[i * step] -= med;
        }
    }
}

// link per-frame spectral peaks into tracks; frame f owns peaks starts[f]..starts[f + 1], each with sub-bin pitch k and level lv
// a track claims the peak nearest its slope prediction (the stronger if close) and dies after max_gap frames without one
// track[i] = track of peak i, order = tracks with at least min_len peaks by end time; returns how many
extern "C" int link_ridges(const int* starts, int F, const double* k, const float* lv, double peak_db,
                           double tol, int max_gap, int min_len, int* track, int* order) {
    struct Tr { int id, last, n; double k, slope; };
    struct Pair { double cost; int ti, pi; };
    std::vector<Tr> act, alive;
    std::vector<Pair> pairs;
    std::vector<char> taken, used;
    int ids = 0, n_out = 0;
    for (int f = 0; f < F; ++f) {
        int a = starts[f], np_ = starts[f + 1] - a;
        alive.clear();
        for (const Tr& t : act) {
            if (f - t.last <= max_gap) alive.push_back(t);
            else if (t.n >= min_len) order[n_out++] = t.id;
        }
        act.swap(alive);
        used.assign(np_, 0);
        taken.assign(act.size(), 0);
        pairs.clear();
        for (int ti = 0; ti < (int)act.size(); ++ti) {
            const Tr& t = act[ti];
            int gap = f - t.last;
            double pred = t.k + t.slope * gap;
            // new track has no slope yet, and close passes sweep fast
            double reach = tol + 0.3 * gap + 0.5 * std::fabs(t.slope) + (t.n < 3 ? 3.0 : 0.0);
            for (int pi = 0; pi < np_; ++pi) {
                double dist = std::fabs(pred - k[a + pi]);
                if (dist <= reach) pairs.push_back({dist - (double)(0.1f * (lv[a + pi] - (float)peak_db)), ti, pi});
            }
        }
        std::sort(pairs.begin(), pairs.end(), [](const Pair& x, const Pair& y) {
            return x.cost != y.cost ? x.cost < y.cost : (x.ti != y.ti ? x.ti < y.ti : x.pi < y.pi); });
        for (const Pair& p : pairs) {
            if (taken[p.ti] || used[p.pi]) continue;
            Tr& t = act[p.ti];
            t.slope = 0.5 * t.slope + 0.5 * (k[a + p.pi] - t.k) / (f - t.last);
            t.last = f; t.k = k[a + p.pi]; ++t.n;
            track[a + p.pi] = t.id;
            taken[p.ti] = used[p.pi] = 1;
        }
        for (int pi = 0; pi < np_; ++pi)
            if (!used[pi]) { track[a + pi] = ids; act.push_back({ids++, f, 1, k[a + pi], 0.0}); }
    }
    for (const Tr& t : act) if (t.n >= min_len) order[n_out++] = t.id;
    return n_out;
}
