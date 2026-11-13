"""RESULTS §26: three DETERMINISTIC replacements for CMA (no random numbers anywhere after the eigensolver).

All three end the same way: an order -> acyclic sweep in the balance window -> one acyclic refine.
  cone     order-constrained spectral relaxation (cone-constrained PCA, Deshpande-Montanari-Richard 2014):
             max s'Ws / s'Ds  over s nondecreasing along every DAG edge, s'D1 = 0
           projected power iteration  s <- P_C((D^-1 W + I) s), P_C = D-weighted L2 isotonic projection (l2_ir),
           D-mean removed, D-normalised; start = Fiedler vector; stop when |ds|_D < 1e-4 or 200 iterations.
           Both signs of the start; the one with the larger Rayleigh quotient is kept (the objective, not the cut).
           s is monotone, so every split of order(s, Kahn tie-break) is acyclic: no post-hoc projection.
  melo     multiple-eigenvector linear ordering (Alpert-Kahng-Yao 1999): vertex vectors x_v = sqrt(alpha_j) u_j(v)
           (16 eigvecs of the normalised adjacency), greedy order maximising |sum of chosen vectors|;
           rank -> exact L1 projection -> order; orientation by projected sweep cut.
  compass  compass (pattern) search (Torczon) on theta = (a, b) in R^32 with the exact objective of theta_opt.py;
           start a = -2 e1, b = +2 e1 (near-planar spectral bisection), step 1, poll +-e_i in fixed order,
           accept first improvement, halve on failure; stop at step < 1e-3 or 1000 evaluations.
Compared with §24 pipeline P and §25 theta-CMA.
Cost: cone  iterations x 24 max-flows;  melo O(n^2 k) greedy + 2 exact projections;  compass evals x 2 exact projections.
Usage: python bdp/det.py <mode> <src> [eps]
"""
import sys, os, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
from scipy.sparse.linalg import eigsh, LinearOperator
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from refine import HG, load, bounds, sweep_part, refine, kahn
from exact_ir import l1_ir_exact
from l2_ir import l2_ir
from separability import SAVE

P24 = {"serv": 98, "omsp": 361, "picosoc": 335, "aes": 664, "riscv": 304, "biriscv": 238}
C25 = {"serv": 96, "omsp": 287, "picosoc": 426, "aes": 633, "riscv": 307, "biriscv": 803}


def eig(h, k):
    n = h.n; w = 1.0 / np.bincount(h.netid)
    dv = np.asarray(h.H.sum(1)).ravel(); inv = 1 / np.sqrt(np.maximum(dv, 1))
    mv = lambda x: inv * (h.H @ (w * (h.H.T @ (inv * x))))
    vals, U = eigsh(LinearOperator((n, n), matvec=mv, dtype=float), k=k + 1, which="LA",
                    v0=np.random.default_rng(0).normal(size=n), tol=1e-6)
    o = np.argsort(-vals)[1:]
    return vals[o], U[:, o], dv, inv, w


def decode_exact(h, s, eps):
    out = None
    for sg in (1, -1):
        f = l1_ir_exact(h, sg * s); kr = np.empty(h.n, np.int64); kr[kahn(h, sg * s)] = np.arange(h.n)
        c, p = sweep_part(h, np.lexsort((kr, f)), eps)
        if out is None or c < out[0]: out = (c, p)
    return out


def cone(h, eps):
    vals, U, dv, inv, w = eig(h, 1)
    Wm = lambda x: h.H @ (w * (h.H.T @ x))
    def norm(s):
        s = s - (dv @ s) / dv.sum(); return s / np.sqrt(dv @ (s * s))
    best = None
    for sg in (1, -1):
        s = norm(sg * U[:, 0] * inv); it = 0
        for it in range(1, 201):
            f = norm(l2_ir(h, Wm(s) / dv + s, dv))
            if np.sqrt(dv @ ((f - s) ** 2)) < 1e-4: s = f; break
            s = f
        R = (s @ Wm(s)) / (dv @ (s * s))
        assert np.all(s[h.ev] >= s[h.eu] - 1e-9 * np.abs(s).max()), "not monotone"
        print(f"   cone sign {sg:+d}: {it} iterations, Rayleigh quotient {R:.5f}", flush=True)
        if best is None or R > best[0]: best = (R, s)
    s = best[1]; kr = np.empty(h.n, np.int64); kr[kahn(h, s)] = np.arange(h.n)
    order = np.lexsort((kr, np.round(s, 12)))
    return sweep_part(h, order, eps)


def melo(h, eps):
    vals, U, dv, inv, w = eig(h, 16)
    Xv = U * np.sqrt(np.maximum(vals, 0))
    n = h.n; sq = (Xv * Xv).sum(1); g = np.zeros(n); used = np.zeros(n, bool); rank = np.empty(n)
    for i in range(n):
        sc = 2 * g + sq; sc[used] = -np.inf; v = int(np.argmax(sc))
        used[v] = True; rank[v] = i; g += Xv @ Xv[v]
    return decode_exact(h, rank, eps)


def compass(h, eps, evals=1000):
    vals, U, dv, inv, w = eig(h, 32)
    X = U[:, :16] * inv[:, None]; X = (X - X.mean(0)) / (X.std(0) + 1e-12); d = 16
    memo = {}; nev = [0]
    def J(th):
        key = th.tobytes()
        if key not in memo:
            a, b = th[:d], th[d:]
            s = np.sqrt(((X - a) ** 2).sum(1)) - np.sqrt(((X - b) ** 2).sum(1))
            memo[key] = decode_exact(h, s, eps); nev[0] += 1
        return memo[key]
    th = np.zeros(2 * d); th[0] = -2.0; th[d] = 2.0; cur = J(th); step = 1.0
    while step >= 1e-3 and nev[0] < evals:
        moved = False
        for i in range(2 * d):
            for sgn in (1.0, -1.0):
                cand = th.copy(); cand[i] += sgn * step; r = J(cand)
                if r[0] < cur[0]: th, cur, moved = cand, r, True; break
            if nev[0] >= evals: break
        if not moved: step /= 2
    print(f"   compass: {nev[0]} evaluations, final step {step:g}", flush=True)
    return cur


def main(mode, src, eps):
    name = os.path.basename(src.rstrip("/")).replace("_hg.npz", "")
    h = HG(*load(src)); lo, hi = bounds(h.n, eps)
    tc = h.cut(np.load(os.path.join(SAVE, f"{name}_e{int(eps * 100):02d}.npy")))
    t0 = time.time(); c, p = {"cone": cone, "melo": melo, "compass": compass}[mode](h, eps)
    assert h.closed(p) and lo <= np.sum(p == 0) <= hi
    p = p.copy(); r = refine(h, p, eps)[-1][1]; assert h.closed(p) and lo <= np.sum(p == 0) <= hi
    print(f"   RESULT {mode} {name}: sweep {c} -> refined {r} ({r / tc:.2f}x T) | §24 P {P24[name]} | §25 CMA {C25[name]} "
          f"| T {tc}  [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 0.10)
