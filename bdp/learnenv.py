"""RESULTS §37: jryri's proposal -- a LEARNED, circuit-independent envelope operator Phi_theta.

Features X(theta) = [X_spec | U(theta) | L(theta)], with a parametric envelope along the DAG (his envelope is the
special case alpha = 1, beta = inf, delta = 1e-4):
  forward, in topological (level) order:
      soft_v = (1/beta) * log mean exp(beta * {x_v} U {U_u + delta : u pred of v})     beta -> inf: max; beta -> 0: mean
      U_v    = (1 - alpha) * x_v + alpha * soft_v
  backward (successors, min, - delta) is the mirror image.  theta = (alpha, beta, delta) forward + backward = 6 numbers,
  shared by all coordinates and all circuits.  delta is where DAG depth enters the coordinates.
Training (as proposed): on designs with an oracle partition y, ridge-fit the best hyperplane in closed form,
  w*(theta) = (X'X + rho I)^-1 X'y,  loss = sum_v c_v log(1 + exp(-y_v X_v w*)),  c_v = hypergraph + DAG degree.
  theta has 6 dimensions, so it is optimised derivative-free and deterministically (Nelder-Mead) instead of autograd.
Protocol: 2-fold across circuits.  Fold A trains on (serv, omsp, riscv) and tests on (picosoc, aes, biriscv); B the
  reverse.  Oracle y = best_parts2 (K+repair / dagP-w + refine).  Oracles are used for training and for SCORING
  held-out separability only.
Reported per design: degree-weighted loss and misclassification of the ridge hyperplane for
  X only | his envelope | learned theta (in-sample) | learned theta (held-out),
  and the held-out hyperplane decoded by the method's decoder (projection -> sweep -> FM -> refine).
Usage: python bdp/learnenv.py
"""
import sys, os, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
from scipy.optimize import minimize
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from refine import HG, load
from det import eig
from oracle import decode

HERE = os.path.dirname(os.path.abspath(__file__)); TGT = os.path.join(HERE, "best_parts2")
FOLDS = {"A": ["serv", "omsp", "riscv"], "B": ["picosoc", "aes", "biriscv"]}
RHO = 1e-3


def levels(n, eu, ev):
    lv = np.zeros(n, np.int64)
    while True:
        new = lv.copy(); np.maximum.at(new, ev, lv[eu] + 1)
        if np.array_equal(new, lv): return lv
        lv = new


def plan(n, src, dst):
    """Edges grouped by level of dst, sorted by dst inside a level; returns per-level (src, dst_unique, starts, counts)."""
    lv = levels(n, src, dst); o = np.lexsort((dst, lv[dst])); s, d = src[o], dst[o]; L = lv[d]; out = []
    cuts = np.flatnonzero(np.diff(L)) + 1; b = np.r_[0, cuts, len(L)]
    for i in range(len(b) - 1):
        ss, dd = s[b[i]:b[i + 1]], d[b[i]:b[i + 1]]
        st = np.r_[0, np.flatnonzero(np.diff(dd)) + 1]
        out.append((ss, dd[st], st, np.diff(np.r_[st, len(dd)])))
    return out


def prop(X, P, alpha, beta, delta):
    U = X.copy()
    for ss, du, st, cnt in P:
        V = U[ss] + delta; xv = X[du]
        M = np.maximum(np.maximum.reduceat(V, st, axis=0), xv)
        S = np.exp(beta * (xv - M)) + np.add.reduceat(np.exp(beta * (V - np.repeat(M, cnt, axis=0))), st, axis=0)
        soft = M + np.log(S / (cnt[:, None] + 1)) / beta
        U[du] = (1 - alpha) * xv + alpha * soft
    return U


def unpack(t):
    f = lambda a, b, c: (1 / (1 + np.exp(-a)), float(np.exp(np.clip(b, -3, 4))), float(np.exp(np.clip(c, -12, 3))))
    return f(*t[:3]), f(*t[3:])


def feats(D, t):
    (af, bf, df), (ab, bb, db) = unpack(t)
    U = prop(D["X"], D["Pf"], af, bf, df); Lo = -prop(-D["X"], D["Pb"], ab, bb, db)
    F = np.hstack([D["X"], U, Lo]); return (F - F.mean(0)) / (F.std(0) + 1e-12)


def fit(F, D):
    y, c = D["y"], D["c"]; w = np.linalg.solve(F.T @ F + RHO * len(y) * np.eye(F.shape[1]), F.T @ y)
    z = F @ w; loss = float(np.sum(c * np.logaddexp(0, -y * z)) / c.sum()); mis = float(np.mean((z > 0) != (y > 0)))
    return loss, mis, z


def main():
    data = {}
    for name in FOLDS["A"] + FOLDS["B"]:
        h = HG(*load(f"data/{name}_hg.npz")); vals, U, dv, inv, w = eig(h, 32)
        X = U[:, :16] * inv[:, None]
        for j in range(16):
            if np.sum(X[h.ev, j] - X[h.eu, j]) < 0: X[:, j] *= -1
        X = (X - X.mean(0)) / (X.std(0) + 1e-12); T = np.load(os.path.join(TGT, f"{name}_e10.npy"))
        c = np.bincount(h.pins, minlength=h.n) + np.bincount(h.eu, minlength=h.n) + np.bincount(h.ev, minlength=h.n)
        data[name] = dict(h=h, X=X, y=np.where(T == 1, 1.0, -1.0), c=c.astype(float), T=T, tc=h.cut(T),
                          Pf=plan(h.n, h.eu, h.ev), Pb=plan(h.n, h.ev, h.eu))
        print(f"loaded {name}: n={h.n}, DAG depth {len(data[name]['Pf'])}, target {data[name]['tc']}", flush=True)
    his = np.array([20.0, 4.0, np.log(1e-4), 20.0, 4.0, np.log(1e-4)])     # alpha ~ 1, beta = e^4 ~ 55 (near max), delta 1e-4
    t0v = np.array([2.0, 2.0, -3.0, 2.0, 2.0, -3.0])
    learned = {}
    for fold, names in FOLDS.items():
        t0 = time.time(); obj = lambda t: sum(fit(feats(data[n], t), data[n])[0] for n in names)
        r = minimize(obj, t0v, method="Nelder-Mead", options={"maxfev": 300, "xatol": 1e-2, "fatol": 1e-4})
        learned[fold] = r.x; (af, bf, df), (ab, bb, db) = unpack(r.x)
        print(f"fold {fold} trained on {names}: loss {obj(t0v):.4f} -> {r.fun:.4f} in {r.nfev} evals [{time.time() - t0:.0f}s] | "
              f"forward alpha {af:.2f} beta {bf:.2f} delta {df:.4f} | backward alpha {ab:.2f} beta {bb:.2f} delta {db:.4f}", flush=True)
    print(f"\n{'design':8} | X only (16)    | his envelope   | learned in-sample | learned HELD-OUT | held-out decoded+refined (x target)")
    for fold, names in FOLDS.items():
        other = "B" if fold == "A" else "A"
        for n in names:
            D = data[n]; Xs = (D["X"] - D["X"].mean(0)) / (D["X"].std(0) + 1e-12)
            a = fit(Xs, D); b = fit(feats(D, his), D); c_ = fit(feats(D, learned[fold]), D); d = fit(feats(D, learned[other]), D)
            r, p = decode(D["h"], d[2], 0.10); rx, _ = decode(D["h"], a[2], 0.10)
            print(f"{n:8} | {a[0]:.3f} {100 * a[1]:5.1f}% | {b[0]:.3f} {100 * b[1]:5.1f}% | {c_[0]:.3f} {100 * c_[1]:5.1f}%    | "
                  f"{d[0]:.3f} {100 * d[1]:5.1f}%   | learned {r} ({r / D['tc']:.2f}x) vs X-only {rx} ({rx / D['tc']:.2f}x)", flush=True)


if __name__ == "__main__":
    main()
