"""RESULTS §46: Lang's swept random hyperplanes as the equal-budget baseline for our search.

Same embedding [X|U|L] (exact envelope, 48 dims), same decoder and objective as homotopy_cont._J, same final
refinement; only the SEARCH differs:
  random   N independent directions w ~ N(0, I) (unit-normalised), each decoded and scored, best kept.
           lam = inf: t = 1 (acyclic decoder).  lam = 1: t ~ U[0, 1] per sample.
  N = 1085 per lam, the number of evaluations the homotopy / continuation used per stage (§45).
Fixed seed (rng 0); one seed reported, so this is one draw of a random baseline.
Compared with: homotopy from e1 (§44/§45), its 5-start median (§45), continuation at lam = 1 (§45).
Usage: python bdp/lang.py <src> [workers]
"""
import sys, os, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from refine import HG, load, bounds, refine
import homotopy_cont as hc

N = 1085


def main(src, workers):
    from det import eig
    import learnenv as le
    from tradeoff import prefine, nback
    name = os.path.basename(src.rstrip("/")).replace("_hg.npz", "")
    h = HG(*load(src)); eps = 0.10; lo, hi = bounds(h.n, eps); t0 = time.time()
    vals, U, dv, inv, w = eig(h, 32); X = U[:, :16] * inv[:, None]
    for j in range(16):
        if np.sum(X[h.ev, j] - X[h.eu, j]) < 0: X[:, j] *= -1
    X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    def env(X_, P):
        U_ = X_.copy()
        for ss, du, st, cnt in P: U_[du] = np.maximum(X_[du], np.maximum.reduceat(U_[ss] + 1e-4, st, axis=0))
        return U_
    F = np.hstack([X, env(X, le.plan(h.n, h.eu, h.ev)), -env(-X, le.plan(h.n, h.ev, h.eu))]); F = (F - F.mean(0)) / (F.std(0) + 1e-12)
    d = F.shape[1]; rng = np.random.default_rng(0)
    print(f"== {name} n={h.n} | {N} random directions per lam", flush=True)
    with Pool(workers, initializer=hc._init, initargs=(src, F)) as pool:
        for lam in (np.inf, 1.0):
            W = rng.normal(size=(N, d)); W /= np.linalg.norm(W, axis=1, keepdims=True)
            T = np.ones(N) if np.isinf(lam) else rng.uniform(0, 1, N)
            t1 = time.time(); R = pool.map(hc._J, [(np.r_[W[i], T[i]], lam) for i in range(N)], chunksize=4)
            ok = [r for r in R if r[1] is not None]; Js = np.array([r[0] for r in ok])
            p = min(ok, key=lambda r: r[0])[1].copy()
            if np.isinf(lam): refine(h, p, eps); assert h.closed(p)
            else: prefine(h, p, eps, lam)
            assert lo <= np.sum(p == 0) <= hi
            c, bk = h.cut(p), nback(h, p); J = c + (0 if np.isinf(lam) else lam * bk)
            print(f"   lam={lam:g}: best cut {c} back {bk} J {J:.0f} | sample J before refine: min {Js.min():.0f}, "
                  f"median {np.median(Js):.0f} | {len(ok)}/{N} valid [{time.time() - t1:.0f}s]", flush=True)
    print(f"   done [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    a = sys.argv
    main(a[1], int(a[2]) if len(a) > 2 else 12)
