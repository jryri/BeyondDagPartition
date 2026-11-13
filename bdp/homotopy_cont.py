"""RESULTS §45: lambda-continuation of the search with a decoder that can fall back to acyclic.

Instability measured (§45a): acyclic search spread 2-13% with rare outliers; finite-lambda search from a fresh start
is far less stable, and on biriscv / riscv it ends WORSE than the acyclic solution, which is itself feasible for any
lambda.  That is a pure search failure.

Decoder with mixing parameter t in [0, 1] (searched):
    f = (1 - t) * s + t * Linf(s),   Linf(s) = (U(s) + L(s)) / 2 (levelised, no max-flow)
    order by (f, level, s); the sweep picks the split minimising cut + lam * back (exact per split, O(m));
    then FM at lam (acyclic FM if lam = inf, penalised FM otherwise).
    t = 1 is exactly the acyclic decoder; t = 0 is the no-projection decoder.
Continuation: lam = inf -> 4 -> 1.  Each stage starts at the previous stage's incumbent (w, t); the first stage
starts at w = e_1 (the Fiedler direction) with t = 1.
Guarantee: every finite-lambda stage is seeded with the previous stage's FINAL (refined) partition as its incumbent;
the result is the best evaluated point and the penalised refinement never increases J, so
J_lam(result) <= J_lam(previous final) <= cut of the refined acyclic solution.
Search per stage: the §31 homotopy (per-axis central differences, sigma 2 -> 1/32, 1000 evaluations).
Usage: python bdp/homotopy_cont.py <src> [workers]
"""
import sys, os, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from refine import HG, load, bounds, refine, fm
from homotopy_dec import amax, cut_profile

_G = {}
LAMS = (np.inf, 4.0, 1.0)


def _init(src, F):
    import learnenv as le
    h = HG(*load(src)); _G.update(h=h, F=F, lv=le.levels(h.n, h.eu, h.ev), Pf=le.plan(h.n, h.eu, h.ev), Pb=le.plan(h.n, h.ev, h.eu))


def _J(args):
    th, lam = args; h, F, lv = _G["h"], _G["F"], _G["lv"]; d = F.shape[1]; eps = 0.10
    t = float(np.clip(th[d], 0.0, 1.0)); s0 = F @ th[:d]; a, b = bounds(h.n, eps); out = None
    for sg in (1, -1):
        s = sg * s0; li = 0.5 * (amax(s, _G["Pf"]) - amax(-s, _G["Pb"])); f = (1 - t) * s + t * li
        order = np.lexsort((s, lv, f)); cut, back = cut_profile(h, order)
        obj = cut[a:b + 1] if np.isinf(lam) else cut[a:b + 1] + lam * back[a:b + 1]
        k = a + int(np.argmin(obj)); p = np.ones(h.n, np.int8); p[order[:k]] = 0
        if np.isinf(lam):
            if np.any((p[h.eu] == 1) & (p[h.ev] == 0)): continue          # lam = inf: only acyclic splits count
            c = fm(h, p, eps)
        else:
            from backedge_ul import _pfm
            c = _pfm(h, p, eps, lam)
        if out is None or c < out[0]: out = (c, p)
    return out if out is not None else (float("inf"), None)


def search(pool, th0, lam, seed=None, evals=1000, sigma0=2.0, sigma_min=1 / 32):
    D = len(th0); d = D - 1; E = np.eye(D); memo = {}; inc = [seed, th0.copy() if seed is not None else None]
    def batch(ths):
        new = []; seen = set()
        for th in ths:
            k = th.tobytes()
            if k not in memo and k not in seen: new.append(th); seen.add(k)
        for th, r in zip(new, pool.map(_J, [(x, lam) for x in new], chunksize=2)): memo[th.tobytes()] = r
        for th in ths:
            r = memo[th.tobytes()]
            if r[1] is not None and (inc[0] is None or r[0] < inc[0][0]): inc[0], inc[1] = r, th.copy()
        return [memo[th.tobytes()][0] for th in ths]
    pts = lambda th, s: [th + s * e for e in E] + [th - s * e for e in E]
    def val(v, s):
        Jp, Jm = np.array(v[:D]), np.array(v[D:])
        ok = np.isfinite(Jp) & np.isfinite(Jm); g = np.where(ok, (Jp - Jm) / (2 * s), 0.0)
        allv = np.r_[Jp, Jm]; return np.mean(allv[np.isfinite(allv)]) if np.any(np.isfinite(allv)) else np.inf, g
    def renorm(th):
        th = th.copy(); th[:d] /= np.linalg.norm(th[:d]); th[d] = np.clip(th[d], 0.0, 1.0); return th
    th = th0.copy(); sigma = sigma0
    v = batch([th] + pts(th, sigma)); F_, g = val(v[1:], sigma)
    while sigma >= sigma_min and len(memo) < evals:
        if not np.any(g):
            sigma /= 2; F_, g = val(batch(pts(th, sigma)), sigma); continue
        cand = renorm(th - sigma * g / np.linalg.norm(g))
        v = batch([cand] + pts(cand, sigma)); Fc, gc = val(v[1:], sigma)
        if Fc < F_: th, F_, g = cand, Fc, gc
        else:
            sigma /= 2; F_, g = val(batch(pts(th, sigma)), sigma)
    return inc[0], inc[1], len(memo)


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
    d = F.shape[1]; th = np.zeros(d + 1); th[0] = 1.0; th[d] = 1.0; prev = None
    print(f"== {name} n={h.n}", flush=True)
    with Pool(workers, initializer=_init, initargs=(src, F)) as pool:
        for lam in LAMS:
            seed = None if prev is None else (h.cut(prev) + lam * nback(h, prev), prev.copy())   # previous FINAL partition
            t1 = time.time(); r, th_new, ne = search(pool, th, lam, seed); p = r[1].copy()
            if np.isinf(lam): refine(h, p, eps); assert h.closed(p)
            else: prefine(h, p, eps, lam)
            assert lo <= np.sum(p == 0) <= hi
            c, bk = h.cut(p), nback(h, p); J = c + (0 if np.isinf(lam) else lam * bk)
            print(f"   lam={lam:g}: cut {c} back {bk} J {J:.0f} | t {np.clip(th_new[d], 0, 1):.2f} | {ne} evals [{time.time() - t1:.0f}s]", flush=True)
            th = th_new; prev = p.copy()
    print(f"   done [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    a = sys.argv
    main(a[1], int(a[2]) if len(a) > 2 else 12)
