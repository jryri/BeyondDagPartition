"""RESULTS §40: the few-back-edge axis on the §39 method.  Is "solve acyclic first, relax afterwards" a problem?

Objective  J_lam = cut + lam * back   (back = DAG edges from block 1 to block 0; lam = inf is the acyclic problem).
Two ways to get a solution for a finite lam, same embedding [X|U|L], same plane family, same search:
  (a) RELAX-AFTER   search at lam = inf (= §39) -> refine -> downward continuation lam = 64, 32, ..., with the §21
                    penalised refinement (penalised FM + finite-capacity flow), warm-started each step.
  (c) LAM-INSIDE    the search optimises J_lam directly: each evaluation = exact projection -> acyclic sweep ->
                    penalised FM at this lam; fresh start w0 = e_1 (not warm-started from the acyclic solution);
                    finish = penalised refinement at this lam.
Compared at lam = 4 and lam = 1 by the same number J_lam, plus (cut, back).
If (c) is clearly lower, the acyclic optimum and the few-back-edge optimum lie in different basins and
relax-after is the wrong design; if equal, relax-after is enough (and 1/3 of the cost).
Usage: python bdp/backedge_ul.py <src> [eps] [workers]
"""
import sys, os, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from refine import HG, load, bounds, refine, fm, sweep_part, kahn
from exact_ir import l1_ir_fast

_G = {}
LAMS_C = (4.0, 1.0)


def _init(src, F, eps): _G["h"] = HG(*load(src)); _G["F"] = F; _G["eps"] = eps


def _pfm(h, p, eps, lam):
    from tradeoff import pfm_pass, nback
    J = h.cut(p) + lam * nback(h, p)
    for _ in range(10):
        nj = pfm_pass(h, p, eps, lam)
        if nj >= J: break
        J = nj
    return h.cut(p) + lam * nback(h, p)


def _J(args):
    w, lam = args; h, F, eps = _G["h"], _G["F"], _G["eps"]; s = F @ w; out = None
    for sg in (1, -1):
        f = l1_ir_fast(h, sg * s); kr = np.empty(h.n, np.int64); kr[kahn(h, sg * s)] = np.arange(h.n)
        c, p = sweep_part(h, np.lexsort((kr, f)), eps); p = p.copy()
        c = fm(h, p, eps) if np.isinf(lam) else _pfm(h, p, eps, lam)
        if out is None or c < out[0]: out = (c, p)
    return out


def search(pool, d, lam, evals=1000, sigma0=2.0, sigma_min=1 / 32):
    E = np.eye(d); memo = {}; inc = [None]
    def batch(ths):
        new = []; seen = set()
        for th in ths:
            k = th.tobytes()
            if k not in memo and k not in seen: new.append(th); seen.add(k)
        for th, r in zip(new, pool.map(_J, [(t, lam) for t in new], chunksize=1)): memo[th.tobytes()] = r
        for th in ths:
            r = memo[th.tobytes()]
            if inc[0] is None or r[0] < inc[0][0]: inc[0] = r
        return [memo[th.tobytes()][0] for th in ths]
    pts = lambda th, s: [th + s * e for e in E] + [th - s * e for e in E]
    def val(v, s):
        Jp, Jm = np.array(v[:d]), np.array(v[d:]); return (Jp + Jm).mean() / 2, (Jp - Jm) / (2 * s)
    th = np.zeros(d); th[0] = 1.0; sigma = sigma0
    v = batch([th] + pts(th, sigma)); F_, g = val(v[1:], sigma)
    while sigma >= sigma_min and len(memo) < evals:
        if not np.any(g):
            sigma /= 2; F_, g = val(batch(pts(th, sigma)), sigma); continue
        cand = th - sigma * g / np.linalg.norm(g); cand /= np.linalg.norm(cand)
        v = batch([cand] + pts(cand, sigma)); Fc, gc = val(v[1:], sigma)
        if Fc < F_: th, F_, g = cand, Fc, gc
        else:
            sigma /= 2; F_, g = val(batch(pts(th, sigma)), sigma)
    return inc[0][1].copy(), len(memo)


def main(src, eps, workers):
    from det import eig
    import learnenv as le
    from tradeoff import prefine, nback
    name = os.path.basename(src.rstrip("/")).replace("_hg.npz", "")
    h = HG(*load(src)); lo, hi = bounds(h.n, eps); t0 = time.time()
    vals, U, dv, inv, w = eig(h, 32); X = U[:, :16] * inv[:, None]
    for j in range(16):
        if np.sum(X[h.ev, j] - X[h.eu, j]) < 0: X[:, j] *= -1
    X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    D = dict(X=X, Pf=le.plan(h.n, h.eu, h.ev), Pb=le.plan(h.n, h.ev, h.eu))
    F = le.feats(D, np.array([20.0, 4.0, np.log(1e-4), 20.0, 4.0, np.log(1e-4)])); d = F.shape[1]
    bal = lambda p: lo <= np.sum(p == 0) <= hi
    print(f"== {name} n={h.n} | DAG edges {len(h.eu)}", flush=True)
    with Pool(workers, initializer=_init, initargs=(src, F, eps)) as pool:
        p, ne = search(pool, d, np.inf); refine(h, p, eps); assert h.closed(p) and bal(p)
        print(f"   acyclic (lam=inf): cut {h.cut(p)} | back 0   [{ne} evals, {time.time() - t0:.0f}s]", flush=True)
        after = {}; q = p.copy()
        for lam in (64, 32, 16, 8, 4, 2, 1):
            prefine(h, q, eps, lam); assert bal(q); after[float(lam)] = (h.cut(q), nback(h, q))
        for lam in LAMS_C:
            t1 = time.time(); r, ne = search(pool, d, lam); prefine(h, r, eps, lam); assert bal(r)
            ca, ba = after[lam]; cc, bc = h.cut(r), nback(h, r); Ja, Jc = ca + lam * ba, cc + lam * bc
            print(f"   lam={lam:g}: (a) relax-after cut {ca:5} back {ba:4} J {Ja:7.1f} | (c) lam-inside cut {cc:5} back {bc:4} J {Jc:7.1f} "
                  f"| (c) vs (a): {100 * (Jc - Ja) / Ja:+.1f}%   [{ne} evals, {time.time() - t1:.0f}s]", flush=True)
    print(f"   done [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    a = sys.argv
    main(a[1], float(a[2]) if len(a) > 2 else 0.10, int(a[3]) if len(a) > 3 else 12)
