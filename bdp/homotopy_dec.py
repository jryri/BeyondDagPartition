"""RESULTS §44: can the max-flow projection be removed?  Same embedding, family and search; only the DECODER changes.

Embedding [X | U | L] with the exact envelope (48 dims), free-sign plane s = F w, homotopy as §39, 1000 evaluations.
Decoders (score s -> order -> sweep -> FM), all without max-flow:
  linf   f = (U(s) + L(s)) / 2: exact L-inf isotonic regression, two levelised max/min passes, O(DAG edges)
  alpha  f = a U(s) + (1 - a) L(s), a = sigmoid(t) searched as a 49th parameter (every a in [0, 1] is monotone)
  kahn   topological order closest to s by greedy priority (Kahn with key s), O((n + m) log n)
  nop    NO projection, finite lam only: order by s itself; the sweep picks the split minimising cut + lam * back
         (back edges of every split counted exactly in O(m) by a difference array); then penalised FM at lam
Reference: decoder l1 (exact L1 projection by max-flow) = `homotopy_ul.py ... ule` for lam = inf, §40 (c) for finite lam.
Usage: python bdp/homotopy_dec.py <src> <decoder> [lam] [workers]
"""
import sys, os, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from refine import HG, load, bounds, refine, fm, kahn

_G = {}


def amax(x, P):
    U = x.copy()
    for ss, du, st, cnt in P: U[du] = np.maximum(x[du], np.maximum.reduceat(U[ss], st))
    return U


def cut_profile(h, order):
    n = h.n; pos = np.empty(n, np.int64); pos[order] = np.arange(n); q = pos[h.pins]
    lo_ = np.full(h.M, n); hi_ = np.zeros(h.M, np.int64)
    np.minimum.at(lo_, h.netid, q); np.maximum.at(hi_, h.netid, q)
    cut = np.cumsum(np.bincount(lo_ + 1, minlength=n + 2)[:n + 2] - np.bincount(hi_ + 1, minlength=n + 2)[:n + 2])
    pu, pv = pos[h.eu], pos[h.ev]; k = pv < pu          # edge u->v is a back edge for splits s with pv < s <= pu
    back = np.cumsum(np.bincount(pv[k] + 1, minlength=n + 2)[:n + 2] - np.bincount(pu[k] + 1, minlength=n + 2)[:n + 2])
    return cut, back


def _init(src, F, eps, dec, lam):
    import learnenv as le
    h = HG(*load(src)); _G.update(h=h, F=F, eps=eps, dec=dec, lam=lam, lv=le.levels(h.n, h.eu, h.ev),
                                  Pf=le.plan(h.n, h.eu, h.ev), Pb=le.plan(h.n, h.ev, h.eu))


def _J(th):
    h, F, eps, dec, lam, lv = (_G[k] for k in ("h", "F", "eps", "dec", "lam", "lv")); d = F.shape[1]
    s0 = F @ th[:d]; a, b = bounds(h.n, eps); out = None
    for sg in (1, -1):
        s = sg * s0
        if dec == "kahn": order = kahn(h, s)
        elif dec == "nop": order = np.argsort(s, kind="stable")
        else:
            al = 0.5 if dec == "linf" else 1 / (1 + np.exp(-th[d]))
            f = al * amax(s, _G["Pf"]) - (1 - al) * amax(-s, _G["Pb"]); order = np.lexsort((s, lv, f))
        cut, back = cut_profile(h, order)
        obj = cut[a:b + 1] if np.isinf(lam) else cut[a:b + 1] + lam * back[a:b + 1]
        k = a + int(np.argmin(obj)); p = np.ones(h.n, np.int8); p[order[:k]] = 0
        if np.isinf(lam): c = fm(h, p, eps)
        else:
            from backedge_ul import _pfm
            c = _pfm(h, p, eps, lam)
        if out is None or c < out[0]: out = (c, p)
    return out


def main(src, dec, lam, workers, evals=1000, start=0, sigma0=2.0, sigma_min=1 / 32):
    from det import eig
    import learnenv as le
    from tradeoff import prefine, nback
    name = os.path.basename(src.rstrip("/")).replace("_hg.npz", "")
    h = HG(*load(src)); eps = 0.10; lo, hi = bounds(h.n, eps); t0 = time.time()
    vals, U, dv, inv, w = eig(h, 32); X = U[:, :16] * inv[:, None]
    for j in range(16):
        if np.sum(X[h.ev, j] - X[h.eu, j]) < 0: X[:, j] *= -1
    X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    ex = lambda X_, P: np.column_stack([amax(X_[:, j] if True else 0, P) for j in range(X_.shape[1])])
    def env(X_, P):
        U_ = X_.copy()
        for ss, du, st, cnt in P: U_[du] = np.maximum(X_[du], np.maximum.reduceat(U_[ss] + 1e-4, st, axis=0))
        return U_
    F = np.hstack([X, env(X, le.plan(h.n, h.eu, h.ev)), -env(-X, le.plan(h.n, h.ev, h.eu))]); F = (F - F.mean(0)) / (F.std(0) + 1e-12)
    d = F.shape[1]; D = d + (1 if dec == "alpha" else 0); E = np.eye(D); memo = {}; inc = [None]
    with Pool(workers, initializer=_init, initargs=(src, F, eps, dec, lam)) as pool:
        def batch(ths):
            new = []; seen = set()
            for th in ths:
                k = th.tobytes()
                if k not in memo and k not in seen: new.append(th); seen.add(k)
            for th, r in zip(new, pool.map(_J, new, chunksize=2)): memo[th.tobytes()] = r
            for th in ths:
                r = memo[th.tobytes()]
                if inc[0] is None or r[0] < inc[0][0]: inc[0] = r
            return [memo[th.tobytes()][0] for th in ths]
        pts = lambda th, s: [th + s * e for e in E] + [th - s * e for e in E]
        def val(v, s):
            Jp, Jm = np.array(v[:D]), np.array(v[D:]); return (Jp + Jm).mean() / 2, (Jp - Jm) / (2 * s)
        def renorm(th):
            th = th.copy(); th[:d] /= np.linalg.norm(th[:d]); return th
        th = np.zeros(D); th[start] = 1.0; sigma = sigma0; steps = 0
        v = batch([th] + pts(th, sigma)); F_, g = val(v[1:], sigma)
        while sigma >= sigma_min and len(memo) < evals:
            if not np.any(g):
                sigma /= 2; F_, g = val(batch(pts(th, sigma)), sigma); continue
            cand = renorm(th - sigma * g / np.linalg.norm(g))
            v = batch([cand] + pts(cand, sigma)); Fc, gc = val(v[1:], sigma)
            if Fc < F_: th, F_, g = cand, Fc, gc; steps += 1
            else:
                sigma /= 2; F_, g = val(batch(pts(th, sigma)), sigma)
    ts = time.time() - t0; p = inc[0][1].copy()
    if np.isinf(lam): refine(h, p, eps); assert h.closed(p)
    else: prefine(h, p, eps, lam)
    assert lo <= np.sum(p == 0) <= hi
    c, bk = h.cut(p), nback(h, p); extra = f" | alpha {1 / (1 + np.exp(-th[d])):.2f}" if dec == "alpha" else ""
    print(f"RESULT {name} decoder={dec} lam={lam:g} start=e{start + 1} evals={evals}: cut {c} back {bk} J {c + (0 if np.isinf(lam) else lam * bk):.0f} | "
          f"{steps} steps, {len(memo)} evals, search {ts:.0f}s ({1000 * ts / len(memo):.0f} ms/eval wall, {workers} workers){extra}", flush=True)


if __name__ == "__main__":
    a = sys.argv
    main(a[1], a[2], float(a[3]) if len(a) > 3 else np.inf, int(a[4]) if len(a) > 4 else 12,
         int(a[5]) if len(a) > 5 else 1000, int(a[6]) if len(a) > 6 else 0)
