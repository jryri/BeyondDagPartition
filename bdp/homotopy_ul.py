"""RESULTS §39: the combination suggested by §37/§38, run as a SEARCH (no oracle anywhere).

  embedding  F = [X | U | L]: 16 spectral coordinates (oriented along the DAG) + his L-inf envelopes
             (ancestor-max U, descendant-min L; learnenv.prop with alpha = 1, beta = 55, delta = 1e-4), standardised: 48 dims
  separator  free-sign hyperplane: score s = F w, w in R^48 (the offset is the sweep's job)
  decoder    exact L1 projection of s on the DAG -> acyclic sweep in the balance window -> FM  (= J2), both orientations
  search     the §31/§32 homotopy (per-axis central differences at width sigma, sigma 2 -> 1/32, 1000 evaluations),
             with w renormalised to |w| = 1 after every step (J depends only on the direction of w: §27 showed the
             radial coordinate is a nuisance).  Start w0 = e_1 (the first spectral coordinate): deterministic.
  finish     one full acyclic refine.
`emb=x` runs the same plane search on X only (16 dims) as the control: the only difference is the U/L columns.
Usage: python bdp/homotopy_ul.py <src> [eps] [workers] [ul|x]
"""
import sys, os, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from refine import HG, load, bounds, refine, fm, sweep_part, kahn
from exact_ir import l1_ir_fast

BASE = {"serv": 61, "omsp": 341, "picosoc": 250, "aes": 475, "riscv": 251, "biriscv": 238}    # §33 best single baseline
H32 = {"serv": 71, "omsp": 286, "picosoc": 287, "aes": 551, "riscv": 301, "biriscv": 238}     # §32 two-seed on X
ORA = {"serv": 66, "omsp": 333, "picosoc": 339, "aes": 598, "riscv": 251, "biriscv": 238}     # §38 oracle plane on [X|U|L]
_G = {}


def _init(src, F, eps): _G["h"] = HG(*load(src)); _G["F"] = F; _G["eps"] = eps


def _J2(w):
    h, F, eps = _G["h"], _G["F"], _G["eps"]; s = F @ w; out = None
    for sg in (1, -1):
        f = l1_ir_fast(h, sg * s); kr = np.empty(h.n, np.int64); kr[kahn(h, sg * s)] = np.arange(h.n)
        c, p = sweep_part(h, np.lexsort((kr, f)), eps); p = p.copy(); c = fm(h, p, eps)
        if out is None or c < out[0]: out = (c, p)
    return out


def main(src, eps, workers, emb, evals=1000, sigma0=2.0, sigma_min=1 / 32):
    from det import eig
    import learnenv as le
    name = os.path.basename(src.rstrip("/")).replace("_hg.npz", "")
    h = HG(*load(src)); lo, hi = bounds(h.n, eps); t0 = time.time()
    vals, U, dv, inv, w = eig(h, 32); X = U[:, :16] * inv[:, None]
    for j in range(16):
        if np.sum(X[h.ev, j] - X[h.eu, j]) < 0: X[:, j] *= -1
    X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    if emb == "ul":
        D = dict(X=X, Pf=le.plan(h.n, h.eu, h.ev), Pb=le.plan(h.n, h.ev, h.eu))
        F = le.feats(D, np.array([20.0, 4.0, np.log(1e-4), 20.0, 4.0, np.log(1e-4)]))
    elif emb in ("ule", "sph"):                       # §43: EXACT max/min envelope (his), optionally + spherical normalisation
        def amax(X_, P):
            U_ = X_.copy()
            for ss, du, st, cnt in P: U_[du] = np.maximum(X_[du], np.maximum.reduceat(U_[ss] + 1e-4, st, axis=0))
            return U_
        F = np.hstack([X, amax(X, le.plan(h.n, h.eu, h.ev)), -amax(-X, le.plan(h.n, h.ev, h.eu))])
        F = (F - F.mean(0)) / (F.std(0) + 1e-12)
        if emb == "sph":
            A = np.hstack([F, np.ones((h.n, 1))]); F = A / np.linalg.norm(A, axis=1)[:, None]
    else: F = X
    d = F.shape[1]; E = np.eye(d); te = time.time() - t0
    print(f"== plane homotopy-J2 [emb={emb}, {d} dims] {name} n={h.n} | embedding {te:.0f}s | {workers} workers", flush=True)
    memo = {}; inc = [None]
    with Pool(workers, initializer=_init, initargs=(src, F, eps)) as pool:
        def batch(ths):
            new = []; seen = set()
            for th in ths:
                k = th.tobytes()
                if k not in memo and k not in seen: new.append(th); seen.add(k)
            for th, r in zip(new, pool.map(_J2, new, chunksize=1)): memo[th.tobytes()] = r
            for th in ths:
                r = memo[th.tobytes()]
                if inc[0] is None or r[0] < inc[0][0]: inc[0] = r
            return [memo[th.tobytes()][0] for th in ths]
        pts = lambda th, s: [th + s * e for e in E] + [th - s * e for e in E]
        def val(v, s):
            Jp, Jm = np.array(v[:d]), np.array(v[d:]); return (Jp + Jm).mean() / 2, (Jp - Jm) / (2 * s)
        th = np.zeros(d); th[0] = 1.0; sigma = sigma0; steps = 0
        v = batch([th] + pts(th, sigma)); F_, g = val(v[1:], sigma)
        while sigma >= sigma_min and len(memo) < evals:
            if not np.any(g):
                sigma /= 2; F_, g = val(batch(pts(th, sigma)), sigma); continue
            cand = th - sigma * g / np.linalg.norm(g); cand /= np.linalg.norm(cand)
            v = batch([cand] + pts(cand, sigma)); Fc, gc = val(v[1:], sigma)
            if Fc < F_: th, F_, g = cand, Fc, gc; steps += 1
            else:
                sigma /= 2; F_, g = val(batch(pts(th, sigma)), sigma)
            print(f"   sigma {sigma:<8.4g} F {F_:8.1f} | incumbent J {inc[0][0]} | {len(memo)} evals [{time.time() - t0:.0f}s]", flush=True)
    c, p = inc[0]; p = p.copy(); r = refine(h, p, eps)[-1][1]; assert h.closed(p) and lo <= np.sum(p == 0) <= hi
    print(f"   RESULT plane[{emb}] {name}: J2 {c} -> refined {r} | back edges 0 | §32 two-seed on X {H32[name]} | "
          f"§38 oracle plane {ORA[name]} | best baseline §33 {BASE[name]} | {steps} steps, {len(memo)} evals | {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    a = sys.argv
    main(a[1], float(a[2]) if len(a) > 2 else 0.10, int(a[3]) if len(a) > 3 else 12, a[4] if len(a) > 4 else "ul")
