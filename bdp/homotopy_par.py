"""RESULTS §32: homotopy-J2 (§31) with the SAME evaluations, run faster.  No change to the algorithm or its results.

  1. the 64 probes of a step (+ the candidate) are independent -> evaluated in a process pool (wall-clock only;
     total work unchanged).  The serial code always probes the candidate, so the evaluation set is identical.
  2. exact projection on the conflict core only (exact_ir.l1_ir_fast; identical output).
Usage: python bdp/homotopy_par.py <src> [eps] [workers]
"""
import sys, os, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from refine import HG, load, bounds, refine, fm, sweep_part, kahn
from exact_ir import l1_ir_fast
from separability import SAVE

R31 = {"serv": (71, 326), "omsp": (286, 436), "picosoc": (287, 1073), "aes": (551, 983), "riscv": (301, 1728), "biriscv": (238, 3583)}
_G = {}


def _init(src, X, eps):
    _G["h"] = HG(*load(src)); _G["X"] = X; _G["eps"] = eps


def _J2(th):
    h, X, eps = _G["h"], _G["X"], _G["eps"]; d = X.shape[1]
    a, b = th[:d], th[d:]
    s = np.sqrt(((X - a) ** 2).sum(1)) - np.sqrt(((X - b) ** 2).sum(1)); out = None
    for sg in (1, -1):
        f = l1_ir_fast(h, sg * s); kr = np.empty(h.n, np.int64); kr[kahn(h, sg * s)] = np.arange(h.n)
        c, p = sweep_part(h, np.lexsort((kr, f)), eps); p = p.copy(); c = fm(h, p, eps)
        if out is None or c < out[0]: out = (c, p)
    return out


def main(src, eps, workers, emb="x", evals=1000, sigma0=2.0, sigma_min=1 / 32):
    from det import eig
    name = os.path.basename(src.rstrip("/")).replace("_hg.npz", "")
    h = HG(*load(src)); lo, hi = bounds(h.n, eps)
    tc = h.cut(np.load(os.path.join(SAVE, f"{name}_e{int(eps * 100):02d}.npy")))
    t0 = time.time(); vals, U, dv, inv, w = eig(h, 32)
    X = U[:, :16] * inv[:, None]; X = (X - X.mean(0)) / (X.std(0) + 1e-12); d = 16; E = np.eye(2 * d)
    if emb == "g":
        from dagembed import dag_embed
        X = dag_embed(h, X)          # §35: DAG-aware embedding
    print(f"== homotopy-J2 parallel [emb={emb}] {name} n={h.n} | T {tc} | {workers} workers", flush=True)
    memo = {}; inc = [None]
    with Pool(workers, initializer=_init, initargs=(src, X, eps)) as pool:
        def batch(ths):                                      # evaluate the unseen ones in parallel, in order
            new = []; seen = set()
            for th in ths:
                k = th.tobytes()
                if k not in memo and k not in seen: new.append(th); seen.add(k)
            for th, r in zip(new, pool.map(_J2, new, chunksize=1)): memo[th.tobytes()] = r
            for th in ths:                                   # incumbent updated in the serial order
                r = memo[th.tobytes()]
                if inc[0] is None or r[0] < inc[0][0]: inc[0] = r
            return [memo[th.tobytes()][0] for th in ths]

        def probe_pts(th, sigma): return [th + sigma * e for e in E] + [th - sigma * e for e in E]
        def probe_val(v, sigma):
            Jp, Jm = np.array(v[:2 * d]), np.array(v[2 * d:]); return (Jp + Jm).mean() / 2, (Jp - Jm) / (2 * sigma)

        th = np.zeros(2 * d); th[0] = -2.0; th[d] = 2.0; sigma = sigma0; steps = 0
        v = batch([th] + probe_pts(th, sigma)); F, g = probe_val(v[1:], sigma)
        while sigma >= sigma_min and len(memo) < evals:
            if not np.any(g):
                sigma /= 2; F, g = probe_val(batch(probe_pts(th, sigma)), sigma); continue
            cand = th - sigma * g / np.linalg.norm(g)
            v = batch([cand] + probe_pts(cand, sigma)); Fc, gc = probe_val(v[1:], sigma)
            if Fc < F: th, F, g = cand, Fc, gc; steps += 1
            else:
                sigma /= 2; F, g = probe_val(batch(probe_pts(th, sigma)), sigma)
            print(f"   sigma {sigma:<8.4g} F {F:8.1f} | incumbent J {inc[0][0]} | {len(memo)} evals [{time.time() - t0:.0f}s]", flush=True)
    c, p = inc[0]; p = p.copy(); r = refine(h, p, eps)[-1][1]; assert h.closed(p) and lo <= np.sum(p == 0) <= hi
    el = time.time() - t0; r31, t31 = R31[name]
    print(f"   RESULT parallel {name}: J2 {c} -> refined {r} ({r / tc:.2f}x T) | §31 serial {r31} in {t31}s | "
          f"same result: {r == r31} | {steps} steps, {len(memo)} evals | {el:.0f}s ({t31 / el:.1f}x faster)", flush=True)


if __name__ == "__main__":
    a = sys.argv
    main(a[1], float(a[2]) if len(a) > 2 else 0.10, int(a[3]) if len(a) > 3 else 12, a[4] if len(a) > 4 else "x")
