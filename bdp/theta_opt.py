"""RESULTS §25: a clean method -- optimise the separator parameters directly on the exact objective.

Problem:  min cut(part) + lam * back(part)  s.t. balance;  here lam = inf (acyclic).
Method:
  1. representation  X in R^{n x 16}: first 16 of 32 standardised hypergraph spectral eigvecs (as before)
  2. decoder         theta = (a, b) in R^32  ->  s(x) = |x - a| - |x - b|  (plane = far-seed limit)
                     -> exact L1 isotonic projection on the DAG (exact_ir) -> topological order (f, Kahn by s)
                     -> exact acyclic sweep in the balance window;  both orientations, better kept.
                     Every theta maps to a feasible acyclic balanced partition.
  3. optimiser       CMA-ES with library defaults (popsize 4+3 ln 32 = 14, sigma0 = 1 on standardised X),
                     restarted from a new random x0 whenever it stops, until 1000 evaluations.
                     Fitness = the exact projected sweep cut (no proxy, no pool, no top-k).
  4. finish          one acyclic refine (FM + flow) of the best partition found.
Replaces steps 2, 3, 6, 7 of the §24 pipeline.  Compared with: P (§24 pipeline result) and the §19 pool best
(all 120 pool candidates refined).
Cost: 1000 evaluations x 2 exact projections (~log2 n max-flows each) + 1 refine.
Usage: python bdp/theta_opt.py <src> [eps] [evals]
"""
import sys, os, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
import cma
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from refine import HG, load, bounds, sweep_part, refine, kahn
from dims import spectral
from exact_ir import l1_ir_exact
from separability import SAVE

std = lambda A: (A - A.mean(0)) / (A.std(0) + 1e-12)
HERE = os.path.dirname(os.path.abspath(__file__))
P24 = {"serv": 98, "omsp": 361, "picosoc": 335, "aes": 664, "riscv": 304, "biriscv": 238}   # §24 log


def main(src, eps, evals):
    name = os.path.basename(src.rstrip("/")).replace("_hg.npz", "")
    rng = np.random.default_rng(0); h = HG(*load(src)); lo, hi = bounds(h.n, eps)
    T = np.load(os.path.join(SAVE, f"{name}_e{int(eps * 100):02d}.npy")); tc = h.cut(T)
    pool_best = int(np.load(os.path.join(HERE, f"candsel_{name}.npz"))["truth"].min())
    t0 = time.time(); X = std(spectral(h, 32, rng))[:, :16]; d = X.shape[1]
    best = [None]; nev = [0]

    def J(theta):
        a, b = theta[:d], theta[d:]
        s = np.sqrt(((X - a) ** 2).sum(1)) - np.sqrt(((X - b) ** 2).sum(1)); out = None
        for sg in (1, -1):
            f = l1_ir_exact(h, sg * s)
            kr = np.empty(h.n, np.int64); kr[kahn(h, sg * s)] = np.arange(h.n)
            c, p = sweep_part(h, np.lexsort((kr, f)), eps)
            if out is None or c < out[0]: out = (c, p)
        nev[0] += 1
        if best[0] is None or out[0] < best[0][0]: best[0] = out
        if nev[0] % 100 == 0:
            print(f"   eval {nev[0]:>5}: best exact projected sweep {best[0][0]}  [{time.time() - t0:.0f}s]", flush=True)
        return out[0]

    print(f"== {name} n={h.n} | T {tc} | §24 pipeline P {P24.get(name)} | §19 pool best {pool_best}", flush=True)
    restarts = 0
    while nev[0] < evals:
        x0 = rng.normal(size=2 * d)
        es = cma.CMAEvolutionStrategy(x0, 1.0, {"maxfevals": evals - nev[0], "verbose": -9, "seed": int(rng.integers(1 << 30))})
        while not es.stop():
            P = es.ask(); es.tell(P, [J(np.asarray(x)) for x in P])
        restarts += 1
        print(f"   CMA run {restarts} stopped: {list(es.stop().keys())} after {nev[0]} evals", flush=True)
    c, p = best[0]; p = p.copy(); r = refine(h, p, eps)[-1][1]; assert h.closed(p) and lo <= np.sum(p == 0) <= hi
    print(f"   RESULT {name}: best sweep {c} -> refined {r} ({r / tc:.2f}x T) | §24 P {P24.get(name)} | pool best {pool_best} "
          f"| {restarts} CMA runs, {nev[0]} evals  [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    a = sys.argv
    main(a[1], float(a[2]) if len(a) > 2 else 0.10, int(a[3]) if len(a) > 3 else 1000)
