"""Quest: can a MONOTONE DICTIONARY carry the whole search into vector space?

Idea (RESULTS §17): touch the graph a fixed number of times, then search only in R^D.
  offline : D directions d in the 16-dim spectral space (16 eigvecs, +/-, then random +/-),
            each projected once by L1 isotonic regression on the DAG -> column M[:, j] (nondecreasing along every DAG edge)
  online  : score s = M w, w >= 0  -> still nondecreasing along edges (nonneg sum of monotone functions)
            order = lexsort(topo_rank, s) is topological -> EVERY split is acyclic,
            so the fitness is the TRUE acyclic sweep cut (no proxy, no per-candidate projection)
  final   : refine the top-5 distinct directions (acyclic FM + flow)
Tests per design (eps=.10, T = best known in best_parts/):
  (a) oracle expressibility: L1-SVM with w>=0 on M (D=32/64) -> misclassified %, then decode + refine the oracle score
  (b) search, no oracle: R = 2000 random sparse w>=0;  C = CMA on w=|x| (3000 evals) seeded from R's best
Compare: §16 F0 (proxy search + per-candidate projection), §12 test 2 (8 L1-projected coordinates).
Cost: build D*(Q-1) max-flows once; per evaluation O(n D + n log n + pins).
Usage: python bdp/mono_dict.py <src> [eps] [Q] [Dmax]
"""
import sys, os, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
import cma
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from refine import HG, load, bounds, refine
from dims import spectral
from l1ir import l1_ir, kahn_rank
from separability import lp_fit, SAVE
from search import Sweeper


def build_dict(h, Fs, D, Q, rng):
    dirs = []
    for j in range(Fs.shape[1]):
        e = np.eye(Fs.shape[1])[j]; dirs += [e, -e]
    while len(dirs) < D:
        r = rng.normal(size=Fs.shape[1]); r /= np.linalg.norm(r); dirs += [r, -r]
    M = np.empty((h.n, D)); t = time.time()
    for j, d in enumerate(dirs[:D]):
        M[:, j] = l1_ir(h, Fs @ d, Q)
    M = M / np.maximum(M.max(0), 1)            # levels -> [0,1]; positive scaling keeps monotonicity
    return M, time.time() - t


def main(src, eps, Q, Dmax):
    name = os.path.basename(src.rstrip("/")).replace("_hg.npz", "")
    rng = np.random.default_rng(0); h = HG(*load(src)); sw = Sweeper(h, eps); lo, hi = bounds(h.n, eps)
    T = np.load(os.path.join(SAVE, f"{name}_e{int(eps * 100):02d}.npy")); tc = h.cut(T); y = np.where(T == 1, 1.0, -1.0)
    t0 = time.time(); F = spectral(h, 16, rng); Fs = (F - F.mean(0)) / (F.std(0) + 1e-12)
    M, tb = build_dict(h, Fs, Dmax, Q, rng)
    topo = kahn_rank(h, M.sum(1))              # fixed tie-break, computed once
    print(f"== {name} n={h.n} pins={len(h.pins)} target {tc} | eigsh+dict D={Dmax} Q={Q}: {time.time() - t0:.0f}s (dict {tb:.0f}s)", flush=True)

    def decode(s):
        order = np.lexsort((topo, s)); c, sp_ = sw.eval(order, 0)
        p = np.ones(h.n, np.int8); p[order[:sp_]] = 0
        return c, p

    def finish(cands):                          # refine top-5 distinct directions
        cands.sort(key=lambda x: x[0]); top = []
        for c, w in cands:
            u = w / (np.linalg.norm(w) + 1e-12)
            if all(abs(u @ (v / np.linalg.norm(v))) < 0.99 for _, v in top): top.append((c, w))
            if len(top) == 5: break
        out = []
        for c, w in top:
            c2, p = decode(M[:, :w.size] @ w); assert c2 == c and h.closed(p) and lo <= np.sum(p == 0) <= hi
            r = refine(h, p, eps)[-1][1]; assert h.closed(p) and lo <= np.sum(p == 0) <= hi; out.append(r)
        return top[0][0], out

    for D in sorted({32, 64, Dmax} & set(range(1, Dmax + 1))):
        Md = M[:, :D]; line = f"   D={D:>3}:"
        # (a) oracle
        if D <= 64:
            t = time.time(); Ms = (Md - Md.mean(0)) / (Md.std(0) + 1e-12)
            w, b, _, mis = lp_fit(Ms, y, True)
            wd = w / (Md.std(0) + 1e-12)          # same score on unstandardised columns (positive rescale)
            c, p = decode(Md @ wd); assert h.closed(p); r = refine(h, p, eps)[-1][1]
            line += f" oracle mis {100 * mis / h.n:5.2f}% sweep {c} -> {r} ({r / tc:.2f}xT) [{time.time() - t:.0f}s] |"
        # (b) search
        t = time.time(); pool = []
        for _ in range(2000):
            w = rng.exponential(size=D) * (rng.random(D) < rng.uniform(0.05, 0.5))
            if w.sum() > 0: pool.append((decode(Md @ w)[0], w))
        fr, rr = finish(list(pool)); tr = time.time() - t
        x0 = min(pool, key=lambda x: x[0])[1]; x0 = x0 / np.linalg.norm(x0)
        es = cma.CMAEvolutionStrategy(x0, 0.3, {"maxfevals": 3000, "popsize": 32, "verbose": -9, "seed": 1,
                                      "tolfun": 0, "tolfunhist": 0, "tolflatfitness": 10**6, "tolstagnation": 10**6, "tolx": 0})
        cpool = []
        while not es.stop():
            P = es.ask(); R = [(decode(Md @ np.abs(p))[0], np.abs(np.asarray(p))) for p in P]
            es.tell(P, [x[0] for x in R]); cpool += R
        fc, rc = finish(pool + cpool); tcm = time.time() - t - tr
        line += (f" R sweep {fr} -> {min(rr)} ({min(rr) / tc:.2f}xT) [{tr:.0f}s] |"
                 f" R+C sweep {fc} -> {min(rc)} ({min(rc) / tc:.2f}xT) [{tcm:.0f}s]")
        print(line, flush=True)
    print(f"   [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    a = sys.argv
    main(a[1], float(a[2]) if len(a) > 2 else 0.10, int(a[3]) if len(a) > 3 else 32, int(a[4]) if len(a) > 4 else 128)
