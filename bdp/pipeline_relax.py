"""RESULTS §24: back-edge relaxation as the LAST STAGE of the full research pipeline (no outside start).

Pipeline (identical to §19/§20; nothing from Mt-KaHyPar or best-known T enters it):
  1. embedding   16 standardised hypergraph spectral eigvecs (first 16 of 32)
  2. candidates  2000 two-seed potentials f(x) = |x-a| - |x-b|, a,b = c -/+ g u/2  (same seed as §19)
  3. pool        top 60 by unconstrained sweep + 60 random others (the §19/§20 rule)
  4. projection  exact L1 isotonic regression on the DAG (exact_ir), both orientations -> acyclic sweep, keep better
  5. selection   top 5 by projected sweep cut
  6. refine      acyclic FM + flow on the 5, best = pipeline acyclic result P
  7. relaxation  lam-continuation 64 -> 0 from P with penalised FM + finite-capacity flow (tradeoff.prefine)
Mt-KaHyPar is run ONLY to print the unconstrained reference ref; T is printed only as a reference line.
Usage: python bdp/pipeline_relax.py <src> [eps]
"""
import sys, os, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from refine import HG, load, bounds, sweep_part, refine, kahn, kahypar_parts
from dims import spectral
from exact_ir import l1_ir_exact
from separability import SAVE
from tradeoff import prefine, nback, fdrv, LAMS
import mtkahypar as mk

std = lambda A: (A - A.mean(0)) / (A.std(0) + 1e-12)
GAPS = [0.5, 2.0, 8.0, 32.0, 1e4]


def main(src, eps):
    name = os.path.basename(src.rstrip("/")).replace("_hg.npz", "")
    rng = np.random.default_rng(0); h = HG(*load(src)); lo, hi = bounds(h.n, eps)
    t0 = time.time(); X = std(spectral(h, 32, rng))[:, :16]

    def score(c, u, g):
        a, b = X[c] - g * u / 2, X[c] + g * u / 2
        return np.sqrt(((X - a) ** 2).sum(1)) - np.sqrt(((X - b) ** 2).sum(1))

    cands = []
    for _ in range(2000):
        u = rng.normal(size=16); cands.append((int(rng.integers(h.n)), u / np.linalg.norm(u), GAPS[int(rng.integers(len(GAPS)))]))
    pa = np.array([sweep_part(h, np.argsort(score(*cd), kind="stable"), eps)[0] for cd in cands])
    o = np.argsort(pa, kind="stable"); pool = list(o[:60]) + list(rng.choice(o[60:], 60, replace=False))
    ex, parts = [], []
    for i in pool:
        s = score(*cands[i]); best = None
        for sg in (1, -1):
            fv = l1_ir_exact(h, sg * s)
            kr = np.empty(h.n, np.int64); kr[kahn(h, sg * s)] = np.arange(h.n)
            c, p = sweep_part(h, np.lexsort((kr, fv)), eps)
            if best is None or c < best[0]: best = (c, p)
        ex.append(best[0]); parts.append(best[1])
    res = []
    for j in np.argsort(ex, kind="stable")[:5]:
        p = parts[j].copy(); r = refine(h, p, eps)[-1][1]; assert h.closed(p) and lo <= np.sum(p == 0) <= hi
        res.append((r, p))
    P0, p = min(res, key=lambda x: x[0]); p = p.astype(np.int8); tp = time.time() - t0
    ref = min(c for c, _ in kahypar_parts(h, eps, mk.initialize(8), seeds=3))
    Tp = os.path.join(SAVE, f"{name}_e{int(eps * 100):02d}.npy"); tc = h.cut(np.load(Tp)) if os.path.exists(Tp) else None
    print(f"== {name} n={h.n} | ref (unconstrained, reference only) {ref} | T (reference only) {tc}", flush=True)
    print(f"   pipeline acyclic P = {P0} ({P0 / ref:.2f}x ref" + (f", {P0 / tc:.2f}x T" if tc else "") + f")  [{tp:.0f}s]", flush=True)
    for lam in sorted(LAMS, reverse=True):
        t = time.time(); prefine(h, p, eps, lam); assert lo <= np.sum(p == 0) <= hi
        c, b = h.cut(p), nback(h, p)
        print(f"   lam={lam:<6}: cut {c:>5} ({c / ref:.2f}x ref, {c / P0:.2f}x P) | back {b:>5} | drivers {fdrv(h, p):>4}  [{time.time() - t:.0f}s]", flush=True)
    print(f"   [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 0.10)
