"""Quest (RESULTS §22): application = "fewer back edges is better" -> trace the cut vs back-edge frontier
at the end that matters (few back edges).

§21 went UP in lam from the unconstrained optimum and plateaued (local search cannot remove whole chains), so its
few-back-edge points were poor.  Here: the same penalised refinement (tradeoff.prefine: penalised FM + finite-capacity
flow), but continuation goes DOWN in lam, starting from the acyclic best known partition T (lam = inf).
Lowering lam lets the refinement buy cut reductions with back edges only where a back edge is cheap, so each step is a
small, local relaxation of a feasible-strict solution.
Per lam: cut, back edges, feedback drivers, cut / ref (unconstrained), cut / T.
Usage: python bdp/relax.py <src> [eps]
"""
import sys, os, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from refine import HG, load, bounds, kahypar_parts
from separability import SAVE
from tradeoff import prefine, nback, fdrv, LAMS
import mtkahypar as mk


def main(src, eps):
    name = os.path.basename(src.rstrip("/")).replace("_hg.npz", "")
    h = HG(*load(src)); lo, hi = bounds(h.n, eps); mtk = mk.initialize(8)
    T = np.load(os.path.join(SAVE, f"{name}_e{int(eps * 100):02d}.npy")).astype(np.int8); tc = h.cut(T)
    assert h.closed(T)
    t0 = time.time(); ref = min(c for c, _ in kahypar_parts(h, eps, mtk, seeds=3))
    print(f"== {name} n={h.n} ref(unconstrained) {ref} | start = acyclic best known T {tc} ({tc / ref:.2f}x ref)", flush=True)
    p = T.copy()
    for lam in sorted(LAMS, reverse=True):
        t = time.time(); prefine(h, p, eps, lam); assert lo <= np.sum(p == 0) <= hi
        c, b = h.cut(p), nback(h, p)
        print(f"   lam={lam:<6}: cut {c:>5} ({c / ref:.2f}x ref, {c / tc:.2f}x T) | back {b:>5} | drivers {fdrv(h, p):>4}  [{time.time() - t:.0f}s]", flush=True)
    print(f"   [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 0.10)
