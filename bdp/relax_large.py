"""RESULTS §23: the §22 downward relaxation on the large designs (ariane133 185k, bp_quad 972k cells).

No best-known T exists for these, so the acyclic start is the K+rep pipeline of large.py (§15):
Mt-KaHyPar QUALITY (3 seeds, also gives ref) -> closure repair variants -> rebalance -> 2 lowest-cut refined (acyclic
FM + flow) -> best.  Then lam-continuation DOWN (64 -> 0) with tradeoff.prefine, as in relax.py.
Usage: python bdp/relax_large.py <src> [eps]
"""
import sys, os, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from refine import HG, load, bounds, kahypar_parts, closure_variants, rebalance, refine
from tradeoff import prefine, nback, fdrv, LAMS
import mtkahypar as mk


def main(src, eps):
    name = os.path.basename(src.rstrip("/")).replace("_hg.npz", "")
    h = HG(*load(src)); lo, hi = bounds(h.n, eps); mtk = mk.initialize(8)
    print(f"== {name} n={h.n} nets={h.M} dag_edges={len(h.eu)} eps={eps}", flush=True)
    t0 = time.time(); kp = kahypar_parts(h, eps, mtk, seeds=3); ref = min(c for c, _ in kp)
    print(f"   ref (unconstrained) {ref}  [{time.time() - t0:.0f}s]", flush=True)
    t = time.time(); st = []
    for c, prt in kp:
        for v in closure_variants(h, prt):
            if rebalance(h, v, eps): st.append((h.cut(v), v))
    st.sort(key=lambda x: x[0]); best = None
    for c, v in st[:2]:
        p = v.copy(); f = refine(h, p, eps)[-1][1]; assert h.closed(p) and lo <= np.sum(p == 0) <= hi
        if best is None or f < best[0]: best = (f, p)
    S, p = best[0], best[1].astype(np.int8)
    print(f"   acyclic start (K+rep) {S} ({S / ref:.2f}x ref)  [{time.time() - t:.0f}s]", flush=True)
    for lam in sorted(LAMS, reverse=True):
        t = time.time(); prefine(h, p, eps, lam); assert lo <= np.sum(p == 0) <= hi
        c, b = h.cut(p), nback(h, p)
        print(f"   lam={lam:<6}: cut {c:>6} ({c / ref:.2f}x ref, {c / S:.2f}x start) | back {b:>6} | drivers {fdrv(h, p):>5}  [{time.time() - t:.0f}s]", flush=True)
    print(f"   [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 0.10)
