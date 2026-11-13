"""Is the remaining 2-3.5x gap a real price of acyclicity or a stuck search?
Many diverse starts: Mt-KaHyPar seeds x input imbalance {0.02, 0.10, 0.30} -> closure repair (4 variants)
-> forced rebalance -> FM + flow. If the best keeps dropping with more starts, search is the bottleneck.
Usage: python bdp/probe_multistart.py data/x_hg.npz [eps] [seeds]
"""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from refine import HG, load, kahypar_parts, closure_variants, rebalance, refine, bounds
import mtkahypar as mk

src = sys.argv[1]; eps = float(sys.argv[2]) if len(sys.argv) > 2 else 0.10; S = int(sys.argv[3]) if len(sys.argv) > 3 else 8
h = HG(*load(src)); mtk = mk.initialize(8); t0 = time.time()
ref = min(c for c, _ in kahypar_parts(h, eps, mtk, seeds=3))
print(f"== {os.path.basename(src)} eps {eps}: unconstrained ref {ref}", flush=True)
allc = []; best = 10**9
for ein in (0.02, 0.10, 0.30):
    kp = kahypar_parts(h, ein, mtk, seeds=S)
    for k, (c, prt) in enumerate(kp):
        cand = []
        for v in closure_variants(h, prt):
            if rebalance(h, v, eps): cand.append((h.cut(v), v))
        cand.sort(key=lambda x: x[0])
        for c0, v in cand[:2]:
            hist = refine(h, v, eps); f = hist[-1][1]; allc.append(f); best = min(best, f)
        print(f"  in-eps {ein:.2f} seed {k}: unconstrained {c}, repaired starts {[x[0] for x in cand[:2]]} -> best so far {best} ({best / ref:.2f}x)", flush=True)
print(f"  final best {best} ({best / ref:.2f}x); median of refined starts {int(np.median(allc))}; {len(allc)} starts; {time.time() - t0:.0f}s")
