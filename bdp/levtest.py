"""Levelised tie-break: replace the Python Kahn pass (heap, per node) by a vectorised sort on (f, level, s).
level = longest-path depth, strictly increasing along DAG edges, so (f, level) is a topological order whenever f is monotone."""
import sys, os, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import learnenv as le
from refine import HG, load, sweep_part, kahn, fm
from det import eig
from exact_ir import l1_ir_fast
for name in sys.argv[1:]:
    h = HG(*load(f"data/{name}_hg.npz")); vals, U, dv, inv, w = eig(h, 32); X = U[:, :16] * inv[:, None]
    X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    t = time.time(); lv = le.levels(h.n, h.eu, h.ev); tl = time.time() - t
    t = time.time(); P = le.plan(h.n, h.eu, h.ev); Pb = le.plan(h.n, h.ev, h.eu); tp = time.time() - t
    def exact(X, P):
        U = X.copy()
        for ss, du, st, cnt in P: U[du] = np.maximum(X[du], np.maximum.reduceat(U[ss] + 1e-4, st, axis=0))
        return U
    t = time.time(); Ue = exact(X, P); Le = -exact(-X, Pb); te = time.time() - t
    rng = np.random.default_rng(0); tk = tv = tproj = tfm = 0; res = []
    for k in range(6):
        s = X @ rng.normal(size=16)
        t = time.time(); f = l1_ir_fast(h, s); tproj += time.time() - t
        t = time.time(); kr = np.empty(h.n, np.int64); kr[kahn(h, s)] = np.arange(h.n); o1 = np.lexsort((kr, f)); tk += time.time() - t
        t = time.time(); o2 = np.lexsort((s, lv, f)); tv += time.time() - t
        c1, p1 = sweep_part(h, o1, 0.10); c2, p2 = sweep_part(h, o2, 0.10); assert h.closed(p2)
        t = time.time(); p1 = p1.copy(); a = fm(h, p1, 0.10); tfm += time.time() - t; p2 = p2.copy(); b = fm(h, p2, 0.10)
        res.append(f"{c1}/{c2}->{a}/{b}")
    print(f"{name}: depth {lv.max()} | levels {1000 * tl:.0f} ms, plan {1000 * tp:.0f} ms, exact envelope (16 dims, U+L) {1000 * te:.0f} ms (once per design)\n"
          f"   per projection: max-flow {1000 * tproj / 6:.0f} ms | Kahn tie-break {1000 * tk / 6:.0f} ms | levelised tie-break {1000 * tv / 6:.1f} ms | FM {1000 * tfm / 6:.0f} ms\n"
          f"   sweep cut Kahn/levelised -> after FM: {'  '.join(res)}", flush=True)
