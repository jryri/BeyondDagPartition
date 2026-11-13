"""§42 premise test: replace the max-flow L1 projection of the score by a LEVELISED projection (no max-flow).
  L1    exact L1 isotonic regression (max-flow, current decoder), tie-break (level, s)
  Linf  f = (U + L) / 2, U = ancestor-max of s, L = descendant-min of s: the exact L-inf isotonic regression,
        two level-by-level vectorised passes, O(DAG edges)
  U / L alone: ancestor-max only / descendant-min only
All are monotone, so (f, level, s) is a topological order; then the same sweep and FM.
Directions w: the oracle plane on [X|U|L] (good region) plus Gaussian perturbations of it, both orientations."""
import sys, os, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import learnenv as le
from refine import HG, load, sweep_part, fm
from det import eig
from exact_ir import l1_ir_fast
std = lambda F: (F - F.mean(0)) / (F.std(0) + 1e-12)

def amax(x, P):
    U = x.copy()
    for ss, du, st, cnt in P: U[du] = np.maximum(x[du], np.maximum.reduceat(U[ss], st))
    return U

for name in sys.argv[1:]:
    h = HG(*load(f"data/{name}_hg.npz")); vals, U, dv, inv, w = eig(h, 32); X = U[:, :16] * inv[:, None]
    for j in range(16):
        if np.sum(X[h.ev, j] - X[h.eu, j]) < 0: X[:, j] *= -1
    X = std(X); Pf = le.plan(h.n, h.eu, h.ev); Pb = le.plan(h.n, h.ev, h.eu); lv = le.levels(h.n, h.eu, h.ev)
    ex = lambda X_, P: np.column_stack([amax(X_[:, j], P) for j in range(X_.shape[1])])
    F = std(np.hstack([X, ex(X, Pf), -ex(-X, Pb)])); T = np.load(os.path.join(le.TGT, f"{name}_e10.npy"))
    w0 = LogisticRegression(C=1e4, max_iter=5000).fit(F, T).coef_[0]; w0 /= np.linalg.norm(w0); rng = np.random.default_rng(0)
    tm = dict(L1=0., Linf=0.); rows = []; nproj = 0
    for sig in (0, 0.1, 0.1, 0.3, 0.3, 1.0):
        s0 = F @ (w0 + sig * rng.normal(size=F.shape[1]) / np.sqrt(F.shape[1])); best = {}
        for sg in (1, -1):
            s = sg * s0; nproj += 1
            t = time.time(); f1 = l1_ir_fast(h, s); tm["L1"] += time.time() - t
            t = time.time(); Us = amax(s, Pf); Ls = -amax(-s, Pb); fi = (Us + Ls) / 2; tm["Linf"] += time.time() - t
            for k, f in (("L1", f1), ("Linf", fi), ("U", Us), ("L", Ls)):
                assert np.all(f[h.ev] >= f[h.eu] - 1e-9)
                c, p = sweep_part(h, np.lexsort((s, lv, f)), 0.10); assert h.closed(p); p = p.copy(); c2 = fm(h, p, 0.10)
                if k not in best or c2 < best[k][1]: best[k] = (c, c2)
        rows.append(f"   sigma {sig:<4}: " + " | ".join(f"{k} {best[k][0]:5} -> {best[k][1]:5}" for k in ("L1", "Linf", "U", "L")))
    print(f"== {name} (target {h.cut(T)}): per projection L1 {1000 * tm['L1'] / nproj:.0f} ms, Linf {1000 * tm['Linf'] / nproj:.1f} ms "
          f"({tm['L1'] / tm['Linf']:.0f}x)   [sweep -> after FM, better orientation]", flush=True)
    print("\n".join(rows), flush=True)
