"""Quest: is the SHAPE of the separator in the spectral space (a hyperplane) the limit?

All separators are "score + threshold": cells are sorted by a score s(x), the sweep picks the threshold.
Only the score family changes; downstream is identical (L1 isotonic projection of the score on the DAG, 64 levels
-> lexsort(level, Kahn rank) -> exact acyclic sweep -> acyclic FM + flow).
  lin16  s = w.x                      hyperplane, 16 eigvecs
  lin32  s = w.x                      hyperplane, 32 eigvecs (same #parameters as diag16)
  sph16  s = w.x + a|x|^2             sphere (Miller-Teng-Thurston-Vavasis geometric separators; Leighton-Rao balls)
  diag16 s = w.x + sum a_i x_i^2      axis-aligned quadric
  quad16 s = w.x + x'Ax               general quadric (degree-2 kernel), 152 parameters
(a) ORACLE expressibility: L1-SVM fit of each family to the best-known partition T -> misclassified %,
    then project the fitted score + sweep + refine.
(b) FINDABILITY, no oracle, same budget (2016 candidates) and same protocol for both arms:
    proxy = unconstrained sweep cut; top-5 distinct (partition agreement < .98); both orientations projected;
    better sweep per candidate refined; best of 5 reported.
      L: 16 single eigvecs + 2000 random directions          s = w.x
      S: 2016 balls centred on random cells                  s = |x - x_c|   (radius = sweep threshold)
Cost per candidate: O(n k) score + O(n log n + pins) sweep for both arms; the lift for quad16 is O(n k^2) once.
Usage: python bdp/shape.py <src> [eps]
"""
import sys, os, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from refine import HG, load, bounds, sweep_part, refine
from dims import spectral
from l1ir import l1_ir, kahn_rank
from separability import lp_fit, SAVE

std = lambda A: (A - A.mean(0)) / (A.std(0) + 1e-12)


def main(src, eps):
    name = os.path.basename(src.rstrip("/")).replace("_hg.npz", "")
    rng = np.random.default_rng(0); h = HG(*load(src)); lo, hi = bounds(h.n, eps)
    T = np.load(os.path.join(SAVE, f"{name}_e{int(eps * 100):02d}.npy")); tc = h.cut(T); y = np.where(T == 1, 1.0, -1.0)
    t0 = time.time(); F32 = std(spectral(h, 32, rng)); X = F32[:, :16]
    print(f"== {name} n={h.n} target {tc} (eigsh {time.time() - t0:.0f}s)", flush=True)

    def pipeline(s):
        g = l1_ir(h, s, 64); order = np.lexsort((kahn_rank(h, s), g)); c, p = sweep_part(h, order, eps)
        assert h.closed(p) and lo <= np.sum(p == 0) <= hi
        return c, p

    # ---------- (a) oracle
    iu = np.triu_indices(16); Xc = np.clip(X, -5, 5)      # winsorise before squaring: raw products break HiGHS (outliers)
    fam = {"lin16": X, "lin32": F32, "sph16": np.c_[X, (Xc ** 2).sum(1)], "diag16": np.c_[X, Xc ** 2],
           "quad16": np.c_[X, (Xc[:, :, None] * Xc[:, None, :])[:, iu[0], iu[1]]]}
    for lab, A in fam.items():
        t = time.time(); A = std(A)
        try: w, b, _, mis = lp_fit(A, y, False)
        except Exception as e: print(f"   oracle {lab:<6}: LP failed ({type(e).__name__})  [{time.time() - t:.0f}s]", flush=True); continue
        c, p = pipeline(A @ w); r = refine(h, p, eps)[-1][1]
        print(f"   oracle {lab:<6} ({A.shape[1]:>3} par): mis {100 * mis / h.n:5.2f}% | projected sweep {c} -> refined {r} "
              f"({r / tc:.2f}xT), agrees {100 * np.mean(p == T):.1f}%  [{time.time() - t:.0f}s]", flush=True)

    # ---------- (b) findability
    W = [np.eye(16)[j] for j in range(16)] + [rng.normal(size=16) for _ in range(2000)]
    C = rng.choice(h.n, 2016, replace=False) if h.n >= 2016 else rng.integers(0, h.n, 2016)
    arms = {"L hyperplane": [(lambda w=w: X @ w) for w in W],
            "S ball": [(lambda c=c: np.sqrt(((X - X[c]) ** 2).sum(1))) for c in C]}
    for lab, fns in arms.items():
        t = time.time(); sc = []
        for i, f in enumerate(fns):
            sc.append((sweep_part(h, np.argsort(f(), kind="stable"), eps)[0], i))
        sc.sort(); top = []
        for c0, i in sc:
            p0 = sweep_part(h, np.argsort(fns[i](), kind="stable"), eps)[1]
            if all(max(np.mean(p0 == q), np.mean(p0 != q)) < 0.98 for _, _, q in top): top.append((c0, i, p0))
            if len(top) == 5: break
        ts = time.time() - t; res = []
        for c0, i, _ in top:
            s = fns[i](); c, p = min((pipeline(sg * s) for sg in (1, -1)), key=lambda x: x[0])
            r = refine(h, p, eps)[-1][1]; assert h.closed(p) and lo <= np.sum(p == 0) <= hi
            res.append((c0, c, r))
        best = min(x[2] for x in res)
        print(f"   find {lab:<12}: proxy {[x[0] for x in res]} | projected sweep {[x[1] for x in res]} -> refined "
              f"{[x[2] for x in res]} | best {best} ({best / tc:.2f}xT)  [search {ts:.0f}s, total {time.time() - t:.0f}s]", flush=True)
    print(f"   [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 0.10)
