"""RESULTS §34: which layer is the bottleneck -- embedding, decoder family, or search?  (oracle diagnosis, not a method)

Target y = best known acyclic partition, recomputed: the better of K+repair+refine and dagP-w(10 seeds)+refine (§33).
Representation X = the 16 standardised spectral coordinates used by homotopy (det.eig).
Every score goes through the SAME decoder as the method: exact L1 projection -> sweep -> FM (J2), then full refine.

  L1 embedding : can ANY smooth function of X reproduce y?
      knn   5-fold out-of-fold kNN (k=15) probability   -- flexible, local
      lin   logistic regression on X                       -- a plane (the far-seed limit of the family)
  L2 family    : best two-seed separator s = |x-a| - |x-b|, fitted to y by logistic loss (a, b, scale, offset;
                 L-BFGS, analytic gradient, init a/b = class centroids)
  L3 search    : homotopy-J2 (§32), which never sees y

Reading: knn bad -> embedding is the ceiling.  knn good, two-seed bad -> family.  two-seed good, homotopy bad -> search.
Reported per score: misclassification vs y, refined cut, and node disagreement of the final partition with y.
Usage: python bdp/oracle.py <src> [eps]
"""
import sys, os, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from refine import HG, load, bounds, refine, kahypar_parts, closure_variants, rebalance, sweep_part, kahn, fm
from exact_ir import l1_ir_fast
from dagp_run import run_dagp, OUT as DAGP_DIR
from det import eig
import mtkahypar as mk

HERE = os.path.dirname(os.path.abspath(__file__)); TGT = os.path.join(HERE, "best_parts2")
H32 = {"serv": 71, "omsp": 286, "picosoc": 287, "aes": 551, "riscv": 301, "biriscv": 238}


def target(h, eps, name):
    path = os.path.join(TGT, f"{name}_e{int(eps * 100):02d}.npy")
    if os.path.exists(path): p = np.load(path); return h.cut(p), "saved", p
    lo, hi = bounds(h.n, eps); ok = lambda p: h.closed(p) and lo <= np.sum(p == 0) <= hi; c = []
    kp = kahypar_parts(h, eps, mk.initialize(8), seeds=3); st = []
    for _, prt in kp:
        for v in closure_variants(h, prt):
            if rebalance(h, v, eps) and ok(v): st.append((h.cut(v), v))
    for _, v in sorted(st, key=lambda x: x[0])[:3]:
        q = v.copy(); c.append((refine(h, q, eps)[-1][1], "K+repair", q))
    for s in range(1, 11):
        p = run_dagp(os.path.join(DAGP_DIR, f"{name}_w.dot"), eps, s).astype(np.int8)
        if not h.closed(p): p = (1 - p).astype(np.int8)
        if ok(p): q = p.copy(); c.append((refine(h, q, eps)[-1][1], "dagP-w", q))
    best = min(c, key=lambda x: x[0]); assert ok(best[2])
    os.makedirs(TGT, exist_ok=True); np.save(path, best[2]); return best


def decode(h, s, eps):
    out = None
    for sg in (1, -1):
        f = l1_ir_fast(h, sg * s); kr = np.empty(h.n, np.int64); kr[kahn(h, sg * s)] = np.arange(h.n)
        c, p = sweep_part(h, np.lexsort((kr, f)), eps); p = p.copy(); c = fm(h, p, eps)
        if out is None or c < out[0]: out = (c, p)
    p = out[1].copy(); r = refine(h, p, eps)[-1][1]; return r, p


def fit_two_seed(X, y):
    n, d = X.shape; ys = 2.0 * y - 1                        # y = 1 for block 1
    def f(w):
        a, b, al, c = w[:d], w[d:2 * d], w[2 * d], w[2 * d + 1]
        Da, Db = X - a, X - b; da = np.sqrt((Da * Da).sum(1)) + 1e-9; db = np.sqrt((Db * Db).sum(1)) + 1e-9
        s = da - db; z = al * s - c; m = -ys * z
        L = np.mean(np.logaddexp(0, m)); g = -ys / (1 + np.exp(-m)) / n
        ga = al * (-(g / da)[:, None] * Da).sum(0); gb = al * ((g / db)[:, None] * Db).sum(0)
        return L, np.r_[ga, gb, (g * s).sum(), -g.sum()]
    w0 = np.r_[X[y == 0].mean(0), X[y == 1].mean(0), 1.0, 0.0]
    r = minimize(f, w0, jac=True, method="L-BFGS-B", options={"maxiter": 2000})
    a, b, al, c = r.x[:d], r.x[d:2 * d], r.x[2 * d], r.x[2 * d + 1]
    s = al * (np.sqrt(((X - a) ** 2).sum(1)) - np.sqrt(((X - b) ** 2).sum(1))) - c
    return s, np.linalg.norm(a - b)


def main(src, eps):
    name = os.path.basename(src.rstrip("/")).replace("_hg.npz", "")
    h = HG(*load(src)); t0 = time.time()
    tc, tsrc, T = target(h, eps, name); y = T.astype(int)
    vals, U, dv, inv, w = eig(h, 32)
    X = U[:, :16] * inv[:, None]; X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    print(f"== {name} n={h.n} | target {tc} ({tsrc}) | homotopy-J2 {H32[name]} [{time.time() - t0:.0f}s]", flush=True)
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    pk = cross_val_predict(KNeighborsClassifier(15), X, y, cv=cv, method="predict_proba")[:, 1]
    lr = LogisticRegression(C=1e4, max_iter=5000).fit(X, y); sl = lr.decision_function(X)
    s2, gap = fit_two_seed(X, y)
    rows = [("L1 knn (5-fold OOF)", pk - 0.5), ("L1 lin (plane)", sl), (f"L2 two-seed (|a-b| {gap:.1f})", s2)]
    for lab, s in rows:
        mis = np.mean((s > 0).astype(int) != y); r, p = decode(h, s, eps)
        print(f"   {lab:26} misclass {100 * mis:5.1f}% | refined cut {r:5} ({r / tc:.2f}x target) "
              f"| final differs from target on {100 * np.mean(p != T):5.1f}% of cells [{time.time() - t0:.0f}s]", flush=True)
    print(f"   L3 homotopy-J2 (no oracle) refined cut {H32[name]:5} ({H32[name] / tc:.2f}x target)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 0.10)
