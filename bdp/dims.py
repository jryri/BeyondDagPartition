"""Do more spectral dimensions make the best-known acyclic cut linearly expressible?
For k in (8, 16, 32): X_k = first k nontrivial eigenvectors (same operator as his_embed), no envelopes.
Oracle: free-sign L1-SVM LP fit of target T on X_k -> score -> L1 isotonic projection on the DAG -> sweep -> refine.
Caveat: the LP is supervised on T, so more dims also means more freedom to fit; the repaired+refined cut is the
meaningful number. Findability (no oracle) gets harder with k and is not tested here.
Usage: python bdp/dims.py <src> [eps]
"""
import sys, os, time
import numpy as np
from scipy.sparse.linalg import eigsh, LinearOperator
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from refine import HG, load, sweep_part, refine
from separability import lp_fit, SAVE
from l1ir import l1_ir, kahn_rank


def spectral(h, k, rng):
    n = h.n; w = 1.0 / np.bincount(h.netid)
    dv = np.asarray(h.H.sum(1)).ravel(); inv = 1 / np.sqrt(np.maximum(dv, 1))
    mv = lambda x: inv * (h.H @ (w * (h.H.T @ (inv * x))))
    vals, F = eigsh(LinearOperator((n, n), matvec=mv, dtype=float), k=k + 1, which="LA", v0=rng.normal(size=n), tol=1e-6)
    return F[:, np.argsort(-vals)[1:]] * inv[:, None]


def main(src, eps):
    name = os.path.basename(src.rstrip("/")).replace("_hg.npz", "")
    rng = np.random.default_rng(0); t0 = time.time()
    h = HG(*load(src)); T = np.load(os.path.join(SAVE, f"{name}_e{int(eps * 100):02d}.npy")); tc = h.cut(T)
    y = np.where(T == 1, 1.0, -1.0)
    F32 = spectral(h, 32, rng)
    out = [f"== {name} n={h.n} target {tc}  (eigsh {time.time() - t0:.0f}s)"]
    for k in (8, 16, 32):
        X = F32[:, :k]; Xs = (X - X.mean(0)) / (X.std(0) + 1e-12)
        w, b, slack, mis = lp_fit(Xs, y, False); s = Xs @ w
        g = l1_ir(h, s, 64); order = np.lexsort((kahn_rank(h, s), g))
        c, p = sweep_part(h, order, eps); assert h.closed(p)
        r = refine(h, p.copy(), eps)[-1][1]
        out.append(f"   k={k:>2}: LP misclassified {100 * mis / h.n:5.2f}% | L1-IR repair sweep {c} ({c / tc:.2f}x T), "
                   f"agrees {100 * np.mean(p == T):.1f}% | refined {r} ({r / tc:.2f}x T)")
    out.append(f"   [{time.time() - t0:.0f}s]")
    print("\n".join(out), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 0.10)
