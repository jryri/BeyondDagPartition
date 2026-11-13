"""§38: fair comparison of the DAG part of the embedding, same protocol as §37 (logistic free-sign plane on the oracle,
then projection -> sweep -> FM -> refine):  [X] | [X|G] (per-coordinate L1 projection) | [X|U|L] (his L-inf envelope)
| [U|L] only (X dropped) | [X|U|L] with w >= 0 on U,L and X unused (his original decoder family)."""
import sys, os, warnings
import numpy as np
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import learnenv as le
from refine import HG, load
from det import eig
from dagembed import dag_embed
from oracle import decode
his = np.array([20.0, 4.0, np.log(1e-4), 20.0, 4.0, np.log(1e-4)])
std = lambda F: (F - F.mean(0)) / (F.std(0) + 1e-12)

def nonneg_fit(F, y):                         # logistic with w >= 0 (his family: monotone columns, nonneg weights)
    n, d = F.shape; ys = 2.0 * y - 1
    def f(p):
        z = F @ p[:d] - p[d]; m = -ys * z; g = -ys / (1 + np.exp(-m)) / n
        return np.mean(np.logaddexp(0, m)), np.r_[F.T @ g, -g.sum()]
    r = minimize(f, np.r_[np.full(d, 0.1), 0.0], jac=True, method="L-BFGS-B", bounds=[(0, None)] * d + [(None, None)])
    return F @ r.x[:d] - r.x[d]

print(f"{'design':8} | [X] 16          | [X|G] L1, 32     | [X|U|L] Linf, 48 | [U|L] only, 32   | [U|L], w>=0 (his family)", flush=True)
for name in ["serv", "omsp", "riscv", "picosoc", "aes", "biriscv"]:
    h = HG(*load(f"data/{name}_hg.npz")); vals, U, dv, inv, w = eig(h, 32); X = U[:, :16] * inv[:, None]
    for j in range(16):
        if np.sum(X[h.ev, j] - X[h.eu, j]) < 0: X[:, j] *= -1
    X = std(X); T = np.load(os.path.join(le.TGT, f"{name}_e10.npy")); tc = h.cut(T)
    D = dict(X=X, Pf=le.plan(h.n, h.eu, h.ev), Pb=le.plan(h.n, h.ev, h.eu)); F3 = le.feats(D, his); UL = F3[:, 16:]
    out = []
    for F in (X, np.hstack([X, dag_embed(h, X)]), F3, UL):
        z = LogisticRegression(C=1e4, max_iter=5000).fit(F, T).decision_function(F)
        r, _ = decode(h, z, 0.10); out.append(f"{100 * np.mean((z > 0) != (T == 1)):5.1f}% -> {r:4}")
    z = nonneg_fit(UL - UL.min(0), T.astype(float)); r, _ = decode(h, z, 0.10)
    out.append(f"{100 * np.mean((z > 0) != (T == 1)):5.1f}% -> {r:4}")
    print(f"{name:8} | " + " | ".join(out) + f"   (target {tc})", flush=True)
