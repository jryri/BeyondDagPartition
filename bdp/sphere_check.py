"""§43 premise check of the "augmented spherical normalisation" proposal: x_hat_v = [x_v, 1] / sqrt(|x_v|^2 + 1).
Claims tested: (1) a few nodes have row norms 10-20 (leverage); (2) that is why ridge fits badly (32.6% on biriscv)
while logistic is fine; (3) the normalisation fixes ridge without changing what a hyperplane can express.
Features: [X | U | L] with the exact envelope, standardised (48 dims).  Oracle = best_parts2."""
import sys, os, warnings
import numpy as np
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import learnenv as le
from refine import HG, load
from det import eig
std = lambda F: (F - F.mean(0)) / (F.std(0) + 1e-12)
def amax(X, P):
    U = X.copy()
    for ss, du, st, cnt in P: U[du] = np.maximum(X[du], np.maximum.reduceat(U[ss] + 1e-4, st, axis=0))
    return U
def ridge_mis(F, y):
    w = np.linalg.solve(F.T @ F + 1e-3 * len(y) * np.eye(F.shape[1]), F.T @ y); return 100 * np.mean((F @ w > 0) != (y > 0))
print(f"{'design':8} | row norm median / 99% / max | ridge: raw -> sphere | logistic: raw -> sphere | share of |w|-leverage held by top 1% rows: raw -> sphere")
for name in ["serv", "omsp", "riscv", "picosoc", "aes", "biriscv"]:
    h = HG(*load(f"data/{name}_hg.npz")); vals, U, dv, inv, w = eig(h, 32); X = U[:, :16] * inv[:, None]
    for j in range(16):
        if np.sum(X[h.ev, j] - X[h.eu, j]) < 0: X[:, j] *= -1
    X = std(X); F = std(np.hstack([X, amax(X, le.plan(h.n, h.eu, h.ev)), -amax(-X, le.plan(h.n, h.ev, h.eu))]))
    T = np.load(os.path.join(le.TGT, f"{name}_e10.npy")); y = np.where(T == 1, 1.0, -1.0)
    A = np.hstack([F, np.ones((h.n, 1))]); nr = np.linalg.norm(A, axis=1); S = A / nr[:, None]
    lev = lambda M: 100 * np.sort((M * M).sum(1))[-max(1, h.n // 100):].sum() / (M * M).sum()
    lg = lambda M: 100 * np.mean((LogisticRegression(C=1e4, max_iter=5000, fit_intercept=False).fit(M, T).decision_function(M) > 0) != (T == 1))
    print(f"{name:8} | {np.median(nr):5.1f} / {np.quantile(nr, .99):5.1f} / {nr.max():6.1f}      | {ridge_mis(A, y):5.1f}% -> {ridge_mis(S, y):5.1f}%   | "
          f"{lg(A):5.1f}% -> {lg(S):5.1f}%       | {lev(A):5.1f}% -> {lev(S):5.1f}%", flush=True)
