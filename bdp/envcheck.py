"""Review check: homotopy_ul uses a SOFT envelope (learnenv.prop, beta = 55, mean-normalised log-sum-exp), not his exact
max/min envelope.  How far apart are they, and does the oracle result (§38) hold with the exact one?"""
import sys, os, warnings
import numpy as np
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import learnenv as le
from refine import HG, load
from det import eig
from oracle import decode
std = lambda F: (F - F.mean(0)) / (F.std(0) + 1e-12)

def exact(X, P, delta):                       # his envelope: U_v = max(x_v, max_pred U_u + delta)
    U = X.copy()
    for ss, du, st, cnt in P: U[du] = np.maximum(X[du], np.maximum.reduceat(U[ss] + delta, st, axis=0))
    return U

his = np.array([20.0, 4.0, np.log(1e-4), 20.0, 4.0, np.log(1e-4)])
for name in ["serv", "omsp", "riscv", "picosoc", "aes", "biriscv"]:
    h = HG(*load(f"data/{name}_hg.npz")); vals, U, dv, inv, w = eig(h, 32); X = U[:, :16] * inv[:, None]
    for j in range(16):
        if np.sum(X[h.ev, j] - X[h.eu, j]) < 0: X[:, j] *= -1
    X = std(X); T = np.load(os.path.join(le.TGT, f"{name}_e10.npy")); tc = h.cut(T)
    D = dict(X=X, Pf=le.plan(h.n, h.eu, h.ev), Pb=le.plan(h.n, h.ev, h.eu))
    Fs = le.feats(D, his); Ue = exact(X, D["Pf"], 1e-4); Le = -exact(-X, D["Pb"], 1e-4); Fe = std(np.hstack([X, Ue, Le]))
    corr = np.mean([abs(np.corrcoef(Fs[:, j], Fe[:, j])[0, 1]) for j in range(16, 48)])
    mono_s = np.mean(Fs[h.ev, 16:32] >= Fs[h.eu, 16:32]); mono_e = np.mean(Fe[h.ev, 16:32] >= Fe[h.eu, 16:32])
    out = []
    for F in (Fs, Fe):
        z = LogisticRegression(C=1e4, max_iter=5000).fit(F, T).decision_function(F); r, _ = decode(h, z, 0.10)
        out.append(f"{100 * np.mean((z > 0) != (T == 1)):5.1f}% -> {r:4}")
    print(f"{name:8} | corr(soft, exact) {corr:.4f} | U monotone on edges: soft {100 * mono_s:.1f}%, exact {100 * mono_e:.1f}% | "
          f"oracle soft {out[0]} | oracle exact {out[1]} (target {tc})", flush=True)
