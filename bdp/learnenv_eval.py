"""§37 follow-up: same learned thetas, but the hyperplane is fitted by logistic regression (the ridge fit of the
proposal is a poor separability measure: biriscv X-only 32.6% by ridge vs 2.5% by logistic), and EVERY feature set
is decoded (projection -> sweep -> FM -> refine), including his envelope.  Held-out theta only."""
import sys, os, warnings
import numpy as np
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import learnenv as le
from refine import HG, load
from det import eig
from oracle import decode

inv_sig = lambda p: np.log(p / (1 - p))
TH = {"A": np.array([inv_sig(0.86), np.log(0.77), np.log(2.0824), inv_sig(0.97), np.log(2.11), np.log(0.44)]),
      "B": np.array([inv_sig(0.97), np.log(4.27), np.log(0.0575), inv_sig(0.999), np.log(1.26), np.log(0.6109)])}
his = np.array([20.0, 4.0, np.log(1e-4), 20.0, 4.0, np.log(1e-4)])
print(f"{'design':8} | X only           | his envelope     | learned (held-out theta)   [misclass -> decoded+refined (x target)]", flush=True)
for fold, names in le.FOLDS.items():
    other = "B" if fold == "A" else "A"
    for name in names:
        h = HG(*load(f"data/{name}_hg.npz")); vals, U, dv, inv, w = eig(h, 32); X = U[:, :16] * inv[:, None]
        for j in range(16):
            if np.sum(X[h.ev, j] - X[h.eu, j]) < 0: X[:, j] *= -1
        X = (X - X.mean(0)) / (X.std(0) + 1e-12); T = np.load(os.path.join(le.TGT, f"{name}_e10.npy")); tc = h.cut(T)
        D = dict(X=X, Pf=le.plan(h.n, h.eu, h.ev), Pb=le.plan(h.n, h.ev, h.eu)); out = []
        for F in (X, le.feats(D, his), le.feats(D, TH[other])):
            lr = LogisticRegression(C=1e4, max_iter=5000).fit(F, T); z = lr.decision_function(F)
            r, _ = decode(h, z, 0.10); out.append(f"{100 * np.mean((z > 0) != (T == 1)):5.1f}% -> {r:4} ({r / tc:.2f}x)")
        print(f"{name:8} | " + " | ".join(out), flush=True)
