"""RESULTS §35a: is the best-known cut locally consistent in the DAG-aware embedding G, compared with X?
Same targets (best_parts2) and kNN protocol as §34 (5-fold OOF, k=15), plus the kNN score decoded by the method's
decoder (projection -> sweep -> FM -> refine).  Usage: python bdp/oracle2.py <src> [eps]"""
import sys, os, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from refine import HG, load
from det import eig
from dagembed import dag_embed
from oracle import target, decode

def main(src, eps):
    name = os.path.basename(src).replace("_hg.npz", ""); h = HG(*load(src)); t0 = time.time()
    tc, tsrc, T = target(h, eps, name); y = T.astype(int)
    vals, U, dv, inv, w = eig(h, 32); X = U[:, :16] * inv[:, None]; X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    G = dag_embed(h, X); tg = time.time() - t0
    print(f"== {name} n={h.n} | target {tc} ({tsrc}) | G built [{tg:.0f}s]", flush=True)
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    for lab, F in (("X  (undirected, 16)", X), ("G  (DAG-aware, 16)", G), ("[X,G] (32)", np.hstack([X, G]))):
        pk = cross_val_predict(KNeighborsClassifier(15), F, y, cv=cv, method="predict_proba")[:, 1]
        mis = np.mean((pk > 0.5).astype(int) != y); r, p = decode(h, pk - 0.5, eps)
        print(f"   kNN on {lab:20} misclass {100 * mis:5.1f}% | decoded+refined {r:5} ({r / tc:.2f}x target)", flush=True)

if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 0.10)
