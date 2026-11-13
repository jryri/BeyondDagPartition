"""Quest: "shape-free" separators in the spectral space -- a cut is a level set of a scalar function f,
and the families below let the data decide the shape instead of fixing it (plane / sphere / quadric, see shape.py).

Same protocol as shape.py (b): 2016 candidates per arm, proxy = unconstrained sweep cut, top-5 distinct,
both orientations L1-projected on the DAG (64 levels), better sweep refined, best of 5 reported.
  P1  potential, 1 seed pair     f = |x-a| - |x-b|                 (hyperboloid sheet; far seeds -> plane)
  P3  potential, 3+3 seeds       f = min_i|x-a_i| - min_j|x-b_j|   (union of regions; ARV "distance to a core")
  T3  pseudotime / polyline      f = arc length of the projection of x on the path c1-c2-c3 (principal-curve analogue)
  G   geodesic on the quotient   k-means K=128 cells (density modes), kNN graph of centroids, geodesic distances;
                                 f = geo(cell(x), a) - geo(cell(x), b), ties by |x-a|-|x-b|   (Isomap-style manifold ball)
Seeds are random cells (P, T) or random pairs of k-means cells (G).
Cost per candidate: P1 O(nk), P3 O(6nk), T3 O(2nk), G O(n) after one O(nkK) k-means; then O(n log n + pins) sweep.
Usage: python bdp/abstract.py <src> [eps]
"""
import sys, os, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
from scipy.cluster.vq import kmeans2
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from refine import HG, load, bounds, sweep_part, refine
from dims import spectral
from l1ir import l1_ir, kahn_rank
from separability import SAVE

std = lambda A: (A - A.mean(0)) / (A.std(0) + 1e-12)
NC = 2016


def main(src, eps):
    name = os.path.basename(src.rstrip("/")).replace("_hg.npz", "")
    rng = np.random.default_rng(0); h = HG(*load(src)); lo, hi = bounds(h.n, eps)
    T = np.load(os.path.join(SAVE, f"{name}_e{int(eps * 100):02d}.npy")); tc = h.cut(T)
    t0 = time.time(); X = std(spectral(h, 32, rng))[:, :16]
    dist = lambda c: np.sqrt(((X - X[c]) ** 2).sum(1))

    def pipeline(s):
        g = l1_ir(h, s, 64); order = np.lexsort((kahn_rank(h, s), g)); c, p = sweep_part(h, order, eps)
        assert h.closed(p) and lo <= np.sum(p == 0) <= hi
        return c, p

    def P(m):
        def f(A, B): return np.min([dist(a) for a in A], 0) - np.min([dist(b) for b in B], 0)
        S = [rng.choice(h.n, 2 * m, replace=False) for _ in range(NC)]
        return [(lambda s=s: f(s[:m], s[m:])) for s in S]

    def poly(cs):
        pts = X[cs]; best = np.full(h.n, np.inf); sc = np.zeros(h.n); cum = 0.0; m = len(cs) - 1
        for j in range(m):
            a = pts[j]; v = pts[j + 1] - a; L = np.linalg.norm(v) + 1e-12
            t = ((X - a) @ v) / L ** 2
            t = np.clip(t, -np.inf if j == 0 else 0, np.inf if j == m - 1 else 1)
            d = ((X - a - t[:, None] * v) ** 2).sum(1); k = d < best
            sc[k] = cum + t[k] * L; best[k] = d[k]; cum += L
        return sc

    K = min(128, h.n // 20); cen, lab = kmeans2(X, K, minit="++", seed=0)
    D = np.sqrt(((cen[:, None] - cen[None]) ** 2).sum(2)); nn = np.argsort(D, 1)[:, 1:7]
    r = np.repeat(np.arange(K), 6); G = csr_matrix((D[r, nn.ravel()], (r, nn.ravel())), shape=(K, K))
    geo = shortest_path(G, directed=False); geo[~np.isfinite(geo)] = 2 * geo[np.isfinite(geo)].max()
    rep = np.array([np.argmin(((X - c) ** 2).sum(1)) for c in cen])          # a real cell per centroid (tie-break)
    pairs = [tuple(rng.choice(K, 2, replace=False)) for _ in range(NC)]
    Gf = [(lambda a=a, b=b: geo[lab, a] - geo[lab, b] + 1e-3 * (dist(rep[a]) - dist(rep[b]))) for a, b in pairs]
    print(f"== {name} n={h.n} target {tc} (setup {time.time() - t0:.0f}s, K={K})", flush=True)

    arms = {"P1 potential pair": P(1), "P3 potential 3+3": P(3),
            "T3 polyline": [(lambda c=c: poly(c)) for c in [rng.choice(h.n, 3, replace=False) for _ in range(NC)]],
            "G geodesic cells": Gf}
    for an, fns in arms.items():
        t = time.time(); sc = sorted((sweep_part(h, np.argsort(f(), kind="stable"), eps)[0], i) for i, f in enumerate(fns)); top = []
        for c0, i in sc:
            p0 = sweep_part(h, np.argsort(fns[i](), kind="stable"), eps)[1]
            if all(max(np.mean(p0 == q), np.mean(p0 != q)) < 0.98 for _, _, q in top): top.append((c0, i, p0))
            if len(top) == 5: break
        ts = time.time() - t; res = []
        for c0, i, _ in top:
            s = fns[i](); c, p = min((pipeline(sg * s) for sg in (1, -1)), key=lambda x: x[0])
            rc = refine(h, p, eps)[-1][1]; assert h.closed(p) and lo <= np.sum(p == 0) <= hi
            res.append((c0, c, rc))
        best = min(x[2] for x in res)
        print(f"   find {an:<18}: proxy {[x[0] for x in res]} | projected sweep {[x[1] for x in res]} -> refined "
              f"{[x[2] for x in res]} | best {best} ({best / tc:.2f}xT)  [search {ts:.0f}s, total {time.time() - t:.0f}s]", flush=True)
    print(f"   [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 0.10)
