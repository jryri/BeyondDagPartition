"""Weighted L2 isotonic regression on a DAG:  min sum_v d_v (f_v - y_v)^2  s.t. f_u <= f_v for every edge u -> v.

Threshold theorem for separable convex costs (Hochbaum 2001): {v : f*_v > t} is the minimal min cut where node v
"prefers above" with weight |f'_v(t)| = 2 d_v |y_v - t| (source side if y_v > t, sink side if y_v < t), closure
edges u -> v infinite.  Bisection on the value range, all subproblems of one depth in ONE max-flow (as exact_ir).
Capacities are rounded to integers (max 1000 per node per round), so the result is exact up to that rounding and
to the final interval width (range / 2^rounds).
"""
import numpy as np, scipy.sparse as sp
from scipy.sparse.csgraph import maximum_flow, breadth_first_order

INF = 1 << 30


def l2_ir(h, y, d=None, rounds=24):
    n = h.n; s, t = n, n + 1
    d = np.ones(n) if d is None else d
    lo = np.full(n, float(y.min())); hi = np.full(n, float(y.max()))
    for _ in range(rounds):
        mid = (lo + hi) / 2
        w = d * np.abs(y - mid); K = 1000.0 / max(w.max(), 1e-300)
        cap = np.rint(w * K).astype(np.int64)
        up = (y > mid) & (cap > 0); dn = (y < mid) & (cap > 0)
        keep = (lo[h.eu] == lo[h.ev]) & (hi[h.eu] == hi[h.ev])
        eu, ev = h.eu[keep], h.ev[keep]; a1, a0 = np.flatnonzero(up), np.flatnonzero(dn)
        rows = np.r_[eu, np.full(len(a1), s), a0]; cols = np.r_[ev, a1, np.full(len(a0), t)]
        caps = np.r_[np.full(len(eu), INF), cap[a1], cap[a0]].astype(np.int32)
        G = sp.csr_matrix((caps, (rows, cols)), shape=(n + 2, n + 2)); G.sum_duplicates()
        res = maximum_flow(G, s, t, method="dinic")
        R = (G - res.flow).tocsr(); R.data[R.data < 0] = 0; R.eliminate_zeros()
        A = breadth_first_order(R, s, directed=True, return_predecessors=False)
        S = np.zeros(n + 2, bool); S[A] = True; S = S[:n]
        lo = np.where(S, mid, lo); hi = np.where(S, hi, mid)
    return (lo + hi) / 2


if __name__ == "__main__":
    from types import SimpleNamespace

    def pava(y, w):                       # weighted isotonic regression on a chain 0 -> 1 -> ... -> n-1
        blocks = []
        for yi, wi in zip(y, w):
            blocks.append([yi * wi, wi, 1])
            while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
                b = blocks.pop(); blocks[-1][0] += b[0]; blocks[-1][1] += b[1]; blocks[-1][2] += b[2]
        return np.concatenate([[b[0] / b[1]] * b[2] for b in blocks])

    rng = np.random.default_rng(0); worst = 0.0
    for trial in range(20):
        n = int(rng.integers(30, 300)); y = rng.normal(size=n) + np.linspace(0, 1, n); w = rng.uniform(0.5, 3, n)
        perm = rng.permutation(n); inv = np.argsort(perm)          # chain in a scrambled node numbering
        h = SimpleNamespace(n=n, eu=perm[:-1], ev=perm[1:])
        f = l2_ir(h, y[inv], w[inv])[perm]; g = pava(y, w)
        worst = max(worst, np.abs(f - g).max() / (y.max() - y.min()))
    print(f"20 weighted chains: max |L2 IR - PAVA| / range = {worst:.1e}")
