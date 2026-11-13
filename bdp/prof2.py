"""Per-round flow time and conflict-core size in exact L1 IR (how much of each flow network is actually needed)."""
import sys, os, time, warnings
import numpy as np, scipy.sparse as sp
warnings.filterwarnings("ignore")
from scipy.sparse.csgraph import maximum_flow, breadth_first_order
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from refine import HG, load
from det import eig
src = sys.argv[1]; h = HG(*load(src)); n = h.n
vals, U, dv, inv, w = eig(h, 32); X = U[:, :16] * inv[:, None]; X = (X - X.mean(0)) / (X.std(0) + 1e-12)
a = np.zeros(16); b = np.zeros(16); a[0] = -2; b[0] = 2
y = np.sqrt(((X - a) ** 2).sum(1)) - np.sqrt(((X - b) ** 2).sum(1))
vals, idx = np.unique(y, return_inverse=True); lo = np.zeros(n, np.int64); hi = np.full(n, len(vals) - 1, np.int64); s, t = n, n + 1; r = 0
while (lo < hi).any():
    act = lo < hi; r += 1
    mid = (lo + hi + 1) // 2; high = act & (idx >= mid); low = act & (idx < mid)
    keep = act[h.eu] & (lo[h.eu] == lo[h.ev]) & (hi[h.eu] == hi[h.ev]); eu, ev = h.eu[keep], h.ev[keep]
    A = sp.csr_matrix((np.ones(len(eu), np.int8), (eu, ev)), shape=(n + 1, n + 1)).tolil()
    # forward reach from high, backward reach from low (super node n)
    F = sp.csr_matrix((np.ones(len(eu) + high.sum(), np.int8), (np.r_[eu, np.full(high.sum(), n)], np.r_[ev, np.flatnonzero(high)])), shape=(n + 1, n + 1))
    B = sp.csr_matrix((np.ones(len(eu) + low.sum(), np.int8), (np.r_[ev, np.full(low.sum(), n)], np.r_[eu, np.flatnonzero(low)])), shape=(n + 1, n + 1))
    fr = np.zeros(n + 1, bool); fr[breadth_first_order(F, n, return_predecessors=False)] = True
    br = np.zeros(n + 1, bool); br[breadth_first_order(B, n, return_predecessors=False)] = True
    core = fr[:n] & br[:n]; ce = core[eu] & core[ev]
    a1, a0 = np.flatnonzero(high), np.flatnonzero(low)
    G = sp.csr_matrix((np.r_[np.full(len(eu), 10**6), np.ones(len(a1)), np.ones(len(a0))].astype(np.int32), (np.r_[eu, np.full(len(a1), s), a0], np.r_[ev, a1, np.full(len(a0), t)])), shape=(n + 2, n + 2))
    t0 = time.time(); res = maximum_flow(G, s, t, method="dinic"); tf = time.time() - t0
    R = (G - res.flow).tocsr(); R.data[R.data < 0] = 0; R.eliminate_zeros()
    S = np.zeros(n + 2, bool); S[breadth_first_order(R, s, directed=True, return_predecessors=False)] = True; S = S[:n]
    print(f"round {r:2}: kept edges {keep.sum():6} | core nodes {core.sum():6} ({100 * core.mean():4.1f}%) core edges {ce.sum():6} | flow value {res.flow_value:5} | {1000 * tf:6.1f} ms")
    up = act & S; dn = act & ~S; lo[up] = mid[up]; hi[dn] = mid[dn] - 1
