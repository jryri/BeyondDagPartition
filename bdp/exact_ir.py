"""EXACT L1 isotonic regression on a DAG (no quantisation parameter Q).

min sum_v |f_v - y_v|  s.t.  f_u <= f_v for every DAG edge u -> v.

Divide and conquer on the value range (Hochbaum-style threshold decomposition):
  every node carries an interval [lo, hi] of candidate values (indices into the sorted distinct y).
  One round: each active node gets the threshold t = vals[mid] of its interval;
    S = minimal successor-closed set (edges INSIDE the same interval only) minimising
        #{v in S : y_v < t} + #{v not in S : y_v >= t}    -> one min s-t cut;
    v in S -> [mid, hi], else -> [lo, mid-1].
  Edges between different intervals are already satisfied (the earlier cut put u's interval below v's),
  so all subproblems of one recursion depth are disjoint and are solved together in ONE max-flow.
Rounds = ceil(log2(#distinct values)) <= ceil(log2 n)  (55k cells: 16; 1M: 20), vs Q-1 = 63 for l1ir.l1_ir(Q=64).
Returns the exact optimal f (values taken from y).
"""
import numpy as np, scipy.sparse as sp
from scipy.sparse.csgraph import maximum_flow, breadth_first_order

INF = 10**6


def l1_ir_exact(h, y, return_rounds=False):
    n = h.n; s, t = n, n + 1
    vals, idx = np.unique(y, return_inverse=True)
    lo = np.zeros(n, np.int64); hi = np.full(n, len(vals) - 1, np.int64); rounds = 0
    while True:
        act = lo < hi
        if not act.any(): break
        rounds += 1
        mid = (lo + hi + 1) // 2; high = act & (idx >= mid); low = act & (idx < mid)
        keep = act[h.eu] & (lo[h.eu] == lo[h.ev]) & (hi[h.eu] == hi[h.ev])
        eu, ev = h.eu[keep], h.ev[keep]; a1, a0 = np.flatnonzero(high), np.flatnonzero(low)
        rows = np.r_[eu, np.full(len(a1), s), a0]; cols = np.r_[ev, a1, np.full(len(a0), t)]
        caps = np.r_[np.full(len(eu), INF), np.ones(len(a1)), np.ones(len(a0))].astype(np.int32)
        G = sp.csr_matrix((caps, (rows, cols)), shape=(n + 2, n + 2)); G.sum_duplicates()
        res = maximum_flow(G, s, t, method="dinic")
        R = (G - res.flow).tocsr(); R.data[R.data < 0] = 0; R.eliminate_zeros()
        A = breadth_first_order(R, s, directed=True, return_predecessors=False)
        S = np.zeros(n + 2, bool); S[A] = True; S = S[:n]
        up = act & S; dn = act & ~S
        lo[up] = mid[up]; hi[dn] = mid[dn] - 1
    f = vals[lo]
    return (f, rounds) if return_rounds else f


if __name__ == "__main__":
    # correctness check against an LP on random DAGs
    from types import SimpleNamespace
    from scipy.optimize import linprog
    rng = np.random.default_rng(1); worst = 0.0
    for trial in range(30):
        n = int(rng.integers(20, 80)); m = int(rng.integers(n, 4 * n))
        u = rng.integers(0, n, m); v = rng.integers(0, n, m); k = u < v; u, v = u[k], v[k]   # u<v -> acyclic
        perm = rng.permutation(n); u, v = perm[u], perm[v]
        y = np.round(rng.normal(size=n), 1) if trial % 2 else rng.normal(size=n)              # with / without ties
        h = SimpleNamespace(n=n, eu=u, ev=v)
        f = l1_ir_exact(h, y)
        assert np.all(f[u] <= f[v] + 1e-12), "not monotone"
        # LP: vars [f (n), p (n), q (n)], f - y = p - q, min sum p+q, f_u - f_v <= 0
        c = np.r_[np.zeros(n), np.ones(2 * n)]
        Aeq = sp.hstack([sp.eye(n), -sp.eye(n), sp.eye(n)]); beq = y
        Aub = sp.csr_matrix((np.r_[np.ones(len(u)), -np.ones(len(u))], (np.r_[np.arange(len(u)), np.arange(len(u))], np.r_[u, v])), shape=(len(u), 3 * n))
        r = linprog(c, A_ub=Aub, b_ub=np.zeros(len(u)), A_eq=Aeq, b_eq=beq, bounds=[(None, None)] * n + [(0, None)] * 2 * n, method="highs")
        ours = np.abs(f - y).sum(); worst = max(worst, ours - r.fun)
        assert ours <= r.fun + 1e-6, f"trial {trial}: ours {ours} > LP {r.fun}"
    print(f"30 random DAGs: exact L1-IR matches the LP optimum (max excess {worst:.2e})")


def l1_ir_fast(h, y):
    """Same result as l1_ir_exact, smaller flow networks (RESULTS §32).

    In one round a flow path is  s -> high node -> ... -> low node -> t.  Only the CONFLICT CORE can carry flow:
      high nodes with a low descendant, low nodes with a high ancestor (inside the same interval).
    Outside the core the minimal min cut is known without a flow: a high node with no low descendant is on the
    source side (its unit arc can never be saturated), a low node with no high ancestor is on the sink side
    (nothing reaches it in the residual graph).  Two BFS passes find the core; the max-flow runs on the core only.
    """
    n = h.n
    vals, idx = np.unique(y, return_inverse=True)
    lo = np.zeros(n, np.int64); hi = np.full(n, len(vals) - 1, np.int64)
    while True:
        act = lo < hi
        if not act.any(): break
        mid = (lo + hi + 1) // 2; high = act & (idx >= mid); low = act & (idx < mid)
        keep = act[h.eu] & (lo[h.eu] == lo[h.ev]) & (hi[h.eu] == hi[h.ev])
        eu, ev = h.eu[keep], h.ev[keep]; a1, a0 = np.flatnonzero(high), np.flatnonzero(low)
        one = np.ones(len(eu) + max(len(a1), len(a0)), np.int8)
        F = sp.csr_matrix((one[:len(eu) + len(a1)], (np.r_[eu, np.full(len(a1), n)], np.r_[ev, a1])), shape=(n + 1, n + 1))
        B = sp.csr_matrix((one[:len(eu) + len(a0)], (np.r_[ev, np.full(len(a0), n)], np.r_[eu, a0])), shape=(n + 1, n + 1))
        fr = np.zeros(n + 1, bool); fr[breadth_first_order(F, n, return_predecessors=False)] = True
        br = np.zeros(n + 1, bool); br[breadth_first_order(B, n, return_predecessors=False)] = True
        core = fr[:n] & br[:n]
        S = high & ~core                                   # non-core: high -> source side, low -> sink side
        k = int(core.sum())
        if k:
            cid = np.full(n, -1, np.int64); cn = np.flatnonzero(core); cid[cn] = np.arange(k); s, t = k, k + 1
            ke = core[eu] & core[ev]; cu, cv = cid[eu[ke]], cid[ev[ke]]
            c1, c0 = cid[cn[high[cn]]], cid[cn[low[cn]]]
            rows = np.r_[cu, np.full(len(c1), s), c0]; cols = np.r_[cv, c1, np.full(len(c0), t)]
            caps = np.r_[np.full(len(cu), INF), np.ones(len(c1)), np.ones(len(c0))].astype(np.int32)
            G = sp.csr_matrix((caps, (rows, cols)), shape=(k + 2, k + 2)); G.sum_duplicates()
            res = maximum_flow(G, s, t, method="dinic")
            R = (G - res.flow).tocsr(); R.data[R.data < 0] = 0; R.eliminate_zeros()
            A = breadth_first_order(R, s, directed=True, return_predecessors=False)
            Sc = np.zeros(k + 2, bool); Sc[A] = True; S[cn[Sc[:k]]] = True
        up = act & S; dn = act & ~S
        lo[up] = mid[up]; hi[dn] = mid[dn] - 1
    return vals[lo]
