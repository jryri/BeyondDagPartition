"""Profile one J2 evaluation (RESULTS §32): where does the time go?  Usage: python bdp/prof.py <src> [n_theta]"""
import sys, os, time, warnings
import numpy as np, scipy.sparse as sp
warnings.filterwarnings("ignore")
from scipy.sparse.csgraph import maximum_flow, breadth_first_order
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from refine import HG, load, sweep_part, kahn, fm, fm_pass
from det import eig
import exact_ir

def main(src, K):
    name = os.path.basename(src).replace("_hg.npz", ""); h = HG(*load(src)); eps = 0.10
    vals, U, dv, inv, w = eig(h, 32)
    X = U[:, :16] * inv[:, None]; X = (X - X.mean(0)) / (X.std(0) + 1e-12); d = 16
    T = dict(score=0., ir_build=0., ir_flow=0., ir_resid=0., ir_other=0., kahn=0., sweep=0., fm=0.); rounds = 0; passes = 0
    # instrumented copy of l1_ir_exact
    def ir(y):
        nonlocal rounds
        n = h.n; s, t = n, n + 1; t0 = time.time()
        vals, idx = np.unique(y, return_inverse=True)
        lo = np.zeros(n, np.int64); hi = np.full(n, len(vals) - 1, np.int64); T["ir_other"] += time.time() - t0
        while True:
            act = lo < hi
            if not act.any(): break
            rounds += 1; t0 = time.time()
            mid = (lo + hi + 1) // 2; high = act & (idx >= mid); low = act & (idx < mid)
            keep = act[h.eu] & (lo[h.eu] == lo[h.ev]) & (hi[h.eu] == hi[h.ev])
            eu, ev = h.eu[keep], h.ev[keep]; a1, a0 = np.flatnonzero(high), np.flatnonzero(low)
            rows = np.r_[eu, np.full(len(a1), s), a0]; cols = np.r_[ev, a1, np.full(len(a0), t)]
            caps = np.r_[np.full(len(eu), exact_ir.INF), np.ones(len(a1)), np.ones(len(a0))].astype(np.int32)
            G = sp.csr_matrix((caps, (rows, cols)), shape=(n + 2, n + 2)); G.sum_duplicates(); T["ir_build"] += time.time() - t0
            t0 = time.time(); res = maximum_flow(G, s, t, method="dinic"); T["ir_flow"] += time.time() - t0
            t0 = time.time(); R = (G - res.flow).tocsr(); R.data[R.data < 0] = 0; R.eliminate_zeros()
            A = breadth_first_order(R, s, directed=True, return_predecessors=False)
            S = np.zeros(n + 2, bool); S[A] = True; S = S[:n]
            up = act & S; dn = act & ~S; lo[up] = mid[up]; hi[dn] = mid[dn] - 1; T["ir_resid"] += time.time() - t0
        return vals[lo]
    rng = np.random.default_rng(0); t00 = time.time()
    for k in range(K):
        th = np.zeros(2 * d); th[0] = -2; th[d] = 2; th += rng.normal(size=2 * d) * (0 if k == 0 else 1.0)
        t0 = time.time(); a, b = th[:d], th[d:]; s_ = np.sqrt(((X - a) ** 2).sum(1)) - np.sqrt(((X - b) ** 2).sum(1)); T["score"] += time.time() - t0
        for sg in (1, -1):
            f = ir(sg * s_)
            t0 = time.time(); kr = np.empty(h.n, np.int64); kr[kahn(h, sg * s_)] = np.arange(h.n); T["kahn"] += time.time() - t0
            t0 = time.time(); c, p = sweep_part(h, np.lexsort((kr, f)), eps); T["sweep"] += time.time() - t0
            t0 = time.time(); p = p.copy(); cut = h.cut(p)
            for _ in range(10):
                nc = fm_pass(h, p, eps); passes += 1
                if nc >= cut: break
                cut = nc
            T["fm"] += time.time() - t0
    tot = time.time() - t00
    print(f"== {name} n={h.n} pins={len(h.pins)} dag edges={len(h.eu)} | {K} thetas, {1000 * tot / K:.0f} ms/eval | "
          f"{rounds / (2 * K):.1f} flow rounds/projection, {passes / (2 * K):.1f} FM passes/orientation")
    for k, v in sorted(T.items(), key=lambda kv: -kv[1]): print(f"   {k:9} {1000 * v / K:8.0f} ms/eval  {100 * v / tot:5.1f}%")

if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 6)
