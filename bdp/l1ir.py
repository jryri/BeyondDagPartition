"""Replace his L-inf projection (U/L envelopes) by an L1 isotonic regression on the DAG.

Background (RESULTS §11): U = ancestor-max, L = descendant-min, (U+L)/2 = optimal L-inf isotonic regression (IR).
L-inf minimises the WORST deviation, so whole cones get overwritten. L1 IR minimises the TOTAL deviation.

L1 IR via threshold decomposition (quantised to Q levels):
  for each threshold t_k (quantiles of y): S_k = successor-closed set minimising #{v in S: y_v < t_k} + #{v not in S: y_v >= t_k}
  -> min s-t cut: s->v cap 1 if y_v >= t_k, v->t cap 1 otherwise, u->v INF for every DAG edge u->v
     (u high => v high).  Minimal source side; nested by intersection.  g(v) = #levels containing v.
  Each S_k is the minimum-Hamming acyclic repair of the split {y >= t_k}.

Tests (eps=.10; targets from separability.py in best_parts/):
  1. distortion: share of cells whose quantised value changes, L-inf (U+L)/2 vs L1 IR, per spectral coordinate
  2. his search, swapped projection: same 2000 random sparse nonneg combos on Z=[U,L] (his) vs G=L1-IR(X)
     (order = lexsort(score, envelope score) -> topological), exact sweep, refine best
  3. oracle expressibility of G (LP fit, w>=0) vs Z
  4. score-level: oracle free-sign w on [X,Z] (test B of §10), repair by L1 IR of the score instead of Kahn
Usage: python bdp/l1ir.py <src> [eps] [Q]
"""
import sys, os, time
import numpy as np, scipy.sparse as sp
from scipy.sparse.csgraph import maximum_flow, breadth_first_order
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from refine import HG, load, his_embed, sweep_part, refine, kahn
from separability import lp_fit, SAVE

INF = 10**6


def l1_ir(h, y, Q=32):
    """Quantised L1 isotonic regression of y on the DAG (nondecreasing along edges). Returns integer levels 0..Q-1."""
    n = h.n; s, t = n, n + 1
    ts = np.quantile(y, np.arange(1, Q) / Q)
    prev = np.ones(n, bool); g = np.zeros(n, np.int64)
    base_r, base_c = h.eu, h.ev
    for tk in ts:
        hi = y >= tk
        a1, a0 = np.flatnonzero(hi), np.flatnonzero(~hi)
        rows = np.r_[base_r, np.full(len(a1), s), a0]; cols = np.r_[base_c, a1, np.full(len(a0), t)]
        caps = np.r_[np.full(len(base_r), INF), np.ones(len(a1)), np.ones(len(a0))].astype(np.int32)
        G = sp.csr_matrix((caps, (rows, cols)), shape=(n + 2, n + 2)); G.sum_duplicates()
        res = maximum_flow(G, s, t, method="dinic")
        R = (G - res.flow).tocsr(); R.data[R.data < 0] = 0; R.eliminate_zeros()
        A = breadth_first_order(R, s, directed=True, return_predecessors=False)
        S = np.zeros(n + 2, bool); S[A] = True; S = S[:n] & prev
        g += S; prev = S
    return g


def qlevel(y, Q):
    return np.searchsorted(np.quantile(y, np.arange(1, Q) / Q), y, side="right")


def main(src, eps, Q):
    name = os.path.basename(src.rstrip("/")).replace("_hg.npz", "")
    rng = np.random.default_rng(0); t0 = time.time()
    h = HG(*load(src)); X, Z = his_embed(h, rng)
    T = np.load(os.path.join(SAVE, f"{name}_e{int(eps * 100):02d}.npy")); tc = h.cut(T)
    y_t = np.where(T == 1, 1.0, -1.0)
    out = [f"== {name} n={h.n} eps={eps} target {tc}"]
    # --- 1. L1 IR per coordinate + distortion
    U, L = Z[:, :8], Z[:, 8:]
    Gm = np.zeros((h.n, 8)); ch_inf, ch_1 = [], []
    for j in range(8):
        q = qlevel(X[:, j], Q)
        g = l1_ir(h, X[:, j], Q); Gm[:, j] = g
        # L-inf IR in the same quantised scale: midpoint of envelopes recomputed exactly on levels
        A = q.astype(float).copy(); B = q.astype(float).copy()
        topo = kahn(h, np.zeros(h.n))
        for u in topo:
            sc = h.succ(u)
            if len(sc): A[sc] = np.maximum(A[sc], A[u])
        for u in topo[::-1]:
            sc = h.succ(u)
            if len(sc): B[u] = min(B[u], B[sc].min())
        ch_inf.append(np.mean(np.round((A + B) / 2) != q)); ch_1.append(np.mean(g != q))
        assert np.all(g[h.ev] >= g[h.eu]), "L1 IR not monotone"
    out.append(f"   cells changed per coordinate: L-inf (U+L)/2 mean {100 * np.mean(ch_inf):.0f}% "
               f"[{', '.join(f'{100 * c:.0f}' for c in ch_inf)}] | L1 IR mean {100 * np.mean(ch_1):.0f}% "
               f"[{', '.join(f'{100 * c:.0f}' for c in ch_1)}]   ({time.time() - t0:.0f}s)")
    Gs = (Gm - Gm.min(0)) / np.maximum(np.ptp(Gm, 0), 1e-12)
    # --- 2. his random search, envelope vs L1 IR features
    W = []
    for i in range(2000):
        wv = rng.exponential(size=16) * (rng.random(16) < rng.uniform(0.2, 1))
        if wv.sum() > 0: W.append(wv)
    def search(score_fn):
        best = None
        for wv in W:
            c, p = sweep_part(h, score_fn(wv), eps)
            if best is None or c < best[0]: best = (c, p)
        p = best[1].copy(); r = refine(h, p, eps)[-1][1]
        return best[0], r, float(np.mean(p == T))
    zord = lambda wv: np.argsort(Z @ wv, kind="stable")
    # G has 8 columns: use first 8 weights on G, all 16 on Z as the strictly monotone tie-break
    gord = lambda wv: np.lexsort((Z @ wv, Gs @ wv[:8]))
    for lab, fn in (("his envelope Z", zord), ("L1-IR G (+Z tiebreak)", gord)):
        c, r, ag = search(fn)
        out.append(f"   random search {lab:<22}: sweep {c} ({c / tc:.2f}x T) -> refined {r} ({r / tc:.2f}x T)")
    # --- 3. oracle expressibility of G
    for lab, F in (("Z (envelope)", Z), ("G (L1 IR)", Gs), ("[G,Z]", np.hstack([Gs, Z]))):
        Fs = (F - F.mean(0)) / (F.std(0) + 1e-12)
        w, b, slack, mis = lp_fit(Fs, y_t, True)
        out.append(f"   oracle LP w>=0 on {lab:<13}: misclassified {100 * mis / h.n:.2f}%")
    # --- 4. score-level: oracle free-sign score on [X,Z], repair by Kahn vs by L1 IR
    F = np.hstack([X, Z]); Fs = (F - F.mean(0)) / (F.std(0) + 1e-12)
    w, b, slack, mis = lp_fit(Fs, y_t, False); s_ = Fs @ w
    for lab, order in (("Kahn", kahn(h, s_)),
                       ("L1 IR", None)):
        if order is None:
            g = l1_ir(h, s_, 64); order = np.lexsort((kahn_rank(h, s_), g))
        c, p = sweep_part(h, order, eps); assert h.closed(p)
        r = refine(h, p.copy(), eps)[-1][1]
        out.append(f"   oracle free score + {lab:<6} repair: sweep {c} ({c / tc:.2f}x T), agrees {100 * np.mean(p == T):.1f}% -> refined {r} ({r / tc:.2f}x T)")
    out.append(f"   [{time.time() - t0:.0f}s]")
    print("\n".join(out), flush=True)


def kahn_rank(h, x):
    o = kahn(h, x); r = np.empty(h.n, np.int64); r[o] = np.arange(h.n); return r


if __name__ == "__main__":
    a = sys.argv
    main(a[1], float(a[2]) if len(a) > 2 else 0.10, int(a[3]) if len(a) > 3 else 32)
