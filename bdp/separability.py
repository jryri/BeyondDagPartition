"""Is his decoder family able to EXPRESS a good acyclic cut, or does it just fail to FIND it?

Decoder family (his order mode): score = Z w, w >= 0, Z = [U, L] envelopes (16 dims, each strictly increasing
along DAG edges) -> sort -> one threshold (exact sweep).  Any such cut is acyclic.

Target T = best-known acyclic balanced partition (portfolio: his decoder, KaHyPar+repair, dagP; all refined).
Test A (his family): hinge-loss LP  min sum xi  s.t. y_i (z_i.w - b) >= 1 - xi_i,  w >= 0.
        zero slack  <=> T is exactly a threshold of some w>=0 combination.
        Then order by Z w -> exact sweep -> cut, and -> acyclic refine -> cut.
Test B (linear upper bound): same LP on [X, Z] with free sign w; order is Kahn(score) to stay acyclic.
Baseline: his random search (2000 nonneg combos + sweep), raw and refined.

Reading: if A's sweep cut is close to T, the representation is fine and the search (CMA / random) is the
bottleneck. If A is far from T, a single nonneg hyperplane over these features cannot express good cuts.
Usage: python bdp/separability.py <src> [eps]
"""
import sys, os, time
import numpy as np, scipy.sparse as sp
from scipy.optimize import linprog
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from refine import (HG, load, bounds, refine, kahypar_parts, closure_variants, rebalance, his_embed, sweep_part, kahn)
from dagp_run import run_dagp, OUT as DAGP_DIR
import mtkahypar as mk

SAVE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_parts")


def lp_fit(F, y, nonneg):
    """L1-SVM: min sum xi  s.t. y_i (F_i w - b) >= 1 - xi_i. Variables [w (d), b, xi (n)]."""
    n, d = F.shape
    A = sp.hstack([sp.csr_matrix(-y[:, None] * F), sp.csr_matrix(y[:, None].astype(float)), -sp.eye(n)]).tocsr()
    c = np.r_[np.zeros(d + 1), np.ones(n)]
    bnd = [(0, None) if nonneg else (None, None)] * d + [(None, None)] + [(0, None)] * n
    r = linprog(c, A_ub=A, b_ub=-np.ones(n), bounds=bnd, method="highs")
    w, b, xi = r.x[:d], r.x[d], r.x[d + 1:]
    mis = int(np.sum(y * (F @ w - b) <= 0))
    return w, b, float(xi.sum()), mis


def best_known(h, eps, Z, mtk, name, rng):
    cands = []
    # his decoder
    best = None
    for i in range(2000):
        wv = rng.exponential(size=16) * (rng.random(16) < rng.uniform(0.2, 1))
        if wv.sum() == 0: continue
        c, p = sweep_part(h, np.argsort(Z @ wv, kind="stable"), eps)
        if best is None or c < best[0]: best = (c, p)
    his_raw = best[0]; p = best[1].copy(); his_ref = refine(h, p, eps)[-1][1]; cands.append((his_ref, "his", p))
    # KaHyPar + repair
    kp = kahypar_parts(h, eps, mtk, seeds=3); ref = min(c for c, _ in kp)
    st = []
    for c, prt in kp:
        for v in closure_variants(h, prt):
            if rebalance(h, v, eps): st.append((h.cut(v), v))
    st.sort(key=lambda x: x[0])
    for c, v in st[:3]:
        p = v.copy(); cands.append((refine(h, p, eps)[-1][1], "K+repair", p))
    # dagP
    lo, hi = bounds(h.n, eps)
    for tag in ("w", "u"):
        path = os.path.join(DAGP_DIR, f"{name}_{tag}.dot")
        if not os.path.exists(path): continue
        for s in (1, 2, 3):
            p = run_dagp(path, eps, s).astype(np.int8)
            if not h.closed(p): p = (1 - p).astype(np.int8)
            if h.closed(p) and lo <= np.sum(p == 0) <= hi:
                cands.append((refine(h, p, eps)[-1][1], f"dagP-{tag}", p))
    cands.sort(key=lambda x: x[0])
    return ref, his_raw, his_ref, cands[0]


def main(src, eps):
    name = os.path.basename(src.rstrip("/")).replace("_hg.npz", "")
    rng = np.random.default_rng(0); t0 = time.time()
    h = HG(*load(src)); mtk = mk.initialize(8)
    X, Z = his_embed(h, rng)
    ref, his_raw, his_ref, (tc, tsrc, T) = best_known(h, eps, Z, mtk, name, rng)
    assert h.closed(T)
    os.makedirs(SAVE, exist_ok=True); np.save(os.path.join(SAVE, f"{name}_e{int(eps * 100):02d}.npy"), T)
    y = np.where(T == 1, 1.0, -1.0)
    out = [f"== {name} n={h.n} eps={eps}: unconstrained ref {ref} | target {tc} ({tc / ref:.2f}x, {tsrc}) "
           f"| his random search raw {his_raw} ({his_raw / ref:.2f}x) -> refined {his_ref} ({his_ref / ref:.2f}x)"]
    for label, F, nonneg in (("A  Z, w>=0 (his family)", Z, True), ("B  [X,Z], free w", np.hstack([X, Z]), False)):
        Fs = (F - F.mean(0)) / (F.std(0) + 1e-12)   # positive rescaling keeps monotonicity for w>=0
        w, b, slack, mis = lp_fit(Fs, y, nonneg)
        score = Fs @ w
        order = np.argsort(score, kind="stable") if nonneg else kahn(h, score)
        c, p = sweep_part(h, order, eps); assert h.closed(p)
        agree = float(np.mean(p == T))
        pr = p.copy(); cr = refine(h, pr, eps)[-1][1]
        out.append(f"   {label}: LP slack {slack:.0f}, misclassified {mis} ({100 * mis / h.n:.2f}%), nnz(w) {int(np.sum(np.abs(w) > 1e-9))} "
                   f"| sweep cut {c} ({c / ref:.2f}x), agrees with target {100 * agree:.1f}% | +refine {cr} ({cr / ref:.2f}x)")
    out.append(f"   [{time.time() - t0:.0f}s]")
    print("\n".join(out), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 0.10)
