"""Quest: on large designs, (1) how big is the price of acyclicity, (2) does any method family have an edge, at what time?
Methods (eps given, unit weights, cut = #nets; every acyclic result asserted closed + balanced):
  ref     Mt-KaHyPar QUALITY, unconstrained, 3 seeds (8 threads)
  K+rep   Mt-KaHyPar seeds -> closure repair (4 variants) -> rebalance -> best 2 refined (acyclic FM + flow)
  dagP-w  dagP (no METIS) on net-weighted DAG, 2 seeds -> refine
  his     his envelope decoder: 8 spectral -> U/L -> 2000 random sparse w>=0 -> exact sweep -> refine best
  L1      spectral score route: 16 eigvecs, 2000 random directions scored by UNCONSTRAINED sweep cut (cheap proxy),
          top 5 -> L1 isotonic projection on the DAG (64 levels) -> exact sweep -> refine best
Usage: python bdp/large.py <src> [eps]
"""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from refine import HG, load, bounds, refine, kahypar_parts, closure_variants, rebalance, his_embed, sweep_part
from dagp_run import run_dagp, write_dot, OUT as DAGP_DIR
from dims import spectral
from l1ir import l1_ir, kahn_rank
import mtkahypar as mk


def main(src, eps):
    name = os.path.basename(src.rstrip("/")).replace("_hg.npz", "")
    rng = np.random.default_rng(0)
    h = HG(*load(src)); mtk = mk.initialize(8); lo, hi = bounds(h.n, eps)
    print(f"== {name} n={h.n} nets={h.M} pins={len(h.pins)} dag_edges={len(h.eu)} eps={eps}", flush=True)
    R = {}

    def ok(p):
        assert h.closed(p) and lo <= int(np.sum(p == 0)) <= hi, "not acyclic/balanced"

    def report(tag, start, final, t):
        R[tag] = final
        print(f"   {tag:<7} start {start} -> final {final} ({final / ref:.2f}x ref)  [{t:.0f}s]", flush=True)

    t = time.time(); kp = kahypar_parts(h, eps, mtk, seeds=3); ref = min(c for c, _ in kp)
    print(f"   ref     unconstrained {ref} (seeds {[c for c, _ in kp]})  [{time.time() - t:.0f}s]", flush=True)

    # K+rep
    t = time.time(); st = []
    for c, prt in kp:
        for v in closure_variants(h, prt):
            if rebalance(h, v, eps): st.append((h.cut(v), v))
    st.sort(key=lambda x: x[0]); best = None
    for c, v in st[:2]:
        p = v.copy(); f = refine(h, p, eps)[-1][1]; ok(p)
        if best is None or f < best[1]: best = (c, f)
    if best: report("K+rep", best[0], best[1], time.time() - t)
    else: print("   K+rep   no balanced repaired start", flush=True)

    # dagP-w
    t = time.time(); path = os.path.join(DAGP_DIR, f"{name}_w.dot")
    if not os.path.exists(path): write_dot(h, path, True)
    best = None
    for s in (1, 2):
        try:
            p = run_dagp(path, eps, s).astype(np.int8)
        except Exception as e:
            print(f"   dagP seed {s} failed: {str(e)[:200]}", flush=True); continue
        if not h.closed(p): p = (1 - p).astype(np.int8)
        if not (h.closed(p) and lo <= np.sum(p == 0) <= hi): continue
        c = h.cut(p); f = refine(h, p, eps)[-1][1]; ok(p)
        if best is None or f < best[1]: best = (c, f)
    if best: report("dagP-w", best[0], best[1], time.time() - t)

    # his envelope decoder
    t = time.time(); X, Z = his_embed(h, rng); te = time.time() - t; bc = None
    for i in range(2000):
        wv = rng.exponential(size=16) * (rng.random(16) < rng.uniform(0.2, 1))
        if wv.sum() == 0: continue
        c, p = sweep_part(h, np.argsort(Z @ wv, kind="stable"), eps)
        if bc is None or c < bc[0]: bc = (c, p)
    p = bc[1].copy(); f = refine(h, p, eps)[-1][1]; ok(p)
    report("his", bc[0], f, time.time() - t); print(f"           (embedding+envelopes {te:.0f}s)", flush=True)

    # L1 route
    t = time.time(); F = spectral(h, 16, rng); Fs = (F - F.mean(0)) / (F.std(0) + 1e-12); cand = []
    W = [np.eye(16)[j] for j in range(16)] + [rng.normal(size=16) for _ in range(2000)]   # single eigvecs + random
    for w in W:
        s = Fs @ w
        c, _ = sweep_part(h, np.argsort(s, kind="stable"), eps)      # unconstrained proxy (ignores direction)
        cand.append((c, w))
    cand.sort(key=lambda x: x[0]); best = None
    for c0, w in cand[:5]:
        s = Fs @ w
        for sgn in (1, -1):          # which end of the score becomes the earlier block
            g = l1_ir(h, sgn * s, 64); order = np.lexsort((kahn_rank(h, sgn * s), g))
            c, p = sweep_part(h, order, eps); ok(p)
            if best is None or c < best[0]: best = (c, p, c0)
    p = best[1].copy(); f = refine(h, p, eps)[-1][1]; ok(p)
    report("L1", best[0], f, time.time() - t); print(f"           (best unconstrained proxy {cand[0][0]})", flush=True)
    print(f"   SUMMARY {name}: ref {ref} | " + " | ".join(f"{k} {v} ({v / ref:.2f}x)" for k, v in R.items()), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 0.10)
