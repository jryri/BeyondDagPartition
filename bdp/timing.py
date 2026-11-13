"""Wall-clock of each pipeline component (CPU, this Windows PC, Python). Usage: python bdp/timing.py <src> [eps]"""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from refine import HG, load, sweep_part, refine, kahn, kahypar_parts, closure_variants, rebalance
from dims import spectral
from l1ir import l1_ir, kahn_rank
import mtkahypar as mk


def tm(f):
    t = time.perf_counter(); r = f(); return r, time.perf_counter() - t


def main(src, eps):
    name = os.path.basename(src.rstrip("/")).replace("_hg.npz", ""); rng = np.random.default_rng(0)
    (h, tl) = tm(lambda: HG(*load(src)))
    out = [f"== {name} n={h.n} nets={h.M} pins={len(h.pins)}  load {tl:.1f}s"]
    F, t = tm(lambda: spectral(h, 16, rng)); out.append(f"   eigsh 16 vectors            {t:7.1f}s")
    s = F @ rng.normal(size=16)
    _, t = tm(lambda: sweep_part(h, np.argsort(s), eps)); out.append(f"   one sweep eval (argsort)    {t * 1000:7.1f}ms")
    o, t = tm(lambda: kahn(h, s)); out.append(f"   one Kahn order (Python)     {t * 1000:7.1f}ms")
    _, t = tm(lambda: l1_ir(h, s, 64)); out.append(f"   one L1-IR repair (64 lvls)  {t:7.1f}s")
    mtk = mk.initialize(8)
    kp, t = tm(lambda: kahypar_parts(h, eps, mtk, seeds=1)); out.append(f"   Mt-KaHyPar 1 run (8 thr)    {t:7.1f}s  cut {kp[0][0]}")
    def rep():
        for v in closure_variants(h, kp[0][1]):
            if rebalance(h, v, eps): return v
    v, t = tm(rep); out.append(f"   closure repair+rebalance    {t:7.1f}s")
    if v is not None:
        hist, t = tm(lambda: refine(h, v, eps)); out.append(f"   acyclic refine (FM+flow)    {t:7.1f}s  {hist[0][1]} -> {hist[-1][1]}")
    print("\n".join(out), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 0.10)
