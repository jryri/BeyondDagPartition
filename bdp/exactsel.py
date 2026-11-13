"""Quest (RESULTS §20): can the EXACT L1 projection (exact_ir.py, no Q) replace the 64-level one, and is
"rank by the exact projected sweep cut" a parameter-free selection rule?

Re-creates the identical §19 pool (same seed / family / pool rule) and reuses its saved truth and predictors
(bdp/candsel_<design>.npz: truth = refined cut after the 64-level pipeline; f = 64-level projected sweep cut).
Per pool candidate, both orientations: exact projection -> lexsort(f, Kahn rank) -> exact acyclic sweep.
Reported:
  time      exact projection vs l1_ir(Q=64), per projection, and #rounds (max-flows) of the exact solver
  sweep     exact projected sweep cut vs 64-level projected sweep cut (same candidate, better orientation)
  select    Spearman(exact sweep, truth); top-5 by exact sweep -> refine their EXACT partitions -> best,
            vs top-5 by the 64-level sweep (saved truth) and vs the pool's best truth
Usage: python bdp/exactsel.py <src> [eps]
"""
import sys, os, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from refine import HG, load, bounds, sweep_part, refine, kahn
from dims import spectral
from l1ir import l1_ir
from exact_ir import l1_ir_exact
from separability import SAVE

std = lambda A: (A - A.mean(0)) / (A.std(0) + 1e-12)
GAPS = [0.5, 2.0, 8.0, 32.0, 1e4]
HERE = os.path.dirname(os.path.abspath(__file__))


def main(src, eps):
    name = os.path.basename(src.rstrip("/")).replace("_hg.npz", "")
    rng = np.random.default_rng(0); h = HG(*load(src)); lo, hi = bounds(h.n, eps)
    T = np.load(os.path.join(SAVE, f"{name}_e{int(eps * 100):02d}.npy")); tc = h.cut(T)
    X = std(spectral(h, 32, rng))[:, :16]

    def score(c, u, g):
        a, b = X[c] - g * u / 2, X[c] + g * u / 2
        return np.sqrt(((X - a) ** 2).sum(1)) - np.sqrt(((X - b) ** 2).sum(1))

    cands = []
    for _ in range(2000):
        u = rng.normal(size=16); cands.append((int(rng.integers(h.n)), u / np.linalg.norm(u), GAPS[int(rng.integers(len(GAPS)))]))
    pa = np.array([sweep_part(h, np.argsort(score(*cd), kind="stable"), eps)[0] for cd in cands])
    o = np.argsort(pa, kind="stable"); pool = list(o[:60]) + list(rng.choice(o[60:], 60, replace=False))
    sv = np.load(os.path.join(HERE, f"candsel_{name}.npz")); truth, f64 = sv["truth"], sv["f"]
    assert np.allclose(sv["a"], pa[pool]), "pool differs from §19"
    print(f"== {name} n={h.n} target {tc}", flush=True)

    ex, parts, te, rnd = [], [], 0.0, []
    for i in pool:
        s = score(*cands[i]); best = None
        for sg in (1, -1):
            t = time.time(); fv, r = l1_ir_exact(h, sg * s, True); te += time.time() - t; rnd.append(r)
            kr = np.empty(h.n, np.int64); kr[kahn(h, sg * s)] = np.arange(h.n)
            c, p = sweep_part(h, np.lexsort((kr, fv)), eps); assert h.closed(p) and lo <= np.sum(p == 0) <= hi
            if best is None or c < best[0]: best = (c, p)
        ex.append(best[0]); parts.append(best[1])
    ex = np.array(ex)
    t = time.time(); [l1_ir(h, score(*cands[i]), 64) for i in pool[:10]]; t64 = (time.time() - t) / 10
    tex = te / (2 * len(pool))
    print(f"   time per projection: exact {tex:.2f}s ({np.mean(rnd):.1f} rounds) vs Q=64 {t64:.2f}s (63 rounds) -> {t64 / tex:.1f}x", flush=True)
    r = ex / f64
    print(f"   projected sweep, exact / Q64 (same candidate): median {np.median(r):.2f}, lower on {np.mean(ex < f64) * 100:.0f}%, "
          f"higher on {np.mean(ex > f64) * 100:.0f}%", flush=True)
    top_ex = np.argsort(ex, kind="stable")[:5]; top_64 = np.argsort(f64, kind="stable")[:5]
    ref = []
    for j in top_ex:
        p = parts[j].copy(); ref.append(refine(h, p, eps)[-1][1]); assert h.closed(p) and lo <= np.sum(p == 0) <= hi
    pb = truth.min()
    print(f"   select: spearman exact {spearmanr(ex, truth)[0]:+.2f} vs Q64 {spearmanr(f64, truth)[0]:+.2f} | "
          f"top-5 exact -> refined {ref} best {min(ref)} ({min(ref) / tc:.2f}xT) | top-5 Q64 best {truth[top_64].min()} "
          f"({truth[top_64].min() / tc:.2f}xT) | pool best {pb} ({pb / tc:.2f}xT)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 0.10)
