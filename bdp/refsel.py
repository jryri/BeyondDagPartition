"""RESULTS §29: does the optimiser's objective predict the final (refined) cut?  And does a refine-aware proxy?

§28: homotopy and compass reach the same sweep J on biriscv (670 vs 667) but refine to 309 vs 238.
Question: is J a usable objective near the optimum, and does adding a cheap refinement step fix it?
Pool: the exact §19 pool (same seed / candidate stream / 60 top-by-proxy + 60 random), 120 two-seed potentials.
Per candidate and per orientation (both kept, proxy-independent):
  J0  exact L1 projection -> exact acyclic sweep            (what §25-§28 optimise)        2 x ~log2 n max-flows
  J1  J0's partition + ONE acyclic FM pass                   (+ O(pins log n))
  J2  J0's partition + FM to convergence (no flow)           (+ passes x O(pins log n))
  truth = full acyclic refine (FM + flow, 3 rounds) of J0's partition; min over the two orientations.
  Each proxy = min over the two orientations.
Reported: Spearman over the pool and over the better half (by truth: the region an optimiser works in),
          regret@1 and regret@5 (best truth among the proxy's top-k / best truth in the pool), ms per candidate.
Usage: python bdp/refsel.py <src> [eps]
"""
import sys, os, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from refine import HG, load, bounds, sweep_part, refine, kahn, fm_pass, fm
from dims import spectral
from exact_ir import l1_ir_exact
from separability import SAVE

std = lambda A: (A - A.mean(0)) / (A.std(0) + 1e-12)
GAPS = [0.5, 2.0, 8.0, 32.0, 1e4]
HERE = os.path.dirname(os.path.abspath(__file__))


def main(src, eps):
    name = os.path.basename(src.rstrip("/")).replace("_hg.npz", "")
    rng = np.random.default_rng(0); h = HG(*load(src)); lo, hi = bounds(h.n, eps)
    tc = h.cut(np.load(os.path.join(SAVE, f"{name}_e{int(eps * 100):02d}.npy")))
    t0 = time.time(); X = std(spectral(h, 32, rng))[:, :16]

    def score(c, u, g):                                      # identical to candsel.py
        a, b = X[c] - g * u / 2, X[c] + g * u / 2
        return np.sqrt(((X - a) ** 2).sum(1)) - np.sqrt(((X - b) ** 2).sum(1))

    cands = []
    for _ in range(2000):
        u = rng.normal(size=16); cands.append((int(rng.integers(h.n)), u / np.linalg.norm(u), GAPS[int(rng.integers(len(GAPS)))]))
    pa = np.array([sweep_part(h, np.argsort(score(*cd), kind="stable"), eps)[0] for cd in cands])
    o = np.argsort(pa, kind="stable"); pool = list(o[:60]) + list(rng.choice(o[60:], 60, replace=False))
    print(f"== {name} n={h.n} | T {tc} | pool regenerated [{time.time() - t0:.0f}s]", flush=True)

    keys = ["J0", "J1", "J2"]; V = {k: [] for k in keys}; tm = {k: 0.0 for k in keys + ["truth"]}; truth = []
    for i in pool:
        s = score(*cands[i]); vals = {k: [] for k in keys}; tr = []
        for sg in (1, -1):
            t = time.time(); f = l1_ir_exact(h, sg * s); kr = np.empty(h.n, np.int64); kr[kahn(h, sg * s)] = np.arange(h.n)
            c, p = sweep_part(h, np.lexsort((kr, f)), eps); tm["J0"] += time.time() - t; vals["J0"].append(c)
            q = p.copy(); t = time.time(); fm_pass(h, q, eps); vals["J1"].append(h.cut(q)); tm["J1"] += time.time() - t
            q = p.copy(); t = time.time(); vals["J2"].append(fm(h, q, eps)); tm["J2"] += time.time() - t
            q = p.copy(); t = time.time(); tr.append(refine(h, q, eps)[-1][1]); tm["truth"] += time.time() - t
            assert h.closed(q) and lo <= np.sum(q == 0) <= hi
        for k in keys: V[k].append(min(vals[k]))
        truth.append(min(tr))
    truth = np.array(truth); best = truth.min(); good = truth <= np.median(truth)
    np.savez(os.path.join(HERE, f"refsel_{name}.npz"), truth=truth, **{k: np.array(v) for k, v in V.items()})
    print(f"   pool 120: best refined {best} ({best / tc:.2f}x T), median {int(np.median(truth))} | "
          f"refine {1000 * tm['truth'] / len(pool):.0f} ms/cand", flush=True)
    for k in keys:
        v = np.array(V[k], float); r = np.argsort(v, kind="stable")
        print(f"   {k}: spearman all {spearmanr(v, truth)[0]:+.2f}, better half {spearmanr(v[good], truth[good])[0]:+.2f} | "
              f"regret@1 {truth[r[0]] / best:.2f}  @5 {truth[r[:5]].min() / best:.2f} | "
              f"{1000 * tm[k] / len(pool):.0f} ms/cand (incremental)", flush=True)
    print(f"   [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 0.10)
