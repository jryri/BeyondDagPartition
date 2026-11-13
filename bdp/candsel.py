"""Two questions on ONE separator family (RESULTS §19).

Family: two-seed potential  f(x) = |x - a| - |x - b|,  a = c - g u/2,  b = c + g u/2
        c = a random cell's point, u = random unit direction, g = seed gap.  g -> inf is the hyperplane u.x.
Q1 (curvature): is there a design-dependent best gap g, or is the plane (g large) always enough?
Q2 (selection): which CHEAP score predicts the final refined acyclic cut of a candidate?
Pool: 2000 candidates (g uniform over GAPS) scored by the current proxy; pool = top 60 by proxy + 60 random others.
Truth per candidate: both orientations L1-projected (64 levels) -> exact acyclic sweep -> better one refined.
Predictors (all "lower is better"; sign-dependent ones take the min over the two orientations):
  a  unconstrained sweep cut (current proxy)                         O(n log n + pins)
  b  a + 1*back nets on the same sweep (his J, exact)                O(n log n + pins + m)
  c  fraction of DAG edges whose score decreases along the edge      O(m)
  e  Kahn repair: topological order closest to s, exact sweep        O((n+m) log n), no flow
  d  coarse L1 projection, 8 levels (7 max-flows) + sweep            7 F
  f  full L1 projection, 64 levels (63 max-flows) + sweep            63 F   (= current pipeline before refine)
Reported: Spearman rank correlation with truth; regret@5 = best truth among the predictor's top-5 / best truth in pool.
Usage: python bdp/candsel.py <src> [eps]
"""
import sys, os, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from refine import HG, load, bounds, sweep_part, refine, kahn
from dims import spectral
from l1ir import l1_ir
from separability import SAVE
from search import Sweeper

std = lambda A: (A - A.mean(0)) / (A.std(0) + 1e-12)
GAPS = [0.5, 2.0, 8.0, 32.0, 1e4]


def main(src, eps):
    name = os.path.basename(src.rstrip("/")).replace("_hg.npz", "")
    rng = np.random.default_rng(0); h = HG(*load(src)); sw = Sweeper(h, eps); lo, hi = bounds(h.n, eps)
    T = np.load(os.path.join(SAVE, f"{name}_e{int(eps * 100):02d}.npy")); tc = h.cut(T)
    t0 = time.time(); X = std(spectral(h, 32, rng))[:, :16]

    def score(c, u, g):
        a, b = X[c] - g * u / 2, X[c] + g * u / 2
        return np.sqrt(((X - a) ** 2).sum(1)) - np.sqrt(((X - b) ** 2).sum(1))

    cands = []
    for _ in range(2000):
        u = rng.normal(size=16); cands.append((int(rng.integers(h.n)), u / np.linalg.norm(u), GAPS[int(rng.integers(len(GAPS)))]))
    pa = np.array([sweep_part(h, np.argsort(score(*cd), kind="stable"), eps)[0] for cd in cands])
    o = np.argsort(pa, kind="stable"); pool = list(o[:60]) + list(rng.choice(o[60:], 60, replace=False))
    print(f"== {name} n={h.n} target {tc} (setup+proxy {time.time() - t0:.0f}s)", flush=True)

    keys = "abcedf"; P = {k: [] for k in keys}; tm = {k: 0.0 for k in keys}; truth = []; gaps = []
    for i in pool:
        s = score(*cands[i]); gaps.append(cands[i][2]); P["a"].append(pa[i])
        t = time.time(); P["b"].append(min(sw.eval(np.argsort(sg * s, kind="stable"), 1.0)[0] for sg in (1, -1))); tm["b"] += time.time() - t
        t = time.time(); dec = np.mean(s[h.eu] > s[h.ev]); inc = np.mean(s[h.eu] < s[h.ev]); P["c"].append(min(dec, inc)); tm["c"] += time.time() - t
        t = time.time(); ko = [kahn(h, sg * s) for sg in (1, -1)]; P["e"].append(min(sweep_part(h, k_, eps)[0] for k_ in ko)); tm["e"] += time.time() - t
        kr = []
        for k_ in ko:
            r = np.empty(h.n, np.int64); r[k_] = np.arange(h.n); kr.append(r)
        t = time.time(); P["d"].append(min(sweep_part(h, np.lexsort((kr[j], l1_ir(h, sg * s, 8))), eps)[0] for j, sg in enumerate((1, -1)))); tm["d"] += time.time() - t
        t = time.time(); full = [sweep_part(h, np.lexsort((kr[j], l1_ir(h, sg * s, 64))), eps) for j, sg in enumerate((1, -1))]
        c, p = min(full, key=lambda x: x[0]); P["f"].append(c); tm["f"] += time.time() - t
        assert h.closed(p) and lo <= np.sum(p == 0) <= hi
        truth.append(refine(h, p, eps)[-1][1])
    truth = np.array(truth); gaps = np.array(gaps); best = truth.min()
    np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"candsel_{name}.npz"), truth=truth, gaps=gaps, **{k: np.array(v) for k, v in P.items()})
    print(f"   pool 120: best refined {best} ({best / tc:.2f}xT), median {int(np.median(truth))}, worst {truth.max()}", flush=True)
    for k in keys:
        v = np.array(P[k], float); top5 = np.argsort(v, kind="stable")[:5]
        print(f"   predictor {k}: spearman {spearmanr(v, truth)[0]:+.2f} | top-5 best {truth[top5].min()} (regret {truth[top5].min() / best:.2f}) "
              f"| {1000 * tm[k] / len(pool):.0f} ms/cand", flush=True)
    print("   gap      : " + " | ".join(f"g={g:g}: n={np.sum(gaps == g)}, best {truth[gaps == g].min()}, median {int(np.median(truth[gaps == g]))}"
                                       for g in GAPS if np.any(gaps == g)), flush=True)
    print(f"   [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 0.10)
