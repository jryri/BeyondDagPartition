"""Quest: is the direction search the bottleneck of the L1 score route?
Route: 16 spectral eigvecs (standardised) -> score s = F w (free sign) -> [search picks top-5 w] ->
       L1 isotonic projection of s on the DAG (64 levels) -> exact acyclic sweep -> refine ALL 5 -> best.
Arms (same downstream, only the search differs):
  F0  2016 directions (16 single eigvecs + 2000 random), fitness = unconstrained sweep cut (ignores direction)
  F1  CMA-ES on the same unconstrained fitness
  F2  CMA-ES on J = min over the balance window of cut(p) + lam * back(p),
      back(p) = #drivers with a DAG sink placed before them at split p (his J, but on an exact sweep)
Both orientations (+s, -s) are scored; the better one is kept with the direction.
Usage: python bdp/search.py <src> [eps] [lam] [evals]
"""
import sys, os, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
import cma
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from refine import HG, load, bounds, sweep_part, refine
from dims import spectral
from l1ir import l1_ir, kahn_rank
from separability import SAVE


class Sweeper:
    """Exact sweep of cut(p) and back(p) over split positions for an order (vectorised difference arrays)."""
    def __init__(s, h, eps):
        s.h = h; s.n = h.n; s.lo, s.hi = bounds(h.n, eps)
        s.drv = np.unique(h.eu)

    def eval(s, order, lam):
        h, n = s.h, s.n
        pos = np.empty(n, np.int64); pos[order] = np.arange(n)
        q = pos[h.pins]; a = np.full(h.M, n); b = np.zeros(h.M, np.int64)
        np.minimum.at(a, h.netid, q); np.maximum.at(b, h.netid, q)
        cut = np.cumsum(np.bincount(a + 1, minlength=n + 2) - np.bincount(b + 1, minlength=n + 2))
        if lam > 0:
            ms = np.full(n, n); np.minimum.at(ms, h.eu, pos[h.ev])           # earliest DAG sink per driver
            d = s.drv; lo_, hi_ = ms[d] + 1, pos[d]                              # back for p in (ms, pos(d)]
            k = lo_ <= hi_
            back = np.cumsum(np.bincount(lo_[k], minlength=n + 2) - np.bincount(hi_[k] + 1, minlength=n + 2))
            J = cut + lam * back
        else:
            J = cut
        w = J[s.lo:s.hi + 1]; p = s.lo + int(np.argmin(w))
        return int(J[p]), p


def search(arm, Fs, sw, lam, evals, rng):
    """returns list of (fitness, w) for the best 5 distinct directions (sign folded into w)."""
    def fit(w):
        s_ = Fs @ w; best = None
        for sg in (1, -1):
            j, _ = sw.eval(np.argsort(sg * s_, kind="stable"), lam)
            if best is None or j < best[0]: best = (j, sg * w)
        return best
    pool = []
    if arm == "F0":
        W = [np.eye(Fs.shape[1])[j] for j in range(Fs.shape[1])] + [rng.normal(size=Fs.shape[1]) for _ in range(2000)]
        pool = [fit(w) for w in W]
    else:
        d = Fs.shape[1]; per = evals // 2
        for r in range(2):
            es = cma.CMAEvolutionStrategy(rng.normal(size=d), 0.8, {"maxfevals": per, "popsize": 32, "verbose": -9,
                                          "seed": int(rng.integers(1 << 30)), "tolfun": 0, "tolfunhist": 0,
                                          "tolflatfitness": 10**6, "tolstagnation": 10**6, "tolx": 0})
            while not es.stop():
                P = es.ask(); R = [fit(np.asarray(p)) for p in P]; es.tell(P, [x[0] for x in R]); pool += R
    pool.sort(key=lambda x: x[0]); top = []
    for j, w in pool:
        u = w / np.linalg.norm(w)
        if all(abs(u @ (v / np.linalg.norm(v))) < 0.99 for _, v in top): top.append((j, w))
        if len(top) == 5: break
    return top


def main(src, eps, lam, evals):
    name = os.path.basename(src.rstrip("/")).replace("_hg.npz", "")
    rng = np.random.default_rng(0); h = HG(*load(src)); sw = Sweeper(h, eps); lo, hi = bounds(h.n, eps)
    tp = os.path.join(SAVE, f"{name}_e{int(eps * 100):02d}.npy")
    T = h.cut(np.load(tp)) if os.path.exists(tp) else None
    t0 = time.time(); F = spectral(h, 16, rng); Fs = (F - F.mean(0)) / (F.std(0) + 1e-12)
    print(f"== {name} n={h.n} eps={eps} lam={lam} target {T}  (eigsh {time.time() - t0:.0f}s)", flush=True)
    for arm in ("F0", "F1", "F2"):
        t = time.time(); L = lam if arm == "F2" else 0
        top = search(arm, Fs, sw, L, evals, rng); ts = time.time() - t
        res = []
        for j, w in top:
            s_ = Fs @ w; g = l1_ir(h, s_, 64); order = np.lexsort((kahn_rank(h, s_), g))
            c, p = sweep_part(h, order, eps)
            assert h.closed(p) and lo <= np.sum(p == 0) <= hi
            r = refine(h, p, eps)[-1][1]; assert h.closed(p) and lo <= np.sum(p == 0) <= hi
            res.append((j, c, r))
        best = min(res, key=lambda x: x[2])
        tag = f" ({best[2] / T:.2f}x T)" if T else ""
        print(f"   {arm}: search best fitness {top[0][0]}  [{ts:.0f}s] | top5 projected sweep {[x[1] for x in res]} "
              f"-> refined {[x[2] for x in res]} | best {best[2]}{tag}  [{time.time() - t:.0f}s]", flush=True)


if __name__ == "__main__":
    a = sys.argv
    main(a[1], float(a[2]) if len(a) > 2 else 0.10, float(a[3]) if len(a) > 3 else 1.0, int(a[4]) if len(a) > 4 else 3000)
