"""RESULTS §36: lattice combinations of acyclic cuts (premise test for "special separators").

Fact: block 0 of an acyclic cut is a predecessor-closed set, and those are closed under union and intersection.
At the score level: if r_i, r_j are topological ranks (strictly increasing along DAG edges), so are
max(r_i, r_j) and min(r_i, r_j); a sweep prefix of max is the INTERSECTION of the two prefixes, of min the UNION.
So combining two simple separators needs no projection (no max-flow): one sort + sweep + FM.

Test: singles = the §19 pool (120 two-seed potentials x 2 orientations), each decoded as the method does
(exact projection -> sweep -> FM = J2).  Take the 20 best singles by J2; all 190 pairs x {max, min}:
order by the combined rank -> sweep in the balance window -> FM.
Reported: best single vs best combination (sweep, after FM, after full refine); share of combinations that beat
BOTH parents after FM, and the median relative gain of those.
Caveat: best-of-380 combinations vs best-of-240 singles is not an equal-budget comparison; the share that beats
both parents is the unconfounded statistic.
Usage: python bdp/lattice.py <src> [eps] [workers]
"""
import sys, os, time, warnings, itertools
import numpy as np
warnings.filterwarnings("ignore")
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from refine import HG, load, bounds, sweep_part, refine, kahn, fm
from exact_ir import l1_ir_fast

std = lambda A: (A - A.mean(0)) / (A.std(0) + 1e-12)
GAPS = [0.5, 2.0, 8.0, 32.0, 1e4]
BASE = {"serv": 61, "omsp": 341, "picosoc": 250, "aes": 475, "riscv": 251, "biriscv": 238}   # §33 best single baseline
H32 = {"serv": 71, "omsp": 286, "picosoc": 287, "aes": 551, "riscv": 301, "biriscv": 238}
_G = {}


def _init(src, eps): _G["h"] = HG(*load(src)); _G["eps"] = eps


def _single(s):
    h, eps = _G["h"], _G["eps"]; out = []
    for sg in (1, -1):
        f = l1_ir_fast(h, sg * s); kr = np.empty(h.n, np.int64); kr[kahn(h, sg * s)] = np.arange(h.n)
        order = np.lexsort((kr, f)); rank = np.empty(h.n, np.int64); rank[order] = np.arange(h.n)
        c0, p = sweep_part(h, order, eps); p = p.copy(); c2 = fm(h, p, eps)
        out.append((c0, c2, rank.astype(np.int32), p))
    return out


def _combo(args):
    ri, rj, op = args; h, eps = _G["h"], _G["eps"]
    key = np.maximum(ri, rj) if op == "max" else np.minimum(ri, rj)
    c0, p = sweep_part(h, np.lexsort((ri, key)), eps); p = p.copy(); c2 = fm(h, p, eps)
    return c0, c2, p


def main(src, eps, workers):
    from dims import spectral
    name = os.path.basename(src.rstrip("/")).replace("_hg.npz", "")
    rng = np.random.default_rng(0); h = HG(*load(src)); lo, hi = bounds(h.n, eps); t0 = time.time()
    X = std(spectral(h, 32, rng))[:, :16]
    def score(c, u, g):
        a, b = X[c] - g * u / 2, X[c] + g * u / 2
        return np.sqrt(((X - a) ** 2).sum(1)) - np.sqrt(((X - b) ** 2).sum(1))
    cands = []
    for _ in range(2000):
        u = rng.normal(size=16); cands.append((int(rng.integers(h.n)), u / np.linalg.norm(u), GAPS[int(rng.integers(len(GAPS)))]))
    pa = np.array([sweep_part(h, np.argsort(score(*cd), kind="stable"), eps)[0] for cd in cands])
    o = np.argsort(pa, kind="stable"); pool_ = list(o[:60]) + list(rng.choice(o[60:], 60, replace=False))
    with Pool(workers, initializer=_init, initargs=(src, eps)) as pool:
        S = [x for r in pool.map(_single, [score(*cands[i]) for i in pool_], chunksize=1) for x in r]
        ts = time.time() - t0
        S.sort(key=lambda x: x[1]); top = S[:20]
        jobs = [(top[i][2], top[j][2], op) for i, j in itertools.combinations(range(20), 2) for op in ("max", "min")]
        par = [(i, j) for i, j in itertools.combinations(range(20), 2) for _ in range(2)]
        t1 = time.time(); C = pool.map(_combo, jobs, chunksize=4); tc = time.time() - t1
    bs = top[0]; bc = min(C, key=lambda x: x[1])
    wins = [(min(top[i][1], top[j][1]) - c[1]) / min(top[i][1], top[j][1]) for (i, j), c in zip(par, C) if c[1] < min(top[i][1], top[j][1])]
    ps = bs[3].copy(); rs = refine(h, ps, eps)[-1][1]; pc = bc[2].copy(); rc = refine(h, pc, eps)[-1][1]
    for p in (ps, pc): assert h.closed(p) and lo <= np.sum(p == 0) <= hi
    print(f"== {name} n={h.n} | 240 singles in {ts:.0f}s | 380 combinations in {tc:.0f}s ({1000 * tc / 380:.0f} ms each, {workers} workers)", flush=True)
    print(f"   best single      : sweep {bs[0]:5} -> FM {bs[1]:5} -> refined {rs:5}", flush=True)
    print(f"   best combination : sweep {bc[0]:5} -> FM {bc[1]:5} -> refined {rc:5}   ({100 * (rs - rc) / rs:+.1f}% vs best single)", flush=True)
    print(f"   combinations beating BOTH parents after FM: {len(wins)}/380 ({100 * len(wins) / 380:.0f}%), "
          f"median gain {100 * np.median(wins) if wins else 0:.1f}%, max {100 * max(wins) if wins else 0:.1f}%", flush=True)
    print(f"   reference: homotopy-J2 {H32[name]} | best single baseline (§33) {BASE[name]}", flush=True)


if __name__ == "__main__":
    a = sys.argv
    main(a[1], float(a[2]) if len(a) > 2 else 0.10, int(a[3]) if len(a) > 3 else 12)
