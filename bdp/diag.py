"""RESULTS §27: why does the theta-search fail?  Plateau vs local minimum at the optimiser's endpoint.

Question: at the endpoint theta* of §25 CMA / §26 compass, is J (the exact projected sweep cut the optimiser sees)
  (a) flat         -- small moves give the SAME partition / same J  -> no signal; remedy = smoothing / surrogate
  (b) local min    -- moves of every size give worse J             -> trapped; remedy = global exploration
  (c) not a min    -- some moves at some radius give better J      -> optimiser stopped too early / wrong step scale
Method: reproduce theta* deterministically (same seeds, same code path as theta_opt.py / det.py; checked against
the logged sweep value), then probe J(theta* + r u) for 20 fixed random unit directions u in R^32 and
r in {0.01, 0.03, 0.1, 0.3, 1, 3} (X is standardised, so r = 1 is one standard deviation per coordinate).
Per radius: fraction better / equal / worse, fraction with the identical partition, best and median dJ.
Cost: reproduction (as the original run) + 120 evaluations x 2 exact projections.
Usage: python bdp/diag.py <cma|compass> <src> [eps]
"""
import sys, os, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from refine import HG, load, bounds
from det import eig, decode_exact
HERE = os.path.dirname(os.path.abspath(__file__))
RADII = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]


def make_J(h, X, eps):
    d = X.shape[1]; memo = {}; nev = [0]
    def J(th):
        key = th.tobytes()
        if key not in memo:
            a, b = th[:d], th[d:]
            s = np.sqrt(((X - a) ** 2).sum(1)) - np.sqrt(((X - b) ** 2).sum(1))
            memo[key] = decode_exact(h, s, eps); nev[0] += 1
        return memo[key]
    return J, nev


def run_compass(h, eps, evals=1000):                       # identical to det.compass, but returns theta
    vals, U, dv, inv, w = eig(h, 32)
    X = U[:, :16] * inv[:, None]; X = (X - X.mean(0)) / (X.std(0) + 1e-12); d = 16
    J, nev = make_J(h, X, eps)
    th = np.zeros(2 * d); th[0] = -2.0; th[d] = 2.0; cur = J(th); step = 1.0
    while step >= 1e-3 and nev[0] < evals:
        moved = False
        for i in range(2 * d):
            for sgn in (1.0, -1.0):
                cand = th.copy(); cand[i] += sgn * step; r = J(cand)
                if r[0] < cur[0]: th, cur, moved = cand, r, True; break
            if nev[0] >= evals: break
        if not moved: step /= 2
    return X, th, J, f"final step {step:g}, {nev[0]} evals"


def run_cma(h, eps, evals=1000):                           # identical to theta_opt.main's search, tracks theta
    import cma
    from dims import spectral
    rng = np.random.default_rng(0)
    X = spectral(h, 32, rng); X = ((X - X.mean(0)) / (X.std(0) + 1e-12))[:, :16]; d = 16
    J, nev = make_J(h, X, eps); best = [None, None]; info = []
    while nev[0] < evals:
        x0 = rng.normal(size=2 * d)
        es = cma.CMAEvolutionStrategy(x0, 1.0, {"maxfevals": evals - nev[0], "verbose": -9, "seed": int(rng.integers(1 << 30))})
        while not es.stop():
            P = es.ask(); F = []
            for x in P:
                th = np.asarray(x); c = J(th)[0]; F.append(c)
                if best[0] is None or c < best[0]: best = [c, th.copy()]
            es.tell(P, F)
        info.append(f"stop {list(es.stop().keys())}, sigma {es.sigma:.3g}, |mean - best| {np.linalg.norm(es.mean - best[1]):.3g}")
    return X, best[1], J, "; ".join(info) + f", {nev[0]} evals"


def main(opt, src, eps):
    name = os.path.basename(src.rstrip("/")).replace("_hg.npz", "")
    h = HG(*load(src)); t0 = time.time()
    X, th, J, info = {"cma": run_cma, "compass": run_compass}[opt](h, eps)
    c0, p0 = J(th)
    print(f"== {opt} {name} n={h.n}: endpoint sweep J* = {c0}  ({info})  |theta*| = {np.linalg.norm(th):.3g}  "
          f"[{time.time() - t0:.0f}s]", flush=True)
    np.savez(os.path.join(HERE, f"diag_{opt}_{name}.npz"), theta=th, J=c0)
    U = np.random.default_rng(12345).normal(size=(20, th.size)); U /= np.linalg.norm(U, axis=1, keepdims=True)
    print(f"   {'r':>5} | better equal worse | same-partition | best dJ  median dJ", flush=True)
    for r in RADII:
        dJ, same = [], 0
        for u in U:
            c, p = J(th + r * u); dJ.append(c - c0); same += int(np.array_equal(p, p0))
        dJ = np.array(dJ)
        print(f"   {r:>5} | {np.sum(dJ < 0):>6} {np.sum(dJ == 0):>5} {np.sum(dJ > 0):>5} | {same:>14} | "
              f"{dJ.min():>+7} {np.median(dJ):>+9.0f}", flush=True)
    print(f"   done [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 0.10)
