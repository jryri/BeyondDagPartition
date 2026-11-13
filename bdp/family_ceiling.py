"""How good can his decoder family get with a strong search?
Family: order = argsort(Z w), w >= 0 (Z = 16 envelope features) -> exact balanced sweep.  CMA-ES directly on the
sweep cut (w = theta^2), several restarts (random init + one from uniform w).  The result is a strong (not exact)
estimate of the family's best cut.  Compared with the best-known acyclic target (best_parts/, from separability.py).
Usage: python bdp/family_ceiling.py <src> [eps] [restarts] [evals]
"""
import sys, os, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
import cma
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from refine import HG, load, his_embed, sweep_part, refine

SAVE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_parts")


def main(src, eps, R, E):
    name = os.path.basename(src.rstrip("/")).replace("_hg.npz", "")
    t0 = time.time(); rng = np.random.default_rng(0)
    h = HG(*load(src)); X, Z = his_embed(h, rng)
    tp = os.path.join(SAVE, f"{name}_e{int(eps * 100):02d}.npy")
    T = np.load(tp) if os.path.exists(tp) else None
    f = lambda th: sweep_part(h, np.argsort(Z @ (np.asarray(th) ** 2), kind="stable"), eps)[0]
    # seed CMA with the best of 2000 random sparse nonneg combos (his search), then restarts from the top R
    rs = []
    for i in range(2000):
        wv = rng.exponential(size=16) * (rng.random(16) < rng.uniform(0.2, 1))
        if wv.sum() > 0: rs.append((f(np.sqrt(wv)), np.sqrt(wv)))
    rs.sort(key=lambda x: x[0]); best = (rs[0][0], rs[0][1])
    opts = {"maxfevals": E, "verbose": -9, "popsize": 16, "tolfun": 0, "tolfunhist": 0, "tolflatfitness": 10**6,
            "tolx": 0, "tolstagnation": 10**6}
    for r in range(R):
        x0 = rs[r][1] / (np.linalg.norm(rs[r][1]) + 1e-12)
        es = cma.CMAEvolutionStrategy(x0, 0.3, dict(opts, seed=r + 1))
        while not es.stop():
            P = es.ask(); es.tell(P, [f(p) for p in P])
        if es.result.fbest < best[0]: best = (int(es.result.fbest), np.asarray(es.result.xbest))
    c, p = sweep_part(h, np.argsort(Z @ best[1] ** 2, kind="stable"), eps)
    agree = f"{100 * np.mean(p == T):.1f}%" if T is not None else "-"
    tc = h.cut(T) if T is not None else None
    pr = p.copy(); cr = refine(h, pr, eps)[-1][1]
    print(f"== {name} eps={eps}: family best (CMA {R}x{E}) sweep cut {c} -> refined {cr} | target {tc} | "
          f"raw/target {c / tc:.2f}x, refined/target {cr / tc:.2f}x | agrees with target {agree} [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    a = sys.argv
    main(a[1], float(a[2]) if len(a) > 2 else 0.10, int(a[3]) if len(a) > 3 else 4, int(a[4]) if len(a) > 4 else 1500)
