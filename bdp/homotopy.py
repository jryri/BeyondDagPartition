"""RESULTS §28: Gaussian-homotopy (graduated) descent on theta -- the global mechanism suggested by §27.

§27: J(theta) is multimodal at every scale (compass ends in strict local minima) and flat in the far-seed regime
(CMA plateaus).  Smoothing addresses both: at width sigma, minima narrower than sigma vanish and plateaus narrower
than sigma acquire a slope.  Graduated optimisation follows the minimiser of the smoothed objective while sigma shrinks.

Smoothed objective and gradient (fully deterministic, no random directions):
  per coordinate i, the central difference  g_i = (J(theta + sigma e_i) - J(theta - sigma e_i)) / (2 sigma)
  is EXACTLY the i-th partial derivative of J convolved with a uniform kernel of half-width sigma along axis i.
  F_sigma(theta) = mean of the 64 probe values (the smoothed level used for acceptance).
Step: theta <- theta - sigma * g / |g|   (step length = smoothing scale).
Accept while F_sigma decreases; otherwise reject and halve sigma.  sigma: 2 -> stop below 1/32 (§27: below
r ~ 0.03 J is mostly constant, so no information is left), or at 1000 evaluations.
Incumbent = best exact J among all evaluated thetas (every theta is a feasible acyclic balanced partition).
Same representation X, start theta0 = (-2 e1, +2 e1) and budget as §26 compass: the ONLY change is the search rule.
Cost: 64 evaluations per step x 2 exact projections (~log2 n max-flows each); + 1 refine.
Usage: python bdp/homotopy.py <src> [eps]
"""
import sys, os, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from refine import HG, load, bounds, refine, fm, sweep_part, kahn
from exact_ir import l1_ir_exact
from det import eig, P24, C25
from diag import make_J
from separability import SAVE

CMP26 = {"serv": (542, 226), "omsp": (330, 311), "picosoc": (327, 324), "aes": (596, 578), "riscv": (670, 510),
         "biriscv": (667, 238)}                              # §26 compass (sweep, refined)
CMA25 = {"serv": 180, "omsp": 297, "picosoc": 451, "aes": 685, "riscv": 489, "biriscv": 859}   # §25 CMA sweep


def make_J2(h, X, eps):
    """§31: refinement-aware objective (§29): exact projection -> sweep -> FM to convergence, both orientations."""
    d = X.shape[1]; memo = {}; nev = [0]
    def J(th):
        key = th.tobytes()
        if key not in memo:
            a, b = th[:d], th[d:]
            s = np.sqrt(((X - a) ** 2).sum(1)) - np.sqrt(((X - b) ** 2).sum(1)); out = None
            for sg in (1, -1):
                f = l1_ir_exact(h, sg * s); kr = np.empty(h.n, np.int64); kr[kahn(h, sg * s)] = np.arange(h.n)
                c, p = sweep_part(h, np.lexsort((kr, f)), eps); p = p.copy(); c = fm(h, p, eps)
                if out is None or c < out[0]: out = (c, p)
            memo[key] = out; nev[0] += 1
        return memo[key]
    return J, nev


def homotopy(h, eps, evals=1000, sigma0=2.0, sigma_min=1 / 32, obj="j0"):
    vals, U, dv, inv, w = eig(h, 32)
    X = U[:, :16] * inv[:, None]; X = (X - X.mean(0)) / (X.std(0) + 1e-12); d = 16
    J, nev = (make_J if obj == "j0" else make_J2)(h, X, eps); E = np.eye(2 * d)
    th = np.zeros(2 * d); th[0] = -2.0; th[d] = 2.0
    inc = J(th); sigma = sigma0; t0 = time.time()

    def probe(th, sigma):
        nonlocal inc
        Jp = np.array([J(th + sigma * e)[0] for e in E]); Jm = np.array([J(th - sigma * e)[0] for e in E])
        for k in range(2 * d):
            for r in (J(th + sigma * E[k]), J(th - sigma * E[k])):
                if r[0] < inc[0]: inc = r
        return (Jp + Jm).mean() / 2, (Jp - Jm) / (2 * sigma)

    F, g = probe(th, sigma); steps = 0
    while sigma >= sigma_min and nev[0] < evals:
        if not np.any(g):
            sigma /= 2; F, g = probe(th, sigma); continue
        cand = th - sigma * g / np.linalg.norm(g)
        c = J(cand)
        if c[0] < inc[0]: inc = c
        Fc, gc = probe(cand, sigma)
        if Fc < F: th, F, g = cand, Fc, gc; steps += 1
        else:
            sigma /= 2; F, g = probe(th, sigma)
        print(f"   sigma {sigma:<8.4g} F {F:8.1f} | incumbent J {inc[0]} | |theta| {np.linalg.norm(th):.2f} "
              f"| {nev[0]} evals [{time.time() - t0:.0f}s]", flush=True)
    print(f"   homotopy: {steps} accepted steps, {nev[0]} evaluations, final sigma {sigma:g}", flush=True)
    return inc


H28 = {"serv": 116, "omsp": 290, "picosoc": 297, "aes": 579, "riscv": 338, "biriscv": 309}   # §28 homotopy-J0 refined
POOL29 = {"serv": 97, "omsp": 327, "picosoc": 311, "aes": 507, "riscv": 304, "biriscv": 238}  # §29 pool best refined


def main(src, eps, obj="j0"):
    name = os.path.basename(src.rstrip("/")).replace("_hg.npz", "")
    h = HG(*load(src)); lo, hi = bounds(h.n, eps)
    tc = h.cut(np.load(os.path.join(SAVE, f"{name}_e{int(eps * 100):02d}.npy")))
    print(f"== homotopy[{obj}] {name} n={h.n} | T {tc}", flush=True)
    t0 = time.time(); c, p = homotopy(h, eps, obj=obj)
    if obj == "j2":
        p = p.copy(); r = refine(h, p, eps)[-1][1]; assert h.closed(p) and lo <= np.sum(p == 0) <= hi
        print(f"   RESULT homotopy-J2 {name}: J2 {c} -> refined {r} ({r / tc:.2f}x T) | §28 homotopy-J0 {H28[name]} "
              f"| §29 pool best {POOL29[name]} | §24 P {P24[name]} | T {tc}  [{time.time() - t0:.0f}s]", flush=True)
        return
    assert h.closed(p) and lo <= np.sum(p == 0) <= hi
    p = p.copy(); r = refine(h, p, eps)[-1][1]; assert h.closed(p) and lo <= np.sum(p == 0) <= hi
    cs, cr = CMP26[name]
    print(f"   RESULT homotopy {name}: sweep {c} -> refined {r} ({r / tc:.2f}x T) | compass §26 {cs} -> {cr} "
          f"| CMA §25 sweep {CMA25[name]} -> {C25[name]} | §24 P {P24[name]} | T {tc}  [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 0.10, sys.argv[3] if len(sys.argv) > 3 else "j0")
