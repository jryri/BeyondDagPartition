"""RESULTS §33: every baseline reported separately, raw and after our refine (ε given; λ=∞).

Per design:
  Mt-KaHyPar   unconstrained, QUALITY preset, 3 seeds (best): cut and back edges (min over the two block orders).
               Not acyclic -- reference only.
  K+repair     each Mt-KaHyPar partition -> 4 closure variants -> rebalance -> best raw (before refine);
               then the 3 lowest refined (as best_known in separability.py) -> best refined.
  dagP-u/-w    rMLGP (no METIS), 10 seeds each, net cut of the best balanced raw, and best after our refine.
  his decoder  2000 random nonneg weights on the 16 envelope features + sweep (as separability.py) -> raw, refined.
  homotopy-J2  §32 (one method, no pool), refined -- from the log.
Times: wall-clock per baseline (Mt-KaHyPar 8 threads; dagP and Python single process).
Usage: python bdp/baselines.py <src> [eps]
"""
import sys, os, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from refine import HG, load, bounds, refine, kahypar_parts, closure_variants, rebalance, his_embed, sweep_part
from dagp_run import run_dagp, write_dot, OUT as DAGP_DIR
import mtkahypar as mk

H32 = {"serv": 71, "omsp": 286, "picosoc": 287, "aes": 551, "riscv": 301, "biriscv": 238}


def nback(h, p): return int(np.sum((p[h.eu] == 1) & (p[h.ev] == 0)))


def main(src, eps):
    name = os.path.basename(src.rstrip("/")).replace("_hg.npz", "")
    h = HG(*load(src)); lo, hi = bounds(h.n, eps); rng = np.random.default_rng(0)
    print(f"== {name} n={h.n} eps={eps}", flush=True)
    ok = lambda p: h.closed(p) and lo <= np.sum(p == 0) <= hi

    t0 = time.time(); mtk = mk.initialize(8); kp = kahypar_parts(h, eps, mtk, seeds=3); tk = time.time() - t0
    c, p = min(kp, key=lambda x: x[0]); b = min(nback(h, p), nback(h, (1 - p).astype(np.int8)))
    print(f"   Mt-KaHyPar (unconstrained)  cut {c:6} | back edges {b:6} | {tk:.0f}s", flush=True)

    t0 = time.time(); st = []
    for _, prt in kp:
        for v in closure_variants(h, prt):
            if rebalance(h, v, eps) and ok(v): st.append((h.cut(v), v))
    st.sort(key=lambda x: x[0]); raw = st[0][0] if st else None; fin = []
    for c_, v in st[:3]:
        q = v.copy(); fin.append(refine(h, q, eps)[-1][1]); assert ok(q)
    print(f"   K+repair                    raw {raw} -> refined {min(fin) if fin else '-'} | back 0 | "
          f"{tk + time.time() - t0:.0f}s incl. Mt-KaHyPar", flush=True)

    for tag in ("u", "w"):
        path = os.path.join(DAGP_DIR, f"{name}_{tag}.dot")
        if not os.path.exists(path): write_dot(h, path, tag == "w")
        t0 = time.time(); raws = []; parts = []
        for s in range(1, 11):
            p = run_dagp(path, eps, s).astype(np.int8)
            if not h.closed(p): p = (1 - p).astype(np.int8)
            if ok(p): raws.append(h.cut(p)); parts.append(p)
        td = time.time() - t0
        if not raws:
            print(f"   dagP-{tag}: no balanced acyclic result in 10 seeds | {td:.0f}s", flush=True); continue
        t1 = time.time(); fin = []
        for p in parts:
            q = p.copy(); fin.append(refine(h, q, eps)[-1][1]); assert ok(q)
        print(f"   dagP-{tag:2}                     raw {min(raws)} (median {int(np.median(raws))}, {len(raws)}/10 balanced) "
              f"-> refined {min(fin)} | back 0 | dagP {td:.0f}s, refine {time.time() - t1:.0f}s", flush=True)

    t0 = time.time(); X, Z = his_embed(h, rng); best = None
    for _ in range(2000):
        wv = rng.exponential(size=16) * (rng.random(16) < rng.uniform(0.2, 1))
        if wv.sum() == 0: continue
        c, p = sweep_part(h, np.argsort(Z @ wv, kind="stable"), eps)
        if best is None or c < best[0]: best = (c, p)
    q = best[1].copy(); r = refine(h, q, eps)[-1][1]; assert ok(q)
    print(f"   his decoder (random 2000)   raw {best[0]} -> refined {r} | back 0 | {time.time() - t0:.0f}s", flush=True)
    print(f"   homotopy-J2 (§32)           refined {H32.get(name)} | back 0", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 0.10)
