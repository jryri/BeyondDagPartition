"""dagP baseline (Herrmann et al. SISC 2019, multilevel acyclic DAG bisection; built without METIS).

dagP minimises DAG edge cut, not net cut. Two inputs per design:
  u : unit edge weights (plain dagP objective)
  w : edge driver->sink weighted 60/#DAG-sinks of its net (each net contributes ~60 if fully cut; proxy for net cut)
Nets with only FF/macro sinks have no DAG edges and are invisible to dagP (they do not affect acyclicity).
Each dagP partition is checked (acyclic, balanced), evaluated in net cut, then refined with our acyclic FM+flow.
Usage: python bdp/dagp_run.py <design src> [seeds]
"""
import sys, os, subprocess, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from refine import HG, load, bounds, refine, kahypar_parts
import mtkahypar as mk

EXE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools", "dagP_build", "rMLGP.exe")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dagp_in")


def write_dot(h, path, weighted):
    if weighted:
        # number of DAG sinks per (driver, net): approximate by out-degree of the driver within each net is costly;
        # use per-edge weight = 60 / max(1, #DAG successors of driver)  (one output net per gate in our netlists)
        outdeg = np.bincount(h.eu, minlength=h.n)
        w = np.maximum(1, np.round(60 / np.maximum(outdeg[h.eu], 1))).astype(int)
    with open(path, "w") as f:
        f.write("digraph G {\n")
        for v in range(h.n): f.write(f"{v};\n")
        if weighted:
            for a, b, x in zip(h.eu, h.ev, w): f.write(f"{a} -> {b} [weight={x}];\n")
        else:
            for a, b in zip(h.eu, h.ev): f.write(f"{a} -> {b};\n")
        f.write("}\n")


def run_dagp(path, eps, seed):
    r = subprocess.run([EXE, path, "2", "--ratio", f"{1 + eps}", "--seed", str(seed), "--write_parts", "1", "--print", "0"],
                       capture_output=True, text=True, cwd=os.path.dirname(path), timeout=3600)
    pf = f"{path}.partsfile.part_2.seed_{seed}.txt"
    if not os.path.exists(pf):
        raise RuntimeError(r.stdout[-2000:] + r.stderr[-2000:])
    p = np.loadtxt(pf, dtype=np.int64)
    return p


def main(src, seeds):
    os.makedirs(OUT, exist_ok=True)
    name = os.path.basename(src.rstrip("/")).replace("_hg.npz", "")
    h = HG(*load(src)); mtk = mk.initialize(8)
    print(f"\n== {name}  n={h.n} nets={h.M} dag_edges={len(h.eu)}", flush=True)
    paths = {}
    for tag, wt in (("u", False), ("w", True)):
        paths[tag] = os.path.join(OUT, f"{name}_{tag}.dot")
        if not os.path.exists(paths[tag]): write_dot(h, paths[tag], wt)
    for eps in (0.02, 0.10):
        lo, hi = bounds(h.n, eps)
        ref = min(c for c, _ in kahypar_parts(h, eps, mtk, seeds=3))
        print(f" eps {eps:.2f}: Mt-KaHyPar unconstrained ref {ref}", flush=True)
        for tag in ("u", "w"):
            raw, fin, t0 = [], [], time.time()
            for s in range(1, seeds + 1):
                p = run_dagp(paths[tag], eps, s)
                assert len(p) == h.n
                if not h.closed(p.astype(np.int8)):
                    p = 1 - p  # block numbering: make block 0 the earlier one
                part = p.astype(np.int8)
                assert h.closed(part), "dagP partition not acyclic?"
                s0 = int(np.sum(part == 0)); bal = lo <= s0 <= hi
                c = h.cut(part); raw.append((c, bal))
                hist = refine(h, part, eps) if bal else [("start", c)]
                if bal:
                    assert h.closed(part) and lo <= np.sum(part == 0) <= hi
                    fin.append(hist[-1][1])
            rb = [c for c, b in raw if b]
            print(f"   dagP-{tag}: raw net cut best {min(rb) if rb else '-'} ({min(rb) / ref:.2f}x) "
                  f"[{sum(b for _, b in raw)}/{len(raw)} balanced]  +refine best {min(fin) if fin else '-'} "
                  f"({min(fin) / ref:.2f}x)  [{time.time() - t0:.0f}s]" if fin else f"   dagP-{tag}: no balanced result {raw}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 5)
