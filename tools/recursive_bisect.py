#!/usr/bin/env python3
"""K-way partition by recursive K=2 bisection (power of two).

Each bisection calls HySpecPro's existing K=2 path (8-d sign baseline, or
24-d order + affine + J). Sub-hypergraphs keep only pins inside the subset;
DAG edges keep both endpoints inside.
"""
from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch


def read_hgr(path: Path):
    nets = []
    with path.open() as f:
        m, n = map(int, f.readline().split()[:2])
        for line in f:
            pins = [int(x) - 1 for x in line.split()]
            if len(pins) >= 2:
                nets.append(pins)
    return n, nets


def read_dag(path: Path, n: int):
    edges = []
    with path.open() as f:
        for line in f:
            if "->" not in line:
                continue
            a, b = line.replace(";", "").split("->")
            edges.append((int(a), int(b)))
    return edges


def write_hgr(path: Path, nets, n: int):
    with path.open("w") as f:
        f.write(f"{len(nets)} {n}\n")
        for pins in nets:
            f.write(" ".join(str(p + 1) for p in pins) + "\n")


def write_dag(path: Path, n: int, edges):
    with path.open("w") as f:
        f.write("digraph G {\n")
        for i in range(n):
            f.write(f"{i};\n")
        for u, v in edges:
            f.write(f"{u} -> {v};\n")
        f.write("}\n")


def induce(nets, edges, members: np.ndarray):
    """Renumber members to 0..n_sub-1; keep nets/edges fully inside."""
    old = np.asarray(members, dtype=np.int32)
    n_sub = len(old)
    remap = {int(g): i for i, g in enumerate(old.tolist())}
    mem_set = set(remap)
    sub_nets = []
    for pins in nets:
        kept = [remap[p] for p in pins if p in mem_set]
        seen = set()
        uniq = []
        for x in kept:
            if x not in seen:
                seen.add(x)
                uniq.append(int(x))
        if len(uniq) >= 2:
            sub_nets.append(uniq)
    sub_edges = []
    for u, v in edges:
        if u in mem_set and v in mem_set:
            sub_edges.append((remap[u], remap[v]))
    return n_sub, sub_nets, sub_edges, old, remap


def _topo_rank(n: int, edges):
    succ = [[] for _ in range(n)]
    indeg = np.zeros(n, dtype=np.int32)
    for u, v in edges:
        succ[u].append(v)
        indeg[v] += 1
    topo = list(np.flatnonzero(indeg == 0))
    for u in topo:
        for v in succ[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                topo.append(v)
    rank = np.full(n, n, dtype=np.int32)
    if len(topo) == n:
        rank[np.asarray(topo, dtype=np.int32)] = np.arange(n, dtype=np.int32)
    return rank


def _feedback(pi, nets, drv, rank_of):
    F = 0
    for pins, d in zip(nets, drv):
        if d < 0:
            continue
        if len({int(pi[p]) for p in pins}) <= 1:
            continue
        dr = int(rank_of[int(pi[d])])
        if any(int(rank_of[int(pi[p])]) < dr for p in pins):
            F += 1
    return F


def metrics(pi: np.ndarray, nets, edges, k: int, n_random: int = 64):
    """C and F_net. For large k avoid k! : use topo-mean block order + samples."""
    n = len(pi)
    succ = [set() for _ in range(n)]
    for u, v in edges:
        succ[u].add(v)
    C = 0
    drv = []
    for pins in nets:
        parts = {int(pi[p]) for p in pins}
        if len(parts) > 1:
            C += 1
        hits = [p for p in pins if all(q == p or q in succ[p] for q in pins)]
        drv.append(hits[0] if len(hits) == 1 else -1)

    # Exact min over permutations only for small k.
    if k <= 7:
        best_F = None
        for perm in itertools.permutations(range(k)):
            F = _feedback(pi, nets, drv, np.array(perm, dtype=np.int32))
            best_F = F if best_F is None else min(best_F, F)
        bal = np.bincount(pi, minlength=k)
        return C, best_F, bal

    cell_rank = _topo_rank(n, edges)
    mean_r = np.zeros(k, dtype=np.float64)
    counts = np.zeros(k, dtype=np.float64)
    for i, lab in enumerate(pi):
        mean_r[lab] += cell_rank[i]
        counts[lab] += 1
    mean_r /= np.maximum(counts, 1)
    order = np.argsort(mean_r)
    candidates = [order, order[::-1]]
    rng = np.random.default_rng(0)
    for _ in range(n_random):
        candidates.append(rng.permutation(k))
    best_F = None
    for order in candidates:
        rank_of = np.empty(k, dtype=np.int32)
        rank_of[np.asarray(order, dtype=np.int32)] = np.arange(k, dtype=np.int32)
        F = _feedback(pi, nets, drv, rank_of)
        best_F = F if best_F is None else min(best_F, F)
    bal = np.bincount(pi, minlength=k)
    return C, best_F, bal


def run_k2(hgr: Path, out_dir: Path, tag: str, drivers: Path | None, env: dict, lambda_f: float = 1.0) -> np.ndarray:
    out_dir.mkdir(parents=True, exist_ok=True)
    # HySpecPro expects design_root/design.hgr
    design = "part"
    link = out_dir / f"{design}.hgr"
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(hgr.resolve())
    cmd = [
        env["PYTHON"],
        "/home/zixian/HySpecPro/HySpecPro.py",
        "--design_root", str(out_dir) + "/",
        "--design", design,
        "--result_root", str(out_dir) + "/",
        "--tag", tag,
        "--N_CMA_ITE", "1",
        "--UB", "0.02",
        "--KWAY", "2",
        "--device", "cuda:0",
        "--lambda_f", str(lambda_f),
    ]
    if drivers is not None:
        cmd += ["--drivers", str(drivers)]
    print("[run]", " ".join(cmd), flush=True)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    print(proc.stdout[-2000:] if proc.stdout else "", flush=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr[-2000:])
        raise SystemExit(f"HySpecPro failed rc={proc.returncode} tag={tag}")
    print(f"[run] {tag} wall={time.perf_counter()-t0:.1f}s", flush=True)
    sol = out_dir / f"{tag}_{design}_best_solution.pt"
    return torch.load(sol, map_location="cpu").numpy().astype(np.int32)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hgr", type=Path, required=True)
    ap.add_argument("--dag", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument(
        "--mode",
        choices=["ours", "base", "basej"],
        required=True,
        help="base=C only; basej=spectral X + optimize J; ours=order X + optimize J",
    )
    ap.add_argument("--design-name", type=str, default="aes")
    ap.add_argument("--lambda_f", type=float, default=1.0, help="J = C + lambda_f * F_net")
    args = ap.parse_args()
    if args.k < 2 or (args.k & (args.k - 1)) != 0:
        raise SystemExit("k must be a power of two")
    if args.mode in ("ours", "basej") and args.dag is None:
        raise SystemExit(f"{args.mode} needs --dag")

    n, nets = read_hgr(args.hgr)
    edges = read_dag(args.dag, n) if args.dag else []
    args.out_dir.mkdir(parents=True, exist_ok=True)

    nv = Path("/home/eda/anaconda3/envs/gnn_rtx5090/lib/python3.11/site-packages/nvidia")
    lib_dirs = []
    for sub in ("cuda_nvrtc/lib", "nvjitlink/lib", "cuda_runtime/lib", "cublas/lib", "cusolver/lib", "cusparse/lib"):
        d = nv / sub
        if d.is_dir():
            lib_dirs.append(str(d))
    extra = ":".join(lib_dirs)

    env = os.environ.copy()
    env["PATH"] = "/home/eda/anaconda3/envs/gnn_rtx5090/bin:" + env.get("PATH", "")
    env["DGLBACKEND"] = "pytorch"
    env["DGL_LIBRARY_PATH"] = "/home/zixian/dgl/build"
    env["LD_LIBRARY_PATH"] = (
        f"{extra}:/home/eda/anaconda3/envs/gnn_rtx5090/lib:/home/zixian/dgl/build:"
        + env.get("LD_LIBRARY_PATH", "")
    )
    env["PYTHONPATH"] = "/home/zixian/dgl/python:/home/zixian/HySpecPro"
    env["PYTHONUNBUFFERED"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["PYTHON"] = "/home/eda/anaconda3/envs/gnn_rtx5090/bin/python"

    # base: cut only, 8-d spectral. basej: J with spectral X. ours: J with order X.
    env.pop("HYSP_SPEC_ONLY", None)
    if args.mode == "base":
        env["HYSP_BASE8"] = "1"
    elif args.mode == "basej":
        env["HYSP_SPEC_ONLY"] = "1"
        env["HYSP_ORDIM"] = env.get("HYSP_ORDIM", "8")

    # labels grow as a binary tree of bisections
    pi = np.zeros(n, dtype=np.int32)
    # queue of (member_ids, label_base, bits_left)
    import math
    levels = int(math.log2(args.k))
    groups = [np.arange(n, dtype=np.int32)]  # each entry is members of one current block

    for level in range(levels):
        new_groups = []
        for gi, members in enumerate(groups):
            work = args.out_dir / f"L{level}_G{gi}"
            work.mkdir(parents=True, exist_ok=True)
            n_sub, sub_nets, sub_edges, old, _ = induce(nets, edges, members)
            hgr_p = work / "cells.hgr"
            write_hgr(hgr_p, sub_nets, n_sub)
            dag_p = None
            if args.mode in ("ours", "basej"):
                dag_p = work / "cell_dag.dot"
                write_dag(dag_p, n_sub, sub_edges)
            print(
                f"[bisect] level={level} group={gi} n={n_sub} "
                f"nets={len(sub_nets)} dag_edges={len(sub_edges)}",
                flush=True,
            )
            # register design name in batch dict via symlink name aes-like
            # Use design "part" and ensure batch size: add via env HYSP_NBATCH
            env["HYSP_NBATCH"] = "1500"
            sub_pi = run_k2(
                hgr_p, work, tag=f"{args.mode}_l{level}g{gi}",
                drivers=dag_p, env=env, lambda_f=args.lambda_f,
            )
            if len(sub_pi) != n_sub:
                raise SystemExit(f"solution size {len(sub_pi)} != {n_sub}")
            left = old[sub_pi == 0]
            right = old[sub_pi == 1]
            print(f"  split {n_sub} -> {len(left)} / {len(right)}", flush=True)
            new_groups.append(left)
            new_groups.append(right)
        groups = new_groups

    # assign labels 0..k-1 by group order
    for lab, members in enumerate(groups):
        pi[members] = lab

    out_pi = args.out_dir / f"{args.design_name}_k{args.k}_{args.mode}_pi.pt"
    torch.save(torch.from_numpy(pi), out_pi)
    print(f"wrote {out_pi}", flush=True)
    C, F, bal = metrics(pi, nets, edges, args.k)
    J = float(C) + args.lambda_f * float(F)
    summary = {
        "design": args.design_name,
        "k": args.k,
        "mode": args.mode,
        "lambda_f": args.lambda_f,
        "C": int(C),
        "F_net": int(F),
        "J": J,
        "parts": bal.tolist(),
        "pi": str(out_pi),
    }
    (args.out_dir / "SUMMARY.json").write_text(
        __import__("json").dumps(summary, indent=2) + "\n"
    )
    print(summary, flush=True)


if __name__ == "__main__":
    main()
