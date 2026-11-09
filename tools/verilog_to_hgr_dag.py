#!/usr/bin/env python3
"""Build cells.hgr + cell_dag.dot from a Genus-mapped Verilog netlist.

Used for the cu3d LVTD aes (hierarchical sbox + top). Pin directions follow
common Liberty naming for this library (Z/ZN/Q/... out, A*/D/CP/... in).
Flip-flop D/CP/CDN/... inputs do not become DAG edges (same cut as nd_dagp).
"""
from __future__ import annotations

import argparse
import re
import time
from collections import defaultdict
from pathlib import Path

INST_NAME = r"(?:\\[^\s(]+|[A-Za-z_][\w$]*)"
INST_FULL_RE = re.compile(
    rf"^\s*(\w+)\s+({INST_NAME})\s*\((.*)\)\s*;\s*$", re.S
)
INST_START_RE = re.compile(rf"^\s*(\w+)\s+({INST_NAME})\s*\(")
INST_NAME_ONLY_RE = re.compile(rf"^\s*(\w+)\s+({INST_NAME})\s*$")
PORT_RE = re.compile(r"\.(\w+)\s*\(\s*([^)]*?)\s*\)", re.S)
KEYWORDS = {
    "module", "endmodule", "input", "output", "inout", "wire",
    "assign", "supply0", "supply1",
}

OUT_PINS = {
    "Z", "ZN", "Q", "QN", "CO", "S", "SUM", "Y", "YN", "X", "XN",
}
IN_PINS = {
    "A", "A1", "A2", "A3", "A4", "B", "B1", "B2", "B3", "B4",
    "C", "C1", "C2", "CI", "I", "I1", "I2", "D", "CP", "CDN", "SDN",
    "SN", "RN", "CLK", "CK", "SL", "SI", "SE", "E", "TE",
}
SEQ_RE = re.compile(r"(DFF|DFC|DFR|LAT|JKFF)", re.I)


def parse_modules(text: str) -> dict[str, str]:
    return {
        m.group(1): m.group(2)
        for m in re.finditer(r"module\s+(\w+)\s*\([^;]*\);\s*(.*?)endmodule", text, re.S)
    }


def parse_instances(body: str) -> list[dict]:
    lines = body.splitlines()
    insts: list[dict] = []
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        ms = INST_START_RE.match(line) or INST_NAME_ONLY_RE.match(line)
        if not ms:
            i += 1
            continue
        cell = ms.group(1)
        if cell in KEYWORDS:
            i += 1
            continue
        buf = line
        while ");" not in buf and i + 1 < n:
            i += 1
            buf += " " + lines[i].strip()
        m = INST_FULL_RE.match(buf)
        if not m:
            i += 1
            continue
        cell, iname, ports = m.group(1), m.group(2), m.group(3)
        conns = {
            pm.group(1): pm.group(2).strip()
            for pm in PORT_RE.finditer(ports)
            if pm.group(2).strip()
        }
        insts.append({"cell": cell, "name": iname, "conns": conns})
        i += 1
    return insts


def expand_bits(expr: str) -> list[str]:
    """Expand a port expression into bit-level net names."""
    expr = expr.strip()
    if expr.startswith("{") and expr.endswith("}"):
        parts = []
        inner = expr[1:-1]
        # split on commas not inside brackets (none expected deeper)
        for tok in inner.split(","):
            parts.extend(expand_bits(tok.strip()))
        return parts
    m = re.match(r"^([\\A-Za-z_][\w$\\]*)\[(\d+):(\d+)\]$", expr)
    if m:
        name, hi, lo = m.group(1), int(m.group(2)), int(m.group(3))
        step = -1 if hi >= lo else 1
        return [f"{name}[{i}]" for i in range(hi, lo + step, step)]
    m = re.match(r"^([\\A-Za-z_][\w$\\]*)\[(\d+)\]$", expr)
    if m:
        return [f"{m.group(1)}[{m.group(2)}]"]
    if expr.startswith("1'b") or expr.startswith("1'h"):
        return [expr]
    return [expr]


def port_width_from_body(body: str, port: str) -> int | None:
    m = re.search(
        rf"(?:input|output|inout)\s+\[(\d+):(\d+)\]\s+{re.escape(port)}\s*;",
        body,
    )
    if not m:
        return None
    hi, lo = int(m.group(1)), int(m.group(2))
    return abs(hi - lo) + 1


def flatten(modules: dict[str, str], top: str) -> list[dict]:
    """Return leaf instances with hierarchical names and top-level net names."""
    top_body = modules[top]
    top_insts = parse_instances(top_body)
    leaves: list[dict] = []
    for inst in top_insts:
        cell = inst["cell"]
        if cell in modules and cell != top:
            sub_body = modules[cell]
            # map submodule port bits -> parent nets
            port_map: dict[str, str] = {}
            for port, expr in inst["conns"].items():
                bits = expand_bits(expr)
                w = port_width_from_body(sub_body, port)
                if w is None:
                    # scalar port
                    if len(bits) != 1:
                        raise ValueError(f"scalar port {port} got {bits}")
                    port_map[port] = bits[0]
                else:
                    # Genus may pass a whole bus as a bare name: .d (sa00_sub)
                    if len(bits) == 1 and not bits[0].startswith("1'"):
                        mdecl = re.search(
                            rf"(?:input|output|inout)\s+\[(\d+):(\d+)\]\s+{re.escape(port)}\s*;",
                            sub_body,
                        )
                        hi, lo = int(mdecl.group(1)), int(mdecl.group(2))
                        step = -1 if hi >= lo else 1
                        bits = [f"{bits[0]}[{i}]" for i in range(hi, lo + step, step)]
                    if len(bits) != w:
                        raise ValueError(
                            f"port {cell}.{port} width {w} got {len(bits)} bits: {expr}"
                        )
                    # port declared [hi:lo] with hi>=lo typically [7:0]
                    m = re.search(
                        rf"(?:input|output|inout)\s+\[(\d+):(\d+)\]\s+{re.escape(port)}\s*;",
                        sub_body,
                    )
                    hi, lo = int(m.group(1)), int(m.group(2))
                    step = -1 if hi >= lo else 1
                    for i, bit in zip(range(hi, lo + step, step), bits):
                        port_map[f"{port}[{i}]"] = bit

            prefix = inst["name"]
            for sub in parse_instances(sub_body):
                if sub["cell"] in modules:
                    raise ValueError(f"nested hierarchy not supported: {sub['cell']}")
                conns = {}
                for pin, net in sub["conns"].items():
                    if net.startswith("1'b") or net.startswith("1'h"):
                        continue
                    # local net or port
                    if net in port_map:
                        conns[pin] = port_map[net]
                    elif re.match(r"^(\w+)\[(\d+)\]$", net) and net in port_map:
                        conns[pin] = port_map[net]
                    else:
                        # bit of a port bus written as d[0]
                        m = re.match(r"^(\w+)\[(\d+)\]$", net)
                        if m and f"{m.group(1)}[{m.group(2)}]" in port_map:
                            conns[pin] = port_map[f"{m.group(1)}[{m.group(2)}]"]
                        elif net in port_map:
                            conns[pin] = port_map[net]
                        else:
                            conns[pin] = f"{prefix}/{net}"
                leaves.append(
                    {
                        "cell": sub["cell"],
                        "name": f"{prefix}/{sub['name']}",
                        "conns": conns,
                    }
                )
        else:
            # leaf in top
            conns = {
                pin: net
                for pin, net in inst["conns"].items()
                if not (net.startswith("1'b") or net.startswith("1'h"))
            }
            # expand bus-valued connections on leaf? rare; keep as-is if scalar
            leaves.append({"cell": cell, "name": inst["name"], "conns": conns})
    return leaves


def pin_dir(pin: str) -> str:
    p = pin.upper()
    if p in OUT_PINS:
        return "OUTPUT"
    if p in IN_PINS:
        return "INPUT"
    # FA / AO naming leftovers
    if p.startswith("Z") or p in {"Q", "QN"}:
        return "OUTPUT"
    return "INPUT"


def is_seq(cell: str) -> bool:
    return bool(SEQ_RE.search(cell))


def build_hgr_and_dag(leaves: list[dict], out_dir: Path) -> None:
    names = sorted(inst["name"] for inst in leaves)
    idx = {n: i for i, n in enumerate(names)}
    cell_of = {inst["name"]: inst["cell"] for inst in leaves}

    net_pins: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for inst in leaves:
        for pin, net in inst["conns"].items():
            # explode rare bus on leaf pin
            bits = expand_bits(net) if net.startswith("{") else [net]
            for b in bits:
                if b.startswith("1'b") or b.startswith("1'h"):
                    continue
                net_pins[b].append((inst["name"], pin))

    # hgr: 1-based indices
    hedges = []
    for pins in net_pins.values():
        cells = []
        seen = set()
        for inst, _ in pins:
            if inst not in seen:
                seen.add(inst)
                cells.append(idx[inst] + 1)
        if len(cells) >= 2:
            hedges.append(cells)

    hgr = out_dir / "cells.hgr"
    with hgr.open("w") as f:
        f.write(f"{len(hedges)} {len(names)}\n")
        for he in hedges:
            f.write(" ".join(map(str, he)) + "\n")

    # DAG: driver -> sink; skip edges into sequential inputs
    edges: set[tuple[int, int]] = set()
    unknown_pins = set()
    for pins in net_pins.values():
        drivers, sinks = [], []
        for inst, pin in pins:
            d = pin_dir(pin)
            if d == "OUTPUT":
                drivers.append(inst)
            elif d == "INPUT":
                if is_seq(cell_of[inst]) and pin.upper() in {
                    "D", "CP", "CDN", "SDN", "SN", "RN", "CLK", "CK", "SI", "SE", "SL",
                }:
                    continue
                sinks.append(inst)
            else:
                unknown_pins.add(pin)
        for d in drivers:
            for s in sinks:
                if d != s:
                    edges.add((idx[d], idx[s]))

    dot = out_dir / "cell_dag.dot"
    with dot.open("w") as f:
        f.write("digraph G {\n")
        for i in range(len(names)):
            f.write(f"{i};\n")
        for u, v in sorted(edges):
            f.write(f"{u} -> {v};\n")
        f.write("}\n")

    # name map for debugging
    (out_dir / "cell_names.txt").write_text("\n".join(names) + "\n")

    # acyclicity check
    succ = [[] for _ in range(len(names))]
    indeg = [0] * len(names)
    for u, v in edges:
        succ[u].append(v)
        indeg[v] += 1
    q = [i for i, d in enumerate(indeg) if d == 0]
    seen_n = 0
    for u in q:
        seen_n += 1
        for v in succ[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)

    print(f"cells={len(names)} hyperedges={len(hedges)} dag_edges={len(edges)}")
    print(f"topo_ordered={seen_n}/{len(names)} cyclic={seen_n != len(names)}")
    if unknown_pins:
        print(f"unclassified_pins={sorted(unknown_pins)[:20]}")
    print(f"wrote {hgr}")
    print(f"wrote {dot}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--netlist", type=Path, required=True)
    ap.add_argument("--top", type=str, default="aes_cipher_top")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    t0 = time.perf_counter()
    text = args.netlist.read_text()
    modules = parse_modules(text)
    if args.top not in modules:
        raise SystemExit(f"top {args.top} not in {list(modules)}")
    leaves = flatten(modules, args.top)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    build_hgr_and_dag(leaves, args.out_dir)
    print(f"wall_s={time.perf_counter() - t0:.2f}")


if __name__ == "__main__":
    main()
