"""Premise check: can RTL modules serve as intact units for acyclic partitioning?

Gate DAG = driver -> sink over nets, edges into sequential cells (FF/latch/macro) dropped
(same convention as BeyondDagPartition's cell_dag). For each hierarchy level, contract modules
and measure the strongly connected components of the module quotient graph.
If the largest SCC holds > ~50% of cells, no balanced acyclic bipartition exists that keeps
those modules intact.
"""
import sys, time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, r"C:\Users\YL\Desktop\CIA LAB\HyperGraphPartition\quest1")
import eval_all
eval_all.COMPRESS = True


def gate_dag(z):
    net, node, drv, types = z["net"], z["node"], z["drv"], z["types"]
    n = len(types)
    seq = np.array([("DFF" in t) or ("LATCH" in t) or t.startswith("$_SR") or not t.startswith("$_")
                    for t in types])
    dn, dg = net[drv], node[drv]
    sk = (~drv) & (~seq[node])
    sn, sg = net[sk], node[sk]
    order = np.argsort(dn, kind="stable"); dn, dg = dn[order], dg[order]
    lo = np.searchsorted(dn, sn, "left"); hi = np.searchsorted(dn, sn, "right")
    cnt = hi - lo
    one = cnt == 1
    src = [dg[lo[one]]]; dst = [sg[one]]
    for i in np.where(cnt > 1)[0]:
        src.append(dg[lo[i]:hi[i]]); dst.append(np.full(cnt[i], sg[i]))
    src = np.concatenate(src); dst = np.concatenate(dst)
    keep = src != dst
    A = sp.csr_matrix((np.ones(keep.sum()), (src[keep], dst[keep])), shape=(n, n))
    A.data[:] = 1
    return A, seq


def main(path):
    t0 = time.time()
    z = np.load(path)
    A, seq = gate_dag(z)
    n = A.shape[0]
    ncomp, lab = connected_components(A, directed=True, connection="strong")
    loopcells = n - ncomp  # crude: cells beyond one per SCC
    ids, leaf, trunc = eval_all.label_tree(z["paths"])
    r, c = A.nonzero()
    print(f"\n== {path}  n={n}  edges={len(r)}  seq={seq.sum()}  gate-level cyclic cells={loopcells}  "
          f"depth={ids.shape[0]}")
    print(f"{'level':>5} {'#mod':>6} {'cross%':>7} {'xedge_inSCC%':>12} {'bigSCC_mods':>11} {'bigSCC_cells%':>13}")
    keys = [t + ("#",) for t in trunc]
    D = max(len(k) for k in keys)
    for l in range(1, D + 1):
        g = np.unique(["/".join(k[:l]) for k in keys], return_inverse=True)[1]
        m = g.max() + 1
        gr, gc = g[r], g[c]
        x = gr != gc
        if m < 2 or x.sum() == 0:
            continue
        Q = sp.csr_matrix((np.ones(x.sum()), (gr[x], gc[x])), shape=(m, m))
        _, sl = connected_components(Q, directed=True, connection="strong")
        same = sl[gr[x]] == sl[gc[x]]
        sz_cells = np.bincount(sl[g], minlength=sl.max() + 1)
        sz_mods = np.bincount(sl)
        b = sz_cells.argmax()
        print(f"{l:>5} {m:>6} {100*x.mean():>7.2f} {100*same.mean():>12.1f} {sz_mods[b]:>11} "
              f"{100*sz_cells[b]/n:>13.1f}")
    print(f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        main(p)
