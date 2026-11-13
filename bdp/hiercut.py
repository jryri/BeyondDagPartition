"""Oracle test: are RTL module boundaries a good reference for min-cut bisection?

Uses ground-truth hierarchy (upper bound for anything R2 could recover).
 1. enrichment: P(net crosses leaf modules | net is cut by KaHyPar) / P(net crosses leaf modules)
 2. module splitting: how many leaf modules does the (near-)optimal partition split?  (unconstrained and acyclic-repaired)
 3. module-atomic bisection: contract leaf (or level-l) modules to weighted nodes, partition, compare cut to flat.
Usage: python bdp/hiercut.py data/<design>_hg.npz [...]
"""
import sys, os, tempfile
import numpy as np, scipy.sparse as sp
from scipy.sparse.csgraph import breadth_first_order as bfo
import mtkahypar as mk
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import eval_all
eval_all.COMPRESS = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modscc import gate_dag

mtk = mk.initialize(8)
EPS = (0.02, 0.10)


def write_hgr(nets, n, w=None):
    f = tempfile.NamedTemporaryFile("w", suffix=".hgr", delete=False)
    f.write(f"{len(nets)} {n}" + (" 10" if w is not None else "") + "\n")
    for p in nets: f.write(" ".join(str(x + 1) for x in p) + "\n")
    if w is not None:
        for x in w: f.write(f"{int(x)}\n")
    f.close(); return f.name


def kahypar(nets, n, eps, w=None, seeds=3):
    path = write_hgr(nets, n, w); best = None
    try:
        for s in range(seeds):
            mk.set_seed(s)
            ctx = mtk.context_from_preset(mk.PresetType.QUALITY); ctx.set_partitioning_parameters(2, eps, mk.Objective.CUT); ctx.logging = False
            hg = mtk.hypergraph_from_file(path, ctx, mk.FileFormat.HMETIS)
            p = hg.partition(ctx)
            prt = np.array([p.block_id(v) for v in range(n)])
            ww = np.ones(n) if w is None else np.asarray(w, float)
            if np.bincount(prt, weights=ww, minlength=2).max() > (1 + eps) * np.ceil(ww.sum() / 2) + 1e-9: continue
            if best is None or p.cut() < best[0]:
                best = (p.cut(), prt)
    except Exception as e:
        best = ("fail:" + type(e).__name__, None)
    os.unlink(path); return best


def main(path):
    z = np.load(path)
    net, node = z["net"], z["node"]
    o = np.lexsort((node, net)); net, node = net[o], node[o]
    k = np.r_[True, (net[1:] != net[:-1]) | (node[1:] != node[:-1])]; net, node = net[k], node[k]
    cnt = np.bincount(net); ok = cnt[net] >= 2
    _, netid = np.unique(net[ok], return_inverse=True); pins = node[ok]
    n = len(z["types"]); M = netid.max() + 1
    order = np.argsort(netid, kind="stable"); ptr = np.r_[0, np.cumsum(np.bincount(netid))]; pp = pins[order]
    nets = [pp[ptr[i]:ptr[i + 1]] for i in range(M)]
    ids, leaf, trunc = eval_all.label_tree(z["paths"])
    leafid = leaf[1]; nleaf = leafid.max() + 1
    # nets crossing leaf modules
    def crossing(lab):
        lo = np.full(M, 10**9); hi = np.full(M, -1)
        np.minimum.at(lo, netid, lab[pins]); np.maximum.at(hi, netid, lab[pins]); return lo != hi
    xleaf = crossing(leafid)
    print(f"\n== {os.path.basename(path)}  n={n} nets={M} leaf modules={nleaf} depth={ids.shape[0]}  "
          f"nets crossing leaf modules: {xleaf.mean():.3f}")
    A, seq = gate_dag(z); A = A.tocsr(); eu, ev = A.nonzero(); AT = A.T.tocsr()

    def closure(seed, Adj):
        kk = len(seed)
        if kk == 0: return np.zeros(n, bool)
        B = sp.vstack([sp.csr_matrix((np.ones(kk), (np.zeros(kk, int), seed)), shape=(1, n)), Adj]).tocsr()
        B = sp.hstack([sp.csr_matrix((n + 1, 1)), B]).tocsr()
        r = bfo(B, 0, directed=True, return_predecessors=False); m_ = np.zeros(n + 1, bool); m_[r] = True; return m_[1:]

    def cutof(part):
        lo = np.full(M, 9); hi = np.full(M, -1)
        np.minimum.at(lo, netid, part[pins]); np.maximum.at(hi, netid, part[pins]); return lo != hi

    def split_stats(part):
        s = np.bincount(leafid, weights=part, minlength=nleaf); tot = np.bincount(leafid, minlength=nleaf)
        minority = np.minimum(s, tot - s) / tot
        sp_ = minority > 0.05
        return sp_.sum(), (tot[sp_].sum() / n), np.sum(np.minimum(s, tot - s)) / n

    for eps in EPS:
        c, part = kahypar(nets, n, eps)
        cutn = cutof(part)
        enr = xleaf[cutn].mean() / xleaf.mean()
        ns, fc, mv = split_stats(part)
        print(f" eps{eps:.2f} flat KaHyPar cut {c}: cut nets crossing leaf modules {xleaf[cutn].mean():.2f} "
              f"(enrichment {enr:.1f}x); leaf modules split {ns}/{nleaf} ({fc:.0%} of cells in split modules; "
              f"min cells to move to make it module-pure {mv:.1%})")
        # acyclic repair of the same partition (best balanced variant)
        lo_, hi_ = n * (1 - eps) / 2, n * (1 + eps) / 2; best = None
        for early in (0, 1):
            Sx = part == early
            for Sr in (~closure(np.flatnonzero(~Sx), A), closure(np.flatnonzero(Sx), AT)):
                if lo_ <= Sr.sum() <= hi_:
                    cc = cutof(Sr.astype(int)).sum()
                    if best is None or cc < best[0]: best = (cc, Sr.astype(int))
        if best:
            ns, fc, mv = split_stats(best[1])
            print(f"          acyclic (closure repair) cut {best[0]}: leaf modules split {ns}/{nleaf} ({fc:.0%} of cells)")
        # module-atomic bisection at leaf level and at each tree level
        for lvl in list(range(ids.shape[0] - 1, -1, -1)):
            lab = ids[lvl].copy(); miss = lab < 0
            if miss.any():  # cells whose path ends above this level keep their deepest id
                for l2 in range(lvl - 1, -1, -1):
                    m2 = miss & (ids[l2] >= 0); lab[m2] = ids[l2][m2] + 10**7 * (l2 + 1); miss &= ~m2
            _, lab = np.unique(lab, return_inverse=True); K = lab.max() + 1
            if K < 2: continue
            cn = [np.unique(lab[p]) for p in nets]; cn = [p for p in cn if len(p) >= 2]
            w = np.bincount(lab, minlength=K)
            r = kahypar(cn, K, eps, w)
            tag = "leaf" if lvl == ids.shape[0] - 1 else f"level {lvl}"
            val = r[0] if r is not None else "infeasible"
            ratio = f"({r[0] / c:.2f}x)" if r is not None and isinstance(r[0], (int, np.integer)) else ""
            print(f"          module-atomic [{tag}, {K} units, largest {w.max() / n:.0%}]: cut {val} {ratio}")
            if r is None or not isinstance(r[0], (int, np.integer)): break


if __name__ == "__main__":
    for p in sys.argv[1:]:
        main(p)
