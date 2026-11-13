"""Price of acyclicity: unconstrained Mt-KaHyPar cut vs acyclic cuts (exact sweep), K=2.

Input: his format (dir with cells.hgr + cell_dag.dot) or our npz (gate DAG with edges into FF/macro dropped).
Acyclic candidates (no FM): Fiedler + Kahn-priority repair; envelope U/L random nonneg combos (his order mode).
"""
import sys, os, heapq, time, tempfile
import numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import eigsh, LinearOperator
import mtkahypar as mk

src = sys.argv[1]; NR = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
rng = np.random.default_rng(0)
t0 = time.time()
if src.endswith(".npz"):
    z = np.load(src)
    net, node, drv, types = z["net"], z["node"], z["drv"], z["types"]
    n = len(types)
    seq = np.array([("DFF" in t) or ("LATCH" in t) or t.startswith("$_SR") or not t.startswith("$_") for t in types])
    o = np.lexsort((node, net)); net, node, drv = net[o], node[o], drv[o]
    keep = np.r_[True, (net[1:] != net[:-1]) | (node[1:] != node[:-1])]
    net, node, drv = net[keep], node[keep], drv[keep]
    cnt = np.bincount(net); ok = cnt[net] >= 2
    _, netid = np.unique(net[ok], return_inverse=True); pins = node[ok]
    dn, dg = net[drv], node[drv]; sk = (~drv) & (~seq[node]); sn, sg = net[sk], node[sk]
    oo = np.argsort(dn, kind="stable"); dn, dg = dn[oo], dg[oo]
    lo = np.searchsorted(dn, sn, "left"); hi = np.searchsorted(dn, sn, "right")
    src_, dst_ = [dg[lo[hi - lo == 1]]], [sg[hi - lo == 1]]
    for i in np.where(hi - lo > 1)[0]:
        src_.append(dg[lo[i]:hi[i]]); dst_.append(np.full(hi[i] - lo[i], sg[i]))
    e = np.unique(np.stack([np.concatenate(src_), np.concatenate(dst_)], 1), axis=0); e = e[e[:, 0] != e[:, 1]]
    eu, ev = e[:, 0], e[:, 1]
else:
    L = [l.split() for l in open(src + "/cells.hgr")]
    m_, n = map(int, L[0][:2])
    nl = [np.array(sorted(set(int(x) - 1 for x in l))) for l in L[1:m_ + 1]]; nl = [p for p in nl if len(p) >= 2]
    pins = np.concatenate(nl); netid = np.repeat(np.arange(len(nl)), [len(p) for p in nl])
    ed = [tuple(map(int, l.replace(";", "").split("->"))) for l in open(src + "/cell_dag.dot") if "->" in l]
    e = np.unique(np.array(ed), axis=0); eu, ev = e[:, 0], e[:, 1]
M = netid.max() + 1; sizes = np.bincount(netid)
E = len(eu)
succ_ptr = np.r_[0, np.cumsum(np.bincount(eu, minlength=n))]; so = np.argsort(eu, kind="stable"); succ_v = ev[so]
print(f"== {os.path.basename(src)}  n={n} nets={M} pins={len(pins)} dag_edges={E}", flush=True)

# ---------- unconstrained reference ----------
mtk = mk.initialize(8)
with tempfile.NamedTemporaryFile("w", suffix=".hgr", delete=False) as f:
    f.write(f"{M} {n}\n")
    order = np.argsort(netid, kind="stable"); ptr = np.r_[0, np.cumsum(sizes)]
    pp = pins[order] + 1
    for i in range(M): f.write(" ".join(map(str, pp[ptr[i]:ptr[i + 1]])) + "\n")
    hgr = f.name
ref = {}; kparts = {}
for eps in (0.02, 0.10):
    best = None
    for seed in range(3):
        mk.set_seed(seed)
        ctx = mtk.context_from_preset(mk.PresetType.QUALITY); ctx.set_partitioning_parameters(2, eps, mk.Objective.CUT); ctx.logging = False
        hg = mtk.hypergraph_from_file(hgr, ctx, mk.FileFormat.HMETIS); p = hg.partition(ctx)
        prt = np.array([p.block_id(v) for v in range(n)])
        kparts.setdefault(eps, []).append(prt)
        if best is None or p.cut() < best[0]:
            best = (p.cut(), prt)
    part = best[1]; f01 = np.sum((part[eu] == 0) & (part[ev] == 1)); f10 = np.sum((part[eu] == 1) & (part[ev] == 0))
    ref[eps] = best[0]
    print(f"  Mt-KaHyPar QUALITY eps{eps:.2f}: cut {best[0]}  (crossing DAG edges {f01} fwd / {f10} bwd)", flush=True)
os.unlink(hgr)

# ---------- topological machinery ----------
indeg0 = np.bincount(ev, minlength=n)
def kahn(x):
    indeg = indeg0.copy(); h = [(x[v], v) for v in np.flatnonzero(indeg == 0)]; heapq.heapify(h); out = []
    while h:
        _, u = heapq.heappop(h); out.append(u)
        for v in succ_v[succ_ptr[u]:succ_ptr[u + 1]]:
            indeg[v] -= 1
            if indeg[v] == 0: heapq.heappush(h, (x[v], v))
    assert len(out) == n, "DAG has a cycle"
    return np.array(out)
topo = kahn(np.zeros(n))
EPS = (0.02, 0.10)
def sweep(order):
    pos = np.empty(n, np.int64); pos[order] = np.arange(n); q = pos[pins]
    lo = np.full(M, n); hi = np.zeros(M, np.int64)
    np.minimum.at(lo, netid, q); np.maximum.at(hi, netid, q)
    cut = np.cumsum(np.bincount(lo + 1, minlength=n + 2)[:n + 2] - np.bincount(hi + 1, minlength=n + 2)[:n + 2])
    return [int(cut[a:b + 1].min()) for a, b in
            [(int(np.ceil(n * (1 - e) / 2)), int(np.floor(n * (1 + e) / 2))) for e in EPS]]
def show(name, r):
    print(f"  {name:<44} " + "  ".join(f"eps{e:.2f}:{c:>6} ({c / ref[e]:.2f}x)" for e, c in zip(EPS, r)), flush=True)

# ---------- spectral ----------
w = 1.0 / sizes
H = sp.csr_matrix((np.ones(len(pins)), (pins, netid)), shape=(n, M))
dv = np.asarray(H @ (w * 0 + 1) if False else H.sum(1)).ravel(); inv = 1 / np.sqrt(np.maximum(dv, 1))
mv = lambda x: inv * (H @ (w * (H.T @ (inv * x))))
vals, F = eigsh(LinearOperator((n, n), matvec=mv, dtype=float), k=9, which="LA", v0=rng.normal(size=n), tol=1e-6)
F = F[:, np.argsort(-vals)[1:]] * inv[:, None]
for j in range(8):
    if np.sum(F[ev, j] - F[eu, j]) < 0: F[:, j] *= -1
show("[not acyclic] Fiedler plain sweep", sweep(np.argsort(F[:, 0])))
show("Fiedler + Kahn-priority repair", sweep(kahn(F[:, 0])))

# ---------- envelopes ----------
X = (F - F.min(0)) / np.ptp(F, 0); delta = 1e-6
U = X.copy(); Lo = X.copy()
for u in topo:
    s = succ_v[succ_ptr[u]:succ_ptr[u + 1]]
    if len(s): U[s] = np.maximum(U[s], U[u] + delta)
for u in topo[::-1]:
    s = succ_v[succ_ptr[u]:succ_ptr[u + 1]]
    if len(s): Lo[u] = np.minimum(Lo[u], Lo[s].min(0) - delta)
Z = np.hstack([U, Lo]); Z = (Z - Z.min(0)) / np.ptp(Z, 0)
def topo_ok(o):
    pos = np.empty(n, np.int64); pos[o] = np.arange(n); return np.all(pos[eu] < pos[ev])
o = np.argsort(Z[:, 0], kind="stable"); assert topo_ok(o), "envelope not monotone"
show("best single U/L feature", min((sweep(np.argsort(Z[:, j], kind="stable")) for j in range(16)), key=lambda r: r[0]))
best = [10**9] * 2; curve = []
for i in range(NR):
    wv = rng.exponential(size=16) * (rng.random(16) < rng.uniform(0.2, 1))
    if wv.sum() == 0: continue
    r = sweep(np.argsort(Z @ wv, kind="stable")); best = [min(a, b) for a, b in zip(best, r)]
    if i + 1 in (100, NR // 2, NR): curve.append((i + 1, best[0]))
show(f"envelope nonneg combos x{NR} (his order mode)", best)
print(f"  convergence eps.02: {curve}   ({time.time() - t0:.0f}s)", flush=True)

# ---------- exact flow refinement (Lawler net model + infinite reverse DAG arcs = closure constraint) ----------
from scipy.sparse.csgraph import maximum_flow, breadth_first_order
INF = 10**6
def flow_cut(order, a, b):
    """Exact min hypergraph cut over predecessor-closed S with order[:a] in S and order[b:] not in S."""
    s, t = n + 2 * M, n + 2 * M + 1; N = t + 1
    ein, eout = n + netid, n + M + netid
    rows = [pins, eout, n + np.arange(M), ev, np.full(a, s), order[b:]]
    cols = [ein, pins, n + M + np.arange(M), eu, order[:a], np.full(n - b, t)]
    caps = [np.full(len(pins), INF), np.full(len(pins), INF), np.ones(M), np.full(E, INF), np.full(a, INF), np.full(n - b, INF)]
    G = sp.csr_matrix((np.concatenate(caps).astype(np.int32), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N))
    G.sum_duplicates()
    res = maximum_flow(G, s, t, method="dinic")
    R = (G - res.flow).tocsr(); R.data[R.data < 0] = 0; R.eliminate_zeros()
    # residual also contains reverse arcs with positive flow: flow matrix is antisymmetric, G - flow handles that
    Sset = breadth_first_order(R, s, directed=True, return_predecessors=False)
    inS = np.zeros(N, bool); inS[Sset] = True
    return res.flow_value, inS[:n]
def check(inS):
    assert not np.any(inS[ev] & ~inS[eu]), "not predecessor-closed"
    lo = np.ones(M, bool); hi = np.zeros(M, bool)
    np.logical_and.at(lo, netid, inS[pins]); np.logical_or.at(hi, netid, inS[pins])
    return int(np.sum(hi & ~lo)), int(inS.sum())
# start from the best envelope combo order found above (re-search, keep order)
bestw, bestc = [None, None], [10**9, 10**9]
rng = np.random.default_rng(0)
for i in range(NR):
    wv = rng.exponential(size=16) * (rng.random(16) < rng.uniform(0.2, 1))
    if wv.sum() == 0: continue
    c = sweep(np.argsort(Z @ wv, kind="stable"))
    for k in range(2):
        if c[k] < bestc[k]: bestc[k], bestw[k] = c[k], wv
for k, eps in enumerate(EPS):
    order0 = np.argsort(Z @ bestw[k], kind="stable")
    a, b = int(np.ceil(n * (1 - eps) / 2)), int(np.floor(n * (1 + eps) / 2))
    order = order0; hist = []
    for it in range(4):
        fv, inS = flow_cut(order, a, b); c, sz = check(inS); hist.append(c)
        # new topological order: S first, keep previous relative order (Kahn with priority (block, old pos))
        pos = np.empty(n); pos[order] = np.arange(n)
        order = kahn(pos + n * (~inS))
    print(f"  flow window refinement eps{eps:.2f} from sweep {bestc[k]}: cuts per iter {hist}  -> {min(hist)} ({min(hist) / ref[eps]:.2f}x)", flush=True)
# wider freedom: seeds = 30% / 30%, report cut and balance
for frac in (0.30, 0.15):
    a, b = int(frac * n), n - int(frac * n)
    fv, inS = flow_cut(np.argsort(Z @ bestw[1], kind='stable'), a, b); c, sz = check(inS)
    print(f"  flow, seeds {frac:.0%}/{frac:.0%}: cut {c}, |S|/n = {sz / n:.3f}  ({c / ref[0.10]:.2f}x of eps.10 ref)", flush=True)
print(f"  total {time.time() - t0:.0f}s")

# ---------- baseline: KaHyPar (unconstrained) + closure repair ----------
from scipy.sparse.csgraph import breadth_first_order as bfo
Adj = sp.csr_matrix((np.ones(E), (eu, ev)), shape=(n, n)); AdjT = Adj.T.tocsr()
def closure(seed, A):
    """all nodes reachable from seed set via A (multi-source BFS through a super node)."""
    k = len(seed)
    if k == 0: return np.zeros(n, bool)
    B = sp.vstack([sp.csr_matrix((np.ones(k), (np.zeros(k, int), seed)), shape=(1, n)), A]).tocsr()
    B = sp.hstack([sp.csr_matrix((n + 1, 1)), B]).tocsr()
    r = bfo(B, 0, directed=True, return_predecessors=False)
    m_ = np.zeros(n + 1, bool); m_[r] = True; return m_[1:]
for eps in EPS:
    lo_, hi_ = int(np.ceil(n * (1 - eps) / 2)), int(np.floor(n * (1 + eps) / 2))
    out = []
    for prt in kparts[eps]:
        for early in (0, 1):
            Sx = prt == early
            T1 = closure(np.flatnonzero(~Sx), Adj)          # later block + all its descendants
            S1 = ~T1
            S2 = closure(np.flatnonzero(Sx), AdjT)          # earlier block + all its ancestors
            for Sr in (S1, S2):
                c, sz = check(Sr); out.append((c, sz, lo_ <= sz <= hi_))
    feas = [o for o in out if o[2]]
    bestf = min(feas)[0] if feas else None
    print(f"  KaHyPar+closure repair eps{eps:.2f}: best balanced {bestf} "
          f"({bestf / ref[eps]:.2f}x)" if feas else f"  KaHyPar+closure repair eps{eps:.2f}: none balanced", 
          "| best any:", min(out)[:2], flush=True)
