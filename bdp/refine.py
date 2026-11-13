"""Quest: does acyclic refinement close the gap?  (K=2, cut = #nets cut, unit weights)

Starts:
  (i)  his-style decoder: spectral U/L envelopes, best of NR nonneg combos + exact sweep
  (ii) Mt-KaHyPar (unconstrained) + closure repair (+ forced rebalance)
Refinement:
  FM   : 2-way FM, moves restricted to keep the partition predecessor-closed (Popp-style counters:
         v in block 0 movable iff no successor in 0; v in 1 movable iff no predecessor in 1);
         gain-bucket replaced by lazy heaps, rollback to best balanced prefix.
  FLOW : Lawler net model + infinite reverse arc per DAG edge (closure constraint); region around the cut
         limited so that ANY cut inside it stays balanced (KaHyPar-MF style); nodes outside region are seeds.
Reference: Mt-KaHyPar QUALITY unconstrained (best of 3 seeds).
Usage: python bdp/refine.py <dir with cells.hgr+cell_dag.dot | data/x_hg.npz> [NR]
"""
import sys, os, heapq, time, tempfile
import numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import eigsh, LinearOperator
from scipy.sparse.csgraph import maximum_flow, breadth_first_order
import mtkahypar as mk

INF = 10**6


# ---------------------------------------------------------------- loading
def load(src):
    if src.endswith(".npz"):
        z = np.load(src)
        net, node, drv, types = z["net"], z["node"], z["drv"], z["types"]
        n = len(types)
        seq = np.array([("DFF" in t) or ("LATCH" in t) or t.startswith("$_SR") or not t.startswith("$_") for t in types])
        o = np.lexsort((node, net)); net, node, drv = net[o], node[o], drv[o]
        k = np.r_[True, (net[1:] != net[:-1]) | (node[1:] != node[:-1])]
        net, node, drv = net[k], node[k], drv[k]
        ok = np.bincount(net)[net] >= 2
        _, netid = np.unique(net[ok], return_inverse=True); pins = node[ok]
        dn, dg = net[drv], node[drv]; sk = (~drv) & (~seq[node]); sn, sg = net[sk], node[sk]
        oo = np.argsort(dn, kind="stable"); dn, dg = dn[oo], dg[oo]
        lo = np.searchsorted(dn, sn, "left"); hi = np.searchsorted(dn, sn, "right")
        a, b = [dg[lo[hi - lo == 1]]], [sg[hi - lo == 1]]
        for i in np.where(hi - lo > 1)[0]:
            a.append(dg[lo[i]:hi[i]]); b.append(np.full(hi[i] - lo[i], sg[i]))
        e = np.unique(np.stack([np.concatenate(a), np.concatenate(b)], 1), axis=0)
    else:
        L = [l.split() for l in open(src + "/cells.hgr")]
        m_, n = map(int, L[0][:2])
        nl = [np.array(sorted(set(int(x) - 1 for x in l))) for l in L[1:m_ + 1]]; nl = [p for p in nl if len(p) >= 2]
        pins = np.concatenate(nl); netid = np.repeat(np.arange(len(nl)), [len(p) for p in nl])
        e = np.unique(np.array([tuple(map(int, l.replace(";", "").split("->")))
                                for l in open(src + "/cell_dag.dot") if "->" in l]), axis=0)
    e = e[e[:, 0] != e[:, 1]]
    return n, pins.astype(np.int64), netid.astype(np.int64), e[:, 0].astype(np.int64), e[:, 1].astype(np.int64)


class HG:
    def __init__(s, n, pins, netid, eu, ev):
        s.n, s.pins, s.netid, s.eu, s.ev = n, pins, netid, eu, ev
        s.M = int(netid.max()) + 1
        o = np.argsort(netid, kind="stable"); s.np_ = pins[o]; s.nptr = np.r_[0, np.cumsum(np.bincount(netid, minlength=s.M))]
        o = np.argsort(pins, kind="stable"); s.vn = netid[o]; s.vptr = np.r_[0, np.cumsum(np.bincount(pins, minlength=n))]
        o = np.argsort(eu, kind="stable"); s.sv = ev[o]; s.sptr = np.r_[0, np.cumsum(np.bincount(eu, minlength=n))]
        o = np.argsort(ev, kind="stable"); s.pv = eu[o]; s.pptr = np.r_[0, np.cumsum(np.bincount(ev, minlength=n))]
        s.H = sp.csr_matrix((np.ones(len(pins)), (pins, netid)), shape=(n, s.M))
        sz = np.bincount(netid); s.Hs = s.H[:, sz <= 100].tocsr()

    def counts(s, part):
        c1 = np.bincount(s.netid, weights=part[s.pins], minlength=s.M).astype(np.int64)
        c0 = np.bincount(s.netid, minlength=s.M) - c1
        return c0, c1

    def cut(s, part):
        c0, c1 = s.counts(part); return int(np.sum((c0 > 0) & (c1 > 0)))

    def closed(s, part):  # block 0 = earlier = predecessor-closed  <=> no DAG edge 1 -> 0
        return not np.any((part[s.eu] == 1) & (part[s.ev] == 0))

    def succ(s, v): return s.sv[s.sptr[v]:s.sptr[v + 1]]
    def pred(s, v): return s.pv[s.pptr[v]:s.pptr[v + 1]]
    def nets(s, v): return s.vn[s.vptr[v]:s.vptr[v + 1]]
    def npins(s, e): return s.np_[s.nptr[e]:s.nptr[e + 1]]


def bounds(n, eps):
    return int(np.ceil(n * (1 - eps) / 2)), int(np.floor(n * (1 + eps) / 2))  # allowed |block 0|


# ---------------------------------------------------------------- acyclic FM
def fm_pass(h, part, eps, max_bad=1000, rebalance=False):
    """One FM pass (or forced rebalance). part: int8 array, modified in place. Returns new cut."""
    n = h.n; lo, hi = bounds(n, eps)
    c = list(h.counts(part)); c[0] = c[0].copy(); c[1] = c[1].copy()
    # gains: moving v from p to 1-p
    f = np.where(part[h.pins] == 0, c[0][h.netid], c[1][h.netid])
    t = np.where(part[h.pins] == 0, c[1][h.netid], c[0][h.netid])
    gain = np.bincount(h.pins, weights=(f == 1).astype(int) - (t == 0).astype(int), minlength=n).astype(np.int64)
    s_in0 = np.bincount(h.eu, weights=(part[h.ev] == 0), minlength=n).astype(np.int64)  # successors in 0
    p_in1 = np.bincount(h.ev, weights=(part[h.eu] == 1), minlength=n).astype(np.int64)  # predecessors in 1
    locked = np.zeros(n, bool); stamp = np.zeros(n, np.int64)
    heaps = [[], []]  # heaps[p]: candidates currently in block p

    def movable(v):
        return (s_in0[v] == 0) if part[v] == 0 else (p_in1[v] == 0)

    def push(v):
        stamp[v] += 1; heapq.heappush(heaps[part[v]], (-gain[v], stamp[v], v))

    for v in np.flatnonzero((part == 0) & (s_in0 == 0)): push(v)
    for v in np.flatnonzero((part == 1) & (p_in1 == 0)): push(v)
    size0 = int(np.sum(part == 0)); cut = int(np.sum((c[0] > 0) & (c[1] > 0)))
    feas = lambda s0: lo <= s0 <= hi
    best = (cut if feas(size0) else 10**9, 0); log = []; bad = 0

    def top(p):
        H_ = heaps[p]
        while H_:
            g, st, v = H_[0]
            if locked[v] or st != stamp[v] or part[v] != p or not movable(v): heapq.heappop(H_); continue
            return v
        return None

    while True:
        cand = []
        for p in (0, 1):
            ns0 = size0 - 1 if p == 0 else size0 + 1
            if rebalance:
                ok = (size0 > hi and p == 0) or (size0 < lo and p == 1)
            else:
                ok = feas(ns0) or (not feas(size0) and abs(ns0 - n / 2) < abs(size0 - n / 2))
            if ok:
                v = top(p)
                if v is not None: cand.append((gain[v], p, v))
        if not cand: break
        g, p, v = max(cand)
        heapq.heappop(heaps[p]); q = 1 - p
        # --- apply move v: p -> q, FM gain updates
        for e in h.nets(v):
            P = h.npins(e)
            if c[q][e] == 0:
                for u in P:
                    if not locked[u] and u != v: gain[u] += 1; push(u) if movable(u) else None
            elif c[q][e] == 1:
                for u in P:
                    if part[u] == q and not locked[u]: gain[u] -= 1; push(u) if movable(u) else None
            c[p][e] -= 1; c[q][e] += 1
            if c[p][e] == 0:
                for u in P:
                    if not locked[u] and u != v: gain[u] -= 1; push(u) if movable(u) else None
            elif c[p][e] == 1:
                for u in P:
                    if part[u] == p and u != v and not locked[u]: gain[u] += 1; push(u) if movable(u) else None
        part[v] = q; locked[v] = True; cut -= g; size0 += -1 if p == 0 else 1
        if p == 0:   # v went 0 -> 1
            for u in h.pred(v):
                s_in0[u] -= 1
                if s_in0[u] == 0 and part[u] == 0 and not locked[u]: push(u)
            for w in h.succ(v): p_in1[w] += 1
        else:        # v went 1 -> 0
            for w in h.succ(v):
                p_in1[w] -= 1
                if p_in1[w] == 0 and part[w] == 1 and not locked[w]: push(w)
            for u in h.pred(v): s_in0[u] += 1
        log.append(v)
        if feas(size0) and cut < best[0]:
            best = (cut, len(log)); bad = 0
        else:
            bad += 1
        if rebalance and feas(size0):
            return cut
        if bad >= max_bad: break
    for v in log[best[1]:][::-1]: part[v] ^= 1   # rollback (moves keep closure in reverse order too)
    return best[0] if best[0] < 10**9 else cut


def fm(h, part, eps, passes=10):
    cut = h.cut(part)
    for _ in range(passes):
        nc = fm_pass(h, part, eps)
        if nc >= cut: cut = nc; break
        cut = nc
    return h.cut(part)


def rebalance(h, part, eps):
    lo, hi = bounds(h.n, eps)
    for _ in range(50):
        s0 = int(np.sum(part == 0))
        if lo <= s0 <= hi: return True
        fm_pass(h, part, eps, rebalance=True)
    s0 = int(np.sum(part == 0)); return lo <= s0 <= hi


# ---------------------------------------------------------------- acyclic flow refinement
def grow(h, part, side, cap, boundary):
    """BFS (through nets <=100 pins) from boundary nodes of `side`, inside that side, up to cap nodes."""
    inside = part == side
    reached = np.zeros(h.n, bool); frontier = boundary & inside
    order = []
    while frontier.any() and len(order) < cap:
        idx = np.flatnonzero(frontier)
        take = idx[:cap - len(order)]; order.extend(take); reached[take] = True
        nxt = (h.Hs @ (h.Hs.T @ frontier.astype(float)) > 0) & inside & ~reached
        frontier = nxt
    m = np.zeros(h.n, bool); m[np.array(order, dtype=np.int64)] = True; return m


def flow_step(h, part, eps, alpha=1.0):
    n, M = h.n, h.M; lo, hi = bounds(n, eps)
    s0 = int(np.sum(part == 0)); s1 = n - s0
    c0, c1 = h.counts(part); cutnet = (c0 > 0) & (c1 > 0)
    boundary = np.zeros(n, bool); boundary[h.pins[cutnet[h.netid]]] = True
    cap0 = int(alpha * (hi - s0 + (s0 - lo)) ) if False else None
    # region sizes such that any cut inside is balanced: B0 can move to side 1 -> |S'| >= s0-|B0| >= lo
    capB0 = max(0, s0 - lo); capB1 = max(0, hi - s0)
    B0 = grow(h, part, 0, capB0, boundary); B1 = grow(h, part, 1, capB1, boundary)
    seed0 = (part == 0) & ~B0; seed1 = (part == 1) & ~B1
    s, t = n + 2 * M, n + 2 * M + 1; N = t + 1
    ein, eout = n + h.netid, n + M + h.netid
    a0, a1 = np.flatnonzero(seed0), np.flatnonzero(seed1)
    rows = [h.pins, eout, n + np.arange(M), h.ev, np.full(len(a0), s), a1]
    cols = [ein, h.pins, n + M + np.arange(M), h.eu, a0, np.full(len(a1), t)]
    caps = [np.full(len(h.pins), INF), np.full(len(h.pins), INF), np.ones(M), np.full(len(h.eu), INF),
            np.full(len(a0), INF), np.full(len(a1), INF)]
    G = sp.csr_matrix((np.concatenate(caps).astype(np.int32), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N))
    G.sum_duplicates()
    res = maximum_flow(G, s, t, method="dinic")
    R = (G - res.flow).tocsr(); R.data[R.data < 0] = 0; R.eliminate_zeros()
    A = breadth_first_order(R, s, directed=True, return_predecessors=False)
    Bt = breadth_first_order(R.T.tocsr(), t, directed=True, return_predecessors=False)
    inA = np.zeros(N, bool); inA[A] = True; inB = np.zeros(N, bool); inB[Bt] = True
    cands = [np.where(inA[:n], 0, 1).astype(np.int8), np.where(inB[:n], 1, 0).astype(np.int8)]
    best = None
    for pc in cands:
        sz = int(np.sum(pc == 0))
        if lo <= sz <= hi and h.closed(pc):
            c = h.cut(pc)
            key = (c, abs(sz - n / 2))
            if best is None or key < best[0]: best = (key, pc)
    return (res.flow_value, best[1]) if best else (res.flow_value, None)


def flow(h, part, eps, iters=8):
    cut = h.cut(part)
    for _ in range(iters):
        fv, pc = flow_step(h, part, eps)
        if pc is None: break
        nc = h.cut(pc)
        if nc < cut or (nc == cut and abs(np.sum(pc == 0) - h.n / 2) < abs(np.sum(part == 0) - h.n / 2) and nc < cut):
            part[:] = pc; cut = nc
        else:
            break
    return cut


def refine(h, part, eps, rounds=3):
    hist = [("start", h.cut(part))]
    for r in range(rounds):
        before = h.cut(part)
        hist.append(("fm", fm(h, part, eps)))
        hist.append(("flow", flow(h, part, eps)))
        if h.cut(part) >= before: break
    return hist


# ---------------------------------------------------------------- starts
def kahn(h, x):
    indeg = np.bincount(h.ev, minlength=h.n); q = [(x[v], v) for v in np.flatnonzero(indeg == 0)]
    heapq.heapify(q); out = []
    while q:
        _, u = heapq.heappop(q); out.append(u)
        for v in h.succ(u):
            indeg[v] -= 1
            if indeg[v] == 0: heapq.heappush(q, (x[v], v))
    return np.array(out)


def sweep_part(h, order, eps):
    n = h.n; pos = np.empty(n, np.int64); pos[order] = np.arange(n); q = pos[h.pins]
    lo_ = np.full(h.M, n); hi_ = np.zeros(h.M, np.int64)
    np.minimum.at(lo_, h.netid, q); np.maximum.at(hi_, h.netid, q)
    cut = np.cumsum(np.bincount(lo_ + 1, minlength=n + 2)[:n + 2] - np.bincount(hi_ + 1, minlength=n + 2)[:n + 2])
    a, b = bounds(n, eps); s = a + int(np.argmin(cut[a:b + 1]))
    part = np.ones(n, np.int8); part[order[:s]] = 0
    return int(cut[s]), part


def his_embed(h, rng):
    """His order-mode features: 8 spectral coords (X) and their forward-max / backward-min envelopes Z=[U,L].
    Every column of Z strictly increases along DAG edges, so any w>=0 (nonzero) gives a topological order."""
    n = h.n; sizes = np.bincount(h.netid); w = 1.0 / sizes
    dv = np.asarray(h.H.sum(1)).ravel(); inv = 1 / np.sqrt(np.maximum(dv, 1))
    mv = lambda x: inv * (h.H @ (w * (h.H.T @ (inv * x))))
    vals, F = eigsh(LinearOperator((n, n), matvec=mv, dtype=float), k=9, which="LA", v0=rng.normal(size=n), tol=1e-6)
    F = F[:, np.argsort(-vals)[1:]] * inv[:, None]
    for j in range(8):
        if np.sum(F[h.ev, j] - F[h.eu, j]) < 0: F[:, j] *= -1
    X = (F - F.min(0)) / np.ptp(F, 0); topo = kahn(h, np.zeros(n)); d = 1e-6
    U = X.copy(); Lo = X.copy()
    for u in topo:
        s_ = h.succ(u)
        if len(s_): U[s_] = np.maximum(U[s_], U[u] + d)
    for u in topo[::-1]:
        s_ = h.succ(u)
        if len(s_): Lo[u] = np.minimum(Lo[u], Lo[s_].min(0) - d)
    Z = np.hstack([U, Lo]); Z = (Z - Z.min(0)) / np.ptp(Z, 0)
    return X, Z


def his_start(h, eps, NR, rng):
    X, Z = his_embed(h, rng)
    best = None
    for i in range(NR):
        wv = rng.exponential(size=16) * (rng.random(16) < rng.uniform(0.2, 1))
        if wv.sum() == 0: continue
        c, p = sweep_part(h, np.argsort(Z @ wv, kind="stable"), eps)
        if best is None or c < best[0]: best = (c, p)
    return best[1]


def kahypar_parts(h, eps, mtk, seeds=3):
    f = tempfile.NamedTemporaryFile("w", suffix=".hgr", delete=False)
    f.write(f"{h.M} {h.n}\n")
    for e in range(h.M): f.write(" ".join(str(x + 1) for x in h.npins(e)) + "\n")
    f.close(); out = []
    for sd in range(seeds):
        mk.set_seed(sd)
        ctx = mtk.context_from_preset(mk.PresetType.QUALITY); ctx.set_partitioning_parameters(2, eps, mk.Objective.CUT); ctx.logging = False
        hg = mtk.hypergraph_from_file(f.name, ctx, mk.FileFormat.HMETIS); p = hg.partition(ctx)
        out.append((p.cut(), np.array([p.block_id(v) for v in range(h.n)], np.int8)))
    os.unlink(f.name); return out


def closure_variants(h, prt):
    A = sp.csr_matrix((np.ones(len(h.eu)), (h.eu, h.ev)), shape=(h.n, h.n)); AT = A.T.tocsr()
    def reach(mask, Adj):
        seed = np.flatnonzero(mask); k = len(seed)
        if k == 0: return np.zeros(h.n, bool)
        B = sp.vstack([sp.csr_matrix((np.ones(k), (np.zeros(k, int), seed)), shape=(1, h.n)), Adj]).tocsr()
        B = sp.hstack([sp.csr_matrix((h.n + 1, 1)), B]).tocsr()
        r = breadth_first_order(B, 0, directed=True, return_predecessors=False); m = np.zeros(h.n + 1, bool); m[r] = True
        return m[1:]
    out = []
    for early in (0, 1):
        S = prt == early
        T1 = reach(~S, A); out.append(np.where(T1, 1, 0).astype(np.int8))       # later block + descendants
        S2 = reach(S, AT); out.append(np.where(S2, 0, 1).astype(np.int8))       # earlier block + ancestors
    return out


# ---------------------------------------------------------------- main
def main(src, NR):
    t0 = time.time(); rng = np.random.default_rng(0)
    h = HG(*load(src)); mtk = mk.initialize(8)
    print(f"\n== {os.path.basename(src.rstrip('/'))}  n={h.n} nets={h.M} pins={len(h.pins)} dag_edges={len(h.eu)}", flush=True)
    for eps in (0.02, 0.10):
        lo, hi = bounds(h.n, eps)
        kp = kahypar_parts(h, eps, mtk); ref = min(c for c, _ in kp)
        print(f" eps {eps:.2f}: Mt-KaHyPar unconstrained ref cut {ref}", flush=True)
        def show(name, part, hist, t):
            assert h.closed(part), "not acyclic!"; s0 = int(np.sum(part == 0)); assert lo <= s0 <= hi, "unbalanced!"
            assert h.cut(part) == hist[-1][1]
            steps = " -> ".join(f"{k}:{c}" for k, c in hist)
            print(f"   {name:<26} {steps}   final {hist[-1][1]} ({hist[-1][1] / ref:.2f}x)  [{t:.0f}s]", flush=True)
        # (i) his-style start
        t1 = time.time(); part = his_start(h, eps, NR, rng)
        hist = refine(h, part, eps); show("his decoder + refine", part, hist, time.time() - t1)
        # (ii) KaHyPar + closure repair (+ rebalance), all variants, keep best start
        t1 = time.time(); starts = []
        for c, prt in kp:
            for v in closure_variants(h, prt):
                if rebalance(h, v, eps): starts.append((h.cut(v), v))
        starts.sort(key=lambda x: x[0])
        best = None
        for c, v in starts[:3]:
            p = v.copy(); hist = refine(h, p, eps)
            if best is None or hist[-1][1] < best[0][-1][1]: best = (hist, p)
        if best: show("KaHyPar+repair + refine", best[1], best[0], time.time() - t1)
        else: print("   KaHyPar+repair: no balanced start", flush=True)
    print(f" total {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    NR = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    main(sys.argv[1], NR)
