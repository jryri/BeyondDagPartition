"""Quest (RESULTS §21): if a few back edges are allowed, how much cut is saved?  (cut vs back-edge trade-off)

One procedure, no portfolio: penalised objective  J = cut + lam * back,  back = #DAG edges from block 1 to block 0,
traced by CONTINUATION (homotopy) in lam: start at lam=0 from the unconstrained Mt-KaHyPar best (3 seeds, orientation
with fewer back edges), then raise lam step by step, each step warm-started from the previous solution and refined by
  penalised FM   : any cell may move; gain = cut gain - lam * (change in back edges)   (DAG-neighbour gains updated)
  penalised flow : same Lawler network as refine.flow_step, but each DAG edge gets FINITE capacity lam instead of INF,
                   so the min cut minimises nets cut + lam * back edges inside the balanced region.
lam = 0 is the unconstrained problem, lam -> inf the acyclic one; every step keeps balance.
Reported per lam: cut, back edges, feedback drivers (distinct tails of back edges), cut / ref, cut / T (acyclic best known).
Cost per step: FM O(passes * (pins * avg net size + m) log n), flow O(F) on the region graph; same order as refine().
Usage: python bdp/tradeoff.py <src> [eps]
"""
import sys, os, time, heapq, warnings
import numpy as np, scipy.sparse as sp
from scipy.sparse.csgraph import maximum_flow, breadth_first_order
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from refine import HG, load, bounds, grow, kahypar_parts, closure_variants, rebalance, refine, INF
from separability import SAVE
import mtkahypar as mk

LAMS = [0, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64]
SC = 8                                          # capacity scale so that lam = 1/8 is an integer capacity


def nback(h, p): return int(np.sum((p[h.eu] == 1) & (p[h.ev] == 0)))
def fdrv(h, p):
    k = (p[h.eu] == 1) & (p[h.ev] == 0); return int(np.unique(h.eu[k]).size)


def pfm_pass(h, part, eps, lam, max_bad=1000):
    n = h.n; lo, hi = bounds(n, eps)
    c = list(h.counts(part)); c[0] = c[0].copy(); c[1] = c[1].copy()
    f = np.where(part[h.pins] == 0, c[0][h.netid], c[1][h.netid])
    t = np.where(part[h.pins] == 0, c[1][h.netid], c[0][h.netid])
    gain = np.bincount(h.pins, weights=(f == 1).astype(int) - (t == 0).astype(int), minlength=n).astype(np.int64)
    s_in0 = np.bincount(h.eu, weights=(part[h.ev] == 0), minlength=n).astype(np.int64)   # successors in 0
    p_in1 = np.bincount(h.ev, weights=(part[h.eu] == 1), minlength=n).astype(np.int64)   # predecessors in 1
    locked = np.zeros(n, bool); stamp = np.zeros(n, np.int64); heaps = [[], []]

    def dback(v):   # change in #back edges if v switches side
        return (s_in0[v] - p_in1[v]) if part[v] == 0 else (p_in1[v] - s_in0[v])

    def push(v):
        stamp[v] += 1; heapq.heappush(heaps[part[v]], (-(gain[v] - lam * dback(v)), stamp[v], v))

    for v in range(n): push(v)
    size0 = int(np.sum(part == 0)); cut = int(np.sum((c[0] > 0) & (c[1] > 0))); back = nback(h, part)
    feas = lambda s0: lo <= s0 <= hi
    best = (cut + lam * back if feas(size0) else np.inf, 0); log = []; bad = 0

    def top(p):
        H_ = heaps[p]
        while H_:
            _, st, v = H_[0]
            if locked[v] or st != stamp[v] or part[v] != p: heapq.heappop(H_); continue
            return v
        return None

    while True:
        cand = []
        for p in (0, 1):
            ns0 = size0 - 1 if p == 0 else size0 + 1
            if feas(ns0) or (not feas(size0) and abs(ns0 - n / 2) < abs(size0 - n / 2)):
                v = top(p)
                if v is not None: cand.append((gain[v] - lam * dback(v), p, v))
        if not cand: break
        _, p, v = max(cand); heapq.heappop(heaps[p]); q = 1 - p
        g = int(gain[v]); db = int(dback(v))
        for e in h.nets(v):
            P = h.npins(e)
            if c[q][e] == 0:
                for u in P:
                    if not locked[u] and u != v: gain[u] += 1; push(u)
            elif c[q][e] == 1:
                for u in P:
                    if part[u] == q and not locked[u]: gain[u] -= 1; push(u)
            c[p][e] -= 1; c[q][e] += 1
            if c[p][e] == 0:
                for u in P:
                    if not locked[u] and u != v: gain[u] -= 1; push(u)
            elif c[p][e] == 1:
                for u in P:
                    if part[u] == p and u != v and not locked[u]: gain[u] += 1; push(u)
        part[v] = q; locked[v] = True; cut -= g; back += db; size0 += -1 if p == 0 else 1
        if p == 0:
            for u in h.pred(v): s_in0[u] -= 1
            for w in h.succ(v): p_in1[w] += 1
        else:
            for w in h.succ(v): p_in1[w] -= 1
            for u in h.pred(v): s_in0[u] += 1
        for w in np.r_[h.pred(v), h.succ(v)]:
            if not locked[w]: push(w)
        log.append(v)
        J = cut + lam * back
        if feas(size0) and J < best[0]: best = (J, len(log)); bad = 0
        else: bad += 1
        if bad >= max_bad: break
    for v in log[best[1]:][::-1]: part[v] ^= 1
    return h.cut(part) + lam * nback(h, part)


def pflow_step(h, part, eps, lam):
    n, M = h.n, h.M; lo, hi = bounds(n, eps); s0 = int(np.sum(part == 0))
    c0, c1 = h.counts(part); cutnet = (c0 > 0) & (c1 > 0)
    boundary = np.zeros(n, bool); boundary[h.pins[cutnet[h.netid]]] = True
    db = np.zeros(n, bool); k = (part[h.eu] == 1) & (part[h.ev] == 0); db[h.eu[k]] = True; db[h.ev[k]] = True
    boundary |= db                                             # back-edge endpoints are also boundary
    B0 = grow(h, part, 0, max(0, s0 - lo), boundary); B1 = grow(h, part, 1, max(0, hi - s0), boundary)
    seed0 = (part == 0) & ~B0; seed1 = (part == 1) & ~B1
    s, t = n + 2 * M, n + 2 * M + 1; N = t + 1
    ein, eout = n + h.netid, n + M + h.netid
    a0, a1 = np.flatnonzero(seed0), np.flatnonzero(seed1); dc = int(round(lam * SC))
    rows = [h.pins, eout, n + np.arange(M), np.full(len(a0), s), a1]
    cols = [ein, h.pins, n + M + np.arange(M), a0, np.full(len(a1), t)]
    caps = [np.full(len(h.pins), INF), np.full(len(h.pins), INF), np.full(M, SC), np.full(len(a0), INF), np.full(len(a1), INF)]
    if dc > 0: rows.append(h.ev); cols.append(h.eu); caps.append(np.full(len(h.eu), dc))   # back edge v(0) <- u(1) costs lam
    G = sp.csr_matrix((np.concatenate(caps).astype(np.int32), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N))
    G.sum_duplicates()
    res = maximum_flow(G, s, t, method="dinic")
    R = (G - res.flow).tocsr(); R.data[R.data < 0] = 0; R.eliminate_zeros()
    A = breadth_first_order(R, s, directed=True, return_predecessors=False)
    Bt = breadth_first_order(R.T.tocsr(), t, directed=True, return_predecessors=False)
    inA = np.zeros(N, bool); inA[A] = True; inB = np.zeros(N, bool); inB[Bt] = True
    best = None
    for pc in (np.where(inA[:n], 0, 1).astype(np.int8), np.where(inB[:n], 1, 0).astype(np.int8)):
        if lo <= np.sum(pc == 0) <= hi:
            J = h.cut(pc) + lam * nback(h, pc)
            if best is None or J < best[0]: best = (J, pc)
    return best


def prefine(h, part, eps, lam, rounds=4):
    J = h.cut(part) + lam * nback(h, part)
    for _ in range(rounds):
        J0 = J
        for _ in range(10):
            nj = pfm_pass(h, part, eps, lam)
            if nj >= J: J = nj; break
            J = nj
        r = pflow_step(h, part, eps, lam)
        if r is not None and r[0] < J: part[:] = r[1]; J = r[0]
        if J >= J0: break
    return J


def main(src, eps):
    name = os.path.basename(src.rstrip("/")).replace("_hg.npz", "")
    h = HG(*load(src)); lo, hi = bounds(h.n, eps); mtk = mk.initialize(8)
    T = np.load(os.path.join(SAVE, f"{name}_e{int(eps * 100):02d}.npy")); tc = h.cut(T)
    t0 = time.time(); kp = kahypar_parts(h, eps, mtk, seeds=3); ref, p = min(kp, key=lambda x: x[0])
    p = p.astype(np.int8)
    if nback(h, 1 - p) < nback(h, p): p = (1 - p).astype(np.int8)
    print(f"== {name} n={h.n} dag_edges={len(h.eu)} ref(unconstrained) {ref} | acyclic best known T {tc} ({tc / ref:.2f}x ref)", flush=True)
    print(f"   start (Mt-KaHyPar): cut {h.cut(p)} back {nback(h, p)} drivers {fdrv(h, p)}", flush=True)
    for lam in LAMS:
        t = time.time(); prefine(h, p, eps, lam); assert lo <= np.sum(p == 0) <= hi
        c, b = h.cut(p), nback(h, p)
        print(f"   lam={lam:<6}: cut {c:>5} ({c / ref:.2f}x ref, {c / tc:.2f}x T) | back {b:>5} | drivers {fdrv(h, p):>4} "
              f"| acyclic {h.closed(p)}  [{time.time() - t:.0f}s]", flush=True)
    # endpoint: the continuation can stall with a few back edges (single-cell moves cannot remove a chain);
    # close it with the existing repair: lowest-cut balanced closure variant -> acyclic refine
    if not h.closed(p):
        t = time.time(); st = []
        for v in closure_variants(h, p):
            if rebalance(h, v, eps): st.append((h.cut(v), v))
        if st:
            c0, v = min(st, key=lambda x: x[0]); v = v.copy(); r = refine(h, v, eps)[-1][1]
            assert h.closed(v) and lo <= np.sum(v == 0) <= hi
            print(f"   repair -> acyclic: cut {c0} -> refined {r} ({r / ref:.2f}x ref, {r / tc:.2f}x T) | back 0  [{time.time() - t:.0f}s]", flush=True)
    print(f"   [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 0.10)
