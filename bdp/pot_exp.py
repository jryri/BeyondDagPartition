"""Which potential gives the best acyclic balanced cut under the same exact sweep?

Protocol (identical for every candidate): potential x -> topological order by Kahn's algorithm with priority x
(for monotone x this is just argsort) -> prefix sweep, net cut iff min pos < s <= max pos ->
best cut with prefix size in the balance window. Every candidate is acyclic by construction; no FM.
"""
import sys, heapq, time
import numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import eigsh, LinearOperator, splu

d = sys.argv[1]
rng = np.random.default_rng(0)
L = [l.split() for l in open(d + "/cells.hgr")]
m, n = map(int, L[0][:2])
nets = [np.array(sorted(set(int(x) - 1 for x in l))) for l in L[1:m + 1]]
nets = [p for p in nets if len(p) >= 2]
succ = [set() for _ in range(n)]
for l in open(d + "/cell_dag.dot"):
    if "->" in l:
        a, b = l.replace(";", "").split("->"); succ[int(a)].add(int(b))
eu = np.array([u for u, s in enumerate(succ) for v in s]); ev = np.array([v for s in succ for v in s])
E = len(eu)
sizes = np.array([len(p) for p in nets]); pins = np.concatenate(nets); netid = np.repeat(np.arange(len(nets)), sizes)
print(f"n={n} nets={len(nets)} dag_edges={E}")

# ---------- exact sweep ----------
EPS = (0.02, 0.10)
def topo_order(x):
    o = np.argsort(x, kind="stable"); pos = np.empty(n, int); pos[o] = np.arange(n)
    if np.all(pos[eu] < pos[ev]):
        return o
    indeg = np.bincount(ev, minlength=n)
    h = [(x[v], v) for v in np.flatnonzero(indeg == 0)]; heapq.heapify(h); out = []
    while h:
        _, u = heapq.heappop(h); out.append(u)
        for v in succ[u]:
            indeg[v] -= 1
            if indeg[v] == 0: heapq.heappush(h, (x[v], v))
    return np.array(out)

def sweep(order):
    pos = np.empty(n, int); pos[order] = np.arange(n)
    pp = pos[pins]
    lo = np.full(len(nets), n); hi = np.zeros(len(nets), int)
    np.minimum.at(lo, netid, pp); np.maximum.at(hi, netid, pp)
    diff = np.zeros(n + 2); np.add.at(diff, lo + 1, 1); np.add.at(diff, hi + 1, -1)
    cut = np.cumsum(diff)  # cut[s] = nets cut when prefix = first s nodes
    res = []
    for e in EPS:
        a, b = int(np.ceil(n * (1 - e) / 2)), int(np.floor(n * (1 + e) / 2))
        s = a + np.argmin(cut[a:b + 1]); res.append(int(cut[s]))
    return res

def evalx(x):
    return sweep(topo_order(x))

results = {}
def report(name, r, extra=""):
    results[name] = r
    print(f"  {name:<38} " + "  ".join(f"eps{e:.2f}:{c:>5}" for e, c in zip(EPS, r)) + f"  {extra}", flush=True)

# ---------- hypergraph Laplacian (clique expansion, 1/(|e|-1)) ----------
w = 1.0 / (sizes - 1)
H = sp.csr_matrix((np.repeat(w, sizes) ** 0.5, (pins, netid)), shape=(n, len(nets)))
W = (H @ H.T).tocsr(); W.setdiag(0); W.eliminate_zeros()
deg = np.asarray(W.sum(1)).ravel()
LH = sp.diags(deg) - W
Dm12 = sp.diags(1 / np.sqrt(deg))

def fiedler(M, k=2):
    """smallest nontrivial eigvecs of D^-1/2 M D^-1/2 via shift-invert."""
    N = (Dm12 @ M @ Dm12).tocsc()
    lu = splu((N + 1e-6 * sp.identity(n)).tocsc())
    op = LinearOperator((n, n), matvec=lu.solve, dtype=float)
    vals, V = eigsh(op, k=k + 1, which="LM", v0=rng.normal(size=n), tol=1e-8)
    V = V[:, np.argsort(-vals)]  # largest of inverse = smallest of N
    return Dm12 @ V[:, 1:], 1 / np.sort(vals)[::-1][1:] - 1e-6

def orient(x):
    return x if np.sum(x[ev] - x[eu]) >= 0 else -x

t0 = time.time()
# ---------- (a) topology only ----------
indeg = np.bincount(ev, minlength=n); topo = []; q = list(np.flatnonzero(indeg == 0)); deg_in = indeg.copy()
for u in q:
    topo.append(u)
    for v in succ[u]:
        deg_in[v] -= 1
        if deg_in[v] == 0: q.append(v)
asap = np.zeros(n)
for u in topo:
    for v in succ[u]: asap[v] = max(asap[v], asap[u] + 1)
blev = np.zeros(n)
for u in reversed(topo):
    for v in succ[u]: blev[u] = max(blev[u], blev[v] + 1)
alap = blev.max() - blev
print("(a) topology only")
report("ASAP level", evalx(asap + 1e-9 * rng.random(n)))
report("ALAP level", evalx(alap + 1e-9 * rng.random(n)))
report("mid (ASAP+ALAP)/2", evalx((asap + alap) / 2 + 1e-9 * rng.random(n)))
report("random topological order", evalx(rng.random(n)))

# ---------- (b) undirected spectral, repaired by Kahn priority ----------
F, lam = fiedler(LH, k=8)
print(f"(b) undirected spectral (lambda2={lam[0]:.2e}) -> Kahn-priority repair")
best = min((evalx(s * F[:, j]) for j in range(8) for s in (1, -1)), key=lambda r: r[0])
report("Fiedler (oriented)", evalx(orient(F[:, 0])))
report("best of 8 coords x 2 signs", best)
# price of acyclicity: same Fiedler sorted WITHOUT topological constraint
o = np.argsort(F[:, 0]); report("[reference] Fiedler sweep, NOT acyclic", sweep(o))

# ---------- (c) his envelope features, nonneg combinations ----------
X = np.empty((n, 8))
for j in range(8):
    X[:, j] = orient(F[:, j])
X = (X - X.min(0)) / np.ptp(X, 0)
delta = 1e-4 * X.std(0); U = X.copy(); Lo = X.copy()
for u in topo:
    for v in succ[u]: np.maximum(U[v], U[u] + delta, out=U[v])
for u in reversed(topo):
    for v in succ[u]: np.minimum(Lo[u], Lo[v] - delta, out=Lo[u])
Z = np.hstack([U, Lo]); Z = (Z - Z.min(0)) / np.ptp(Z, 0)
print("(c) envelope features U/L (his order mode), exact sweep")
single = [evalx(Z[:, j]) for j in range(16)]
report("best single U/L feature", min(single, key=lambda r: r[0]))
NR = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
bestc, curve = [10**9, 10**9], []
for i in range(NR):
    wv = rng.exponential(size=16) * (rng.random(16) < rng.uniform(0.2, 1))
    if wv.sum() == 0: continue
    r = sweep(np.argsort(Z @ wv, kind="stable"))
    bestc = [min(a, b) for a, b in zip(bestc, r)]
    if i + 1 in (100, 500, NR): curve.append((i + 1, bestc[0]))
report(f"best of {NR} random nonneg combos", bestc, "curve " + str(curve))

# ---------- (d) Pasadakis et al. 2025: L_dag - c d d^T, c = 1/(2|E|) ----------
A = sp.csr_matrix((np.ones(E), (eu, ev)), shape=(n, n)); A = ((A + A.T) > 0).astype(float)
Ld = sp.diags(np.asarray(A.sum(1)).ravel()) - A
dvec = np.bincount(eu, minlength=n) - np.bincount(ev, minlength=n); c = 1 / (2 * E)
op = LinearOperator((n, n), matvec=lambda x: Ld @ x - c * dvec * (dvec @ x), dtype=float)
# smallest eigs of PSD op via largest of (sigma I - op)
sig = 2 * Ld.diagonal().max()
vals, V = eigsh(LinearOperator((n, n), matvec=lambda x: sig * x - op @ x, dtype=float), k=3, which="LA",
                v0=rng.normal(size=n), tol=1e-8)
V = V[:, np.argsort(-vals)]
vk = [V[:, j] for j in range(3) if abs(np.corrcoef(V[:, j], np.ones(n) + 1e-12 * rng.random(n))[0, 1]) < 0.5 and np.std(V[:, j]) > 1e-8]
xd = orient(vk[0])
print(f"(d) Pasadakis directed spectral (DAG Laplacian, rank-one term); fwd edges {np.mean(xd[ev] > xd[eu]):.2f}")
report("directed Fiedler -> Kahn-priority", evalx(xd))
# same trick on hypergraph Laplacian
dd = dvec.astype(float); c2 = 1 / (2 * np.sum(np.abs(dd)))
# ---------- (e) monotone-cone Fiedler via active-set contraction ----------
print("(e) monotone-cone Fiedler (active set: violated DAG edges get stiff springs, accumulate)")
rho = 100 * deg.max()
x = orient(F[:, 0]); active = np.zeros(E, bool); traj = []
for it in range(40):
    viol = x[eu] > x[ev] + 1e-12
    frac = viol.mean()
    r = evalx(x); traj.append((it, round(float(frac), 4), r[0]))
    if frac == 0: break
    active |= viol
    LA_ = sp.csr_matrix((np.full(active.sum(), rho), (eu[active], ev[active])), shape=(n, n)); LA_ = LA_ + LA_.T
    M = LH + sp.diags(np.asarray(LA_.sum(1)).ravel()) - LA_
    xn = fiedler(M, k=1)[0][:, 0]
    x = xn if np.dot(xn, x * deg) >= 0 else -xn
print("   it, viol frac, cut(eps.02):", traj)
report("monotone-cone Fiedler (final)", evalx(x), f"active edges {active.mean():.2f}")
report("monotone-cone Fiedler (best iter)", [min(t[2] for t in traj), -1])
print(f"total {time.time() - t0:.0f}s")
