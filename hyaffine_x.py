"""Build X = [X_spec | X_max | X_min] from HySpecPro's bipartite Laplacian.

X_spec: 8 cell coordinates from the smallest nontrivial eigenvectors.
X_max / X_min: forward and backward envelopes along driver -> sink edges.
The sign decoder is not used here.
"""
import numpy as np
from numpy.linalg import norm
from scipy import sparse
from scipy.sparse.linalg import eigsh


def load_hgr(path):
    nets = []
    with open(path) as f:
        header = f.readline().split()
        m, n = int(header[0]), int(header[1])
        for line in f:
            pins = [int(x) - 1 for x in line.split()]
            if len(pins) >= 2:
                nets.append(np.asarray(sorted(set(pins)), dtype=np.int32))
    if len(nets) != m:
        raise ValueError(f"expected {m} nets, read {len(nets)}")
    return n, nets


def load_dag(path, n):
    succ = [[] for _ in range(n)]
    with open(path) as f:
        for line in f:
            if "->" not in line:
                continue
            a, b = line.replace(";", "").split("->")
            u, v = int(a), int(b)
            succ[u].append(v)
    for u in range(n):
        succ[u] = sorted(set(succ[u]))
    return succ


def topo_order(succ):
    n = len(succ)
    indeg = np.zeros(n, dtype=np.int32)
    for u, vs in enumerate(succ):
        for v in vs:
            indeg[v] += 1
    topo = list(np.flatnonzero(indeg == 0))
    for u in topo:
        for v in succ[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                topo.append(v)
    if len(topo) != n:
        raise ValueError(f"driver graph has a cycle, ordered {len(topo)}/{n}")
    rank = np.empty(n, dtype=np.int32)
    rank[np.asarray(topo, dtype=np.int32)] = np.arange(n)
    return topo, rank


def spectral_cells(nets, n, dim=8):
    """Cell rows of the smallest nontrivial bipartite-Laplacian eigenvectors."""
    m = len(nets)
    pins = np.concatenate(nets)
    net_id = np.repeat(np.arange(m), [len(p) for p in nets])
    sizes = np.array([len(p) for p in nets], dtype=np.float64)
    w = 1.0 / sizes[net_id]
    # bipartite nodes: cells 0..n-1, nets n..n+m-1
    N = n + m
    rows = np.concatenate([pins, net_id + n])
    cols = np.concatenate([net_id + n, pins])
    data = np.concatenate([w, w])
    A = sparse.coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()
    A.sum_duplicates()
    deg = np.asarray(A.sum(1)).ravel()
    inv = np.zeros_like(deg)
    nz = deg > 0
    inv[nz] = 1.0 / np.sqrt(deg[nz])
    # smallest eigenvectors of I - D^{-1/2} A D^{-1/2}
    # equal to largest of the normalized adjacency, shifted
    S = A.copy().astype(np.float64)
    S.data *= inv[S.indices]
    # row-scale
    S = sparse.diags(inv) @ S
    evals, vecs = eigsh(S, k=dim + 1, which="LA", tol=1e-8)
    order = np.argsort(evals)  # smallest Laplacian eig = largest adjacency eig at the end
    # adjacency eigenvalues near 1 are Laplacian eigenvalues near 0
    adj = evals[order]
    vecs = vecs[:, order]
    # drop the trivial mode (adjacency eigenvalue closest to 1)
    # Column -1 is the trivial adjacency eigenvalue near 1.
    # The next `dim` columns are the smallest nontrivial Laplacian modes.
    cells = vecs[:n, -(dim + 1):-1][:, ::-1]
    if cells.shape[1] != dim:
        raise ValueError(f"expected {dim} coordinates, got {cells.shape[1]}")
    print("adjacency eigenvalues, trivial last:", " ".join(f"{v:.6f}" for v in adj[-(dim + 1):]))
    return cells


def orient_and_scale(F, succ):
    """Fix sign so edges increase, then min-max and divide by std."""
    eu, ev = [], []
    for u, vs in enumerate(succ):
        for v in vs:
            eu.append(u)
            ev.append(v)
    eu = np.asarray(eu)
    ev = np.asarray(ev)
    for j in range(F.shape[1]):
        if np.sum(F[ev, j] - F[eu, j]) < 0:
            F[:, j] *= -1
    ptp = np.maximum(np.ptp(F, axis=0), 1e-12)
    F = (F - F.min(0)) / ptp
    std = np.maximum(F.std(0), 1e-12)
    return F / std, F, std


def envelopes(F, succ, topo):
    delta = 1e-4 * np.maximum(F.std(0), 1e-12)
    U = F.copy()
    L = F.copy()
    for u in topo:
        for v in succ[u]:
            U[v] = np.maximum(U[v], U[u] + delta)
    for u in reversed(topo):
        for v in succ[u]:
            L[u] = np.minimum(L[u], L[v] - delta)
    scale = np.maximum(np.concatenate((U, L), axis=1).std(0), 1e-12)
    Xo = np.concatenate((U, L), axis=1) / scale
    return U, L, Xo, delta


def check_monotone(U, L, succ, delta):
    up = down = 0
    up_ok = down_ok = 0
    for u, vs in enumerate(succ):
        for v in vs:
            up += 1
            down += 1
            up_ok += bool(np.all(U[v] + 1e-9 >= U[u] + delta))
            down_ok += bool(np.all(L[u] <= L[v] - delta + 1e-9))
    return up, up_ok, down_ok


def main():
    hgr = "/home/zixian/CFFET_ZiXian/newAsymcell/syn_sha256/nd_part400/cells.hgr"
    dag = "/home/zixian/CFFET_ZiXian/newAsymcell/syn_sha256/nd_dagp/cell_dag.dot"
    n, nets = load_hgr(hgr)
    succ = load_dag(dag, n)
    topo, rank = topo_order(succ)
    raw = spectral_cells(nets, n, 8)
    Xs, F, std = orient_and_scale(raw, succ)
    U, L, Xo, delta = envelopes(F, succ, topo)
    X = np.concatenate((Xs, Xo), axis=1)
    nedge, up_ok, down_ok = check_monotone(U, L, succ, delta)
    np.savez(
        "/tmp/sha_hyaffine_X.npz",
        X=X.astype(np.float32),
        Xs=Xs.astype(np.float32),
        X_max=Xo[:, :8].astype(np.float32),
        X_min=Xo[:, 8:].astype(np.float32),
        rank=rank,
    )
    print(f"cells={n} nets={len(nets)} dag_edges={nedge} X={X.shape}")
    print(f"delta={delta}")
    print(f"forward envelope holds on {up_ok}/{nedge} edges")
    print(f"backward envelope holds on {down_ok}/{nedge} edges")
    print(f"X std {X.std(0)}")
    print("saved /tmp/sha_hyaffine_X.npz")


if __name__ == "__main__":
    main()
