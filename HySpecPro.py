# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dgl
import dgl.function as fn
import torch
import numpy as np
from helper import read_hgr

import os, argparse, itertools
import cma
import cupy as cp
from torch.utils.dlpack import to_dlpack, from_dlpack
from cupyx.scipy.sparse import coo_matrix
from cupyx.scipy.sparse.linalg import eigsh

# batch size dictionary
batch_size_dict = {
    "ibm01": 13000,
    "ibm02": 10000,
    "ibm03": 7000,
    "ibm04": 6000,
    "ibm05": 5000,
    "ibm06": 5000,
    "ibm07": 4000,
    "ibm08": 3500,
    "ibm09": 3500,
    "ibm10": 2600,
    "ibm11": 2200,
    "ibm12": 2000,
    "ibm13": 1500,
    "ibm14": 1200,
    "ibm15": 1000,
    "ibm16": 900,
    "ibm17": 900,
    "ibm18": 800,
    "sparcT1_core": 1860,
    "neuron": 1700,
    "stereo_vision": 1700,
    "des90": 1300,
    "SLAM_spheric": 1200,
    "cholesky_mc": 1300,
    "segmentation": 1060,
    "bitonic_mesh": 760,
    "dart": 760,
    "openCV": 660,
    "stap_qrd": 620,
    "minres": 600,
    "cholesky_bdti": 590,
    "denoise": 580,
    "sparcT2_core": 560,
    "gsm_switch": 420,
    "mes_noc": 380,
    "LU230": 380,
    "LU_Network": 360,
    "sparcT1_chip2": 300,
    "directrf": 240,
    "bitcoin_miner": 240,
    "Bump_2911.mtx": 180,
    "CurlCurl_4.mtx": 200,
    "Ga41As41H72.mtx": 200,
    "Geo_1438.mtx": 250,
    "HV15R.mtx": 130,
    "StocF-1465.mtx": 200,
    "circuit5M.mtx": 100,
    "dgreen.mtx": 300,
    "sha256": 1200,
    "aes": 1500,
    "part": 1500,
}

PARTITION_DTYPE = torch.uint8

# build bipartite Laplacian from dgl graph
def bipartite_L_from_dgl(g, etype=('cell', 'connect', 'net'), device="cuda:0", symmetric=False):
    u_cell, v_net = g.edges(etype=etype)
    u_cell = u_cell.to(device)
    v_net = v_net.to(device)

    n_cells = g.num_nodes(etype[0])
    n_nets = g.num_nodes(etype[2])
    n_total = n_cells + n_nets
    net_offset = n_cells

    # === 1️⃣ Compute net degrees |e| (how many cells per net)
    net_deg = torch.zeros(n_nets, device=device, dtype=torch.float32)
    net_deg.scatter_add_(0, v_net, torch.ones_like(v_net, dtype=torch.float32))
    net_deg = net_deg.clamp(min=1.0)

    # === 2️⃣ Edge weights normalization
    if symmetric:
        # symmetric scaling: 1/sqrt(|e|)
        w_edge = 1.0 / torch.sqrt(net_deg[v_net])
    else:
        # Zhou normalization: 1/|e|
        w_edge = 1.0 / net_deg[v_net]

    # Duplicate edges for undirected bipartite graph
    src_all = torch.cat([u_cell, v_net + net_offset], dim=0).long()
    dst_all = torch.cat([v_net + net_offset, u_cell], dim=0).long()
    data_all = torch.cat([w_edge, w_edge], dim=0).float()

    # === 3️⃣ Convert to CuPy COO for GPU sparse ops
    src_cp = cp.fromDlpack(to_dlpack(src_all))
    dst_cp = cp.fromDlpack(to_dlpack(dst_all))
    data_cp = cp.fromDlpack(to_dlpack(data_all))

    A_cp = coo_matrix((data_cp, (src_cp, dst_cp)), shape=(n_total, n_total), dtype=cp.float32)
    A_cp.sum_duplicates()

    # === 4️⃣ Degree
    deg_cp = cp.asarray(A_cp.sum(axis=1)).ravel()
    deg_inv_sqrt = 1.0 / cp.sqrt(deg_cp + 1e-12)

    # === 5️⃣ Normalize adjacency
    row_idx = A_cp.row.astype(cp.int64)
    col_idx = A_cp.col.astype(cp.int64)
    data_scaled = A_cp.data * deg_inv_sqrt[row_idx] * deg_inv_sqrt[col_idx]
    A_norm = coo_matrix((data_scaled, (row_idx, col_idx)), shape=(n_total, n_total), dtype=cp.float32)
    A_norm.sum_duplicates()
    A_norm = 0.5 * (A_norm + A_norm.T)  # enforce symmetry

    # === 6️⃣ Laplacian
    eye_idx = cp.arange(n_total, dtype=cp.int64)
    I_cp = coo_matrix((cp.ones(n_total, dtype=cp.float32), (eye_idx, eye_idx)), shape=(n_total, n_total))
    L_cp = I_cp - A_norm

    # === 7️⃣ Sanity checks
    ones = cp.ones(n_total, dtype=cp.float32)
    print("L*1 max abs:", float(cp.max(cp.abs(L_cp @ ones))))

    sqrt_deg = cp.sqrt(deg_cp + 1e-12)
    vec = sqrt_deg  # D^{1/2} * 1
    res = L_cp @ vec
    print("L_sym * D^{1/2}1 max abs:", float(cp.max(cp.abs(res))))

    return L_cp

# compute k smallest eigenvectors of bipartite Laplacian using cupyx eigsh
def bipartite_spectral_embeddings_from_dgl(g, etype=('cell','connect','net'), k=16, which='SA', tol=1e-6, maxiter=5000, device="cuda:0"):
    L_cp = bipartite_L_from_dgl(g, etype=etype, device=device)

    eigvals_cp, eigvecs_cp = eigsh(L_cp, k=k, which=which, tol=tol, maxiter=maxiter)

    n_cells = g.num_nodes(etype[0])
    eigvecs_cells_cp = eigvecs_cp[:n_cells, :]  # (n_cells, k)

    # Convert eigvecs_cells_cp to torch tensor via DLPack (zero-copy)
    eigvecs_cells_torch = from_dlpack(eigvecs_cells_cp.toDlpack()).to(device=device)
    eigvals_cp = from_dlpack(eigvals_cp.toDlpack()).to(device=device)

    # Return cupy eigenvalues and torch embeddings for cells
    return eigvals_cp, eigvecs_cells_torch


def load_driver_edges(path):
    """Driver to sink edges. Accepts a DAG dot file or a JSON list of [src, dst]."""
    text = open(path).read().strip()
    if text.startswith("{") or text.startswith("["):
        import json
        raw = json.loads(text)
        return [(int(a), int(b)) for a, b in raw]
    edges = []
    for line in text.splitlines():
        if "->" not in line:
            continue
        a, b = line.replace(";", "").split("->")
        edges.append((int(a), int(b)))
    return edges


def topo_from_edges(n, edges):
    succ = [[] for _ in range(n)]
    indeg = np.zeros(n, dtype=np.int32)
    for u, v in edges:
        if v not in succ[u]:
            succ[u].append(v)
            indeg[v] += 1
    for u in range(n):
        succ[u].sort()
    topo = [int(i) for i in np.flatnonzero(indeg == 0)]
    for u in topo:
        for v in succ[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                topo.append(v)
    if len(topo) != n:
        raise ValueError(f"driver graph has a cycle, ordered {len(topo)}/{n}")
    return succ, topo


def order_embedding_from_spectral(spec, edges, dim=8):
    """X = [X_spec | X_max | X_min] on the cell coordinates already on GPU.

    spec is the smallest nontrivial bipartite-Laplacian eigenvectors.
    Takes the first `dim` columns; output width is 3*dim.
    Sign is fixed so driver to sink increases.
    U/L are hard envelopes along the driver DAG (max forward / min backward).
    """
    if spec.shape[1] < dim:
        raise ValueError(f"need {dim} spectral columns, got {spec.shape[1]}")
    device = spec.device
    F = spec[:, :dim].detach().float().cpu().numpy()
    n = F.shape[0]
    succ, topo = topo_from_edges(n, edges)
    eu = np.asarray([u for u, vs in enumerate(succ) for v in vs], dtype=np.int64)
    ev = np.asarray([v for vs in succ for v in vs], dtype=np.int64)
    for j in range(F.shape[1]):
        if np.sum(F[ev, j] - F[eu, j]) < 0:
            F[:, j] *= -1.0
    ptp = np.maximum(np.ptp(F, axis=0), 1e-12)
    F = (F - F.min(0)) / ptp
    std = np.maximum(F.std(0), 1e-12)
    Xs = F / std
    delta = 1e-4 * std
    U = F.copy()
    L = F.copy()
    for u in topo:
        for v in succ[u]:
            np.maximum(U[v], U[u] + delta, out=U[v])
    for u in reversed(topo):
        for v in succ[u]:
            np.minimum(L[u], L[v] - delta, out=L[u])
    Xo = np.concatenate((U, L), axis=1)
    Xo = Xo / np.maximum(Xo.std(0), 1e-12)
    X = np.concatenate((Xs, Xo), axis=1).astype(np.float32)
    return torch.from_numpy(X).to(device)

# main function to run the partitioner
def run_partitioner(design_root, design, device, tag, result_root, N_CMA_ITE, KWAY, UB, drivers_path=None, lambda_f=1.0):

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    config = {
        "hgr_file": design_root + design + ".hgr",
        "hMetis_solution_head": "",
        "solution_file": result_root + tag + "_" + design + "_best_solution.pt",
        "score_file": result_root + tag + "_" + design + "_best_score.pt",
        "device": device,
        "use_weight": False,
        "N_batch": batch_size_dict[design],
        "K": KWAY,
        "e": UB
    }

    hgr_file = config["hgr_file"]
    use_weight = config["use_weight"]
    N_batch = config["N_batch"]
    if os.environ.get("HYSP_NBATCH"):
        N_batch = int(os.environ["HYSP_NBATCH"])
    K = config["K"]
    e = config["e"]

    g = read_hgr(hgr_file)
    cell_cnt, net_cnt = g.num_nodes("cell"), g.num_nodes("net")
    print(g)
    g = g.to(device)

    W = cell_cnt
    th_l = W*(1/K - e)
    th_u = W*(1/K + e)
    print("Partition size constraints: ", th_l, th_u)
    driver_of_net = None
    use_affine = False
    if drivers_path:
        succ = [set() for _ in range(cell_cnt)]
        for u, v in load_driver_edges(drivers_path):
            succ[u].add(v)
        cells_e, nets_e = g.edges(etype="connect")
        pins_of = [[] for _ in range(net_cnt)]
        for c, e in zip(cells_e.tolist(), nets_e.tolist()):
            pins_of[e].append(c)
        drv = [-1] * net_cnt
        known = 0
        for e, pins in enumerate(pins_of):
            hits = [p for p in pins if all(q == p or q in succ[p] for q in pins)]
            if len(hits) == 1:
                drv[e] = hits[0]
                known += 1
        driver_of_net = torch.tensor(drv, device=device, dtype=torch.long)
        # K=2: affine halfspace sign(Xw+b). K>2: HySpecPro argmax projection.
        use_affine = (K == 2)
        dec = "affine" if use_affine else f"proj-{K}"
        print(f"J = C + {lambda_f} * F_net, drivers on {known}/{net_cnt} nets, decoder={dec}")

    # batch evaluation function
    def evaluation(g, population):
        cell_cnt, B = population.shape
        with torch.no_grad():
            pop = population.to(device=device)
            deg_col = g.nodes["net"].data.get("deg_col", None)
            if deg_col is None:
                deg = g.in_degrees(torch.arange(g.num_nodes('net'), device=device), etype='connect').to(torch.float32)
                deg_col = deg.view(-1, 1)
                g.nodes["net"].data["deg_col"] = deg_col

            if K == 2:
                # Fast binary path: part-1 pin counts.
                assign_b = pop.to(dtype=torch.float32)
                g.nodes["cell"].data["assign_b"] = assign_b
                g['connect'].update_all(fn.copy_u('assign_b', 'm'), fn.sum('m', 'net_sum1'), etype='connect')
                net_sum1 = g.nodes["net"].data["net_sum1"].to(torch.float32)
                net_cut = (net_sum1 > 0.0) & (net_sum1 < deg_col)
                N_cut = net_cut.sum(dim=0).to(torch.float32)
                if use_weight:
                    temp = torch.zeros([cell_cnt, B, K], device=device, dtype=torch.float32)
                    temp.scatter_(2, pop.unsqueeze(2).long(), 1)
                    w = torch.sum(g.nodes["cell"].data["weight"].reshape(-1, 1, 1) * temp, 0)
                    penalty = 50 * torch.sum(torch.relu(w - th_u) + torch.relu(th_l - w), 1)
                else:
                    ones_per_batch = assign_b.sum(dim=0)
                    zeros_per_batch = float(cell_cnt) - ones_per_batch
                    weight_mat = torch.stack([zeros_per_batch, ones_per_batch], dim=1)
                    penalty = 50.0 * torch.sum(torch.relu(weight_mat - th_u) + torch.relu(th_l - weight_mat), dim=1)
                feedback = torch.zeros(B, device=device)
                if driver_of_net is not None:
                    known = driver_of_net >= 0
                    drv_part = assign_b[driver_of_net.clamp(min=0)]
                    back_01 = ((drv_part > 0.5) & (net_sum1 < deg_col) & known.unsqueeze(1)).sum(0)
                    back_10 = ((drv_part < 0.5) & (net_sum1 > 0.0) & known.unsqueeze(1)).sum(0)
                    feedback = torch.minimum(back_01, back_10).float()
            else:
                # K-way cut: a net is cut if it touches more than one part.
                parts_present = []
                for k in range(K):
                    mask_k = (pop == k).to(torch.float32)
                    g.nodes["cell"].data["assign_k"] = mask_k
                    g['connect'].update_all(fn.copy_u('assign_k', 'm'), fn.sum('m', 'net_k'), etype='connect')
                    parts_present.append(g.nodes["net"].data["net_k"] > 0)
                n_parts = torch.stack(parts_present, dim=-1).sum(dim=-1)
                net_cut = n_parts > 1
                N_cut = net_cut.sum(dim=0).to(torch.float32)
                if use_weight:
                    temp = torch.zeros([cell_cnt, B, K], device=device, dtype=torch.float32)
                    temp.scatter_(2, pop.unsqueeze(2).long(), 1)
                    w = torch.sum(g.nodes["cell"].data["weight"].reshape(-1, 1, 1) * temp, 0)
                else:
                    w = torch.stack([(pop == k).to(torch.float32).sum(dim=0) for k in range(K)], dim=1)
                penalty = 50.0 * torch.sum(torch.relu(w - th_u) + torch.relu(th_l - w), dim=1)
                feedback = torch.zeros(B, device=device)
                if driver_of_net is not None:
                    # F_net = min over block orders. Exact for K<=7; else random sample.
                    known = driver_of_net >= 0
                    drv_cell = driver_of_net.clamp(min=0)
                    best_fb = None
                    if K <= 7:
                        perm_iter = itertools.permutations(range(K))
                    else:
                        rng = np.random.default_rng(0)
                        perm_iter = [tuple(range(K))] + [
                            tuple(rng.permutation(K).tolist()) for _ in range(64)
                        ]
                    for perm in perm_iter:
                        rank_of = torch.tensor(perm, device=device, dtype=torch.long)
                        cell_rank = rank_of[pop.long()]
                        g.nodes["cell"].data["rank"] = cell_rank.to(torch.float32)
                        g['connect'].update_all(fn.copy_u('rank', 'm'), fn.min('m', 'net_min_rank'), etype='connect')
                        net_min = g.nodes["net"].data["net_min_rank"]
                        drv_rank = cell_rank[drv_cell].to(torch.float32)
                        fb = ((net_min < drv_rank) & known.unsqueeze(1) & net_cut).sum(0).float()
                        best_fb = fb if best_fb is None else torch.minimum(best_fb, fb)
                    feedback = best_fb
            score = (N_cut + lambda_f * feedback + penalty).cpu()
            return score

    # conduct CMA-ES
    def CMA_ES_iterations(g, embed_list):
        best_solution_list = []
        best_score_list = []

        # conduct 5 runs of CMA-ES for each embedding
        for _ in range(N_CMA_ITE):
            for embed_idx, embed in enumerate(embed_list):
                print("embed_idx: ", embed_idx)
                # conduct CMA-ES
                with torch.no_grad():
                    N_nodes, N_embed = embed.shape
                    N_cma_ite = 50
                    best_solution = None
                    best_score = None
                    sigma = 0.3
                    Np = 3 # population size = N_batch * Np * Ns, where Np is the parallelization factor and Ns is the sequential factor
                    Ns = 1
                    Na = 1
                    # Affine (K=2): |w|+1. Else HySpecPro projections: N_embed * K.
                    cma_dim = (N_embed + 1) if use_affine else (N_embed * K * Na)
                    es = cma.CMAEvolutionStrategy(np.zeros(cma_dim), sigma, {'popsize': N_batch * Np * Ns})
                    unchange_cnt = 0
                    global_min = 100000000.0
                    RELU = torch.nn.ReLU().to(device)
                    for _ in range(N_cma_ite):
                        # stop if the best score has not improved for 3 consecutive iterations
                        if unchange_cnt >= 5:
                            break
                        # sample a new population
                        assign_W = es.ask()
                        assign_W_np = np.array(assign_W, dtype=np.float32)
                        temp_score_list = []
                        if use_affine:
                            # y_v = 1[x_v^T w + b > 0]. Same GPU matmul cost as origin-forced sign.
                            wb = torch.from_numpy(assign_W_np).to(device).float()
                            w = wb[:, :N_embed]
                            b = wb[:, N_embed]
                            logits = embed @ w.T + b
                            solution = (logits > 0).to(PARTITION_DTYPE)
                            del wb, w, b, logits
                            score = evaluation(g, solution)
                            temp_score_list.append(score.detach())
                            if best_score is None:
                                best_score = score.detach()
                                best_solution = solution.detach()
                            else:
                                mask = score < best_score
                                best_score[mask] = score[mask]
                                best_solution[:, mask] = solution[:, mask].detach()
                                del mask
                            del solution, score
                        else:
                            assign_W_torch = torch.from_numpy(assign_W_np).reshape(-1, Na, N_embed, K).to(device).float()

                            # evaluate the population
                            for s in range(Ns):
                                assign_W_torch_s = assign_W_torch[N_batch * Np * s : N_batch * Np * (s+1)]
                                if K == 2:
                                    # Binary case: argmax([x^T w0, x^T w1]) is
                                    # equivalent to x^T(w1-w0) > 0.  This avoids
                                    # materializing logits with shape [N, B, 2].
                                    w_delta = (assign_W_torch_s[..., 1] - assign_W_torch_s[..., 0]).sum(dim=1)
                                    logits_delta = embed @ w_delta.T
                                    solution = (logits_delta > 0).to(PARTITION_DTYPE)
                                    del w_delta, logits_delta
                                else:
                                    logits = torch.einsum('ne,paek->pank', embed, assign_W_torch_s)
                                    logits = logits.sum(dim=1)
                                    logits = logits.permute(1, 0, 2)
                                    solution = torch.argmax(logits, 2).to(PARTITION_DTYPE)
                                    del logits
                                score = evaluation(g, solution)
                                temp_score_list.append(score.detach())

                                if best_score is None:
                                    best_score = score.detach()
                                    best_solution = solution.detach()
                                else:
                                    mask = score < best_score
                                    best_score[mask] = score[mask]
                                    best_solution[:,mask] = solution[:,mask].detach()
                                    del mask

                                del solution, score, assign_W_torch_s
                            del assign_W_torch

                        temp_score_list = torch.cat(temp_score_list, 0).cpu().tolist()
                        # print("temp_score_list: ", len(temp_score_list))
                        temp_min = min(temp_score_list)
                        if temp_min < global_min:
                            global_min = temp_min
                            unchange_cnt = 0
                        else:
                            unchange_cnt += 1

                        # update the CMA-ES with the population scores
                        es.tell(assign_W, temp_score_list)
                        es.disp()
                        del assign_W
                    del es
                    print("best score after CMA-ES: ", best_score.min())

                scores_sorted, indices = torch.sort(best_score)
                best_score = best_score[indices[:3]]
                best_solution = best_solution[:, indices[:3]]
                del indices, scores_sorted

                best_solution_list.append(best_solution)
                best_score_list.append(best_score)


        best_solution_all = torch.cat(best_solution_list, 1)
        best_score_all = torch.cat(best_score_list, 0)
        return best_solution_all, best_score_all

    # conduct batched FM/KL refinement
    def fm_kl_refine_batch(g, assign_batch, th_l, th_u, max_passes=2, max_iters=1000):
        """
        Batched FM/KL refinement.
        - assign_batch: LongTensor [cell_cnt, B] of {0,1}
        Returns: (refined_assign_batch [cell_cnt, B], improved_mask [B])
        """
        assign_batch = assign_batch.to(dtype=PARTITION_DTYPE)
        device_local = assign_batch.device
        cell_cnt = g.num_nodes('cell')
        B = assign_batch.shape[1]

        # Baseline scores per head
        with torch.no_grad():
            baseline_scores = evaluation(g, assign_batch)
        baseline_cut = baseline_scores.to(device_local).float()  # [B]

        part0_sizes = (assign_batch == 0).sum(dim=0).float()  # [B]
        part1_sizes = (cell_cnt - part0_sizes)

        improved_any = torch.zeros(B, dtype=torch.bool, device=device_local)

        def compute_gains_batched(assign_b):
            # Build one-hot per head: [cell_cnt, B, 2]
            onehot = torch.zeros(cell_cnt, B, 2, device=device_local, dtype=torch.float32)
            onehot.scatter_(2, assign_b.long().unsqueeze(2), 1.0)
            g.nodes['cell'].data['part_onehot_b'] = onehot
            g['connect'].update_all(fn.copy_u('part_onehot_b','m'), fn.sum('m','net_counts_b'), etype='connect')
            net_counts = g.nodes['net'].data['net_counts_b']  # [net_cnt, B, 2]
            net_cut = (net_counts[:,:,0] > 0) & (net_counts[:,:,1] > 0)  # [net_cnt, B]
            pos0 = (net_cut & (net_counts[:,:,0] == 1)).to(torch.float32)
            neg0 = ((~net_cut) & (net_counts[:,:,1] == 0)).to(torch.float32)
            pos1 = (net_cut & (net_counts[:,:,1] == 1)).to(torch.float32)
            neg1 = ((~net_cut) & (net_counts[:,:,0] == 0)).to(torch.float32)

            cells_idx, nets_idx = g.in_edges(torch.arange(g.num_nodes('net'), device=device_local), etype='connect')
            # edge contributions [E,B]
            edge_c_pos0 = pos0[nets_idx, :]
            edge_c_neg0 = neg0[nets_idx, :]
            edge_c_pos1 = pos1[nets_idx, :]
            edge_c_neg1 = neg1[nets_idx, :]

            gain0 = torch.zeros(cell_cnt, B, device=device_local, dtype=torch.float32)
            gain1 = torch.zeros(cell_cnt, B, device=device_local, dtype=torch.float32)
            gain0.index_add_(0, cells_idx, (edge_c_pos0 - edge_c_neg0))
            gain1.index_add_(0, cells_idx, (edge_c_pos1 - edge_c_neg1))
            gains = torch.where(assign_b == 0, gain0, gain1)
            return gains

        for _pass in range(max_passes):
            moved = torch.zeros(cell_cnt, B, dtype=torch.bool, device=device_local)
            iters = 0
            while iters < max_iters:
                gains = compute_gains_batched(assign_batch)
                gains_masked = gains.masked_fill(moved, float('-inf'))
                best_gain, best_idx = torch.max(gains_masked, dim=0)  # [B]
                try_mask = best_gain > 0
                if not try_mask.any():
                    break

                cols = torch.arange(B, device=device_local)
                cur_part = assign_batch[best_idx, cols]

                # Balance check per head
                new_p0 = torch.where(cur_part == 0, part0_sizes - 1, part0_sizes + 1)
                new_p1 = cell_cnt - new_p0
                balance_ok = (new_p0 >= th_l) & (new_p0 <= th_u) & (new_p1 >= th_l) & (new_p1 <= th_u)
                cand_mask = try_mask & balance_ok
                if not cand_mask.any():
                    # mark tried as moved and continue
                    moved[best_idx[try_mask], cols[try_mask]] = True
                    iters += 1
                    continue

                # Tentatively flip all candidate heads in a copy and evaluate in one shot
                tentative = assign_batch.clone()
                flip_cols = cols[cand_mask]
                flip_rows = best_idx[cand_mask]
                tentative[flip_rows, flip_cols] = 1 - cur_part[cand_mask]

                with torch.no_grad():
                    new_scores = evaluation(g, tentative).to(device_local).float()

                accept_mask = cand_mask & (new_scores < baseline_cut)
                if accept_mask.any():
                    a_cols = cols[accept_mask]
                    a_rows = best_idx[accept_mask]
                    assign_batch[a_rows, a_cols] = 1 - assign_batch[a_rows, a_cols]
                    # update baselines and sizes for accepted
                    improved_any[accept_mask] = True
                    delta = torch.where(cur_part[accept_mask] == 0, -1.0, 1.0)
                    part0_sizes[accept_mask] = part0_sizes[accept_mask] + delta
                    part1_sizes[accept_mask] = cell_cnt - part0_sizes[accept_mask]
                    baseline_cut[accept_mask] = new_scores[accept_mask]
                    moved[a_rows, a_cols] = True
                else:
                    # None accepted; mark tried as moved to avoid reselecting
                    moved[best_idx[cand_mask], cols[cand_mask]] = True

                iters += 1

        return assign_batch, improved_any

    # construct embedding spaces
    embed_list = []
    # Compute spectral embedding from bipartite graph
    k = 48
    eigvals, eigvecs_cells_torch = bipartite_spectral_embeddings_from_dgl(g, etype=('cell','connect','net'), k=k, which='SA', device=device)
    print("Bipartite graph eigvals:", eigvals)
    mask = eigvals > 1e-6
    eigvals = eigvals[mask]
    eigvecs_cells_torch = eigvecs_cells_torch[:, mask]
    if drivers_path:
        _ordim = int(os.environ.get("HYSP_ORDIM", "8"))
        if os.environ.get("HYSP_SPEC_ONLY") == "1":
            # Ablation: same J / affine / drivers, but X is spectral only (no U/L).
            X = eigvecs_cells_torch[:, :_ordim]
            print(f"spectral-only X (no order): {tuple(X.shape)}")
        else:
            # dim spectral + hard U/L envelopes => 3*dim.
            X = order_embedding_from_spectral(
                eigvecs_cells_torch, load_driver_edges(drivers_path), dim=_ordim
            )
            print(f"order embedding X: {tuple(X.shape)}")
        embed_list += [X]
    else:
        if os.environ.get("HYSP_BASE8") == "1":
            embed_list += [eigvecs_cells_torch[:, :8]]
        else:
            embed_list += [eigvecs_cells_torch[:, :4], eigvecs_cells_torch[:, :8], eigvecs_cells_torch[:, :16], eigvecs_cells_torch[:, :24], eigvecs_cells_torch[:, :36]]

    best_solution, best_score = CMA_ES_iterations(g, embed_list)
    print("Best score after CMA-ES on all embedding spaces: ", best_score.min())

    scores_sorted, indices = torch.sort(best_score)
    best_solution = best_solution[:, indices]
    best_score = scores_sorted

    # FM/KL refinement on the best discrete solutions.
    total_edge_degree = int(g.num_edges(etype='connect'))
    topK = 32
    if total_edge_degree >= 100_000_000:
        topK = 2
    elif total_edge_degree >= 50_000_000:
        topK = 8

    if K != 2:
        print(f"FM/KL skipped for K={K} (binary-only refine)")
    else:
        print(f"FM/KL topK: {topK} (total_edge_degree={total_edge_degree})")
        best_assign = best_solution[:,:topK].to(device)
        try:
            refined_assign, did_improve = fm_kl_refine_batch(g, best_assign, th_l, th_u, max_passes=10)
        except torch.cuda.OutOfMemoryError as oom:
            print(f"FM/KL refinement skipped due to CUDA OOM: {oom}")
            torch.cuda.empty_cache()
            refined_assign = best_assign
            did_improve = torch.zeros(topK, dtype=torch.bool, device=device)
        if did_improve.sum() > 0:
            print("FM/KL refinement improved the solution. Re-evaluating...")
            best_solution[:,:topK] = refined_assign.to(best_solution.device)
            best_score[:topK] = evaluation(g, best_solution[:,:topK])
            print(f"Refined final score: {best_score.min(), best_score[0]}")

            scores_sorted, indices = torch.sort(best_score)
            best_solution = best_solution[:, indices]

    torch.save(best_solution[:,0].to("cpu"), config["solution_file"])
    torch.save(best_score.min().detach().cpu(), config["score_file"])
    return best_score.min().item()

def main():
    parser = argparse.ArgumentParser(description="HySpecPro")
    parser.add_argument("--design_root", type=str, default="./benchmark/Titan23_benchmark/", help="Root path to the hgr files")
    parser.add_argument("--design", type=str, default="sparcT1_core", help="Design name")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument("--tag", type=str, default="v1", help="Tag to differentiate different runs")
    parser.add_argument("--result_root", type=str, default="./results/", help="Root path to the results")
    parser.add_argument("--N_CMA_ITE", type=int, default=5, help="Number of CMA-ES iterations")
    parser.add_argument("--KWAY", type=int, default=2, help="Number of partitions")
    parser.add_argument("--UB", type=float, default=0.02, help="Upper bound for the partition size")
    parser.add_argument("--drivers", type=str, default="", help="Driver-to-sink dot or JSON. Builds the 24-d order embedding.")
    parser.add_argument("--lambda_f", type=float, default=1.0, help="Weight of F_net in J = C + lambda_f * F_net")
    args = parser.parse_args()

    best_score = run_partitioner(
        args.design_root, args.design, args.device, args.tag, args.result_root,
        args.N_CMA_ITE, args.KWAY, args.UB, args.drivers or None, args.lambda_f,
    )
    print(f"Final best score: {best_score}")

if __name__ == "__main__":
    main()