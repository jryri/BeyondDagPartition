"""Construct original-DAG longest-path features for controlled ablation."""
import hashlib
import json
from pathlib import Path

import numpy as np

from run_cycle import read_hgr

root = Path(__file__).parent
nets, _, weights = read_hgr(root / 'inputs/arbiter.hgr')
drivers = np.array(json.loads((root / 'inputs/arbiter.drivers.json').read_text()))
n = len(weights)
successors = [set() for _ in range(n)]
for pins, driver in zip(nets, drivers):
    successors[driver].update(int(v) for v in pins if v != driver)
out = root / 'path_ablation'
out.mkdir(exist_ok=True)
for k, layout in ((3, 'direct'), (4, 'recursive')):
    with np.load(root / f'cycle_{layout}_k{k}/embedding.npz') as source:
        spec, order, rank = source['Xs'], source['Xo'], source['rank']
    topo = np.argsort(rank)
    forward = np.zeros(n)
    backward = np.zeros(n)
    for u in topo:
        for v in successors[u]:
            if rank[u] >= rank[v]:
                raise ValueError('Input is not a DAG')
            forward[v] = max(forward[v], forward[u] + 1)
    for u in topo[::-1]:
        for v in successors[u]:
            backward[u] = max(backward[u], backward[v] + 1)
    position, length = forward - backward, forward + backward
    for name, extra in [('length', [length]), ('both', [length, position]), ('position', [position])]:
        columns = np.column_stack(extra)
        columns = (columns - columns.mean(axis=0)) / columns.std(axis=0)
        digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        np.savez_compressed(out / f'k{k}_{name}.npz', Xs=np.column_stack((spec, columns)),
                            Xo=order, rank=rank, k=k,
                            hgr_sha256=digest(root / 'inputs/arbiter.hgr'),
                            drivers_sha256=digest(root / 'inputs/arbiter.drivers.json'))
