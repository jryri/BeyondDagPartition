"""DAG-aware embedding G (RESULTS §35): his envelope idea with the L1 projection instead of the L-inf envelope.

For each spectral coordinate x_j (orientation flipped so DAG edges increase on average, as in his
orient_and_scale), G_j = exact L1 isotonic regression of x_j on the DAG (exact_ir.l1_ir_fast).
Every column of G is nondecreasing along every DAG edge, so the embedding itself carries the dataflow order.
Columns are then standardised.  Cost: d exact projections (~log2 n max-flows each), once per design.
"""
import numpy as np
from exact_ir import l1_ir_fast


def dag_embed(h, X):
    G = np.empty_like(X)
    for j in range(X.shape[1]):
        x = X[:, j] if np.sum(X[h.ev, j] - X[h.eu, j]) >= 0 else -X[:, j]
        G[:, j] = l1_ir_fast(h, x)
        assert np.all(G[h.ev, j] >= G[h.eu, j])
    return (G - G.mean(0)) / (G.std(0) + 1e-12)
