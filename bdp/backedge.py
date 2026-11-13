import sys, os
sys.path.insert(0, 'bdp')
import numpy as np
from refine import HG, load, kahypar_parts
import mtkahypar as mk
mtk = mk.initialize(8)
srcs = ['data/serv_hg.npz','data/omsp_hg.npz','data/sha256_hg.npz','bdp/aes_cu3d','data/picosoc_hg.npz','data/aes_hg.npz','data/riscv_hg.npz','data/biriscv_hg.npz']
for s in srcs:
    h = HG(*load(s)); E = len(h.eu)
    row = []
    for eps in (0.02, 0.10):
        best = None
        for c, p in kahypar_parts(h, eps, mtk, seeds=3):
            p = np.asarray(p)
            for q in (p, 1 - p):  # orientation: block 0 = earlier
                back = (q[h.eu] == 1) & (q[h.ev] == 0)
                nb = int(back.sum()); cutE = int((q[h.eu] != q[h.ev]).sum())
                # nets carrying a backward edge: driver eu's net; count distinct drivers (one output net per gate)
                fn = len(np.unique(h.eu[back]))
                if best is None or (c, nb) < (best[0], best[1]):
                    best = (c, nb, cutE, fn)
        c, nb, cutE, fn = best
        row.append(f"eps{eps:.2f}: cut {c:4d}  back/crossing DAG edges {nb:4d}/{cutE:5d} ({100*nb/max(cutE,1):4.1f}%)  feedback nets {fn}")
    print(f"{os.path.basename(s):14s} E={E:6d} | " + " | ".join(row), flush=True)
