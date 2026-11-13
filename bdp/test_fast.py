import sys, os, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from refine import HG, load
from det import eig
from exact_ir import l1_ir_exact, l1_ir_fast
for name in sys.argv[1:]:
    h = HG(*load(f"data/{name}_hg.npz")); vals, U, dv, inv, w = eig(h, 32)
    X = U[:, :16] * inv[:, None]; X = (X - X.mean(0)) / (X.std(0) + 1e-12); rng = np.random.default_rng(0); te = tf = 0; ok = True
    for k in range(8):
        th = np.zeros(32); th[0] = -2; th[16] = 2; th += rng.normal(size=32) * (k > 0) * [1, 1, 1, 1, 5, 5, 0.1, 0.1][k]
        s = np.sqrt(((X - th[:16]) ** 2).sum(1)) - np.sqrt(((X - th[16:]) ** 2).sum(1))
        for sg in (1, -1):
            t0 = time.time(); a = l1_ir_exact(h, sg * s); te += time.time() - t0
            t0 = time.time(); b = l1_ir_fast(h, sg * s); tf += time.time() - t0
            ok &= np.array_equal(a, b)
    print(f"{name}: identical on 16 projections = {ok} | exact {1000 * te / 16:.0f} ms, fast {1000 * tf / 16:.0f} ms ({te / tf:.2f}x)")
