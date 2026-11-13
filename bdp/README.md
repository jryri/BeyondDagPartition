# bdp: DAG-aware embedding for acyclic / few-back-edge bisection

Research code and results built on top of the order-aware HySpecPro line in this repository.
Full record of every experiment, including negative results: [`RESULTS_bdp.md`](RESULTS_bdp.md) (sections 1–46).

## Current method

1. **Embedding** `[X | U | L]` (48 dims): 16 hypergraph spectral coordinates, oriented along the DAG, plus the
   ancestor-max `U` and descendant-min `L` envelopes of `hyaffine_x.py` (exact max/min, levelised).
2. **Separator**: free-sign hyperplane `s = [X|U|L] w`.
3. **Order**: `f = (1 - t) s + t * Linf(s)`, `Linf(s) = (U(s) + L(s)) / 2` (exact L-inf isotonic regression,
   two level-by-level passes, no max-flow). `t = 1` gives a topological order (acyclic); `t < 1` allows back edges.
4. **Sweep** in the balance window, minimising `cut + lam * back` (exact per split, O(pins + DAG edges)).
5. **FM** (acyclic, or penalised at finite `lam`), then one full refinement at the end.
6. **Search** over `(w, t)` by deterministic homotopy (per-axis central differences, sigma 2 -> 1/32),
   with lam-continuation `inf -> 4 -> 1`, each stage seeded with the previous stage's final partition,
   so a finite-`lam` result is never worse than the acyclic one.

Entry point: `python bdp/homotopy_cont.py data/<design>_hg.npz [workers]`.

## Results so far (eps = 0.10; RESULTS sections 44–46)

| design | acyclic cut | lam = 1: cut / back edges | best single baseline (acyclic) |
|---|---|---|---|
| serv | 61 | 59 / 0 | 61 (Mt-KaHyPar + closure repair + refine) |
| omsp | 316 | 135 / 21 | 341 |
| picosoc | 262 | 260 / 1 | 250 |
| aes | 477 | 226 / 40 | 475 |
| riscv | 263 | 231 / 4 | 251 (dagP + refine) |
| biriscv | 251 (238 at lam = 4) | 238 / 0 | 238 |

Caveats (RESULTS section 41): single runs, search noise of roughly 1–13% between deterministic starts;
six designs of at most 55k cells; one balance setting.

## Files

- `homotopy_cont.py` current method; `homotopy_dec.py` decoder comparison; `homotopy_ul.py` plane search with
  L1 (max-flow) projection; `lang.py` equal-budget random-direction baseline; `baselines.py` Mt-KaHyPar / dagP /
  his-decoder baselines; `refine.py` acyclic FM + flow refinement; `exact_ir.py` exact L1 isotonic regression.
- `best_parts2/` best-known acyclic partitions used as oracle targets in diagnostic sections (34–38).
- `run_*.bat` Windows launchers used for the logged runs.

Netlist hypergraphs (`data/<design>_hg.npz`, Yosys-synthesised; about 3 GB) are not included.
