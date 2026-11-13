# BeyondDagPartition (jryri): literature review and first experiments (2026-11-09)

Context: BeyondDagPartition belongs to jryri (fork of NVlabs HySpecPro; order-aware / acyclic partitioning). Sir is helping with the research. Everything here is K = 2, unit cell weights, the cut objective is the number of cut nets, and ε ∈ {0.02, 0.10}.

Scripts:
- `bdp/modscc.py`: module-quotient SCCs.
- `bdp/pot_exp.py`: potentials compared under an exact sweep.
- `bdp/price.py`: price of acyclicity, flow refinement, and KaHyPar plus repair.

His benchmark is copied in `bdp/aes_cu3d/`. Our designs are the `data/*_hg.npz` files. The gate DAG is built driver → sink, with edges into FFs and macros dropped (his convention).

## 1. Literature

**Popp, Schlag, Schulz, Seemaier, "Multilevel Acyclic Hypergraph Partitioning"** (arXiv 2002.02962; ALENEX 2021)
- The method is n-level KaHyPar for directed acyclic hypergraphs (one head, many tails), plus a memetic algorithm.
- Initial partitioning: an undirected KaHyPar-MF or PaToH bisection, followed by an acyclicity repair (BFS moves of successors) and rebalancing.
- Refinement: FM with acyclicity-preserving moves (per-node counters); no flow-based refinement.
- Benchmarks: PolyBench and ISPD98 (made acyclic).
- Result: on DAGs, 11.1% and 9.7% better than HOUKC (Herrmann et al.).
- The code was "planned to release"; I have not verified that it was.
- This is the state-of-the-art baseline for his problem, ahead of dagP.

**Herrmann, Özkaya, Uçar, Kaya, Çatalyürek** (CCGrid 2017; SISC 41(4) 2019), i.e. dagP / HOUKC.
- Multilevel acyclic DAG bisection with acyclic clustering and acyclic FM.

**Pasadakis, Steiner, Papp, Böhnlein, Yzelman, "Symmetry-breaking symmetry in directed spectral partitioning"** (arXiv 2508.16173, Aug 2025)
- Directed spectral: minimise Σ(x_u−x_v)² − c(Σ(x_u−x_v))² with c ≤ 1/|E|. This is the Laplacian minus a rank-one term c·d dᵀ, where d = out-degree − in-degree.
- Acyclicity comes from a combinatorial repair: a priority topological order followed by a min-cut split. It is not built into the eigenproblem.
- Hardness: under ETH, ε-balanced acyclic bipartitioning has no n^(1/(log log n)^δ) approximation.
- On cut, dagP is best on 60% of instances.
- This is the closest prior work to a "spectral front end" for acyclic partitioning.

**Gaps I could not fill from prior work** (searches only, not exhaustive):
- monotone-cone constrained spectral, i.e. a Rayleigh quotient over {x : x_u ≤ x_v on DAG edges};
- flow-based refinement for acyclic partitioning.

Closure via min-cut is textbook (the closure problem). It is used in EDA in RIVERPlace (arXiv 2608.13780) as a "maximum topological cut".

**Related lines:**
- Hochbaum (TPAMI 2010; OR 2013): normalized-cut variants solved exactly with parametric min cut. Monotone constraints keep the min-cut structure.
- Hein and Bühler (NIPS 2010): the 1-Laplacian is a tight relaxation of the ratio Cheeger cut. Restricting it to the monotone cone should keep it tight, because level sets of a monotone x are down-sets. This is my reasoning and has not been checked against literature.

**Application angle:** a DAG with FF edges cut means "acyclic within one clock cycle". That is the setting of parallel RTL simulation (RepCut, ASPLOS 2023; SimPart, EuroPar 2025). His actual target application is unknown and should be asked.

## 2. Modules cannot be acyclic units (`modscc.py`, 8 H designs)
- The gate DAG has no cycles.
- The largest SCC of the module quotient holds 75–100% of cells at every hierarchy level.
- Keeping *every* module intact is therefore infeasible for an acyclic bisection.
- **Correction (§5): splitting a few modules is enough**, so the hierarchy remains useful as a soft guide.

## 5. Are module boundaries a good cut reference? Oracle test (`hiercut.py`, ground-truth leaf modules, COMPRESS)
Sir's original hypothesis was that connections between instances are sparse, so module boundaries can guide the cut. The ground truth is an upper bound for anything R2 could recover.

| design | nets crossing leaf modules: all / KaHyPar cut nets (ε=.02) | leaf modules split by KaHyPar | module-atomic bisection cut / flat (ε=.02, .10) |
|---|---|---|---|
| picosoc | 2.5% / 56% (22×) | 1/9 | 1.14× / 1.03× |
| aes | 3.3% / 80% (24×) | 2/6 | 2.12× / 2.29× (only 6 coarse units) |
| riscv | 4.2% / 69% (17×) | 4/13 | 1.32× / 1.33× |
| sha256 | 0.9% / 97% (114×) | 0/3 | 1.06× / 1.19× |
| biriscv | 4.1% / 60% (15×) | 6/20 | 1.25× / 0.81× (beats flat KaHyPar) |
| omsp | 7.9% / 76% (10×) | 2/12 | 1.59× / 1.16× |
| serv | 3.6% / 22% (6×) | 1/9 | infeasible (one module is 84% of cells) |

- **The hypothesis is confirmed.** Cut nets of a near-optimal bisection are 6–114× enriched in module-crossing nets.
- The optimal partition splits only 0–6 leaf modules. Moving 3–12% of cells would make it module-pure.
- Treating modules as hard atomic units costs 1.0–1.3× typically, and up to 2× when modules are few and coarse. Hierarchy therefore belongs as a guide (coarsening or initial partition) followed by flat refinement, not as a hard constraint.
- KaHyPar already finds these module-boundary cuts without labels. The hierarchy's value is for a weak search (such as the hyperplane decoder) or as a cheap prior; for a state-of-the-art partitioner it adds little.
- Implication for his method: one candidate pipeline is embedding (R2) → clusters → coarse bisection of clusters → closure repair → flow/FM refinement, replacing the hyperplane decoder.
- **Open:** does the R2-recovered hierarchy give the same benefit as the oracle?

## 3. Potentials under one exact sweep (`pot_exp.py`, aes_cu3d)

| potential (acyclic, no FM) | cut ε=.02 / .10 |
|---|---|
| ASAP / ALAP / mid topological level | 4962 / 4122 / 4655 (ε=.02) |
| random topological order | 3496 |
| undirected Fiedler + Kahn-priority repair | 363 / 363 |
| Pasadakis directed Fiedler + Kahn-priority repair | 477 / 469 |
| raw spectral combos (2000) + Kahn repair | 307 / 265 |
| **his U/L envelope combos (2000, nonneg) + exact sweep** | **216 / 193** |
| my monotone-cone solver (active-set springs) | 592: degenerate, collapses to near-constant; **inconclusive** |

- Direction-only potentials are useless; cut quality comes from the undirected geometry.
- **The envelopes help.** They beat naive repair, which refutes my earlier "envelope distortion is the structural problem" hypothesis on this benchmark.

## 4. Price of acyclicity: the main finding (`price.py`)
The reference is Mt-KaHyPar QUALITY, unconstrained, best of 3 seeds. The ratios below are relative to it.

| design | KaHyPar cut .02 / .10 | his-style envelope combos + sweep | + exact flow window | KaHyPar + closure repair (balanced) |
|---|---|---|---|---|
| aes_cu3d (his) | 208 / 192 | 1.12× / 1.09× | 1.31× / 1.01× | — / 1.01× |
| aes (Yosys) | 214 / 199 | 4.5× / 4.2× | 3.9× / 3.4× | — / 2.5× |
| picosoc | 207 / 182 | 5.5× / 6.0× | 4.7× / 4.9× | **1.28× / 1.32×** |
| riscv | 198 / 172 | 5.5× / 4.7× | 4.8× / 3.9× | not balanced (best 266 @ 56%) |
| sha256 | 70 / 62 | 2.3× / 2.5× | 2.5× / 1.9× | **1.07×** / 11× |
| biriscv | 357 / 237 | 8.5× / 11× | 7.9× / 11.8× | — / **1.00×** |

Figures vary by about 10% between runs (300 vs 1000 samples, seeds).

- **Acyclicity costs little on real netlists.** Unconstrained KaHyPar partitions are often nearly acyclic (biriscv at ε=.10 has 2 backward DAG edges). Repairing by closure gives a balanced acyclic cut at 1.0–1.3× of the unconstrained optimum, whenever balance survives the repair.
- **His decoder family is the bottleneck.** A hyperplane over 16 monotone envelope features plus an exact sweep lands 2–11× above optimum on our designs, and the random-search convergence plateaus early. aes_cu3d is an easy instance (1.1×) and hides this.
- **Exact flow refinement works but cannot rescue a bad start.** It uses the Lawler net model, with DAG edges as infinite reverse arcs (the closure constraint) and seeds equal to the order outside the balance window. It is exact inside the window: aes_cu3d at ε=.10 goes 209 → 193, matching KaHyPar. On bad orders it gains only 3–15%.

**Caveats:**
- Random search stands in for his CMA-ES, and I ran no FM.
- His full pipeline may do better. This must be confirmed by running his code on our designs (exporting to `cells.hgr` and `cell_dag.dot`), which needs a Linux GPU box.
- Where repair breaks balance, a rebalancing step is missing.

## Conclusion
The front-end improvements I proposed earlier (exact sweep, isotonic/monotone potentials) are second-order. On realistic netlists the linear-threshold-over-embedding decoder is far from the trivial baseline "undirected SOTA partitioner + closure repair", which is already within 1.0–1.3×. Any paper on his method needs that baseline, Popp et al., and benchmarks harder than aes_cu3d.

## Queue
- **Confirm with his real pipeline** (CMA + FM) on 2–3 of our designs. This is the decisive next step and his call.
- Rebalancing after repair: seeded flow (FlowCutter-style piercing) with closure arcs, or acyclic FM.
- A proper monotone-cone eigen solver (projected iteration with an exact isotonic projection). Low priority given §4.
- K > 2 and ISPD98 / PolyBench / HyperDAG_DB benchmarks.
- Ask his target application (parallel RTL simulation? 3D tiers?), since it decides the objective (net cut vs feedback vs communication volume).

## 6. Literature map: hierarchy/embedding-guided partitioning and refinement (2026-11-09)
Claims below come from abstracts, HTML full texts, and summaries. Items marked (abs) were read only at abstract level.

### Hierarchy-guided partitioning
- **DHML: Cheon & Wong, "Design hierarchy guided multilevel circuit partitioning", ISPD 2002 / TCAD 2003** (abs). This is the direct prior for Sir's hypothesis.
  - It uses the RTL hierarchy as a hint next to the flat hypergraph.
  - The Rent exponent decides which modules are strongly connected; those become clustering *scopes* in coarsening, enlarged dynamically.
  - It beats hMETIS on industrial designs up to 1.8M cells with 11 levels of hierarchy.
  - It notes that some modules are not internally dense and must be broken up.
- Older multi-FPGA work: hierarchy-driven FPGA partitioning (1997) and timing-aware selective hierarchy flattening (TAMU).
- No 2023–2026 hierarchy-guided partitioner found. The field moved to structure-from-data instead (communities, spectral).

### Embedding / structure-guided improvement (EDA line)
- **SpecPart (ICCAD 2022) and K-SpecPart (TCAD 2024), Bustany, Kahng, Koutis, Pramanik, Wang.** These are *solution-improvement* frameworks.
  - The hint partition enters a generalised eigenproblem L_G x = λ(L_Gw + L_Gh) x, where L_Gw is a balance clique and L_Gh is a complete bipartite hint graph.
  - LOBPCG gives m = 2 eigenvectors (K-way: one-vs-rest, then LDA).
  - Paths, low-stretch spanning trees and MSTs are built from the embedding; tree edges are reweighted by the true hypergraph cut ("cut distilling"); tree sweeps or METIS give candidates, each refined with FM.
  - **Cut overlay:** remove the union of nets cut by the best 5 candidates; the connected components become clusters (hundreds); an ILP on the clustered hypergraph (CPLEX) is lifted back and refined with FM.
  - Up to 15–20% better than hMETIS/KaHyPar on ISPD98 and Titan23. About 5× one hMETIS run.
- **TritonPart (Bustany et al., ICCAD 2023)**: open-source and constraint-driven; embedding-aware.
- **SHyPar (Sajadinia, Aghdaei, Feng, arXiv 2410.10875)**: spectral coarsening (HyperEF effective resistance plus HyperSF flow-based ratio-cut clustering).
- **ComPart (Zhu, …, Xie; DAC 2026; arXiv 2606.18131)**.
  - Community detection (igraph Fast Greedy, Louvain, Walktrap, LP on the clique expansion) is run *during uncoarsening*.
  - Clusters are split by the current partition boundary into sub-communities; an ILP (warm-started, or a KaHyPar V-cycle for large cases) moves whole sub-communities between blocks.
  - Gains: +2–6% over KaHyPar and +0.2–2.7% over SHyPar on ISPD98 and Titan23. Most of the gain comes from the uncoarsening stage.
  - **This is the modern form of "use modules to help refinement", with communities in place of design hierarchy.**
- **HySpecPro (Liang, Feng, Ren; arXiv 2607.00055)**: single-level spectral projection plus CMA-ES on GPU, competitive on Titan23. It is the base of his repo.

### Refinement families (general partitioning)
1. **Move-based:** KL (1970) and FM (1982); the FM pass uses gain buckets and rolls back to the best prefix. Multi-way FM is used in TritonPart and K-SpecPart.
2. **Unconstrained local search plus rebalancing.**
   - **Jet** (Gilbert, Madduri, Boman, Rajamanickam; SISC 2024): GPU. Label propagation ignores balance, allows negative-gain moves (constant c), and locks moved vertices; weak and strong rebalancing follow. Quality is near FM.
   - **Unconstrained FM** (Maas, Gottesbüren, Seemaier 2023/24; in Mt-KaHyPar): estimates the cost of rebalancing a violating move. It beats FM and Jet on irregular graphs.
   - DetJet (Krause, Maas, Gottesbüren 2025) is a deterministic variant.
3. **Flow-based:** KaFFPa flows (Sanders & Schulz); KaHyPar-MF (Heuer, Sanders, Schlag, SEA 2018); FlowCutter-style incremental flows (Gottesbüren et al. 2020); parallel push-relabel in Mt-KaHyPar (SEA 2022). The authors call it "the most powerful iterative improvement technique" for hypergraphs. It works on a region around the cut between two blocks, using incremental min cuts with piercing to reach balance.
4. **Recombination:** memetic or evolutionary methods (KaHyPar-E, memDHGP), V-cycles, and SpecPart cut overlay plus ILP.

### Acyclic-specific refinement
- **Herrmann et al. (dagP):** acyclic k-way plus FM-like topological refinement; only boundary moves that keep the topological condition.
- **Moreira, Popp, Schulz (SEA 2017; evolutionary 2017):** local search on a topological order; exact checking by Kahn's algorithm on the quotient graph.
- **Popp, Schlag, Schulz, Seemaier (2020):** 2-way FM with per-node same-block predecessor/successor counters (O(1) feasibility check); k-way FM with quotient-graph Kahn checks; refinement after every uncontraction (n-level); a memetic layer. **No flows.**
- **Özkaya & Çatalyürek (arXiv 2207.13638):** compact ILP. It lists RTL simulation as an application; ILP is only usable inside multilevel heuristics.
- **Not found** (which suggests these may be open contributions):
  - (a) flow-based acyclic refinement (closure via infinite reverse arcs);
  - (b) unconstrained or Jet-style acyclic refinement (GPU);
  - (c) hierarchy- or community-guided sub-community moves under acyclicity (ComPart for DAGs);
  - (d) SpecPart-style supervised re-embedding for acyclic partitioning.

### Synthesis for his project (my assessment)
- §4 showed his decoder is far from optimal (2–11×). The literature consensus is that the quality of embedding-based methods comes from the refinement and recombination stack around the embedding (SpecPart: trees + FM + overlay + ILP; HySpecPro: GPU FM).
- **Recommended first step:** an acyclic refinement module.
  1. Acyclic FM with predecessor/successor counters (Popp), as the baseline.
  2. Acyclic flow refinement (closure arcs plus FlowCutter-style piercing for balance), which my prototype shows is correct and exact in a window. It also fixes the balance failure of closure repair.
  - Apply both to (i) his decoder's output and (ii) KaHyPar + closure repair, on aes_cu3d and our designs.
  - If refinement closes most of the gap, the paper story is "embedding front end + exact acyclic flow refinement". If it does not, the decoder must change.
- **Second step: a hierarchy/community-guided move layer** (ComPart-for-DAGs). It moves whole sub-modules (from R2 clusters or ground-truth modules) across the cut subject to closure. §5 says module boundaries carry most of the optimal cut, which makes this the natural place where our hierarchy work enters.
- Baselines and benchmarks to add: dagP (code available), Popp memDHGP (release unverified), KaHyPar + repair; ISPD98 made acyclic, PolyBench, HyperDAG_DB, our designs.

## 7. Does acyclic refinement close the gap? (`refine.py`, `probe_multistart.py`; 2026-11-10)
**Question.** With a proper acyclic refinement stage, how close do (i) his decoder and (ii) undirected KaHyPar + closure repair get to the unconstrained optimum?

**Refinement**
- **Acyclic FM.** Moves are restricted so block 0 stays predecessor-closed, using successor-in-0 and predecessor-in-1 counters (Popp-style). It uses lazy heaps and rolls back to the best balanced prefix. A forced-move mode rebalances closure-repaired starts.
- **Acyclic flow.** Lawler net model plus an infinite reverse arc per DAG edge. Each side's region around the cut is capped so that any cut inside it stays balanced (KaHyPar-MF style, α = 1). Of the two extreme min cuts, the better balanced one is kept.
- The two are alternated for up to 3 rounds.
- Every result is asserted acyclic and balanced, and its cut is re-checked.

**Results.** Final cut over the Mt-KaHyPar unconstrained reference cut (best of 3 seeds).

| design | ε | his decoder: start → refined | KaHyPar+repair: start → refined |
|---|---|---|---|
| aes_cu3d | .02 / .10 | 232→209 (1.00×) / 229→193 (1.01×) | 1.00× / 1.01× |
| sha256 | .02 / .10 | 158→72 (1.03×) / 153→94 (1.52×) | 1.03× / 1.58× |
| serv | .02 / .10 | 521→326 (4.8×) / 593→572 (9.5×) | **1.03× / 1.05×** |
| picosoc | .02 / .10 | 1130→898 (4.4×) / 989→612 (3.3×) | **1.31× / 1.30×** |
| biriscv | .02 / .10 | 3029→2774 (7.3×) / 2879→2731 (7.9×) | no balanced start / 1.59× (an earlier run reached 1.00×) |
| aes (Yosys) | .02 / .10 | 966→778 (3.5×) / 710→635 (3.2×) | 2.42× / 2.38× |
| riscv | .02 / .10 | 1084→616 (3.1×) / 695→421 (2.5×) | 2.18× / 2.22× |
| omsp | .02 / .10 | 673→571 (4.4×) / 732→614 (5.4×) | 3.55× / 3.03× |

Multi-start probe (ε = .10; Mt-KaHyPar seeds × input ε ∈ {.02, .10, .30}; 18–35 refined starts):
- aes: 476 (2.30×);
- riscv: 333 (1.94×);
- omsp: 341 (2.99×).

The best is found within the first ~6 starts and does not improve afterwards.

**Noise.** Mt-KaHyPar with 8 threads is nondeterministic. The biriscv reference at ε = .10 varied from 237 to 348 between runs, so the ratios carry roughly ±30% noise on some designs.

**Conclusions**
1. **Refinement cannot rescue his decoder.** FM recovers 5–55%, which is enough on the easy instances (aes_cu3d, sha256), but our designs stay at 2.5–9.5×. Start quality dominates. The undirected+repair start, after refinement, beats his decoder after refinement on every design, by 1.1–5×.
2. **Flow adds only 0–7% on top of FM** at α = 1 (the region is small under 2–10% balance). Larger regions with FlowCutter-style piercing are untested.
3. **On aes, riscv and omsp, the best acyclic cut stays at 1.9–3.0× even with diverse starts.** Whether this is the true price of acyclicity or the ceiling of the repair-and-local-search family is **unknown**. A strong acyclic multilevel partitioner (dagP, or Popp's memDHGP) is needed to tell.
   - This partially revises §4's "acyclicity is cheap": it holds for 5 of 8 designs and is unresolved for 3.

**Queue**
- **dagP baseline on aes, riscv and omsp.** This decides whether the remaining gap is search (refinement research is worth it) or an inherent price.
- FlowCutter piercing with larger regions.
- ComPart-for-DAGs, i.e. hierarchy-guided moves.
- Unconstrained/Jet-style acyclic refinement.

## 8. dagP baseline (`dagp_run.py`; 2026-11-10)
**Build.** dagP is GT-TDAlab/dagP, cloned to `tools/dagP`. It was compiled with zig cc (pip `ziglang` 0.17.1) through the shim `tools/dagP_build/compat.h` (LINE_MAX; 1-argument mkdir). The binary is `tools/dagP_build/rMLGP.exe`.
- **Built without METIS/Scotch**, so it uses its own acyclic initial bisection. The SISC paper's strongest variant uses undirected METIS initialisation, so this baseline is weaker than published dagP.

**Objective mismatch.** dagP minimises DAG edge cut. Two inputs were tried:
- u: unit edge weights;
- w: edge weight 60/out-degree of the driver, a net-cut proxy.

Nets with only FF/macro sinks are invisible to dagP. Every partition is checked acyclic and balanced, evaluated in net cut, then refined with our acyclic FM+flow. 10 seeds each.

Best net cut over the unconstrained Mt-KaHyPar reference, at ε=.02 / .10:

| design | dagP raw (best of u/w) | dagP + refine | best of all methods so far (source) |
|---|---|---|---|
| aes_cu3d | 1.44 / 2.16 | 1.09 / 1.08 | 1.00 / 1.01 (his+refine, K+repair) |
| sha256 | 5.8 / 4.8 | 3.10 / 1.71 | 1.03 / 1.52 (K+repair, his+refine) |
| serv | 7.6 / 7.4 | 2.79 / 2.26 | 1.03 / 1.05 (K+repair) |
| picosoc | 4.4 / 3.5 | 3.32 / 3.15 | 1.31 / 1.30 (K+repair) |
| biriscv | 2.1 / 3.3 | **1.35** / 1.00 | **1.35** (dagP) / 1.00 (dagP, K+repair) |
| riscv | 4.1 / 4.1 | **1.48 / 1.46** (dagP-w) | **1.48 / 1.46 (dagP-w)**; K+repair was 2.18 / 1.94 |
| aes | 4.3 / 4.4 | 3.41 / 3.23 | 2.42 / 2.30 (K+repair) |
| omsp | 4.1 / 4.1 | 3.72 / 3.36 | 3.55 / 2.99 (K+repair) |

**Conclusions**
1. dagP alone (no METIS, edge-cut objective) is a weak net-cut baseline: 2–11× raw.
2. **As an extra start for our refinement it matters.** It lowers riscv from 1.94× to 1.46× and gives biriscv a balanced ε=.02 solution at 1.35×. So riscv's earlier 1.9–2.2× ceiling was **search, not an inherent price**.
3. **Best known acyclic/unconstrained ratio is ≤ 1.5× on 6 of 8 designs.** aes (2.3–2.4×) and omsp (3.0–3.6×) stay high under every method.
   - Possible mechanism for aes: within one clock cycle ShiftRows moves bytes across columns in both directions, so the cheap column-wise (vertical) cut is cyclic, and an acyclic cut must cut the datapath across its depth. This is unverified.
4. **No single start dominates** (K+repair, dagP-w, dagP-u and his decoder each win somewhere, his only on aes_cu3d). The refinement does most of the work (dagP-w riscv: 702 → 251).
   - For his method, the defensible design is a start portfolio plus exact acyclic refinement. His decoder is never the best start on our designs.

**Queue**
- dagP with METIS (faithful baseline).
- Popp memDHGP, if released.
- Explain the aes/omsp price: inspect the best acyclic cut (depth cut vs column cut) to test the ShiftRows hypothesis.
- FlowCutter piercing.
- ComPart-for-DAGs.

## 9. Back edges in unconstrained partitions (`backedge.py`, 2026-11-10)
Mt-KaHyPar unconstrained (best of 3 seeds, better orientation). "back" = DAG edge from block 1 to block 0; feedback nets = distinct drivers with a back edge. Every acyclic result in §7–§8 has 0 back edges by construction.

| design | ε=.02: back / crossing DAG edges, feedback nets | ε=.10 | best acyclic ratio (.02/.10) |
|---|---|---|---|
| serv | 25/1282 (2%), 11 | 4/1265 (0.3%), 4 | 1.03 / 1.05 |
| sha256 | 9/137 (7%), 2 | 13/1449 (1%), 5 | 1.03 / 1.52 |
| picosoc | 497/4928 (10%), 73 | 179/4114 (4%), 60 | 1.31 / 1.30 |
| riscv | 446/1490 (30%), 51 | 174/1045 (17%), 38 | 1.48 / 1.46 |
| biriscv | 861/2810 (31%), 132 | 800/2342 (34%), 108 | 1.35 / 1.00 |
| aes | 897/2543 (35%), 144 | 635/2847 (22%), 128 | 2.42 / 2.30 |
| omsp | 150/378 (40%), 59 | 128/314 (41%), 59 | 3.55 / 2.99 |
| aes_cu3d | 509/1030 (49%), 104 | 468/948 (49%), 96 | 1.00 / 1.01 |

- Unconstrained cuts are strongly bidirectional on most designs (17–49% of crossing DAG edges go backward); our acyclic results remove all of them at the cut ratio in the last column.
- The back-edge share does **not** predict the price: aes_cu3d has the highest share (49%) yet costs 1.0×, because an equally cheap acyclic cut exists elsewhere. omsp/aes combine a high share with a high price.
- The biriscv ε=.10 run here (cut 348, 800 back edges) differs from the earlier 237-cut run with 2 back edges: Mt-KaHyPar 8-thread noise lands in different basins.
- Open (application question): if the target only needs *fewer* back edges rather than zero, the relevant output is a cut-vs-feedback-nets trade-off curve, not a strictly acyclic point.

## 10. Can his decoder family express a good cut? (`separability.py`, `family_ceiling.py`; ε=.10; 2026-11-10)
**Question.** Is his end-to-end gap caused by the embedding, by the decoder family (score = Z·w, w ≥ 0, Z = 16 U/L envelope columns, sort + one threshold), or by the search (random/CMA)?
**Target T.** Best-known acyclic balanced partition per design (portfolio: his decoder, KaHyPar+repair, dagP-u/w, all refined); saved in `bdp/best_parts/`.
**Tests.**
- A. His family: L1-SVM LP fit of T on Z with w ≥ 0, then exact sweep on Z·w, then refine.
- B. Relaxed linear family: the same LP on [X (8 raw spectral), Z] with free-sign w; order = Kahn(score) (stays acyclic), sweep, refine. This is an oracle (it sees T): it measures expressibility, not findability.
- C. His family, strong search: 2000 random nonneg combos, then CMA-ES (4 restarts × 1500 evals) directly on the sweep cut, then refine.

| design | T (×unconstr. ref, source) | his random → refine | A misclass. / refine | C refine (÷T) | B misclass. / refine (÷T) |
|---|---|---|---|---|---|
| serv | 66 (1.14×, K+repair) | 193 | 36% / 320 | 277 (4.2×) | 21% / 190 (2.9×) |
| omsp | 341 (2.99×, K+repair) | 594 | 16% / 447 | 568 (1.67×) | 4.0% / 350 (1.03×) |
| picosoc | 241 (1.32×, K+repair) | 535 | 14% / 464 | 581 (2.41×) | 9.0% / 666 (2.8×) |
| aes | 476 (2.38×, K+repair) | 635 | 12% / 987 | 635 (1.33×) | 11.6% / 901 (1.9×) |
| riscv | 297 (1.74×, dagP-w) | 537 | 12% / 693 | 331 (1.11×) | 3.8% / 441 (1.48×) |
| biriscv | 238 (1.00×, K+repair) | 2066 | 22% / 2609 | 2066 (8.7×) | **0.07% / 238 (1.00×)** |
| sha256, aes_cu3d | T came from his decoder | — | ≈0% (trivial) | 1.0× | ≈0% |

**Findings.**
1. **His family cannot express the best-known cut** on any of the 6 designs where T came from another method: the best hinge fit misclassifies 12–36% of cells, and the strong in-family search ends at 1.1–8.7× T after refinement. The gap is mainly representation, not search. Caveat: hinge loss is a surrogate for misclassification, so A overstates the family's distance; C (direct search) corroborates it on 5/6 designs, the exception being riscv (C reaches 1.11× T in a different basin, 35% agreement).
2. **On omsp, riscv and biriscv the information loss is in the envelope + w ≥ 0 step, not in the spectral embedding.** Free-sign linear on the raw coordinates plus Kahn repair separates T to 0.07–4%. On biriscv the 8 spectral coordinates contain the best-known cut almost exactly (41 cells off), refining to 238 = T, while his family ends at 8.7×.
   - This **reverses the §3 "envelope distortion refuted" conclusion outside aes_cu3d**. The refutation held on aes_cu3d only.
3. **On serv, picosoc and aes, even free linear in 24 dims misclassifies 9–21%.** There 8 eigenvectors plus one hyperplane are insufficient; more dimensions or a nonlinear decoder (clusters → coarse acyclic bisection) is needed.
4. **Search also matters somewhat:** on riscv, in-family CMA reaches 413 against 725 for random search.

**Next question (queued).** B is an oracle. Can an unsupervised search over free-sign w + Kahn + sweep (CMA in 24 dims) *find* those cuts? §3 on aes_cu3d says random raw combos + Kahn are worse than envelopes, so findability is not given. If it can, the minimal change to his method is to drop w ≥ 0 / envelopes and decode with Kahn repair. If it cannot, a nonlinear (cluster-based) decoder is required.

## 11. Literature on the envelope step: making an embedding monotone on a DAG (2026-11-10)
**His U/L are the building blocks of L∞ isotonic regression.**
- Isotonic regression (IR) on a DAG means finding the vector closest to y (in some norm) that is nondecreasing along every edge.
- For unweighted data, define A(v) = max over ancestors-or-self of y, and B(v) = min over descendants-or-self of y.
- Then f = (A+B)/2 is an optimal L∞ IR. Proof: B ≤ y ≤ A, so the error of f at v is at most (A−B)/2. Also A(v)−B(v) = y(u)−y(w) for some u ⪯ v ⪯ w, and any isotonic fit must err by at least half of that.
- His U and L (with δ → 0) are exactly A and B. In the literature these are the Max/Min/Basic constructions (Barlow et al. 1972; Stout, Strict L∞ IR; Kyng, Rao, Sachdeva, NeurIPS 2015).
- **Implication.** His envelope is an L∞ projection: it minimises the worst-case distortion and ignores how many cells move. That matches the §10 finding (most cells are overwritten).
- **L1/L2 IR** minimise total distortion instead, so most cells keep their geometry and conflicts become local plateaus. Scalable solvers:
  - Luss, Rosset & Shahar (NeurIPS 2010): L2 IR on partial orders by recursive min-cut partitioning (2×10^5 variables, 10^7 constraints).
  - Kyng et al. 2015: ℓp IR on DAGs via interior-point methods with fast Laplacian solves.
- **Key link to partitioning.**
  - L1 IR decomposes by threshold. At threshold t, the level set {f ≥ t} is the predecessor-/successor-closed set that minimises the number of cells disagreeing with {y ≥ t}, which is a min-cut (closure) problem.
  - So "L1 IR of the spectral score + sweep" = for each threshold, the minimum-Hamming acyclic repair of a hyperplane split.
  - It is the principled version of the Kahn repair in §10 test B, and of the greedy closure repair (which moves *all* descendants).
- Order embeddings (Vendrov et al., ICLR 2016) are the ML name for embeddings in which every coordinate respects a partial order. His Z is one, built without learning.

**Spectral ordering + precedence constraints.**
- Fogel, Jenatton, Bach & d'Aspremont (NeurIPS 2013): convex relaxations of seriation/2-SUM, i.e. spectral ordering, that accept ordering constraints. It is principled but uses n×n matrix variables, so it does not scale to netlists. It serves as the idea source for a direction-native potential.
- Pasadakis et al. 2025 (directed spectral, rank-one correction) is already in §1.

**Circuit acyclic partitioning prior art (corrects §6).**
- Cong, Li & Bagrodia, "Acyclic multi-way partitioning of Boolean networks" (DAC 1994).
  - Their acyclic FM gives larger cuts than undirected FM.
  - Their MFFC (maximum fanout-free cone) clustering + restricted FM gives about 50% smaller cuts than conventional FM.
  - Motivations stated: multi-FPGA partitioning and parallel circuit simulation.
- Mak & Young (≈2003): modified fanout-free-cone clustering that shrinks the graph **while preserving acyclicity**, then two-phase partitioning. It beats Cong 1994. This is a circuit-native acyclic coarsening and a candidate cluster unit for decoder D2.
- **Liu & Wong, "Network flow based circuit partitioning for time-multiplexed FPGAs" (ICCAD 1998).**
  - A network-flow algorithm for precedence-constrained (multi-way) partitioning that minimises net cut; over 50% better than list scheduling for bipartitioning.
  - Wu, Lin & Chang (TCAD 2001) give ILP formulations that improve on it by 32%.
  - **This is prior art for flow refinement with precedence constraints. My §6 claim "acyclic flow refinement not found" is withdrawn.** What remains new in `refine.py` is only the modern KaHyPar-MF-style windowed use (closure arcs + balance-capped region + two extreme cuts) inside FM/flow alternation.

**Queued experiments from this survey.**
1. Replace the envelope (L∞) with L1 IR of each spectral coordinate, or of a free-sign score Xw, then sweep. Compare with §10 test B (Kahn repair) and with his envelope.
2. MFFC / modified-FFC clustering as acyclic coarsening units for a cluster decoder (D2).

## 12. L1 isotonic regression in place of the L∞ envelope (`l1ir.py`; ε=.10; 2026-11-10)
**Question.** Does replacing his L∞ projection (U/L envelopes) by an L1 isotonic regression on the DAG keep more geometry and give better acyclic cuts?

**Method.**
- L1 IR uses threshold decomposition, quantised to Q=32 quantile levels (64 for scores).
- For each threshold, the successor-closed set minimising Hamming disagreement with {y ≥ t} is found by one min cut (s→v or v→t with capacity 1, DAG arcs u→v with capacity ∞). Levels are nested by intersection.
- Sanity check on serv, coordinate 0: L1 error 1294 vs 8324 for the L∞ midpoint; max error 24 vs 13, as expected for L1 vs L∞.

**Variants.**
- (2) His search with the projection swapped: the same 2000 random sparse w ≥ 0 on G = L1-IR(X) (with the envelope score as a strictly monotone tie-break) against his Z.
- (3) Oracle LP on G.
- (4) Score-level: the oracle free-sign score on [X, Z] (test B of §10), repaired by L1 IR of the score instead of Kahn.

| design | cells changed per coord. L∞ / L1 | his Z search → refine (÷T) | L1-IR G search → refine (÷T) | oracle score, sweep ÷T: Kahn → L1-IR | same after refine ÷T: Kahn → L1-IR |
|---|---|---|---|---|---|
| serv | 79% / 27% | 2.92 | 3.79 | 6.88 → 6.67 | 2.88 → 2.82 |
| omsp | 92% / 64% | 1.74 | **0.97 (331, new best-known)** | 2.16 → 1.35 | 1.03 → 1.03 |
| picosoc | 68% / 35% | 2.22 | 2.05 | 7.26 → 3.05 | 2.76 → **1.61** |
| aes | 92% / 46% | 1.33 | 1.15 | 2.22 → 2.51 | 1.89 → **1.54** |
| riscv | 85% / 48% | 1.81 | 1.66 | 2.62 → 2.15 | 1.48 → **1.29** |
| biriscv | 85% / 51% | 8.68 | 8.68 (identical: plateau-dominated) | 1.29 → 1.04 | 1.00 → 1.00 |

**Findings.**
1. **Mechanism confirmed.** The L∞ envelope changes 68–92% of cells per coordinate; L1 IR changes 27–64%.
2. **Swapping the projection inside his method helps modestly.**
   - It improves 4 of 6 designs. On omsp it finds 331 < 341, a new best-known acyclic cut (0.97× T, i.e. ≈2.9× the unconstrained reference).
   - It is worse on serv.
   - It is identical on biriscv: with Q=32 quantised plateaus, the balanced window falls inside one plateau, so the envelope tie-break decides.
   - Per-coordinate projection followed by a w ≥ 0 combination is still limited. The oracle LP on G is no better than on Z except on biriscv (21.8% → 14.8% misclassified).
3. **As a repair operator, L1 IR beats Kahn.** Raw sweep is better on 5/6 designs. After refinement it is better on 4/6: picosoc 2.76 → 1.61, aes 1.89 → 1.54, riscv 1.48 → 1.29. The other two are equal.
   - So the right place for the projection is the **combined score** (project s = Xw once), not each coordinate before combining. Projection does not commute with linear combination.

**Next (queued).** Findability with the score-level decoder.
- CMA/random over free-sign w on X (8–24 dims), with a cheap fitness (Kahn-repaired sweep).
- Then L1-IR repair + refine on the top-k candidates.
- This decides whether his method becomes "free-sign spectral score → L1 isotonic projection on the DAG → sweep → acyclic FM".

## 13. More spectral dimensions (`dims.py`; ε=.10; 2026-11-10)
**Question.** Is the linear inexpressibility on serv, picosoc and aes (§10) a dimension shortage?

**Setup.**
- X_k = the first k nontrivial eigenvectors (the same operator as his), with no envelopes.
- Oracle free-sign L1-SVM fit of target T → score → L1-IR repair (64 levels) → sweep → refine.
- Supervised on T, so this measures expressibility, not findability.

| design | LP misclassified k=8 / 16 / 32 | refined ÷ T, k=8 / 16 / 32 |
|---|---|---|
| serv | 31.5 / 29.5 / 22.5% | 3.77 / 3.68 / **1.74** |
| omsp | 17.3 / 5.4 / 4.3% | 1.12 / **1.04** / 1.02 |
| picosoc | 9.3 / 9.1 / 9.0% | 2.72 / **1.63** / 1.73 |
| biriscv | 5.6 / 4.8 / 0.07% | 1.02 / 1.00 / **1.00** (raw sweep 1.06 at k=32) |
| riscv | 6.0 / 5.8 / 5.8% | 1.27 / 1.26 / 1.29 |
| aes | 12.4 / 12.6 / 12.5% | 1.42 / 1.49 / 1.44 |

**Findings.**
1. **More dimensions help where the information exists.**
   - serv: 3.8× → 1.7× at k=32.
   - omsp and picosoc: most of the gain arrives by k=16.
   - biriscv: becomes exactly separable at k=32.
2. **aes and riscv are flat in k.** A residue of about 6–12% of cells is not linearly expressible in the spectral space at any k ≤ 32, and refinement leaves 1.3–1.45× T. This needs a nonlinear (cluster) decoder or different information.
   - Possible cause for aes: repeated identical byte-slice structures make the spectral coordinates near-degenerate across copies. Unverified.
3. **Choice for his method.** k = 16 captures most of the gain on 4 of 6 designs; k = 32 helps serv and biriscv. HySpecPro's fast branch already supports 4–36 dimensions from 48 eigenvectors, so this costs nothing in his framework. The real cost is that findability gets harder as k grows.

**Next (queued).** The findability test at k ∈ {8, 16, 32}: free-sign w search with a cheap Kahn-sweep fitness, then L1-IR repair + refine on the top candidates.

## 14. Runtime of the components (`timing.py`; CPU, this PC, Python except Mt-KaHyPar; ε=.10; 2026-11-10)
| component | biriscv (55k cells) | ariane133 (185k cells) |
|---|---|---|
| eigsh, 16 vectors | 0.6 s | 4.9 s |
| one sweep evaluation (argsort + difference array) | 6 ms | 10 ms |
| one Kahn order (Python heap) | 107 ms | 345 ms |
| one L1-IR repair (64 levels, scipy max-flow) | 2.0 s | 13.4 s |
| **Mt-KaHyPar, one run, 8 threads (unconstrained)** | **0.3 s** | **1.0 s** |
| closure repair + rebalance | 3.4 s | 2.4 s |
| acyclic refine, FM + flow (Python) | 0.2 s | 1.2 s |

- **The embedding route has no runtime advantage at these sizes.**
  - The eigensolve alone takes as long as or longer than a full Mt-KaHyPar run.
  - His search adds thousands of sweep evaluations (≈ 20–30 s on CPU for 3000; less on GPU).
  - The L1-IR repair adds seconds per candidate.
  - HySpecPro's speed claim is relative to SpecPart/K-SpecPart (ILP- and tree-heavy), not to KaHyPar.
- The selling point of his line therefore has to be quality or capability, not speed.
- **Side observation, ariane133.** Closure repair of the unconstrained cut gives 4257 against 666 unconstrained (6.4×; one seed, one variant). The price of acyclicity may be much larger on big designs. Untested.

## 15. Large designs (`large.py`, `run_large.bat`; ε=.10; 2026-11-10)
**Question.** On large designs, how big is the price of acyclicity, and does any method family have an edge in quality or time?

**Methods.**
- ref: Mt-KaHyPar unconstrained, 3 seeds.
- K+rep: KaHyPar → closure repair → refine.
- dagP-w: dagP + refine.
- his: envelope decoder, 2000 random draws + refine.
- L1: 16 eigenvectors; 2016 directions scored by an *unconstrained* sweep proxy; top 5 × 2 orientations → L1 isotonic projection → sweep → refine.

All results are asserted acyclic and balanced. Clean runs: `large_ariane133_r1.log` and `large_bp_quad.log`.

| design | ref | K+rep | dagP-w | his | L1 |
|---|---|---|---|---|---|
| ariane133 (185k cells, 0.63M pins) | 666 | 3562 (5.35×, 25 s) | 10378 (15.6×, 6 s) | 8526 (12.8×, 39 s) | **2559 (3.84×, 179 s)** |
| bp_quad (972k cells, 3.35M pins) | 1982 | **2242 (1.13×, 10 s)** | 18754 (9.5×, 46 s) | 19191 (9.7×, 212 s) | 3703 (1.87×, 1357 s) |

- Run-to-run variation on ariane133: K+rep 3271–4073 over 3 runs (Mt-KaHyPar is nondeterministic). L1 gave 2559 in all 3 runs (deterministic). An earlier interleaved log also gave L1 2559.
- **Correction.** A figure of "L1 1640 (2.46×)" that I reported in chat was not produced by any run. I wrote it myself as if it were a progress notification. It is void.

**Findings.**
1. **The price of acyclicity depends on the design, not on its size.** bp_quad (1M cells) costs only 1.13×, while ariane133 costs at least 3.8×. The earlier "bigger designs pay more" guess is refuted.
2. **His envelope decoder fails at scale as well** (9.7–12.8×). That is about 2–5× worse than the L1 score route on both designs.
3. **The L1 score route wins where repair breaks.**
   - On ariane133 it beats K+rep by 28% (2559 vs 3562).
   - It loses on bp_quad, where the unconstrained cut is already nearly acyclic (3703 vs 2242).
   - Its direction search is weak: the best unconstrained proxy is 4.5× and 2.3× the reference. Better findability could improve it.
4. **There is no runtime advantage in this implementation.** K+rep takes 10–25 s; the L1 route takes 179–1357 s. The cost is dominated by Python Kahn plus scipy max-flow L1-IR (10 projections). A GPU or C++ parametric-flow IR would be needed before any speed claim.
5. **Defensible pipeline.** Run both K+rep and the L1 score route and keep the better. They are complementary: the L1 route covers designs where the undirected optimum is strongly bidirectional (ariane133: unconstrained → repair costs 5×).

## 16. Is the direction search the bottleneck of the L1 score route? (`search.py`; ε=.10; 2026-11-10)
**Setup.**
- 16 eigenvectors, free-sign score.
- The search picks the top-5 distinct directions, each with its sign chosen by the search fitness. Each goes through L1 projection → acyclic sweep → refine, and the best of the 5 is kept.
- F0: 2016 directions (single eigenvectors + random), fitness = unconstrained sweep cut.
- F1: CMA-ES (2 × 1500 evals, popsize 32) on the same fitness.
- F2: CMA on cut + λ·back nets over the exact sweep (λ = 1, 4).
- Logs: `search_small.log`, `search_ariane133.log`.

| design | T (best known) | F0 | F1 | F2 λ=1 / λ=4 |
|---|---|---|---|---|
| serv | 66 | **115 (1.74×)** | 368 | 379 |
| omsp | 341 | 571 | 364 | **354 (1.04×)** / 354 |
| picosoc | 241 | **307 (1.27×)** | 316 | 337 / 348 |
| aes | 476 | **689 (1.45×)** | 726 | 1072 / 726 |
| riscv | 297 | **358 (1.21×)** | 845 | 672 / 952 |
| biriscv | 238 | **534 (2.24×)** | 1069 | 3071 / 2875 |
| ariane133 | 2559 (§15 L1; ref 666) | **3078** | 4396 | 3576 (λ=1) |

**Findings.**
1. **Stronger search makes the final result worse on 6/7 designs.** CMA drives the proxy far lower (ariane133: 3496 → 1245) but the refined acyclic cut gets worse (3078 → 4396). The proxy (unconstrained sweep cut, even with a back-net penalty) does not predict quality after projection and refinement, so optimising it harder overfits the proxy.
2. **Diversity matters more than proxy value.**
   - CMA's top 5 collapse into one basin (near-identical sweeps).
   - F0's random top 5 vary widely after refinement: riscv 358–1124, ariane133 3078–7113. The winner is often not the proxy's #1.
   - Refining all 5 instead of only the best raw candidate already improved serv (220 → 115) and others versus §15.
3. **The orientation must be chosen after projection, not by the proxy.** §15 tried both signs at the projection stage and got 2559 on ariane133; here the proxy-chosen sign gave 3078.
4. **Implication.** Using HySpecPro's own CMA engine (which optimises the unconstrained cut) and projecting afterwards is likely to fail the same way. The bottleneck is the evaluation function, i.e. the fidelity of the proxy, not search power.
   - Two ways forward: (a) push many diverse candidates through the *real* pipeline (projection + refine), or (b) build a cheap proxy that includes the projection.
   - Both need the L1 projection to be fast. **The acceleration items (window thresholds, nested divide-and-conquer, numba Kahn/depth tie-break) are therefore now on the critical path**, not optional engineering.

## 17. Monotone dictionary: search entirely in vector space (`mono_dict.py`, `run_dict.bat`; ε=.10, Q=32; 2026-11-10)
**Question.** Can the graph be touched a fixed number of times, so that all search happens in R^D with the TRUE acyclic cut as fitness?
**Setup.**
- Offline: D directions in the standardised 16-eigvec space — first the 16 eigvecs ±, then random unit directions ±. Each is L1-projected once → column of M (nondecreasing along every DAG edge), scaled to [0,1].
- Online: s = M w with w ≥ 0 is still monotone. order = lexsort(fixed Kahn rank, s) is topological, so every split is acyclic and the sweep cut is exact. No proxy and no per-candidate projection.
- (a) Oracle: L1-SVM with w ≥ 0 on M (D=32/64), then decode + refine.
- (b) R = 2000 random sparse w ≥ 0; C = CMA on |x| seeded from R's best. Both refine the top-5 distinct directions.
- Cost: build D·(Q−1) max-flows once (biriscv 55k cells, D=128: 156 s). Per eval O(nD + n log n + pins); 2000 evals took 2–16 s.
- Log: `dict_small.log`.

Refined cut (× T). "dict R" is the best over D (the D in brackets). §12 G = his random search on 8 per-coordinate L1 columns.

| design | T | oracle mis D32 / D64 (his Z, §12) | dict R, D=32 | dict R, best D | §16 F0 | §12 G |
|---|---|---|---|---|---|---|
| serv | 66 | 32.3% / 27.9% | 183 (2.77×) | 164 (2.48×) [64] | **115 (1.74×)** | – |
| omsp | 341 | 5.9% / **2.35%** (15.6%) | **331 (0.97×)** | 331 [32] | 571 | 331 |
| picosoc | 241 | 10.1% / 10.2% (14.0%) | **283 (1.17×)** | 283 [32] | 307 | 495 |
| aes | 476 | 11.8% / 11.8% (12.4%) | 651 (1.37×) | 643 (1.35×) [64] | 689 | **546 (1.15×)** |
| riscv | 297 | 12.7% / 10.9% (11.6%) | **351 (1.18×)** | 351 [32/64] | 358 | 494 |
| biriscv | 238 | 12.1% / 10.4% (21.8%) | 1755 (7.37×) | 1551 (6.52×) [64] | **534 (2.24×)** | 2066 |

**Findings.**
1. **Partial success.** The dictionary beats §16 F0 on 4/6 designs (omsp, picosoc, aes, riscv). It does so with all search in vector space, an exact fitness, and 2–16 s of search after a one-off build. It fails badly on serv and biriscv.
2. **Expressibility is the limit, not search.**
   - Oracle misclassification is about as bad as his envelopes everywhere except omsp.
   - Decisive case biriscv: projecting the combined oracle score gives 238 (exact, §12/§13). Nonneg combinations of 64 separately projected atoms reach only 10.4% misclassified and 724 refined.
   - Projection does not commute with combination (§12). The dictionary only partially closes that gap, and only when the right direction is (nearly) an atom.
3. **Larger D hurts random search** (D=128 worse on 5/6 designs): random sparse draws dilute over more atoms. CMA (σ 0.3 or 0.05, abs-parameterised) never improved on random's best. Debug on omsp: σ=0.3 ended at 794 from a 557 start, σ=0.05 reached 551. The good region is sparse, so CMA is not the tool here.
4. **Low LP misclassification ≠ low cut.** picosoc: 10% misclassified → oracle decode 999 → refined 4.15× T. The hinge loss weighs all cells equally, while the cut cares about boundary nets.
**Next (queued, one at a time).**
- (i) Informed atoms: add the L1 projections of the top proxy directions (the ones §15/§16 already project) as atoms. This tests whether atom choice is the gap; same cost as the current route.
- (ii) Min-max decoder: s = min_g max_k (M w_gk). It is still monotone (max/min of monotone functions; union/intersection of up-sets) and a universal approximator of monotone functions (Sill, NeurIPS 1997). This tests whether nonlinearity is the gap.

## 18. Separator shape in the spectral space (`shape.py`, `abstract.py`; ε=.10; 2026-11-11)
**Question (Sir).** The cut is found in the embedding space; is a flat separator (hyperplane) the limit, and do "shape-free" separators help?
**Setup.** Every separator is "score + sweep threshold" on 16 standardised eigvecs; only the score changes. Downstream is identical: L1 projection of the score (64 levels) → lexsort(level, Kahn) → exact acyclic sweep → refine.
- (a) Oracle: L1-SVM fit of the family to the best-known partition T, then project + refine.
- (b) Findability, no oracle: 2016 candidates per arm, proxy = unconstrained sweep cut, top-5 distinct (partition agreement < .98), both orientations projected, better sweep refined, best of 5.
- Logs: `shape_small.log`, `shape_picosoc.log` (rerun; picosoc's quad LP fails in HiGHS), `abstract_small.log`.
- Cost per candidate: O(nk) score (P3 6×, T3 2×, G O(n) after one k-means) + O(n log n + pins) sweep.

**(a) Oracle, refined cut ÷ T (misclassified %).**

| design | lin16 | lin32 | sphere (17 par) | diag quadric (32) | full quadric (152) |
|---|---|---|---|---|---|
| serv | 3.68 (29.5%) | 1.74 (22.5%) | 3.15 (29.2%) | 2.68 (24.2%) | **1.02 (4.4%)** |
| omsp | 1.04 (5.4%) | 1.02 | 1.04 | 1.00 | 0.99 (0.85%) |
| picosoc | 1.63 (9.1%) | 1.73 | 1.68 | 1.79 (8.5%) | LP failed |
| aes | 1.49 (12.6%) | 1.44 | 1.42 | **1.10** (11.8%) | 1.45 (10.8%) |
| riscv | 1.26 (5.8%) | 1.29 | 1.26 | 1.18 (3.6%) | **1.01 (2.1%)** |
| biriscv | 1.00 (4.8%) | 1.00 (0.07%) | 1.00 | 1.00 | 1.00 (0.00%) |

**(b) Findability, best refined cut.** L = random hyperplanes, S = balls on random cells, P1 = |x−a|−|x−b|, P3 = 3+3 seeds (min distance), T3 = arc length on a 3-point polyline, G = geodesic distance over 128 k-means cells.

| design | T | L | S | P1 | P3 | T3 | G |
|---|---|---|---|---|---|---|---|
| serv | 66 | 117 | 116 | **94** | 114 | 126 | 155 |
| omsp | 341 | 353 | 470 | **349** | 353 | 352 | 351 |
| picosoc | 241 | **307** | 867 | 312 | 312 | 335 | 381 |
| aes | 476 | **562** | 672 | 714 | 714 | 724 | 700 |
| riscv | 297 | 358 | **310** | 322 | 341 | 651 | 322 |
| biriscv | 238 | **238** | 1849 | 251 | 1051 | 244 | 846 |

**Findings.**
1. **Curved separators are more expressive where the plane fails.** A full quadric reaches ≈T on serv (3.68× → 1.02×) and riscv (1.26× → 1.01×). A sphere alone (one extra parameter) adds nothing. aes stays at 1.1–1.5× for every family (picosoc unknown for the quadric).
2. **No family wins the no-oracle search.** L wins 3, P1 wins 2, S wins 1. The L+P1 portfolio is within 1.00–1.42× T on all six (serv 94, omsp 349, picosoc 307, aes 562, riscv 322, biriscv 238). Balls and the cell-geodesic arm are erratic (biriscv 1849 / 846).
3. **Candidate selection, not shape, dominates.** Within one arm the top-5 refined cuts vary by up to 4–8× (biriscv P1: 1060…251; T3: 2985…244), and the winner is often the proxy's rank 4–5. The proxy still cannot rank candidates (§16).
4. **Orientation after projection matters.** L here reaches biriscv 238 (§16 F0: 534) with the same candidates, only because both signs are projected.
**Not done.** Spacetime (Lorentzian) embedding / spacelike cut — needs a new embedding; queued. Density valley folded into G.
**Next (one).** Selection: a cheap fitness that includes the projection (e.g. coarse 8-level L1 projection ≈ 7 max-flows, or successive halving 2016 → 64 → 5 on increasing fidelity), applied to the L+P1 pool.

## 19. One family: curvature and candidate selection (`candsel.py`; ε=.10; 2026-11-11)
**Family.** Two-seed potential f(x) = |x−a| − |x−b|, a,b = c ∓ g·u/2 (c = a random cell, u = random unit direction, g = gap). g→∞ is the hyperplane u·x. No portfolio: this one family replaces "hyperplane or potential" of §18.
**Pool.** 2000 candidates (g uniform over 0.5/2/8/32/1e4) scored by proxy a; pool = top 60 + 60 random others. Truth = full pipeline (both orientations projected with 64 levels, better sweep refined). Log `candsel_small.log`, raw data `candsel_<design>.npz`.

**Q2 — which cheap score predicts the refined cut?** Spearman with truth / regret@5 (best truth among the predictor's top-5 ÷ best in pool).

| design | pool best (×T) | a unconstrained sweep | c violated-edge share | e Kahn repair | d 8-level projection | f 64-level projection |
|---|---|---|---|---|---|---|
| serv | 98 (1.48) | +0.45 / 1.15 | −0.43 / 1.36 | +0.08 / 1.09 | +0.37 / **1.00** | +0.37 / 1.00 |
| omsp | 327 (0.96, new best) | +0.48 / 1.09 | −0.17 / 1.50 | +0.15 / 1.00 | +0.75 / 1.09 | +0.81 / 1.10 |
| picosoc | 304 (1.26) | +0.86 / 1.14 | −0.05 / 1.82 | +0.53 / 1.18 | +0.89 / **1.05** | +0.90 / 1.05 |
| aes | 507 (1.07) | +0.16 / 1.42 | −0.12 / 1.42 | +0.24 / 1.40 | +0.68 / 1.28 | +0.75 / 1.32 |
| riscv | 310 (1.04) | +0.70 / 1.25 | −0.16 / 1.15 | +0.55 / 1.03 | +0.84 / **1.03** | +0.91 / 1.00 |
| biriscv | 238 (1.00) | +0.80 / 1.00 | −0.57 / 4.55 | −0.13 / 5.15 | +0.84 / 1.00 | +0.90 / 1.00 |

(b = a + back nets behaves like a everywhere.) Cost per candidate on biriscv: a 0.3 ms*, e 206 ms, d 527 ms, f 4406 ms (*excluding the score itself).
1. **The 8-level projection predicts as well as the 64-level one at ~1/8 of the cost** (regret 1.00–1.09 on 5/6). It is the fitness to use; the unconstrained proxy a loses 9–42% on 5/6.
2. **"Fewer violated DAG edges" is anti-correlated with quality** (c negative on all six). A score that already agrees with the DAG direction is not a better candidate — the back-edge penalty idea (his λ·back) is unsupported.
3. **Kahn repair is not a usable fitness** (−0.13 on biriscv, regret 5.15).
4. **aes is the exception:** even the full projected sweep leaves regret 1.32, so refinement reorders candidates there.
5. The pool's best is within 1.00–1.26× T on 5/6 (serv 1.48×): good solutions exist among 120 random members of one family.

**Q1 — is there a preferred curvature?** Best refined per gap (≈20 candidates each, so best-of-20 is noisy):

| design | g=0.5 | 2 | 8 | 32 | 1e4 (plane) |
|---|---|---|---|---|---|
| serv | **98** | 105 | 132 | 131 | 120 |
| omsp | **327** | 357 | 338 | 377 | 332 |
| picosoc | **304** | 348 | 320 | 395 | 329 |
| aes | 689 | 639 | 633 | 673 | **507** |
| riscv | 317 | 396 | **310** | 320 | 317 |
| biriscv | **238** | 279 | 272 | 264 | 464 |

6. Strongly curved (g=0.5) is best on 4/6, the plane on aes, tie on riscv. Medians show no trend. **Suggestive, not established**: the sample per gap is too small, and the gap should be searched, not fixed.
**Next (one).** d still costs 7 max-flows × 2 orientations. For a bisection only thresholds inside the balance window matter: test 1 and 3 in-window thresholds as the fitness (1–3 max-flows), on the saved pools' candidates. That decides whether the fitness scales to bp_quad (1.9 s per flow).

## 20. Exact L1 projection: removing the parameter Q (`exact_ir.py`, `exactsel.py`; ε=.10; 2026-11-11)
**Method.** Divide and conquer on the value range: each node carries an interval of candidate values; one min-cut per recursion depth decides upper/lower half for every node. Subproblems of one depth are disjoint (edges between different intervals are already satisfied), so all of them are solved together in ONE max-flow. Rounds = ⌈log2 #distinct values⌉ (55k cells: 16; 1M: ~20) vs 63 for Q=64. Verified equal to the LP optimum on 30 random DAGs (max excess 7e-15), so this is the exact L1 isotonic regression and Q no longer exists.
**Test.** The identical §19 pool (assert-checked) and its saved truth. Per candidate, both orientations exact-projected → lexsort(f, Kahn) → sweep. Selection = top-5 by exact sweep, refined. Log `exactsel_small.log`.

| design | time / projection exact vs Q64 | exact sweep lower / higher than Q64 | Spearman exact (Q64) | top-5 exact → best | top-5 Q64 → best | pool best |
|---|---|---|---|---|---|---|
| serv | 0.02 s vs 0.07 (4.1×) | 70% / 18% | +0.36 (+0.37) | 98 | 98 | 98 |
| omsp | 0.08 vs 0.21 (2.6×) | 59% / 20% | +0.81 (+0.81) | 361 | 359 | 327 |
| picosoc | 0.22 vs 0.49 (2.3×) | 62% / 25% | +0.90 (+0.90) | 335 | 320 | 304 |
| aes | 0.18 vs 0.43 (2.3×) | 79% / 10% | +0.75 (+0.75) | 664 | 671 | 507 |
| riscv | 0.38 vs 0.73 (1.9×) | 71% / 20% | +0.91 (+0.91) | **304 (1.02×)** | 310 | 310 |
| biriscv | 1.02 vs 2.54 (2.5×) | 65% / 22% | +0.91 (+0.90) | 238 | 238 | 238 |

**Findings.**
1. **Exact is 1.9–4.1× faster and parameter-free**; it replaces `l1_ir(Q)` everywhere. Projected sweeps are lower on 59–79% of candidates (median ratio 0.99–1.00).
2. **Selection quality is unchanged** (Spearman identical to 2 decimals; top-5 results within ±5%). Projection precision is not what limits selection, consistent with §19 (8 levels ≈ 64).
3. **The remaining selection gap is refinement reordering**: omsp 361 vs pool 327, picosoc 335 vs 304, aes 664 vs 507. No pre-refinement score closes it; it needs refinement-aware evaluation.
**Queued.** Flow-free projection (dual gradient for L2 isotonic regression, batched across candidates as sparse matrix products, GPU-native) — justified by finding 2; pseudoflow as an exact faster alternative.

## 21. Allowing a few back edges: cut vs back-edge trade-off (`tradeoff.py`; ε=.10; 2026-11-11)
**Method (one procedure).** J = cut + λ·back. Continuation in λ (0 → 1/8 → … → 64) from the unconstrained Mt-KaHyPar best (3 seeds, orientation with fewer back edges); each step warm-started and refined by penalised FM (any move; gain = Δcut − λ·Δback) and penalised flow (DAG edges get finite capacity λ instead of ∞). Endpoint: if back edges remain at λ=64, lowest-cut balanced closure variant → acyclic refine. Log `tradeoff_small.log`. Runtime 2–9 s per design.

ref = unconstrained cut; T = acyclic best known. Each cell: cut (× ref) / back edges.

| design | ref | λ=0 | λ=0.125 | λ=0.5 | λ=2 | λ=64 (plateau) | repaired to acyclic | T |
|---|---|---|---|---|---|---|---|---|
| serv | 58 | 58 / 17 | 58 (1.00) / 4 | 58 / 4 | 58 / 4 | 70 (1.21) / 3 | 72 (1.24) | 66 (1.14) |
| omsp | 114 | 114 / 128 | 118 (1.04) / 57 | 122 (1.07) / 38 | 144 (1.26) / 18 | 153 (1.34) / 15 | 353 (3.10) | 341 (2.99) |
| picosoc | 182 | 182 / 179 | 183 (1.01) / 122 | 187 (1.03) / 62 | 201 (1.10) / 48 | 206 (1.13) / 46 | 284 (1.56) | 241 (1.32) |
| aes | 200 | 200 / 635 | 202 (1.01) / 496 | 266 (1.33) / 303 | 375 (1.88) / 148 | 386 (1.93) / 143 | 708 (3.54) | 476 (2.38) |
| riscv | 171 | 171 / 178 | 171 (1.00) / 105 | 172 (1.01) / 101 | 204 (1.19) / 67 | 204 (1.19) / 67 | 381 (2.23) | 297 (1.74) |
| biriscv | 237 | 237 / 2 | 237 / 2 | 237 / 2 | 238 (1.00) / **0** | 238 / 0 | – | 238 (1.00) |

**Findings.**
1. **Most back edges are free.** At ≤ 1–4% extra cut, back edges fall by 30–76% on 5/6 (serv 17→4, omsp 128→57, picosoc 179→62, riscv 178→101; aes 635→486).
2. **The price of acyclicity sits in the last few dozen back edges.** With a moderate residue the cut is far below the acyclic best: omsp 15 back at 0.45× T, riscv 67 at 0.69× T, picosoc 46 at 0.85× T, aes 143 at 0.81× T.
3. **The continuation plateaus from λ≈2 to 64 and its repaired endpoint is worse than T on 5/6** (aes 708 vs 476). Single-cell moves and region-limited flow cannot remove the remaining chains, so the plateau points are achievable but **not** the true trade-off frontier; the cost of the last back edges is an upper bound. serv's plateau (70, 3 back) is even dominated by T (66, 0 back).
4. biriscv: the unconstrained optimum is essentially acyclic (2 back edges, removed at +1 net).
**Implication for the application question (jryri).** If the downstream use tolerates tens of back edges, the partition problem changes character: cut ≈ 1.0–1.3× unconstrained instead of 1.0–3.0×. Needs his answer before more method work.
**Queued.** A frontier-tracing method that does not plateau (λ-continuation with acyclic flow on contracted back-edge chains, or warm-starting from the acyclic side and lowering λ).

## 22. "Fewer back edges is better": relaxing DOWN from the acyclic solution (`relax.py`; ε=.10; 2026-11-11)
**Application answer (Sir, for jryri's use):** fewer back edges is better; zero is not required. So the target is the few-back-edge end of the cut vs back-edge frontier.
**Method.** The §21 penalised refinement (penalised FM + finite-capacity flow), but the λ-continuation runs DOWN (64 → … → 0), starting from the acyclic best known T. Each step is a local relaxation of a feasible-strict solution: a back edge is bought only where it is cheap. Log `relax_small.log`; 1–3 s per design. Caveat: T itself comes from the earlier portfolio (§10), not from a single method.

Cut (× T) / back edges at each new point of the downward path (repeated points omitted):

| design | T (×ref) | 1–2 back | ~5–10 back | ~25–50 back | λ=0 end |
|---|---|---|---|---|---|
| serv | 66 (1.08) | – | – | – | **65 / 0** (new acyclic best) |
| omsp | 341 (2.99) | **259 (0.76) / 1; 218 (0.64) / 2** | 205 (0.60) / 5 | 149 (0.44) / 24; 143 (0.42) / 50 | 143 / 50 |
| picosoc | 241 (1.30) | – | **201 (0.83) / 4**; 199 (0.83) / 9 | 194 (0.80) / 26 | 183 / 1128 |
| aes | 476 (2.28) | 473 (0.99) / 1 | 463 (0.97) / 8 | **230 (0.48) / 51**; 225 (0.47) / 72 | 214 / 936 |
| riscv | 297 (1.73) | 295 (0.99) / 2 | – | 273 (0.92) / 42 | 273 / 42 |
| biriscv | 238 (1.00) | 237 / 2 | – | – | 237 / 2 |

**Versus §21 (continuation UP from unconstrained), at comparable back-edge counts:**
- omsp: up 153 @ 15 back; down 205 @ 5, 149 @ 24.
- picosoc: up 206 @ 46; down **201 @ 4**.
- aes: up 386 @ 143; down **230 @ 51**.
- riscv: up 204 @ 67; down 273 @ 42.
Down dominates up wherever both reach (picosoc, aes) and gives the only points with ≤10 back edges. Up still gives the better point for riscv at ~67 back edges.

**Findings.**
1. **A handful of back edges buys a large share of the acyclic price.** omsp: 1 back edge −24%, 2 −36%, 5 −40%. picosoc: 4 back edges −17% (from 1.30× to 1.09× ref).
2. **The price can be concentrated in one step.** aes moves from 463 (8 back) to 230 (51 back), i.e. 43 back edges remove 233 nets (−52%). aes's acyclic price (2.28× ref) is almost entirely this group.
3. **Back edges concentrate on few drivers** (picosoc 9 back edges from 3 drivers; aes 72 from 44). If the application counts feedback nets/drivers rather than edges, the picture improves further.
4. **Designs whose acyclic optimum is already ≈ unconstrained (serv, biriscv) gain nothing**, as expected.
5. λ steps are coarse (aes jumps 8 → 51 back edges); a finer λ grid near each jump would fill in the frontier.
**Open.** Edge count vs driver count as the application metric (ask jryri). Run on ariane133 / bp_quad. Start from our own pipeline's acyclic result instead of T.

## 23. Downward relaxation on the large designs (`relax_large.py`, `run_relax_large.bat`; ε=.10; 2026-11-11)
**Setup.** No best-known T exists for these designs, so the acyclic start is K+rep (§15 pipeline: Mt-KaHyPar 3 seeds → closure repair → rebalance → 2 lowest refined → best). Then the §22 λ-continuation 64 → 0. Logs `relax_ariane133.log`, `relax_bp_quad.log`. Wall clock: ariane133 64 s, bp_quad 77 s in total, including Mt-KaHyPar and K+rep.

**ariane133** (n = 185k, ref 603, start 3957 = 6.56× ref; §15's best acyclic was L1 2559):

| λ | 64 | 32 | 16 | 8 | 4 | 2 | 1 | 0.5 | 0.25 | 0 |
|---|---|---|---|---|---|---|---|---|---|---|
| cut | 3811 | 3777 | 2829 | 2549 | 2484 | 2300 | 1236 | 694 | 669 | 628 |
| × ref | 6.32 | 6.26 | 4.69 | 4.23 | 4.12 | 3.81 | 2.05 | 1.15 | 1.11 | 1.04 |
| back edges | 1 | 1 | 10 | 14 | 17 | 25 | 379 | 429 | 504 | 1793 |

**bp_quad** (n = 972k, ref 1980, start 2108 = 1.06× ref): unchanged at λ ≥ 2; λ=1: 2060 / 39 back; λ=0.5: 2056 / 43; λ=0.25: 2040 / 83; λ=0: 2026 / 2169.

**Findings.**
1. **ariane133 behaves like omsp/aes, at a larger scale.**
   - 14 back edges take the cut 3957 → 2549, below §15's best acyclic (2559).
   - 25 back edges reach 2300 (−42%).
   - ~400 back edges reach 1.15× ref. ariane133 has 350k DAG edges, so that is about 0.1%.
   - A strict-acyclic ariane133 costs ≥4× ref; tolerating tens to hundreds of back edges removes most of that.
2. **bp_quad behaves like biriscv/serv.** Its acyclic price is small (1.06×), so relaxation buys at most 4%.
3. **The method scales.** About 1–11 s per λ step on 1M cells (pure Python FM + scipy flow), against 23 min for the §15 L1 route.
**Caveat.** ariane133's start is weak (3957 vs the L1 route's 2559). A better start could shift the few-back-edge points down.

**Correction to §22–§23 (2026-11-11).** Neither experiment ran the research pipeline. §22 started from T (portfolio best-known); §23 started from K+rep (Mt-KaHyPar + repair). No embedding, candidate or L1-projection stage was involved. They show only what downward relaxation does to a given acyclic partition, and say nothing about the embedding method. The pipeline version (embedding → candidates → exact L1 projection → sweep → refine → relaxation) has not been run yet.

## 24. Relaxation as the last stage of the full pipeline (`pipeline_relax.py`; ε=.10; 2026-11-11)
**Pipeline (no outside start).** Embedding (16 eigvecs) → 2000 two-seed potentials → pool (top 60 by unconstrained sweep + 60 random) → exact L1 projection, both orientations → acyclic sweep → top 5 by projected sweep → acyclic refine → best = P → λ-continuation 64 → 0 (penalised FM + finite-capacity flow). Mt-KaHyPar (ref) and T are printed as reference lines only. Log `pipeline_relax_small.log`.

Cut (× P) / back edges at each new point:

| design | ref | P (× ref) | 1–2 back | ~5–30 back | λ=0 end | §22 from T, comparable point |
|---|---|---|---|---|---|---|
| serv | 64 | 98 (1.53) | – | – | 98 / 0 (no move) | 65 / 0 |
| omsp | 114 | 361 (3.17) | 282 (0.78) / 1; 243 (0.67) / 2 | 230 (0.64) / 5 | 135 (0.37) / 42; 129 / 68 | 259 / 1; 205 / 5; 149 / 24 |
| picosoc | 182 | 335 (1.84) | 323 (0.96) / 1 | 250 (0.75) / 15; 228 (0.68) / 19; 223 / 28 | 208 / 5631 | 201 / 4 |
| aes | 200 | 664 (3.32) | – | – | 593 / 133; 571 (0.86) / 144 | 230 / 51 |
| riscv | 171 | 304 (1.78) | 302 / 1; 298 / 2 | – | 265 (0.87) / 139 | 295 / 2; 273 / 42 |
| biriscv | 237 | 238 (1.00) | 237 / 2 | – | 237 / 2 | same |

**Findings.**
1. **Inside the pipeline, relaxation still pays where the acyclic price is high.**
   - omsp: 1 back edge −22%, 5 back edges −36%; 42 back edges reach 1.18× ref.
   - picosoc: 15 back edges −25%, 19 back edges −32%.
2. **The final quality is set by the pipeline's acyclic start, because relaxation is local.**
   - aes: from P = 664 the path stalls at 571 with 144 back edges. From T = 476 it reached 230 with 51.
   - serv: from P = 98 nothing moves. From T it reached 65.
   - The penalised refinement cannot escape the basin of its start.
3. **Where P ≈ T (omsp, riscv, biriscv), the two paths are close.** omsp: 243 at 2 back edges vs 218 from T.
**Implication.** The pipeline's weak point is still the acyclic start on serv, picosoc and aes (1.39–1.48× T). Relaxation does not repair it. The open problem from §19/§20 (refinement reorders candidates; the best candidate in the pool is not selected) is what limits the pipeline-with-relaxation result too.

## 25. Clean method: CMA-ES on θ = (a, b) with the exact objective (`theta_opt.py`; ε=.10; λ=∞; 2026-11-12)
**Method.** Representation X (16 eigvecs) → decoder θ = (a, b) ∈ ℝ³² → s = ‖x−a‖ − ‖x−b‖ → exact L1 projection → acyclic sweep (both orientations). Optimiser: CMA-ES with library defaults (popsize 14, σ0 = 1), restart on stop, 1000 evaluations. Fitness = exact projected sweep cut. Finish: one acyclic refine. This replaces §24 steps 2/3/6/7 (random candidates, pool rule, top-5, best-of-5); no proxy, pool or top-k. Log `theta_opt_small.log`.

| design | T | §24 pipeline P | §19 pool best (120 refined) | **θ-CMA** | time |
|---|---|---|---|---|---|
| serv | 66 | 98 | 98 | **96** (1.45×T) | 52 s |
| omsp | 341 | 361 | 327 | **287 (0.84×T, new best known)** | 213 s |
| picosoc | 241 | 335 | 304 | 426 (1.77×T) | 717 s |
| aes | 476 | 664 | 507 | 633 (1.33×T) | 538 s |
| riscv | 297 | 304 | 310 | 307 (1.03×T) | 1029 s |
| biriscv | 238 | 238 | 238 | 803 (3.37×T) | 2779 s (rerun; first run died at eval 100) |

**Findings (6/6).**
1. **Mixed.** Better than P on serv (96), omsp (287, new best known) and aes (633); equal on riscv; much worse on picosoc (426 vs 335) and biriscv (803 vs 238).
2. **The failures are search failures, not expressibility.** The same family contains the 238 solution on biriscv (§19 pool best). CMA sat on a plateau at a projected sweep of 1084 for 500 evaluations, then reached only 859.
3. **One CMA run from a random x0 is not reliable** in this landscape. Random sampling (§19) found biriscv's basin because it is spread out; CMA from one random point did not.
4. **Cost.** 1000 evaluations = 52 s (serv) … 46 min (biriscv), i.e. 2 exact projections per evaluation. Not viable at 1M cells without faster projection.
5. The optimised projected sweep is still far from the refined result (riscv 489 → 307), so the fitness and the final result still disagree.
**Next (one).** Initialisation: start CMA from the classic spectral bisection (plane along the first eigenvector) instead of a random point. This tests whether the failures come from the start rather than the landscape. It adds no parameter.

## 26. Deterministic replacements for CMA (`det.py`, `l2_ir.py`; ε=.10; λ=∞; 2026-11-12)
**Methods.** No random numbers after the eigensolver. All end with order → acyclic sweep → one acyclic refine. Logs `det_cone.log`, `det_melo.log`, `det_compass.log`.
- **cone:** order-constrained spectral relaxation (cone-constrained PCA, Deshpande–Montanari–Richard 2014). max s'Ws/s'Ds over s nondecreasing along DAG edges, s ⟂_D 1. Projected power iteration with a D-weighted L2 isotonic projection (`l2_ir.py`; verified against PAVA on weighted chains, max error 0.12% of the range). Start from the Fiedler vector; keep the sign with the larger Rayleigh quotient. s is monotone, so no post-hoc projection is needed.
- **melo:** multiple-eigenvector linear ordering (Alpert–Kahng–Yao 1999). Vertex vectors √α_j u_j, greedy order maximising ‖sum‖; rank → exact L1 projection.
- **compass:** compass/pattern search (Torczon) on θ = (a, b) ∈ ℝ³² with the §25 exact objective. Start a = −2e₁, b = +2e₁ (near-planar spectral bisection), step 1, halving; stop at step < 1e-3 or 1000 evaluations (it used 670–764).

| design | T | §24 P | §25 CMA | cone | melo | compass |
|---|---|---|---|---|---|---|
| serv | 66 | **98** | **96** | 297 | 249 | 226 |
| omsp | 341 | 361 | **287** | 486 | 337 | 311 |
| picosoc | 241 | 335 | 426 | 359 | 492 | **324** |
| aes | 476 | 664 | 633 | 712 | 624 | **578** |
| riscv | 297 | **304** | **307** | 690 | 516 | 510 |
| biriscv | 238 | **238** | 803 | 1048 | 2099 | **238** |
| time (biriscv) | | | 46 min | 6 min | 12 s | 28 min |

**Findings.**
1. **No deterministic method replaces the pipeline.** compass is the best of the three: better than P on omsp, picosoc and aes; equal on biriscv; much worse on serv (226 vs 98) and riscv (510 vs 304).
2. **cone fails everywhere** (1.4–4.5× T). The order-constrained relaxation optimum is not a good cut: a high Rayleigh quotient under the monotone constraint does not imply a low acyclic cut.
3. **melo** is excellent per second (omsp 337 and aes 624 in ≤2 s) but erratic (biriscv 2099).
4. **compass vs CMA:** compass wins on picosoc, aes and biriscv; CMA wins on serv, omsp and riscv. Each optimiser fails on different designs, so the issue is how the objective landscape in θ behaves, not the particular optimiser.
**Next (Sir's instruction: if none works, return to the "cannot find a direction" problem).** First diagnose whether the failures are plateaus (perturbations do not change the fitness) or local minima (perturbations only make it worse). The remedies differ: smoothing/surrogate vs global exploration.

## 27. Why the θ-search fails: plateau vs local minimum at the endpoints (`diag.py`; ε=.10; λ=∞; 2026-11-12)

**Question.** §25 CMA and §26 compass fail on different designs. At each endpoint θ*, is J (the exact projected sweep the optimiser sees) flat, a local minimum, or not a minimum at all? The remedy differs for each: smoothing, global exploration, or a longer/better-scaled search.

**Method.**
- θ* is reproduced deterministically with the same seeds and code path. J* matches the logged sweep in all 6 cases.
- Probes: J(θ*+r·u) for 20 fixed random unit directions u ∈ ℝ³² and r ∈ {0.01, 0.03, 0.1, 0.3, 1, 3}. X is standardised, so r = 1 is one standard deviation per coordinate.
- Designs: serv, picosoc and riscv, where one optimiser is good and the other bad on J. biriscv is excluded: both sweeps there are bad (859 and 667), and refinement alone produced compass's 238.
- Cost: reproduction plus 120 evaluations, each 2 exact projections.

| endpoint | J* | \|θ*\| | better/equal/worse at r = 0.03 · 0.3 · 3 | class |
|---|---|---|---|---|
| serv compass (bad) | 542 | 4.9 | 0/20/0 · 0/2/18 · 1/0/19 (−4) | local min, wide basin |
| serv CMA (good) | 180 | 15.7 | 1/16/3 · 8/2/10 (−16) · 2/0/18 | **not a min**: stopped by budget, σ = .54, mean 10 from best |
| picosoc compass (good) | 327 | 2.9 | 0/14/6 · 0/3/17 · 0/0/20 | strict narrow local min |
| picosoc CMA (bad) | 451 | 24.3 | 0/20/0 · 0/17/3 · 1/0/19 (−6) | **plateau** out to r = 0.3 |
| riscv compass (bad) | 670 | 3.5 | 0/4/16 · 0/0/20 · 0/0/20 | strict local min (0 of 120 probes better) |
| riscv CMA (good) | 489 | 31.4 | 0/6/14 · 0/2/18 · 1/0/19 (−3) | isolated dip: median dJ is +8 at every r ≤ 1 |

**Findings.**
1. **Two failure modes, one per optimiser, both caused by the θ-parametrisation.**
   - Compass stays near the origin (|θ| ≈ 3–5). There the separator is strongly curved, and J has many strict local minima. Good (picosoc 327) and bad (riscv 670, serv 542) endpoints look locally the same, so local structure does not separate success from failure. Which basin you start in does.
   - CMA drifts to far seeds (|θ| = 16–31), the near-planar regime. A seed displacement r rotates the plane by only about r/|θ|, so J is flat over a wide region (picosoc: identical J out to r = 0.3) and the search loses its signal.
2. **The radial scale |θ| is a badly conditioned nuisance coordinate.** It trades curvature against flatness, and neither optimiser controls it.
3. **The landscape is multimodal at every scale probed** (r up to 3 ≈ |θ| for compass). No local method, deterministic or not, will find the right basin reliably from one start. That explains why replacing the optimiser (§26) only moved the failures to other designs.
4. **Confound.** CMA uses X = `dims.spectral`; compass uses X = `det.eig` (both 16 hypergraph eigenvectors, normalised differently). The comparison between optimisers is therefore also a comparison between two X. The per-endpoint classification is not affected.

**Implication.** The §19 pool (2000 random seeds, then selection) worked because random sampling is global; it was the only global component of the pipeline. A clean replacement needs a principled global mechanism, not a different local optimiser.

## 28. Gaussian-homotopy descent on θ (`homotopy.py`; ε=.10; λ=∞; 2026-11-12)

**Question.** §27 found J(θ) to be multimodal (compass gets trapped) and flat in the far-seed regime (CMA plateaus). Does a graduated, smoothed descent find good basins without randomness?

**Method.**
- Same X, start θ0 and budget (1000 evaluations) as the §26 compass run; the only change is the search rule.
- Gradient: per-axis central difference at width σ. This is exactly ∂/∂θ_i of J convolved with a uniform kernel of half-width σ along that axis, so no random directions are involved.
- Step: θ ← θ − σ·g/|g|. A step is accepted while the smoothed level F_σ (the mean of the 64 probes) decreases; otherwise σ is halved. σ runs from 2 down to 1/32.
- Incumbent: the best exact J over all evaluated θ. Finish: one acyclic refine.

| design | homotopy J → refined | compass §26 | CMA §25 | §24 P | T | J / best J of the three |
|---|---|---|---|---|---|---|
| serv | 235 → 116 | 542 → 226 | 180 → **96** | 98 | 66 | 1.31 |
| omsp | 332 → 290 | 330 → 311 | 297 → **287** | 361 | 341 | 1.12 |
| picosoc | 304 → **297** | 327 → 324 | 451 → 426 | 335 | 241 | 1.00 |
| aes | 622 → 579 | 596 → **578** | 685 → 633 | 664 | 476 | 1.04 |
| riscv | 505 → 338 | 670 → 510 | 489 → 307 | **304** | 297 | 1.03 |
| biriscv | 670 → 309 | 667 → **238** | 859 → 803 | **238** | 238 | 1.00 |

Accepted steps: 5–9. Final σ = 0.03–0.25: aes reached the floor, the other five ran out of budget first.

**Findings.**
1. **Robust on the optimiser's own objective.** Homotopy's J is within 1.31× of the best of the three optimisers on every design (≤1.04× on four of six). Compass is 3.0× worse on serv and 1.37× on riscv; CMA is 1.48× worse on picosoc and 1.29× on biriscv. Its sweep is never catastrophic: the basin trap of §27 is gone.
2. **Not yet better than the pipeline after refine.** Homotopy beats P on omsp, picosoc and aes, but loses on serv (116 vs 98), riscv (338 vs 304) and biriscv (309 vs 238).
3. **On biriscv the remaining gap is not search.** The sweep J is equal to compass's (670 vs 667), yet refined 309 vs 238. As in §20, the sweep objective and the refined outcome disagree. The optimiser is solving the right proxy only up to the refine basin.
4. **Budget-limited.** 64 probes per step leaves only 5–9 steps, and on five of six designs σ never reached its floor. More evaluations, or a cheaper gradient (fewer axes, shared probes), would continue the descent.

**Conclusion.** The "cannot find a direction" quest is answered. The θ-search failures came from the parametrisation's multimodality and plateaus, and per-axis smoothing (deterministic, parameter-free apart from the σ range on standardised X) removes the catastrophic failures. What remains is (a) budget efficiency and (b) the mismatch between the sweep cut J and the refined cut.

## 30. How to score the cut / back-edge trade-off: a survey of scalarisations (2026-11-12; §29 is running)

Sir's proposal: x·y (cut × back). Any scalar that combines two quantities with different units implies an exchange rate (how many cut nets one back edge is worth); the only question is whether it is explicit.

| rule | form | exchange rate | reaches non-convex Pareto points | defect |
|---|---|---|---|---|
| product (raw) | cut·back | implicit, varies | – | back = 0 gives 0 for every acyclic cut |
| product with 1+ | cut·(1+back) | 1st back edge = ×2 cut; 10→11 = ×1.09 | – | marginal penalty decreases with back; rate scales with design size |
| weighted sum | cut + λ·back | λ, explicit | **no** (convex hull only) | §21–§24 λ-continuation therefore traced only the convex hull of the front |
| ε-constraint (Haimes 1971) | min cut s.t. back ≤ B | none; B = allowed cycles | yes | one run per B |
| Nash bargaining (Nash 1950; generalised: Jaimes–Kreinovich) | max (cut_d − cut)(back_d − back) | none (scale-invariant) | yes | depends on the disagreement point d |
| Chebyshev / achievement (Wierzbicki 1980) | min max_i (f_i − z*_i)/(z^nad_i − z*_i) | none after ideal–nadir normalisation | yes | depends on the ideal and nadir points |
| knee (Das 1999; Branke et al. 2004) | max distance below the line through the front's extremes | none | yes | a selection rule, not a score; several definitions |
| hypervolume (Zitzler & Thiele 1998) | area dominated by the whole front | none | – (set measure) | depends on the reference point; scores curves, not single partitions |
| misaligned-cut fraction (RMCE, Pasadakis et al. 2025; cut imbalance, Cucuringu et al. 2020) | back / cut-edges | – | – | ignores cut size; descriptive only |

The product idea, done correctly, is the Nash bargaining solution: multiply the *improvements* relative to a disagreement point, not the raw values. This is what makes it scale-invariant.

**Applied to the §24 fronts.** Normalisation: ideal = (ref cut, 0 back), nadir = (acyclic P, back at the λ = 0 end).
- With this nadir, Nash, Chebyshev and knee choose the **same point on every design**:
  - omsp 230 / 5 back;
  - picosoc 223 / 28;
  - aes 571 / 144;
  - riscv 298 / 2;
  - biriscv 238 / 0.
- **Sensitivity:** dropping the λ = 0 endpoint from the nadir moves the choice on 3 of 4 designs (e.g. picosoc 223/28 → 250/15; aes Nash 571/144 → 593/133, knee → 664/0). So every parameter-free rule is parameter-free only once the ideal and nadir are **defined by the problem**, not by the run.
  - Problem-defined candidates: ideal cut = unconstrained Mt-KaHyPar cut; nadir cut = best acyclic cut; nadir back = back count of the unconstrained partition.

Sources: arXiv 2508.16173 (Pasadakis, Steiner, Papp, Böhnlein, Yzelman 2025); Cucuringu, Li, Sun, Zanetti, AISTATS 2020; Jaimes & Kreinovich, UTEP TR 09-30; Navon et al., ICML 2022 (Nash-MTL); Branke, Deb, Dierolf, Osswald, PPSN 2004; Das 1999, Struct. Optim. 18.

## 29. Does the optimiser's objective predict the refined cut? (`refsel.py`; ε=.10; λ=∞; 2026-11-12; written after §30)

**Question.** §28 (biriscv: equal sweep J, refined 309 vs 238) suggests the sweep J0 that §25–§28 optimise does not track the final refined cut. Is that so near the optimum, and does a refinement-aware proxy fix it?

**Method.**
- Pool: the exact §19 pool (same seed, 120 two-seed potentials).
- Per candidate and orientation, computed on the same partition:
  - J0 = exact projection → exact acyclic sweep;
  - J1 = J0's partition + one acyclic FM pass;
  - J2 = J0's partition + FM to convergence (no flow);
  - truth = full refine (FM + flow, 3 rounds).
- Each value is the min over the two orientations, so truth does not depend on the proxy.
- "Better half" = the 60 candidates with truth ≤ median, the region where an optimiser works.

| design | Spearman better half J0 / J1 / J2 | regret@1 J0 / J1 / J2 | J0 ms | +J2 ms | refine ms | pool best refined (×T) |
|---|---|---|---|---|---|---|
| serv | **0.00** / 0.48 / 0.97 | 1.02 / 1.02 / 1.00 | 63 | 224 | 315 | 97 (1.47) |
| omsp | 0.74 / 0.75 / 0.99 | 1.13 / 1.09 / 1.00 | 233 | 159 | 272 | 327 (0.96) |
| picosoc | 0.74 / 0.80 / 0.96 | 1.08 / 1.08 / 1.00 | 636 | 214 | 547 | 311 (1.29) |
| aes | **0.15** / 0.69 / 0.82 | **1.36** / 1.25 / 1.00 | 524 | 197 | 547 | 507 (1.07) |
| riscv | 0.84 / 0.93 / 0.95 | 1.02 / 1.19 / 1.00 | 1035 | 243 | 687 | 304 (1.02) |
| biriscv | 0.73 / 0.94 / 0.98 | 1.00 / 1.00 / 1.00 | 2392 | 294 | 1211 | 238 (1.00) |

**Findings.**
1. **Near the optimum, J0 is a poor objective.** Among good candidates its rank correlation with the final cut is 0.00 (serv) to 0.84; selecting by J0 costs up to 36% (aes). §25–§28 optimised a function that, in the region that matters, is partly noise with respect to the result.
2. **J2 (FM to convergence) predicts the final cut almost exactly.** Better-half Spearman is 0.82–0.99 and regret@1 = 1.00 on all six designs. One FM pass (J1) is not enough: riscv regret 1.19, aes 1.25.
3. **Cost.** J2 adds 160–300 ms per candidate. That is +350% on serv but +12% on biriscv: FM is O(pins log n) per pass, while the projection takes ~log₂ n max-flows, so the relative cost of FM falls with size.
4. **This is close to tautological.** J2 is most of the refine (flow rarely adds). The finding is that the refinement must be inside the objective. The structure becomes optimise refine∘decode(θ), a memetic / basin-hopping form.
5. **Headroom.** The pool's best refined candidate beats §28 homotopy on serv (97 vs 116), aes (507 vs 579) and biriscv (238 vs 309). The family can express these solutions; the objective was misdirecting the search.

**Risk for the next step.** Refinement maps many θ to the same local optimum, so J2(θ) is more piecewise-constant than J0 (more plateaus). The smoothing in homotopy (§28) is the mechanism that should handle this, and it must be checked.

**Conclusion.** Quest "is J the right objective" answered: no. Use J2 = FM-converged cut as the optimiser's objective.

## 31. Homotopy on the refinement-aware objective J2 (`homotopy.py ... j2`; ε=.10; λ=∞; 2026-11-12)

**Change from §28.** Only the objective changes: J2 = exact projection → sweep → FM to convergence, both orientations (§29). X, θ0, σ schedule and the 1000-evaluation budget are as in §28. Finish: one full refine. All six runs were in parallel on 12 cores.

| design | T | §24 P | §28 homotopy-J0 | §29 pool best | **§31 homotopy-J2** (×T) | time | incumbent first reached at |
|---|---|---|---|---|---|---|---|
| serv | 66 | 98 | 116 | 97 | **71** (1.08) | 326 s | eval ≤130 (first probe ring) |
| omsp | 341 | 361 | 290 | 327 | **286** (0.84) | 436 s | descent (295→291→286) |
| picosoc | 241 | 335 | 297 | 311 | **287** (1.19) | 1073 s | descent (311→290→288) |
| aes | 476 | 664 | 579 | **507** | 551 (1.16) | 983 s | eval ≤130 (first probe ring) |
| riscv | 297 | 304 | 338 | 304 | **301** (1.01) | 1728 s | descent (378→319→312→302→301) |
| biriscv | 238 | 238 | 309 | 238 | **238** (1.00) | 3583 s | first descent step (242→238) |
| geo-mean ×T | | 1.21 | 1.22 | – | **1.04** | | |
| worst ×T | | 1.48 | 1.76 | – | **1.19** | | |

**Findings.**
1. **Best acyclic results of any clean method so far.**
   - Geometric mean 1.04× T, against 1.21 for the pipeline and 1.22 for homotopy-J0. Worst case 1.19× T, against 1.48 and 1.76.
   - Better than P on 5/6 designs (serv −28%, omsp −21%, picosoc −14%, aes −17%, riscv −1%) and tied on biriscv.
   - The method is deterministic and has no pool and no top-k step.
2. **Better than perfect selection from the §19 pool on 4/6 designs** (serv 71 vs 97, omsp 286 vs 327, picosoc 287 vs 311, riscv 301 vs 304), tied on biriscv, worse on aes (551 vs 507). Searching θ adds over the 120-sample pool even when that pool is ranked by the true refined cut.
3. **Attribution is mixed.**
   - On omsp, picosoc and riscv the incumbent improved through the descent.
   - On serv and aes it was found in the first 64-probe ring around θ0, and the descent then lowered the smoothed level F (serv 216→132) without beating it.
   - So the main gain is the objective (§29). Homotopy's own contribution is real on 3/6 designs and nil on 2/6.
4. **Cost is now the dominant problem.** Each evaluation is 2 exact projections + FM; 1000 evaluations take 12–47× the pipeline's time (pipeline: 7–290 s). At 1M cells this is not viable without cheaper evaluations (warm-started flow, fewer probes per step).

## 32. Cost of homotopy-J2: profile and result-preserving speedups (`prof.py`, `prof2.py`, `exact_ir.l1_ir_fast`, `homotopy_par.py`; 2026-11-12)

**Profile of one J2 evaluation** (2 orientations; serial, machine otherwise idle).

| part | serv (n = 4.2k), 186 ms | riscv (n = 23.6k), 1181 ms |
|---|---|---|
| max-flow inside the projection | 12% | **68%** |
| FM to convergence (pure Python) | **74%** | 19% |
| Kahn tie-break order (pure Python) | 8% | 7% |
| flow-graph build + residual BFS | 6% | 5% |

- ~13–15 flow rounds per projection, ~4 FM passes per orientation.
- Flow time grows faster than n: 5.6× the nodes cost 31× the flow time (scipy Dinic).
- Late rounds are not cheap: rounds 8–14 on riscv keep ~13k edges (the pooled blocks) and a flow value of ~4.8k each.

**Speedups that leave every evaluation's result unchanged.**
1. **Parallel probes.** The 64 probes of a step and the candidate are independent, so they go to a process pool (12 workers). The serial code always probes the candidate, so the evaluation set and order of incumbent updates are identical.
2. **Conflict-core pruning (`l1_ir_fast`).** A flow path is s → high node → … → low node → t, so only high nodes with a low descendant and low nodes with a high ancestor can carry flow. Outside this core the minimal min cut is known: high → source side, low → sink side. Two BFS passes find the core. Output is identical on 48 projections (3 designs), but the gain is only 0.96–1.23×: the core is still ~42% of the nodes on riscv.

| design | result (= §31?) | §31 serial, 6 runs sharing the machine | §32 parallel, alone | §24 pipeline (1 process) |
|---|---|---|---|---|
| serv | 71 (same) | 326 s | 47 s | 7 s |
| omsp | 286 (same) | 436 s | 51 s | 25 s |
| picosoc | 287 (same) | 1073 s | 92 s | 72 s |
| aes | 551 (same) | 983 s | 95 s | 64 s |
| riscv | 301 (same) | 1728 s | 143 s | 116 s |
| biriscv | 238 (same) | 3583 s | 307 s | 290 s |

**Reading.**
- Results, step counts and evaluation counts are identical on all six designs.
- Wall-clock is now 1.06–1.3× the pipeline on the four larger designs (2–7× on serv and omsp).
- **This is wall-clock only.** Total work is unchanged: 12 cores for the homotopy against 1 process for the pipeline, so CPU time is still roughly 8–12× the pipeline's. The true serial-vs-parallel speedup is ~4× on serv (serial alone ≈ 190 s); the 7–12× in the logs is inflated because the §31 runs shared the machine.
- At 1M cells the per-evaluation flow cost is the wall: parallelism does not change it.

**Queue (these change the cost per evaluation or the number of evaluations; not done).**
- Compile FM and Kahn (numba or C): removes ~25–80% of an evaluation depending on size; result-preserving. Needs a package install.
- A faster max-flow than scipy's Dinic (push-relabel / pseudoflow): result-preserving; attacks the 68%.
- Fewer evaluations: the incumbent was final by ~130–600 of 1000 evaluations on every design. A stopping rule would change the algorithm, so it needs its own experiment.
- Multilevel (coarsen, optimise θ on the coarse graph): the only route that changes the scaling in n.

## 33. Every baseline reported separately, raw and refined (`baselines.py`; ε=.10; λ=∞; 2026-11-12)

**Why.** T (§10, `separability.best_known`) is itself a portfolio: the best of his decoder, K+repair and dagP-u/w with 3 seeds, all refined. It hid which method produced what, and it was **not** the best known.
- riscv: §8 already had dagP-w + refine = 251, but best_known ran only 3 dagP seeds and recorded 297.
- serv: Mt-KaHyPar noise; this run's K+repair reaches 61 < 66.

**Setup.** One run per design, sequential, machine otherwise idle.
- Mt-KaHyPar: QUALITY preset, 3 seeds, 8 threads.
- K+repair: closure variants → rebalance → 3 lowest refined.
- dagP: rMLGP without METIS, 10 seeds; each balanced result refined.
- His decoder: 2000 random nonneg weights.
- The dagP binary was checked to really run (~0.07 s per run on picosoc) and to write fresh files.

| design | Mt-KaHyPar unconstr. cut / back edges | K+repair raw → ref | dagP-u raw → ref | dagP-w raw → ref | his decoder raw → ref | homotopy-J2 (§32) |
|---|---|---|---|---|---|---|
| serv | 61 / 19 | 63 → **61** (5 s) | 690 → 261 | 459 → 140 | 504 → 193 | 71 (47 s, 12 cores) |
| omsp | 114 / 126 | 362 → 341 (5 s) | 563 → 431 | 465 → 383 | 663 → 594 | **286** (51 s) |
| picosoc | 187 / 512 | 250 → **250** (6 s) | 948 → 605 | 633 → 574 | 928 → 535 | 287 (92 s) |
| aes | 199 / 635 | 503 → **475** (9 s) | 1008 → 763 | 885 → 650 | 710 → 635 | 551 (95 s) |
| riscv | 172 / 174 | 425 → 381 (7 s) | 878 → 335 | 702 → **251** (3 s) | 725 → 537 | 301 (143 s) |
| biriscv | 237 / 2 | 238 → **238** (10 s) | 786 → **238** | 817 → **238** | 2164 → 2066 | **238** (307 s) |

Times for dagP include its refine; raw dagP ≈ 1 s for 10 seeds.

**Findings.**
1. **Homotopy-J2 is not competitive with the strongest single baseline.**
   - It wins only omsp (286 vs 341, −16%) and ties biriscv.
   - It loses serv (+16% vs K+repair), picosoc (+15%), aes (+16%) and riscv (+20% vs dagP-w + refine).
   - The baselines take 3–10 s on one machine; homotopy takes 47–307 s on 12 cores.
2. **The "1.04× T" of §31/§32 was an artefact of a wrong T.** Corrected best known: serv 61, riscv 251. §24–§32 ratios against T overstate the method.
3. **K+repair is the real competitor.** Its start is a partition already optimised for cut by a multilevel partitioner; closure repair and refine then cost little (serv 63 → 61, biriscv 238). dagP raw is weak (3–7× the unconstrained cut) but is an excellent start for refinement on riscv.
4. **Speed is not the main problem; quality is.** Accelerating a method that loses on 4/6 is premature.

**Next (proposed).** Check expressibility against K+repair: fit θ (two-seed) to the K+repair partition as an oracle, then project + FM.
- If the oracle reaches K+repair → the family can express it, the gap is search.
- If not → the 16-dim spectral representation with a two-seed separator is the ceiling.

## 34. Which layer is the bottleneck: embedding, decoder family or search? (`oracle.py`; ε=.10; λ=∞; 2026-11-12)

**Method (oracle diagnosis, not a method).**
- Target y = the better of K+repair + refine and dagP-w (10 seeds) + refine, recomputed. It is saved in `bdp/best_parts2/`. Mt-KaHyPar noise moves it: serv 66 here vs 61 in §33; omsp 345 vs 341.
- X = the 16 spectral coordinates used by homotopy.
- Every score goes through the method's own decoder (exact projection → sweep → FM → refine).
- The three layers:
  - **L1 embedding:** 5-fold out-of-fold kNN (k = 15). Is y locally consistent in X (do cells that are close in X share the label)? A logistic plane is shown for reference.
  - **L2 family:** two-seed separator fitted to y by logistic loss.
  - **L3 search:** homotopy-J2 without the oracle.

| design (target) | kNN misclass → refined (×) | plane misclass → refined | two-seed misclass → refined | homotopy-J2 (×) | diagnosis |
|---|---|---|---|---|---|
| serv (66) | **18.1%** → 139 (2.11) | 27.3% → 258 | 27.5% → 227 (3.44) | 71 (1.08) | embedding cannot express y; search finds a *different* near-equal cut |
| omsp (345) | 3.1% → 337 (0.98) | 5.6% → 356 | 5.0% → 356 (1.03) | **286 (0.83)** | no bottleneck; search beats the target |
| picosoc (241) | 8.3% → 296 (1.23) | 9.0% → 394 | 9.0% → 264 (1.10) | 287 (1.19) | family/embedding ceiling ~1.10 plus a ~9% search gap |
| aes (475) | **10.6%** → 590 (1.24) | 13.2% → 721 | 11.7% → 752 (1.58) | 551 (1.16) | **embedding**: even the flexible oracle is worse than the search |
| riscv (251) | **1.1% → 256 (1.02)** | 6.6% → 416 | 6.1% → 340 (1.35) | 301 (1.20) | **decoder family**: y is in X, but the two-seed / plane family cannot cut it |
| biriscv (238) | 0.0% → 238 | 2.5% → 238 | 2.9% → 238 | 238 | no bottleneck |

The fitted two-seed separator degenerates to a near-plane (|a−b| = 0.3–0.6) on omsp, picosoc and aes; on riscv and biriscv it goes far-seed (|a−b| = 160–460).

**Findings.**
1. **No single bottleneck.**
   - Embedding: serv, aes.
   - Decoder family: riscv.
   - Search: part of picosoc.
   - None: omsp, biriscv.
2. **Embedding (serv, aes).** The best cut is not locally consistent in the 16 undirected spectral coordinates: cells close in X often sit on opposite sides, so even kNN misclassifies 11–18%. kNN in the same 16 dimensions reaches 0–3% on omsp, riscv and biriscv, so dimensionality alone does not explain it. Homotopy already does better than any oracle fit in X, so better search or a richer family cannot help there.
   - Plausible cause: the spectral embedding is undirected and carries no DAG depth. An acyclic cut on aes may have to cut across logic depth (§8 ShiftRows hypothesis), which X does not encode.
3. **Family (riscv).** y is almost exactly a function of X (kNN 1.1% → 1.02×), but a single plane or two-seed surface cannot express it (1.35–1.66× after fitting).
4. **Search.** It is not the main problem. Homotopy beats the oracle-fitted family on 4/6 designs, because it optimises the cut while the fit optimises misclassification.

**Caveat.** Misclassification against one target is a proxy. Serv shows that a cut of nearly equal value can have a very different structure.

## 35. A DAG-aware embedding G: per-coordinate L1 projection of the spectral coordinates (`dagembed.py`, `oracle2.py`, `homotopy_par.py ... g`; ε=.10; λ=∞; 2026-11-12)

**Context.**
- Sir set the paper's selling point: a DAG-aware embedding that improves and speeds up few-back-edge, low-cut partitioning.
- Sir had understood L1 to be a correction of his U/L envelope distortion *in the embedding*. Since §12 the main line instead projected only the final score, on an undirected X, so the coordinates carried no DAG information. That switch was justified by a confounded comparison (§12: w≥0 random search on G vs a free-sign oracle score on [X,Z]) and was not flagged to Sir.

**G.** For each of the 16 spectral coordinates: orient it so DAG edges increase on average (as his `orient_and_scale`), then apply the exact L1 isotonic regression on the DAG and standardise. Every column is monotone along every DAG edge. Cost: 16 exact projections, 0–15 s once per design.

| design | kNN misclass X → G ([X,G]) | kNN decoded+refined X → G ([X,G]) | homotopy-J2 on X → **on G** | best single baseline |
|---|---|---|---|---|
| serv | 18.1 → 18.8% (18.8) | 139 → 142 (141) | 71 → **98** | 61 K+repair |
| omsp | 3.1 → **1.3%** (1.2) | 337 → 336 (335) | 286 → 290 | 341 K+repair |
| picosoc | 8.3 → 8.1% (8.4) | 296 → 345 (338) | 287 → 289 | 250 K+repair |
| aes | 10.6 → 10.6% (10.3) | 590 → **544** (552) | 551 → 563 | 475 K+repair |
| riscv | 1.1 → **0.6%** (0.8) | 256 → 252 (**244** < 251 known) | 301 → 310 | 251 dagP-w |
| biriscv | 0.0 → 0.1% (0.0) | 238 → 238 | 238 → 238 | 238 |

The homotopy runs on G had the same budget and search as on X, with 12 workers: 47–228 s.

**Findings.**
1. **G does not improve the method on any design.**
   - Homotopy-J2 on G is equal to or worse than on X: serv 71 → 98, aes +2%, riscv +3%; the other three are within 1%.
   - In the oracle diagnosis G helps somewhat: omsp and riscv are more locally consistent, and aes decodes 8% better. The search does not convert this.
2. **Why: an isotonic projection adds no information.** G is a monotone reshaping of the same undirected spectral information. The projection can only pool values along violated edges; it cannot create a direction X does not contain. Its envelope version (his U/L, an L∞ projection) is the same in kind.
3. **For the selling point this is the important negative result.** A DAG-aware embedding built by *projecting* an undirected spectrum does not help. If the embedding is to carry the DAG, the DAG has to be in the **operator** whose eigenvectors are the embedding, not applied afterwards.
4. **New best-known riscv: 244** (kNN oracle on [X,G], decoded and refined; an oracle result, not a method result). Saved targets stay at 251.

**Candidate operators** (all eigsh-cost, O(pins + DAG edges) per matvec, the same as now):
- (a) **Alignment-penalised spectral** (Pasadakis, Steiner, Papp, Böhnlein, Yzelman 2025): min xᵀLx − c·(Σ_{u→v}(x_u − x_v))². It rewards cuts whose edges all point one way, which directly relaxes few back edges with low cut.
- (b) **Magnetic / Hermitian Laplacian** (Cucuringu et al. 2020): direction encoded as a complex phase.
- (c) **Constrained spectral**: min xᵀLx s.t. x_v − x_u ≥ δ, a potential that is acyclic by construction.

## 36. Lattice combinations of acyclic cuts: premise test for special separators (`lattice.py`; ε=.10; λ=∞; 2026-11-12)

**Idea.** Block 0 of an acyclic cut is a predecessor-closed set, and these are closed under union and intersection (a distributive lattice). At the score level: if r_i, r_j are topological ranks, so are max(r_i, r_j) and min(r_i, r_j). So two simple separators combine into a non-planar one with **no projection**: one sort + sweep + FM.

**Test.**
- Singles: the §19 pool (120 two-seed potentials × 2 orientations = 240), each decoded as the method does (exact projection → sweep → FM).
- Combinations: the 20 best singles by FM cut, all 190 pairs × {max, min} = 380; order by the combined rank → sweep → FM.
- Best single and best combination each get one full refine.

| design | best single (refined) | best combination (refined) | gain | combos beating BOTH parents (median gain) | homotopy-J2 | best baseline §33 |
|---|---|---|---|---|---|---|
| serv | 97 | 90 | +7.2% | 7% (7.1%) | 71 | 61 |
| omsp | 327 | 315 | +3.7% | 17% (2.4%) | 286 | 341 |
| picosoc | 311 | 291 | +6.4% | 14% (8.6%) | 287 | 250 |
| aes | 507 | 537 | −5.9% | 6% (1.8%) | 551 | 475 |
| riscv | 304 | 314 | −3.3% | 9% (7.7%) | 301 | 251 |
| biriscv | 238 | 238 | 0 | 19% (9.3%) | 238 | 238 |

Cost per combination: 8–21 ms wall-clock on 12 workers, i.e. ≈0.1–0.25 s CPU, almost all FM. A J2 evaluation with projection is ≈1.2 s CPU at 24k cells, so a combination is ≈7× cheaper and needs no max-flow.

**Findings.**
1. **The lattice combination is valid and cheap, but its gain is modest.** It improves the best single by 4–7% on 3 designs, is neutral on biriscv, and worse on aes and riscv (there the best combination by FM cut refines worse than the best single).
2. **It does not close the gap.** It never beats homotopy-J2 by more than noise (picosoc 291 vs 287) and beats the baseline only on omsp, where the singles already do.
3. **My prediction for riscv was wrong.** §34 said the target is in X but not reachable by one surface, so pairwise min/max should have helped; it did not. Either two-term lattice polynomials are still too weak, or the good cut's shape is not a lattice combination of these particular singles.
4. **Only 6–19% of combinations beat both parents.** A combination is not a reliable improver by itself; it would need its own search.
5. Caveat: best-of-380 against best-of-240 is not an equal-budget comparison, so even the 4–7% is an upper estimate.

**Conclusion.** Separator shape is not the main lever. The result agrees with §34/§35: the limit is the information in the embedding (and the start quality), not how the space is cut.

## 37. jryri's proposal: a learned envelope operator; and the envelope itself re-examined (`learnenv.py`, `learnenv_eval.py`; ε=.10; 2026-11-12)

**Proposal (jryri).** Learn a circuit-independent envelope operator Φ_θ so that an oracle partition becomes linearly separable in X(θ) = [X_spec | U(θ) | L(θ)]; ridge hyperplane in closed form, degree-weighted logistic alignment loss; no RL.

**Implementation.**
- Operator, forward in level order: soft_v = (1/β)·log mean exp(β·{x_v} ∪ {U_u + δ}), U_v = (1−α)x_v + α·soft_v; backward is the mirror image. His envelope is α = 1, β = ∞, δ = 1e-4.
- θ = (α, β, δ) forward + backward = 6 numbers, shared by all 16 coordinates and all circuits. With 6 dimensions it is trained derivative-free and deterministically (Nelder–Mead, 300 evaluations, 19–51 s) instead of autograd.
- Protocol: 2-fold across circuits: A = (serv, omsp, riscv), B = (picosoc, aes, biriscv); train on one, test on the other. Oracle = `best_parts2`.

**Result 1: under the proposed ridge loss, learning adds little.**
- Loss falls 1.397 → 1.350 (A) and 1.467 → 1.381 (B). Held-out ≈ in-sample, so θ transfers.
- But the two folds learn very different θ (forward δ = 2.08 vs 0.058), i.e. the loss is flat in θ.
- The ridge hyperplane is a poor separability measure: biriscv X-only is 32.6% misclassified by ridge and 2.5% by logistic regression.

**Result 2 (the important one): with a proper hyperplane fit, the envelope features already express the oracle.** Logistic regression (free-sign w), then the method's decoder (projection → sweep → FM → refine). Each cell: misclassification → refined cut (× target).

| design | X only (16) | his envelope [X\|U\|L] (48) | learned θ, held-out (48) | best baseline §33 |
|---|---|---|---|---|
| serv | 27.3% → 258 (3.91) | 6.9% → **66 (1.00)** | 3.2% → 66 (1.00) | 61 |
| omsp | 5.6% → 356 (1.03) | 0.6% → **333 (0.97)** | 0.7% → 345 (1.00) | 341 |
| riscv | 6.6% → 416 (1.66) | 1.9% → **251 (1.00)** | 1.7% → 280 (1.12) | 251 |
| picosoc | 9.0% → 394 (1.63) | 8.5% → 339 (1.41) | 9.0% → **298 (1.24)** | 250 |
| aes | 13.2% → 721 (1.52) | 11.5% → **598 (1.26)** | 11.5% → 687 (1.45) | 475 |
| biriscv | 2.5% → 238 | 0.4% → 238 | 0.6% → 238 | 238 |

**Findings.**
1. **His envelope coordinates carry the DAG information our X lacks.** With U/L added and free-sign weights, a single hyperplane reproduces the best-known cut on serv, omsp, riscv and biriscv (0.97–1.00× target). On serv even kNN on X was 18% wrong (§34); here a plane is 6.9% wrong and decodes to the target exactly.
2. **This corrects §35 and §12/§13.**
   - §35 concluded "a projection adds no information" from G (per-coordinate L1 projection). That holds for the L1 projection, which changes few cells. The L∞ envelope is different: U is the ancestor-maximum and L the descendant-minimum, which are non-local DAG quantities.
   - Dropping the envelopes from the coordinates in §12/§13 removed exactly this information. §10 had tested his family with w ≥ 0 (fails) and free-sign only on 8 dimensions.
3. **Learning θ does not beat his fixed envelope consistently.** Better on picosoc (298 vs 339), worse on riscv (280 vs 251) and aes (687 vs 598), equal elsewhere.
4. **picosoc and aes remain limited** (1.24–1.41×): 8.5–11.5% misclassified even with envelopes.

**Caveat.** These are oracle fits: they show the family *contains* the good cuts, not that a search finds them. Free-sign search on [X|U|L] with the J2 objective has never been run (§16 searched X only; §10 searched w ≥ 0 only).

**Next.** Homotopy-J2 over a free-sign hyperplane on his [X|U|L] (48 parameters), no oracle.

## 38. Where the information was lost: envelope vs w ≥ 0 (`envcmp.py`; ε=.10; 2026-11-12)

**Question (Sir).** We replaced his U/L because they distort; why use them again?

**Test.** Same protocol as §37 (logistic free-sign plane fitted to the oracle → projection → sweep → FM → refine), five feature sets. Each cell: misclassification → refined cut.

| design (target) | [X] 16 | [X\|G] L1 per coord., 32 | [X\|U\|L] L∞ envelope, 48 | [U\|L] only, 32 | [U\|L] with w ≥ 0 (his decoder family) |
|---|---|---|---|---|---|
| serv (66) | 27.3% → 258 | 26.5% → 260 | 6.9% → **66** | 18.9% → 202 | 31.8% → 122 |
| omsp (345) | 5.6% → 356 | 0.9% → 350 | 0.6% → **333** | 0.8% → 339 | 2.4% → 350 |
| riscv (251) | 6.6% → 416 | 3.4% → 436 | 1.9% → 251 | 2.0% → **243** | 8.1% → 373 |
| picosoc (241) | 9.0% → 394 | 9.0% → 397 | 8.5% → 339 | 9.3% → **330** | 13.5% → 690 |
| aes (475) | 13.2% → 721 | 11.5% → 701 | 11.5% → **598** | 11.5% → 639 | 12.1% → 763 |
| biriscv (238) | 2.5% → 238 | 0.9% → 238 | 0.4% → 238 | 1.5% → 244 | 32.7% → 3037 |

**Findings.**
1. **The loss was in w ≥ 0, not in the envelope.** The same U/L columns give 66 / 333 / 251 / 238 with free-sign weights and X kept, but 122 / 350 / 373 / 3037 with w ≥ 0. Nonnegative weights cannot subtract, so they cannot use differences such as U − L or x − U.
2. **My §11/§12 diagnosis was wrong.** "The L∞ envelope changes 68–92% of cells, so it loses information" assumed the envelope *replaces* X. When X is kept beside it, the change is the information: U − x says how far an ancestor exceeds the cell. The L1 projection changes few cells, stays close to X, and so adds almost nothing ([X|G] ≈ [X]).
3. **What the L1 work correctly fixed** is the acyclicity guarantee: it moved from "w ≥ 0 on monotone columns" to "exact L1 projection of the final score". That stays.
4. **The right combination, never run as a search:** his coordinates [X|U|L] + free-sign weights + score-level L1 projection + J2.
5. New best-known riscv by oracle: 243 ([U|L] only).

## 39. Search on his envelope coordinates with a free-sign plane (`homotopy_ul.py`; ε=.10; λ=∞; no oracle; 2026-11-12)

**Method.**
- Embedding F = [X | U | L]: 16 spectral coordinates + his L∞ envelopes, 48 dims.
- Separator: free-sign hyperplane s = F·w.
- Decoder: exact L1 projection of s → acyclic sweep → FM (J2), both orientations.
- Search: the §31 homotopy, w renormalised to |w| = 1 after each step, start w0 = e₁, 1000-evaluation budget.
- Finish: one full refine.
- Control: the same plane search on X only (16 dims), so the only difference is the U/L columns.
- 12 workers, one design at a time. All results are strictly acyclic (0 back edges).

| design | two-seed on X (§32) | plane on X (control) | **plane on [X\|U\|L]** | best single baseline §33 | vs baseline | time (UL) |
|---|---|---|---|---|---|---|
| serv | 71 | 96 | **64** | 61 (K+repair) | +5% | 40 s |
| omsp | 286 | **286** | 318 | 341 (K+repair) | −7% | 52 s |
| picosoc | 287 | 283 | **244** | 250 (K+repair) | −2% | 92 s |
| aes | 551 | 463 | **461** | 475 (K+repair) | −3% | 85 s |
| riscv | 301 | 307 | **284** | 251 (dagP-w + refine) | +13% | 138 s |
| biriscv | 238 | **238** | 244 | 238 | +2.5% | 324 s |
| geo-mean × baseline | 1.08 | 1.10 | **1.01** | 1.00 | | |

**Findings.**
1. **First clean method at parity with the best single baseline** (geo-mean 1.01×): it wins omsp, picosoc and aes, and loses serv (5%), riscv (13%) and biriscv (2.5%). Before this section the record was one win and four losses.
2. **The envelope coordinates are the cause on three designs:** serv 96 → 64, picosoc 283 → 244, riscv 307 → 284 against the X-only control.
3. **They hurt on two:** omsp 286 → 318 and biriscv 238 → 244. With 48 instead of 16 dimensions, each step costs 97 evaluations instead of 33, so the same budget allows only 3–6 steps (the control takes up to 12).
4. **aes is a separator effect, not an embedding effect.** Plane on X alone gives 463 (two-seed on X: 551). This contradicts §34's "aes is limited by the embedding": the two-seed parametrisation was hindering the search there.
5. **Search beats the oracle fit** on serv (64 vs 66), omsp, picosoc (244 vs 339) and aes (461 vs 598), because it optimises the cut and not misclassification. On riscv the oracle (251) shows the family contains a better cut than the search found (284).
6. **Cost.** 40–324 s on 12 cores against 3–10 s for the baselines. Speed remains unaddressed.

**Not done.** The few-back-edge axis: every result since §25 is at λ = ∞.

## 40. The few-back-edge axis: relax-after vs λ inside the search (`backedge_ul.py`; ε=.10; 2026-11-12)

**Question (Sir).** Is "shrink to acyclic first, then relax" a problem?

**Setup.** Same embedding [X|U|L], plane family and homotopy as §39. Objective J_λ = cut + λ·back.
- **(a) relax-after:** search at λ = ∞ → refine → downward continuation λ = 64 … 1 with the §21 penalised refinement.
- **(c) λ-inside:** the search optimises J_λ directly (each evaluation: projection → acyclic sweep → penalised FM at λ), from a fresh start w0 = e₁; finish = penalised refinement at λ.

Each cell: cut / back edges (J).

| design | acyclic | λ=4 (a) | λ=4 (c) | λ=1 (a) | λ=1 (c) | unconstrained cut / back (§33) |
|---|---|---|---|---|---|---|
| serv | 64 / 0 | 64 / 0 | 66 / 0 | 64 / 0 (64) | 62 / 1 (63) | 61 / 19 |
| omsp | 318 / 0 | 318 / 0 (318) | **188 / 2 (196)** | 212 / 23 (235) | **136 / 21 (157)** | 114 / 126 |
| picosoc | 244 / 0 | 244 / 0 | 283 / 0 | 230 / 2 (232) | **215 / 0 (215)** | 187 / 512 |
| aes | 461 / 0 | 461 / 0 | 463 / 0 | 461 / 0 (461) | **210 / 45 (255)** | 199 / 635 |
| riscv | 284 / 0 | 284 / 0 | 323 / 0 | 284 / 0 (284) | 300 / 9 (309) | 172 / 174 |
| biriscv | 244 / 0 | 244 / 0 | 244 / 0 | 238 / 0 | 238 / 0 | 237 / 2 |

**Findings.**
1. **Sir's concern is confirmed where the acyclic price is high.**
   - omsp: (c) reaches cut 188 with 2 back edges (−41% vs acyclic); (a) does not move at λ = 4. At λ = 1, (c) is 136 / 21 against (a) 212 / 23: J is 33–38% lower.
   - aes: (a) cannot buy a single back edge at either λ. (c) at λ = 1 reaches 210 with 45 back edges (−54% cut; the unconstrained cut is 199 with 635 back edges).
   - The few-back-edge optimum lies in a different basin from the acyclic optimum, and local relaxation from the acyclic solution does not reach it.
2. **No difference where the price is low** (serv, biriscv): there is nothing to trade.
3. **The search itself is noisy by about ±10–15% between runs.**
   - picosoc λ = 4 (c) gives 283 and riscv λ = 4 (c) gives 323, both worse than the acyclic search (244, 284) with 0 back edges. Any λ-inside result with 0 back edges is also a valid acyclic solution, so these are search failures, not a property of λ.
   - Conversely, picosoc λ = 1 (c) found **215 with 0 back edges**: an acyclic cut better than the acyclic search (244), K+repair (250) and every earlier result. New best-known acyclic picosoc.
4. **Consequence.** One search does not give a reliable point for a given λ. The Pareto set should be read across all evaluated solutions of all λ, and the search needs to be made more stable (budget in 48 dimensions; §39 finding 3).

**Reading across all runs** (best cut for a back-edge budget B, from the solutions above):

| design | B = 0 | B ≤ 2 | B ≤ 10 | B ≤ 50 |
|---|---|---|---|---|
| serv | 64 | 62 | 62 | 62 |
| omsp | 318 | **188** | 188 | **136** |
| picosoc | **215** | 215 | 215 | 215 |
| aes | 461 | 461 | 461 | **210** |
| riscv | 284 | 284 | 284 | 284 |
| biriscv | 238 | 238 | 238 | 238 |

## 41. Review of the §39/§40 method: one implementation check and open concerns (`envcheck.py`; 2026-11-12)

**Check.** `homotopy_ul.py` and `backedge_ul.py` build U/L with `learnenv.prop` at β = 55, a *soft*, mean-normalised envelope, not his exact max/min.
- Correlation with the exact envelope per column: 0.991–0.999.
- The soft U is monotone on only 49–64% of DAG edges (the exact one on 100%), because of the −log(indeg+1)/β term.
- The §38 oracle result holds with the exact envelope and is slightly better: serv 66, omsp 336, riscv 249, picosoc 328, aes 576, biriscv 238.
- So the §37–§40 conclusions stand, but the method should use the exact envelope from now on.

**Open concerns (not yet addressed).**
1. **Significance.** The search varies ±10–15% between runs (§40), every number is a single run, and Mt-KaHyPar baselines vary too (serv 61 vs 66). Wins of 2–3% (picosoc, aes in §39) are inside the noise.
2. **No baseline on the back-edge axis.** §40 compares our two variants only. "K+repair tightened upward" / "Mt-KaHyPar + penalised refinement" (§21) has not been rerun in this protocol.
3. **The table "best cut for a back-edge budget" in §40 reads across runs**, which is a best-of-N.
4. **Speed.** 40–324 s on 12 cores (48-dim: up to 577 s on biriscv in §40) against 3–10 s. The "acceleration" half of the selling point has no support yet.
5. **λ-inside still decodes through an acyclic sweep** and then relaxes locally inside each evaluation; the exact projection is paid although the final answer may contain back edges.
6. **Unequal budgets in §39:** the 48-dim runs used 1065–1160 evaluations, the 16-dim control 520–880 (it stopped at the σ floor).
7. **Scope.** Six designs ≤ 55k cells, ε = 0.10 only; back edges counted as edges, not drivers.

**Levelisation (Sir's suggestion; `levtest.py`).**
- The envelope is already levelised (one vectorised max/min per DAG level): exact U+L for 16 dims takes 2 ms (serv), 15 ms (riscv), 32 ms (biriscv), once per design. It is not a cost.
- The per-evaluation Python Kahn tie-break can be levelised: sort by (f, level, s) instead, where level = longest-path depth. 7 → 0.3 ms (serv), 44 → 1.9 ms (riscv), 107 → 4.4 ms (biriscv), i.e. ~25×, worth ≈10% of an evaluation.
- It is a different tie-break, not an identical result: the sweep cut is equal in 14 of 18 test projections, and one case was clearly worse after FM (riscv 1558 vs 2126).
- Per projection on biriscv the remaining cost is max-flow 774 ms and FM 110 ms. Levelisation does not touch the max-flow.

## 42. Levelising the projection: L∞ instead of max-flow L1 (`linf_decode.py`; ε=.10; premise test; 2026-11-12)

**Idea (Sir: levelise the max-flow step).**
- The max-flow itself is already Dinic (`scipy.sparse.csgraph.maximum_flow(method="dinic")`, compiled), which is a level-graph algorithm. Its levels are BFS levels of the *residual* graph, rebuilt every phase; they are not the DAG's fixed levels, because augmenting must reroute earlier flow (the problem is a matching of high nodes to low descendants).
- What can be levelised exactly is a different projection: the L∞ isotonic regression of the score is f = (U + L)/2 with U = ancestor-max and L = descendant-min, i.e. two vectorised passes over the DAG levels, O(DAG edges). This is his envelope applied to the final score.

**Test.** Same directions w, four monotone repairs of s = F·w, then the same sweep and FM. Directions: the oracle plane on [X|U|L] and Gaussian perturbations of it. Cut after FM, better orientation.

| design | per projection L1 → L∞ | σ=0: L1 / L∞ / U only / L only | wins over 6 directions (L∞ ≤ L1) |
|---|---|---|---|
| serv | 17 → 0.2 ms (70×) | 131 / 179 / **66** / 73 | 1 of 6 |
| omsp | 84 → 0.6 ms (143×) | **336** / 348 / 375 / 339 | 1 of 6 |
| picosoc | 148 → 1.0 ms (142×) | **328** / 760 / 646 / 401 | 3 of 6 |
| aes | 149 → 0.9 ms (164×) | 640 / 763 / 728 / **586** | 5 of 6 |
| riscv | 303 → 1.5 ms (209×) | **249** / 251 / 281 / 256 | 4 of 6 |
| biriscv | 827 → 2.5 ms (330×) | **240** / **240** / 243 / 238 | 4 of 6 |

**Findings.**
1. **The levelised projection is 70–330× cheaper than the max-flow one,** and the ratio grows with size.
2. **Per direction it is not a drop-in replacement.** At the oracle direction it equals L1 on riscv and biriscv, is slightly worse on omsp, and much worse on picosoc (760 vs 328) and serv. Over perturbed directions it wins about as often as it loses (18 of 36).
3. **No repair dominates.** U-only gives 66 on serv (the target) where L1 gives 131; L-only gives 586 on aes where L1 gives 640. The decoded cut is very sensitive to the repair and to small changes of w for every variant, L1 included.
4. **The right test is a search with the levelised decoder**, since the search adapts w to the decoder. An evaluation would then cost one FM (~0.1 s) instead of ~1 s, so the same time buys ~10× the evaluations (FM becomes the bottleneck).

## 43. jryri's augmented spherical normalisation (`sphere_check.py`, `homotopy_ul.py ... ule|sph`; ε=.10; λ=∞; 2026-11-12/13)

**Proposal.** x̂_v = [x_v, 1] / √(|x_v|² + 1): the sign of any affine cut is unchanged, but high-norm ("deep") nodes lose their leverage; offered as the reason ridge misclassifies 32.6% where logistic gets 2.5%.

**Premise check** on [X|U|L] with the exact envelope (48 dims), oracle = `best_parts2`.

| design | row norm median / max | top-1% rows' share of squared norm: raw → sphere | ridge misclass raw → sphere | logistic raw → sphere |
|---|---|---|---|---|
| serv | 3.5 / 38 | 15.5% → 1% | 28.0 → 21.6% | 5.9 → 7.3% |
| omsp | 4.4 / 24 | 7.7% → 1% | 3.0 → 2.5% | 0.7 → 0.8% |
| riscv | 2.8 / 65 | 15.0% → 1% | 8.2 → 8.2% | 1.0 → 2.0% |
| picosoc | 4.2 / 40 | 17.0% → 1% | 9.6 → 9.2% | 8.1 → 8.3% |
| aes | 6.5 / 31 | 5.5% → 1% | 11.7 → 11.7% | 11.0 → 11.6% |
| biriscv | 2.0 / 95 | 23.2% → 1% | 12.6 → 5.8% | 0.3 → 1.7% |

**Search test.** Same plane homotopy-J2 (L1 decoder), exact envelope, with and without the normalisation. Refined acyclic cut.

| design | soft envelope (§39) | exact envelope (control) | exact + sphere |
|---|---|---|---|
| serv | 64 | 64 | 64 |
| omsp | 318 | 315 | 314 |
| picosoc | 244 | 242 | 292 |
| aes | 461 | 465 | 460 |
| riscv | 284 | 313 | 300 |
| biriscv | 244 | 271 | 244 |

**Findings.**
1. **The diagnosis is right:** a few rows carry 5–48× the median norm, and the normalisation removes that leverage.
2. **It explains ridge only partly:** ridge improves on serv and biriscv, not on the other four, and stays far from logistic.
3. **It does not help the search.** Four designs are unchanged; biriscv is 10% better and picosoc 21% worse. No systematic gain.
4. **Why it is not our bottleneck:** the decoder uses the *order* of the scores, not their magnitudes; an extreme node sits at an end of the order and hardly affects the order near the split.
5. **Caveat on "100% equivalent".** The invariance holds for the sign at threshold 0. Our sweep uses every threshold of the ordering, and the normalisation changes the ordering, so the family of cuts is not the same.
6. **Noise measured.** Soft vs exact envelope (column correlation 0.99) changes riscv by 10% and biriscv by 11%. Differences of this size between single runs are not evidence.

## 44. Removing the max-flow: decoder comparison (`homotopy_dec.py`; ε=.10; 2026-11-13)

**Setup.** Same embedding ([X|U|L], exact envelope, 48 dims), free-sign plane, homotopy, 1000 evaluations, 12 workers. Only the decoder (score → order) changes. Reference L1 = exact projection by max-flow (`homotopy_ul.py ... ule`).

**Acyclic (λ = ∞).** Refined cut; search wall-clock in brackets.

| design | L1 max-flow | **L∞ = (U(s)+L(s))/2, levelised** | αU+(1−α)L, α searched (α found) | Kahn | best baseline §33 |
|---|---|---|---|---|---|
| serv | 64 [39 s] | **61** [38 s] | 61 (0.34) | 67 | 61 |
| omsp | 315 [51 s] | 316 [28 s] | 305 (0.49) | **297** | 341 |
| picosoc | **242** [101 s] | 262 [46 s] | 281 (0.52) | 274 | 250 |
| aes | **465** [91 s] | 477 [40 s] | 468 (0.42) | 469 | 475 |
| riscv | 313 [230 s] | **263** [47 s] | 263 (0.49) | 336 | 251 |
| biriscv | 271 [581 s] | **251** [61 s] | 251 (0.50) | 558 | 238 |

**Few back edges, no projection at all** (`nop`: order by s; the sweep minimises cut + λ·back with exact per-split back-edge counts in O(m); penalised FM). λ = 4; cut / back (J).

| design | L1-inside (§40 c) | no projection | 
|---|---|---|
| serv | 66 / 0 (66) | 62 / 0 (62) |
| omsp | 188 / 2 (196) | 174 / 6 (198) |
| picosoc | 283 / 0 (283) | 266 / 4 (282) |
| aes | 463 / 0 (463) | **247 / 32 (375)** |
| riscv | 323 / 0 (323) | 447 / 4 (463) |
| biriscv | 244 / 0 (244) | 244 / 0 (244) |

λ = 1 (run on 2026-11-13 after the interrupted session); cut / back (J).

| design | L1-inside (§40 c) | no projection |
|---|---|---|
| serv | 62 / 1 (63) | **59 / 0 (59)** |
| omsp | 136 / 21 (157) | 135 / 21 (156) |
| picosoc | **215 / 0 (215)** | 187 / 64 (251) |
| aes | 210 / 45 (255) | 210 / 40 (250) |
| riscv | 300 / 9 (309) | 278 / 40 (318) |
| biriscv | **238 / 0 (238)** | 404 / 7 (411) |

- Equal on omsp, aes and riscv (J within 3%); better on serv; worse on picosoc (J +17%) and a clear failure on biriscv (404 vs 238).
- serv 59 with 0 back edges is an acyclic cut below K+repair (61): new best-known acyclic serv.
- Over both λ: no-projection is equal or better on 7 of 12 runs and clearly worse on 3 (riscv λ=4, picosoc λ=1, biriscv λ=1). It is cheaper per evaluation than L1-inside but not more reliable.

**Findings.**
1. **The max-flow can be removed for the acyclic problem.** The levelised L∞ projection matches L1 within the search noise (better on serv, riscv, biriscv; worse by 0.3–8% on omsp, aes, picosoc) and the search is up to 9.5× faster (biriscv 581 → 61 s). On small designs FM dominates, so the time is unchanged there.
2. **A searched α is unnecessary.** The search settles at α ≈ 0.5 (0.34–0.52) and the results equal the fixed midpoint within noise. Use the midpoint: no extra parameter.
3. **Kahn is rejected:** equal on the four small designs, much worse on riscv (336 vs 263) and biriscv (558 vs 251), and not faster (Python loop).
4. **No-projection for finite λ is mixed.** Same J as L1-inside on serv, omsp, picosoc, biriscv; much better on aes (J 375 vs 463: 32 back edges buy −46% cut already at λ = 4); much worse on riscv (J 463 vs 323). The riscv result is a search failure (the acyclic search reaches 263 / 0).
5. **Per-evaluation cost is now FM** (pure Python): 26–57 ms wall on 12 workers for λ = ∞, 88–337 ms with the penalised FM.
6. **Current acyclic method (L∞ decoder) vs best single baseline:** serv 61 / 61, omsp 316 / 341, picosoc 262 / 250, aes 477 / 475, riscv 263 / 251, biriscv 251 / 238: geo-mean 1.01×, with all differences except omsp inside the ±10% noise.

## 45. Search instability: measurement and a remedy by λ-continuation (`homotopy_dec.py` starts, `homotopy_cont.py`; ε=.10; 2026-11-13)

**Measurement (no change to the method; deterministic starts w0 = e_k, k = 1..5).**

Acyclic, L∞ decoder: refined cut.

| design | e1 | e2 | e3 | e4 | e5 | median | spread |
|---|---|---|---|---|---|---|---|
| serv | 61 | 64 | 66 | 69 | 66 | 66 | 13% |
| omsp | 316 | 311 | 391 | 328 | 319 | 319 | 5% + one outlier |
| picosoc | 262 | 265 | 292 | 281 | 273 | 273 | 11% |
| aes | 477 | 475 | 479 | 477 | 468 | 477 | 2% |
| riscv | 263 | 266 | 266 | 266 | 266 | 266 | 1% |
| biriscv | 251 | 326 | 238 | 238 | 238 | 238 | 5% + one outlier |

Finite λ = 1, no-projection decoder, fresh starts: J = cut + back.

| design | e1 | e2 | e3 | acyclic (e1) |
|---|---|---|---|---|
| serv | 59 | 64 | 65 | 61 |
| omsp | 156 | 153 | 149 | 316 |
| picosoc | 251 | 270 | 212 | 262 |
| aes | 250 | 252 | 279 | 477 |
| riscv | 318 | **345** | 264 | 263 |
| biriscv | **411** | **625** | 238 | 251 |

- The acyclic search is fairly stable: 1–13%, with rare outliers. The fixed start e1 (the Fiedler direction, i.e. spectral bisection) is never the outlier.
- The finite-λ search from a fresh start is the unstable part. On biriscv and riscv it ends *worse than the acyclic solution*, which is itself feasible at any λ: a pure search failure.

**Remedy.**
- Decoder with a mixing parameter t ∈ [0, 1], searched: f = (1−t)·s + t·L∞(s). t = 1 is the acyclic decoder; t = 0 is no projection.
- λ-continuation ∞ → 4 → 1. Each stage starts from the previous incumbent (w, t) and is seeded with the previous stage's **final refined** partition.
- Guarantee: J_λ(result) ≤ J_λ(previous final) ≤ refined acyclic cut.
- A first version seeded only the search-level incumbent and violated the guarantee on serv (64 > 61); fixed before the run below.

| design | λ=∞ | λ=4 cut/back (J) | λ=1 cut/back (J) | λ=1 fresh starts J (min / median / max) |
|---|---|---|---|---|
| serv | 61 | 61 / 0 (61) | **59 / 0 (59)** | 59 / 64 / 65 |
| omsp | 316 | 217 / 2 (225) | 135 / 21 (156) | 149 / 153 / 156 |
| picosoc | 262 | 262 / 0 (262) | 260 / 1 (261) | 212 / 251 / 270 |
| aes | 477 | 461 / 0 (461) | 226 / 40 (266) | 250 / 252 / 279 |
| riscv | 263 | 251 / 2 (259) | **231 / 4 (235)** | 264 / 318 / 345 |
| biriscv | 251 | **238 / 0 (238)** | 238 / 0 (238) | 238 / 411 / 625 |

Time: 180–522 s per design for all three stages (12 workers).

**Findings.**
1. **The failure mode is gone.** No finite-λ result is worse than the acyclic one. biriscv λ=1: 238 against fresh-start 411/625. riscv λ=1: J 235 against 318/345, better than every fresh start; cut 231 with 4 back edges, against 251/0 for dagP-w + refine.
2. **Continuation also improves the acyclic end:** biriscv λ=4 finds 238 with 0 back edges (λ=∞ stage: 251).
3. **The price is less exploration.** Continuation stays near t = 1 on 4 of 6 designs and misses large jumps that some fresh starts found: picosoc λ=1 (261 vs best fresh 212), aes λ=4 (461 vs 375 in §44), omsp λ=4 (225 vs 196 in §40). Against the *median* fresh start it is better on 3 designs, equal on 2, 6% worse on aes.
4. Fresh-start multi-start would recover those jumps only as a best-of-N, which is excluded as a method.

**Queue.** Exploration within the continuation (e.g. start each stage from a larger σ, or let t start below 1); the equal-budget random-direction baseline (Lang); logistic feedback from P0; TV as a starting direction.

## 46. Equal-budget baseline: Lang's swept random hyperplanes (`lang.py`; ε=.10; 2026-11-13)

**Setup.**
- Same embedding, decoder, objective and final refinement as §45; only the search differs.
- Random baseline: 1085 random unit directions (the per-stage evaluation count of §45), best kept.
  - λ = ∞: t = 1.
  - λ = 1: t ~ U[0, 1].
- One seed (rng 0).

| design | λ=∞ random | λ=∞ homotopy e1 (5-start median) | λ=1 random J (cut / back) | λ=1 continuation J (cut / back) |
|---|---|---|---|---|
| serv | 67 | **61** (66) | 59 (59 / 0) | 59 (59 / 0) |
| omsp | **307** | 316 (319) | 156 (135 / 21) | 156 (135 / 21) |
| picosoc | 300 | **262** (273) | 312 (247 / 65) | **261** (260 / 1) |
| aes | **463** | 477 (477) | 270 (230 / 40) | **266** (226 / 40) |
| riscv | 266 | 263 (266) | 248 (246 / 2) | **235** (231 / 4) |
| biriscv | 245 | 251 (**238**) | 316 (312 / 4) | **238** (238 / 0) |
| geo-mean, ours / random | | 0.97 (median 0.99) | | **0.92** |

The median random direction is 2–14× worse than the best (e.g. biriscv: median 3548 vs best 345 before refine).

**Findings.**
1. **Acyclic: the homotopy search is not better than random sampling at equal budget** (geo-mean 0.97, 0.99 against the 5-start median, inside the noise). It wins serv and picosoc and loses omsp, aes and biriscv. The search mechanism contributes little at λ = ∞; the embedding, the decoder and FM do the work.
2. **Few back edges: the continuation clearly beats random sampling** (geo-mean 0.92). It is equal on serv, omsp and aes and better on picosoc (−16%), riscv (−5%) and biriscv (−25%). The gain is on the large designs, where random sampling of (w, t) wastes its budget.
3. **Consequence for the paper.** The defensible search contribution is the λ-continuation (warm start + guarantee), not homotopy smoothing per se. For λ = ∞ a deterministic start (e1) followed by homotopy and an equal-budget random sweep are interchangeable.
