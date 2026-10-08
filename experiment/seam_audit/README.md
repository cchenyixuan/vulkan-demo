# Seam audit

Per-particle comparison of a multi-slab chain (test **B**, K = 2 or 4) against a
reference run (**A**, normally V5 with K = 1), binned by voxel-column distance to
B's seams and measured against the reference's own run-to-run noise floor.

## Why this tool exists

The V5 1-D slab decomposition (1-voxel-column ghost, voxel size = kernel support h)
is not equivalent to a single GPU at the seam:

1. **Stale ghost density.** Phase C force for an own column-0 particle reads the ghost
   neighbours' density and pressure from step n. The replicas are packed in phase A3
   (ghost_send), before this step's density, while a single GPU's force reads step n+1.
2. **Departed migrant.** A migrant killed by the sender in A3 is, during the sender's
   C2-C5 of that step, neither in the sender's own lists nor in its ghost lists. The
   sender's column-0 particles miss one neighbour for one step.

The existing checks cannot see either effect. `experiment/v5/_verify_cascade_force.py`
compares one chain against another chain, and the envelope of `_run_v5_equivalence.py`
already contains |K1 - K2|. This tool compares any two configurations particle by
particle, for example V5 K=1 against V5 or V6 with K = 2 or 4.

## What it measures

Each configuration runs twice. A1 and A2 are the reference trials, B1 and B2 the
test trials. Every selected particle i of B1 is matched to a particle a(i) of A1,
and a(i) to a particle b of A2. On those same particles:

    test  = |B1_i - A1_a|      (decomposition effect + floating-point noise)
    noise = |A1_a - A2_b|      (reference run-to-run floor: atomicAdd slot order)
    ratio = statistic(test) / statistic(noise)   for mean, rms, p99, max

Both sides of the ratio are computed on the same particle set. Fields:

- acceleration, shift and velocity: norm of the vector difference
- density, pressure and kernel_sum: absolute difference

The primary fields are acceleration, shift and density.

When B2 exists, the tool also reports:

- the replicate pair (B2 vs A2, with noise A2 vs A1)
- the test configuration's own noise, |B1 - B2| against |A1 - A2|

**Bins.** Columns come from B's seams. The global column is
c = floor((x_B - origin_x) / h). For the nearest cut s:

- signed column: c - s. So -2 is left slab column 1, -1 is left column 0,
  0 is right column 0 and 1 is right column 1.
- unsigned distance: d = c - s when c >= s, otherwise s - 1 - c.

The bins are d = 0..15 plus `interior` (d >= 16, or no seam at all).

**Column-0 split.** This uses B1's `crossed_last_step` flag (the owning slab at N
differs from the one at N-1):

| group | meaning |
|---|---|
| flagged | another particle that crossed a seam in the last step lies closer than h |
| unflagged | no such particle |
| crossed_self | the particle crossed itself |
| unflagged_not_crossed | no crossing nearby and not crossed itself: the cleanest effect-(1)-only set |
| flagged_departed | a crossing neighbour left this particle's slab: effect (2) |
| flagged_arrived_only | every crossing neighbour arrived into this slab |
| flagged_including_self | the literal reading that also counts the particle itself |

"Closer than h" is strict because the Wendland kernel gives W(r >= h) = 0.

**Also reported:**

- column-0 kernel_sum distributions for B1 and the matched A1, and the signed
  B1 - A1 difference, for every column-0 group. A departed neighbour shows up as a
  lower kernel_sum in `flagged_departed` (`kernel_sum_column0.png` plots departed,
  arrived-only and unflagged separately).
- the mean |shift| profile across the seam, signed columns -16..15.
- matching quality.
- the invariants of every input run.

## Files

| file | role |
|---|---|
| `solver_adapter.py` | `load_solver("v5", "v6" or "v7")` returns load_case, compute_chain_partition, Simulator, Orchestrator, Context and env_prefix. Set the environment before calling it, because the solvers read their switches at import time. |
| `dump_state.py` | GPU worker. One process runs one configuration and writes one dump per horizon. `--version v7` (E39) names every switch it sets with the V7_ prefix (`V7_TRANSPORT_EXTENSION=1`); v5 / v6 runs set what they always set. |
| `analyze.py` | CPU analysis. Importable `analyze(...)`, a CLI, and `--self-test`. |
| `run_matrix.py` | Sequential campaign: subprocess runs, timeouts, resume, analysis, summary. |
| `matrix_v5_baseline.json` | V5 K=2 and K=4 against V5 K=1: 2-D 1M, 2-D 4M and 3-D 1M (K=2 only for 3-D), N = 300 and 2000, two trials each. |
| `matrix_v6.json` | V6 `KEEP_DEPARTED` / `GHOST_LAYERS` variants (keep0_layers1, keep1_layers1, keep1_layers2) at K=2 and K=4, against the same V5 K=1 reference. |
| `canonical_dump.py` | E37 bit-identity harness: a K-slab run from the initial state with canonical voxel lists, the full per-particle state dumped by global id, `--compare` bit for bit; `--repo` runs another checkout (e.g. v6-rc1); `--monitor --timestamps` is the E36 section 8.3 timing method. |
| `wall_option_timing.py` | E37 timing tables of the wall option (simple / adami, K = 1) from `canonical_dump --monitor --timestamps` runs and chain-bench logs. |
| `e39_ensemble.py` | E39 accuracy B: E33's ensemble test at K = 1 with the solver as the arm (v6-rc2 vs v7, 6 restarts each from the N = 2000 snapshots of 2-D 1M and 3-D 1M). `run` (GPU; `--dry-run [--preflight]` plans, estimates and checks on the CPU), `analyze` (all-fluid and near-wall bins, E33's permutation tests and joint null, between term, family-wise min-p, resolving power), `selftest` (synthetic dumps with known offsets). |

## How to run

The commands below run from the repository root.

```bash
# CPU only, safe while the GPUs are busy
.venv/Scripts/python.exe -m experiment.seam_audit.analyze --self-test
.venv/Scripts/python.exe -m experiment.seam_audit.run_matrix --matrix experiment/seam_audit/matrix_v5_baseline.json --out logs/seam_audit/campaign --list
.venv/Scripts/python.exe -m experiment.seam_audit.run_matrix --matrix experiment/seam_audit/matrix_v5_baseline.json --out logs/seam_audit/campaign_preflight --dry-run-workers

# GPU campaign. Use the same --out for both matrices so the V5 K=1 reference dumps are reused.
.venv/Scripts/python.exe -m experiment.seam_audit.run_matrix --matrix experiment/seam_audit/matrix_v5_baseline.json --out logs/seam_audit/campaign
.venv/Scripts/python.exe -m experiment.seam_audit.run_matrix --matrix experiment/seam_audit/matrix_v6.json --out logs/seam_audit/campaign

# Redo only the CPU analysis and summary, optionally filtered
.venv/Scripts/python.exe -m experiment.seam_audit.run_matrix --matrix ... --out ... --analyze-only [--only-cases cavity2d_1m] [--only-tests v5_K2]

# A single run, or a single comparison
.venv/Scripts/python.exe -m experiment.seam_audit.dump_state --version v5 --case cases/lid_driven_cavity_2d_gen/case.yaml --slabs 2 --device-map 0,1 --horizons 300,2000 --out-dir DIR --run-name v5_K2_t1
.venv/Scripts/python.exe -m experiment.seam_audit.analyze --reference A1.npz A2.npz --test B1.npz B2.npz --out DIR [--match id] [--particles all]
```

`--dry-run-workers` pre-flights every planned run on the CPU. Each worker loads the
case, partitions it and checks the global-id mask against the partition, then stops
before any Vulkan context exists. Run it after changing a matrix, or after V6
changes its partitioner.

`--self-test` runs two synthetic scenarios. The first has one seam and every
particle in every run, and checks the expected contrasts (flagged and departed far
above 1, far bins near 1, a lowered kernel_sum for departed). The second has three
seams; every run misses a different 1 % of the particles, and 25 particles of B1 and
of A2 sit beyond the match tolerance. It checks the matching, bins, column-0 groups
and ratios, for both match methods, against an independent brute-force computation.

A relative `--out` is taken from the current directory; the default is
`<repository>/logs/seam_audit/<matrix stem>`. Ctrl+C stops a campaign within about a
second: the running worker is killed and the attempt is recorded as `interrupted`.

**Production switches.** run_matrix gives every run these switches (P is `V5_` or `V6_`):

- `VK_LOADER_LAYERS_DISABLE=VK_LAYER_KHRONOS_validation`
- `<P>WORKER_COUNT_AWARE=1`
- `<P>SPLIT_TRANSFER_QUEUES=1`
- `<P>CASCADE_FORCE=1`
- `<P>BAND_VOXEL_DISPATCH=1`
- `<P>GHOST_POOL_FACTOR=0.25` in 2-D, `1.0` in 3-D

The configuration's own `env` is added on top. For v6 every switch the caller did not set
first gets its pre-E6b default (`partition_v6.LEGACY_DEFAULTS`), so the matrix files keep
their meaning now that the release set is the code default. Every sidecar records the `V5_*`,
`V6_*` and `VK_*` environment the run actually saw.

**Device indices.** These use the V5/V6 discrete-first order: 0 and 1 are the two
5090s. The reference uses device 1, the headless 5090.

## Crossing window (two-pass) — `crossing_window.py`, `window_analysis.py`

The single dumped frame rarely has a seam crossing in it: at these horizons a
2-D seam sees a few crossings per hundred frames (the lid-driven flow has not
reached the cavity centre at N = 300; at N = 2000 crossings come mostly from the
lid boundary layer). The `flagged` column-0 groups above are therefore usually
empty, so the matrices add a window before the listed `window_horizons`:

1. **Test runs** (`--window W`) step the last W frames one at a time. After each
   frame a cheap readback (own range of positions + ids, persistent mapped
   staging) finds the particles whose x changed side of any audit cut line
   (`--audit-slab-counts`: the equal-weight cuts of every K of the case); only
   then the full state is read and every particle within 1.5 h of a crossing is
   stored with its frame (`<run>_N<h>_window.npz`).
2. `run_matrix --build-capture-requests` unions the (frame, id) rows of every
   test window of a case (all matrices sharing `--out`) into
   `capture_requests/<case>/requests_N<h>.npz`.
3. **Reference runs** (`--capture-requests-dir`) step the same window and store
   exactly the requested rows. A K = 1 run crosses the same lines with the same
   particles, but not necessarily in the same frame (particles oscillate at the
   line, trajectories diverge), so capturing around its own crossings left every
   test row unmatched; the request pass makes the reference rows exist.
4. `window_analysis.analyze_window` pools all window frames: for every test
   column-0 fluid particle of frame f, the groups are `flagged` (another particle
   that crossed during f is within h), `departed` (one of them left this
   particle's slab — the V5 defect (2)), `arrived` (all of them arrived into it)
   and `control` (between h and 1.5 h from a crossing: same place and time, no
   crossing neighbour). test = |B1 - A1|, noise = |A1 - A2| on the same
   particles at the same frame, as in the main analysis.

Campaign order with windows (one `--out` for every matrix):

```bash
.venv/Scripts/python.exe -m experiment.seam_audit.run_matrix --matrix experiment/seam_audit/matrix_v5_baseline.json --out logs/seam_audit/campaign --roles test --no-analysis
.venv/Scripts/python.exe -m experiment.seam_audit.run_matrix --matrix experiment/seam_audit/matrix_v6.json --out logs/seam_audit/campaign --roles test --no-analysis
.venv/Scripts/python.exe -m experiment.seam_audit.run_matrix --matrix experiment/seam_audit/matrix_v5_baseline.json --out logs/seam_audit/campaign --build-capture-requests
.venv/Scripts/python.exe -m experiment.seam_audit.run_matrix --matrix experiment/seam_audit/matrix_v5_baseline.json --out logs/seam_audit/campaign --roles reference --no-analysis
.venv/Scripts/python.exe -m experiment.seam_audit.run_matrix --matrix experiment/seam_audit/matrix_v5_baseline.json --out logs/seam_audit/campaign --analyze-only
.venv/Scripts/python.exe -m experiment.seam_audit.run_matrix --matrix experiment/seam_audit/matrix_v6.json --out logs/seam_audit/campaign --analyze-only
```

(`logs/seam_audit/run_campaign.sh` is exactly this sequence.)
`python experiment/seam_audit/window_analysis.py` runs its synthetic self-test.

## Performance campaign — `perf_campaign.py`

v5 against the three v6 seam configurations, one process per run, trials
interleaved (trial, then case, then configuration). Each run: steady fps of the
production depth-2 loop without timers; then BenchTimers attached, the step
command buffers re-recorded and depth-1 steps with per-frame GPU timestamps
(phase C, the three band kernels, append_departed, DMA); bytes per link per frame
(count-aware host copy and DMA staging size); drift / stamps / overflow /
far-migration invariants. `--summarize-only` rebuilds `summary.md`.

## Output layout (`--out`)

```
dumps/<case>/<run>_N<horizon>.npz       uncompressed; arrays sorted by global id
dumps/<case>/<run>_N<horizon>.json      sidecar: config, env, cuts, h, origin_x, statuses, invariants
logs/<case>/<run>.log                   stdout + stderr, appended per attempt
runs.jsonl                              one record per attempt (status valid/invalid/failed/timeout)
analysis/<case>/<test>_K<k>_N<horizon>/ report.json, report.md, difference_by_column.png,
                                        shift_profile.png, kernel_sum_column0.png
summary.md, summary.json                one row per (case, test, horizon)
summary_<matrix>.md / .json             the same, named after the matrix file
```

**Run names.**

- Tests: `<test name>_K<k>_t<trial>`.
- The reference: `reference_<version>_dev<devices>[_env<hash>]_K<k>_t<trial>`. The name
  is built from the reference's content, not from the matrix file, so two matrices
  share it. Keep the case names identical across matrix files.

**Dump arrays.** These are per particle:

- `id` (uint32)
- `slab`, plus `previous_slab` (owner at N-1, 255 = unknown), both uint8
- `crossed_last_step` (bool)
- `position`, `velocity`, `acceleration`, `shift`: float32 with shape (n, dimension). In 2-D, z is dropped.
- `density`, `pressure`, `kernel_sum`: float32
- `material` (uint16 material group; the kinds are in the sidecar)

**Disk use.** A dump takes about 53 B per particle in 2-D and 69 B in 3-D. That is
about 55 MB for 2-D 1M, 220 MB for 2-D 4M and 115 MB for 3-D 1M, per run and per
horizon. The v5 baseline matrix needs about 4.3 GB. The v6 matrix adds about 8 GB,
since it reuses the reference. run_matrix stops when the output drive has less than
`--minimum-free-gb` free (default 4).

## How to read the ratios

| ratio | reading |
|---|---|
| about 1 | the decomposition is indistinguishable from reordering the floating-point sums |
| d = 0 above d >= 1 | a seam-local defect |
| flagged above unflagged, mainly `flagged_departed` | effect (2), the departed migrant |
| unflagged_not_crossed above 1 | effect (1), stale ghost density/pressure in the force |
| kernel_sum lower in B1 than in A1 for flagged_departed (`kernel_sum B1-A1 mean` below 0 in the column-0 table) | the missing neighbour, directly |
| `seam_excess` | acceleration rms ratio at d = 0. A single headline number, not a PASS/FAIL |

Some effects grow over time:

- **Chaos growth.** At N = 2000 the noise floor itself has grown, and a seam
  perturbation has had about 2000 steps to spread. Sound covers about 0.15 h per
  step, about 300 columns in 2000 steps. A far-bin ratio well above 1 at large N
  therefore means the perturbation has propagated; it is not a local defect. Compare
  N = 300 with N = 2000, and use the d = 0 to d >= 8 contrast.
- **Matching.** KD-tree matching, the default, uses a tolerance of 0.05 h. Once
  trajectories diverge by more than that, particles go unmatched, and the matched set
  leans towards calm regions. Check `unmatched` and the id agreement rate in
  `report.md`. `--match id` matches every particle by identity. Its "id offset" shows
  how far trajectories have separated, in units of h.
- **Second pair and test noise.** `B2 vs A2` should reproduce the B1 vs A1 ratios. If
  `|B1-B2| / |A1-A2|` is well above 1, the decomposed solver is also noisier, not just
  offset.
- **Few crossings per frame.** Early in the cavity flow (N = 300 and 2000), the number
  of particles crossing a seam in one frame can be small, possibly zero. Check
  `bookkeeping.crossed_last_step_count` and `crossings_by_direction` in the sidecar.
  The flagged group of a single dumped frame can then be small or empty (the ratio
  shows "-"). Pooling over many frames needs a capture window rather than one frame
  per horizon.
- **Particle filter.** The default is `particles=fluid`. Walls have zero acceleration
  and zero shift in every run. `--particles all` includes them.

## Invariants (per dump, `valid` in the sidecar)

All of these must be zero for a dump to be valid:

- drift: alive particles minus the initial total
- missing ids, duplicate ids, undecodable ids
- non-zero `extension_fields.x/y` (that would mean the solver started using the field)
- GPU `stamp_error_count`
- the host stamp errors of every worker
- every `global_status` key starting with `overflow_`

The overflow list is not hard-coded, so V6 can add counters. Each counter is the
per-sim maximum over every observation (the defrag boundaries plus the dump), summed
over sims.

V6's `far_migration_count` counts migrants found in the outer ghost column, a
two-column jump that CFL should make impossible. It is recorded the same way and
listed under `invariants.warnings` when non-zero, but it does not change `valid`.

The worker's exit code is 0 when everything is valid, 3 when the run completed but
is invalid, and 1 on an error. The last stdout line is always
`[seam_audit] RESULT {json}`.

## Implementation notes

- **Global id without touching the solver.** dump_state monkeypatches the simulator's
  `_build_initial_data` inside its own process. The payload also carries
  `extension_fields`: z = id // 2^20 and w = id % 2^20, both exact in float32. The
  id is the row index in the degenerate global case. ghost_send, install_migrations
  and defrag copy the field bit-exactly, so the id follows every particle through
  migration and defrag. Each slab's ids use exactly the mask of
  `_filter_particles_by_x_range` and are asserted bit-identical to the slab's initial
  arrays. After bootstrap, every slab is checked to own exactly its ids.
- **Frame loop.** `run_frames` mirrors `ChainOrchestratorV5.run_pipelined`'s default
  loop, with continuing frame numbers. `run_pipelined` restarts at frame 0, and the
  timeline values and the workers' stamp check need monotonic frames. For each
  horizon N the run goes to N-1, snapshots ownership, runs one frame and dumps.
  Pipeline drains and readbacks do not change GPU state. A defrag on a horizon runs
  after the dump. A defrag on the final horizon is skipped, because no frames follow.
  The submit and defrag order was checked against the real `run_pipelined` on a fake
  orchestrator. `V5_PER_SIM_PIPELINE` is ignored, and a warning is printed.
- **Failure exit.** The Vulkan objects are torn down only after a successful run. On
  any error (a frame stall, a dead transport worker, Ctrl+C) the worker prints the
  traceback and the RESULT line and leaves through `os._exit` without teardown.
  Frames still in flight wait on timeline values that will never be signalled, so
  `vkDeviceWaitIdle` in `sim.destroy()` would block forever. Process exit releases
  the devices, as the matrix runner's kill does.
