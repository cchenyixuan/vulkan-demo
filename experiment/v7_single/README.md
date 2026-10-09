# experiment/v7_single

The v7 kernels (`experiment/v7/shaders/spv`, verified against `MANIFEST.txt` at start-up) with a host
side that does only what one GPU needs: one compute queue, EXCLUSIVE buffers, no ghost / migrant /
departed pools, no timeline semaphores, no phases, no worker threads, no chain orchestration. Kept: the
bootstrap, one pre-recorded single-command time step (barriers as in v7's single-command step), the
adami wall pass, the defrag command and cadence, the status-block readback, one fence per step and a
ring of N fences (N steps in flight). Design, inventory, acceptance and measurements:
`docs/seam_audit/v7_single.md`.

| file | role |
|---|---|
| `utils/vulkan_context_single.py` | instance + one compute queue + one command pool; optional validation layer with a message counter |
| `utils/spirv_manifest.py` | sha256 check of every file the v7 manifest lists; loads the SPIR-V |
| `utils/switches.py` | the three kernel switches (`V7_FUSED_CORRECTION_DENSITY`, `V7_DENSITY_COPY_COMPUTE`, `V7_DEEP_WALL_SKIP`) |
| `utils/deep_wall_single.py` | E39 B4's auto rule and candidate count (v7's, copied) |
| `utils/simulator_single.py` | `SphSimulatorSingle`: buffers, pipelines, the step / bootstrap / defrag / restart recordings, the fence ring, readbacks |
| `utils/runner_single.py` | the step loop with a drained defrag boundary; `SingleRunner`, the orchestrator-shaped wrapper the cavity runner drives |
| `_run_single_bench.py` | headless throughput runner (N in flight, drift / overflow check, `RESULT {json}`) |

The case comes from v7's loader (`experiment.v7.utils.case_loader_v7.load_case_v7`); nothing else of
`experiment/v7`'s Python is imported.
