# SPH V1 Multi-GPU Experiment

Isolated workspace for the V1 dual-GPU SPH rewrite. **V0 (production single-GPU) lives in `utils/sph/`, `shaders/sph/`, `_run_viewer.py` and is frozen** — V1 work happens entirely under this directory.

## Why isolated

V0 is the production single-GPU codepath: minimal, fast, no ghost / migration / sync overhead. Most users on single-GPU rigs should run V0 unchanged. V1 layers multi-GPU support on top with cross-vendor CPU-staged transport, ghost particles, migration channels, and double-sync — all of which are pure overhead in single-GPU mode.

Keeping V1 separate means:
- V0 stays tight; it never has to opt-in or branch around V1 features
- V1 can iterate fast with bigger structural changes (descriptor layout, pipeline phases, ...) without breaking V0
- If V1 doesn't pan out, we throw away one folder

## Import rules

V1 (this folder) MAY import from `utils/sph/*` for unchanged shared infrastructure:
- `utils.sph.case` (case YAML loading)
- `utils.sph.grid` (grid math)
- `utils.sph.obj_loader` (geometry loading)
- `utils.sph.vulkan_context` (Vulkan platform setup; already gained an opt-in `device_index` kwarg)

V1 MUST NOT modify any file under `utils/sph/` or `shaders/sph/`. The moment V1 needs different behaviour from one of those files, copy it into `experiment/v1/utils/` (Python) or `experiment/v1/shaders/` (GLSL) and modify the local copy.

## Layout

```
experiment/v1/
├── README.md                       # this file
├── _run_v1.py                      # CLI entry point (planned: dual-GPU smoke + V1 run)
├── compile_shaders_v1.py           # planned: compiles experiment/v1/shaders/*.comp
├── shaders/                        # currently a copy of shaders/sph/; will diverge
│   ├── common.glsl
│   ├── helpers.glsl
│   ├── predict.comp
│   ├── update_voxel.comp           # future: + migration channel
│   ├── correction.comp             # future: + ghost neighbour iteration
│   ├── density.comp
│   ├── force.comp
│   └── ...
└── utils/
    ├── multi_gpu.py                # MultiGPUContext (dual VulkanContext holder)
    └── gpu_capability.py           # KNOWN_GPU_SPH_WEIGHT lookup (calibrated)
```

## Reference

Design: `docs/sph_v1_design.md` (kept at repo top-level alongside `sph_v0_design.md` and `sph_design.md`).
