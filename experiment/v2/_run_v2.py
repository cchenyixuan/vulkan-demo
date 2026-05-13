"""
_run_v2.py — V2 dual-GPU SPH headless runner (skeleton).

v1.0 entry point. Walks the V2 stack end-to-end:

    1. Build 2 V2 VulkanContext (device_indices=[0, 1])
    2. Build 2 SphSimulatorV2 with V1-equivalent test case (slab partition)
    3. Build DualGpuOrchestratorV2
    4. Bootstrap + run_until(max_steps)
    5. Print instrumentation summary; readback alive_count for sanity

Renderer / GLFW path lives in a separate `_run_v2_viewer.py` (Phase 5+).

Usage (run from repo root):
    .venv/Scripts/python.exe experiment/v2/_run_v2.py
"""

from __future__ import annotations

import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main() -> None:
    # Phase 2-4 fills in the actual wiring. Phase 1 just verifies imports work.
    from experiment.v2.utils.simulator_v2 import SphSimulatorV2
    from experiment.v2.utils.transport_v2 import GhostMigrationWorker
    from experiment.v2.utils.orchestrator_v2 import DualGpuOrchestratorV2

    _ = (SphSimulatorV2, GhostMigrationWorker, DualGpuOrchestratorV2)
    print("[run_v2] Phase 1 skeleton imports OK; runtime not implemented yet.")


if __name__ == "__main__":
    main()
