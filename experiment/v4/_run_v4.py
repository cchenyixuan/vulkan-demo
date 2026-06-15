"""
_run_v4.py — V4 dual-GPU SPH headless runner (skeleton).

v1.0 entry point. Walks the V4 stack end-to-end:

    1. Build 2 V4 VulkanContext (device_indices=[0, 1])
    2. Build 2 SphSimulatorV4 with V1-equivalent test case (slab partition)
    3. Build DualGpuOrchestratorV4
    4. Bootstrap + run_until(max_steps)
    5. Print instrumentation summary; readback alive_count for sanity

Renderer / GLFW path lives in a separate `_run_v4_viewer.py` (Phase 5+).

Usage (run from repo root):
    .venv/Scripts/python.exe experiment/v4/_run_v4.py
"""

from __future__ import annotations

import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main() -> None:
    # Phase 2-4 fills in the actual wiring. Phase 1 just verifies imports work.
    from experiment.v4.utils.simulator_v4 import SphSimulatorV4
    from experiment.v4.utils.transport_v4 import GhostMigrationWorker
    from experiment.v4.utils.orchestrator_v4 import DualGpuOrchestratorV4

    _ = (SphSimulatorV4, GhostMigrationWorker, DualGpuOrchestratorV4)
    print("[run_v4] Phase 1 skeleton imports OK; runtime not implemented yet.")


if __name__ == "__main__":
    main()
