"""
_run_v3.py — V3 dual-GPU SPH headless runner (skeleton).

v1.0 entry point. Walks the V3 stack end-to-end:

    1. Build 2 V3 VulkanContext (device_indices=[0, 1])
    2. Build 2 SphSimulatorV3 with V1-equivalent test case (slab partition)
    3. Build DualGpuOrchestratorV3
    4. Bootstrap + run_until(max_steps)
    5. Print instrumentation summary; readback alive_count for sanity

Renderer / GLFW path lives in a separate `_run_v3_viewer.py` (Phase 5+).

Usage (run from repo root):
    .venv/Scripts/python.exe experiment/v3/_run_v3.py
"""

from __future__ import annotations

import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main() -> None:
    # Phase 2-4 fills in the actual wiring. Phase 1 just verifies imports work.
    from experiment.v3.utils.simulator_v3 import SphSimulatorV3
    from experiment.v3.utils.transport_v3 import GhostMigrationWorker
    from experiment.v3.utils.orchestrator_v3 import DualGpuOrchestratorV3

    _ = (SphSimulatorV3, GhostMigrationWorker, DualGpuOrchestratorV3)
    print("[run_v3] Phase 1 skeleton imports OK; runtime not implemented yet.")


if __name__ == "__main__":
    main()
