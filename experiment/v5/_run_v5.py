"""
_run_v5.py — V5 dual-GPU SPH headless runner (skeleton).

v1.0 entry point. Walks the V5 stack end-to-end:

    1. Build 2 V5 VulkanContext (device_indices=[0, 1])
    2. Build 2 SphSimulatorV5 with V1-equivalent test case (slab partition)
    3. Build DualGpuOrchestratorV5
    4. Bootstrap + run_until(max_steps)
    5. Print instrumentation summary; readback alive_count for sanity

Renderer / GLFW path lives in a separate `_run_v5_viewer.py` (Phase 5+).

Usage (run from repo root):
    .venv/Scripts/python.exe experiment/v5/_run_v5.py
"""

from __future__ import annotations

import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main() -> None:
    # Phase 2-4 fills in the actual wiring. Phase 1 just verifies imports work.
    from experiment.v5.utils.simulator_v5 import SphSimulatorV5
    from experiment.v5.utils.transport_v5 import GhostMigrationWorker
    from experiment.v5.utils.orchestrator_v5 import DualGpuOrchestratorV5

    _ = (SphSimulatorV5, GhostMigrationWorker, DualGpuOrchestratorV5)
    print("[run_v5] Phase 1 skeleton imports OK; runtime not implemented yet.")


if __name__ == "__main__":
    main()
