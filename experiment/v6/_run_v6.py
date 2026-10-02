"""
_run_v6.py — V5 dual-GPU SPH headless runner (skeleton).

v1.0 entry point. Walks the V5 stack end-to-end:

    1. Build 2 V5 VulkanContext (device_indices=[0, 1])
    2. Build 2 SphSimulatorV6 with V1-equivalent test case (slab partition)
    3. Build DualGpuOrchestratorV6
    4. Bootstrap + run_until(max_steps)
    5. Print instrumentation summary; readback alive_count for sanity

Renderer / GLFW path lives in a separate `_run_v6_viewer.py` (Phase 5+).

Usage (run from repo root):
    .venv/Scripts/python.exe experiment/v6/_run_v6.py
"""

from __future__ import annotations

import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main() -> None:
    # Phase 2-4 fills in the actual wiring. Phase 1 just verifies imports work.
    from experiment.v6.utils.simulator_v6 import SphSimulatorV6
    from experiment.v6.utils.transport_v6 import GhostMigrationWorker
    from experiment.v6.utils.orchestrator_v6 import DualGpuOrchestratorV6

    _ = (SphSimulatorV6, GhostMigrationWorker, DualGpuOrchestratorV6)
    print("[run_v6] Phase 1 skeleton imports OK; runtime not implemented yet.")


if __name__ == "__main__":
    main()
