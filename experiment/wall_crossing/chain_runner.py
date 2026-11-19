"""
chain_runner.py - run a case on the solver's production path with a callback at sample steps.

    run_chain(case, device_map, total_steps, sample_steps, on_sample, ...)

The path is the one experiment/validation/cavity_runner.py takes: compute_chain_partition (K = the number of
devices; K = 1 is a chain of one slab), one SphSimulatorV7 per slab (per-direction sync), ChainOrchestratorV7,
bootstrap_all, run_pipelined at depth 2. on_sample(step, simulators, slab_cases) is called at every multiple
of sample_steps (itself a multiple of the defrag cadence): run_pipelined drains every frame before its defrag
callback, so the buffers can be read there, before that boundary's defrag.

restart (a dict {"state", "stored_offset", "step"}: the merged restart state of every slab as
experiment/validation/cavity_runner.read_restart_state reads it, the stored density offset, the step it was taken
at): the chain starts from that state the way the cavity runner resumes (partition_v7.restart_slab_rows splits it,
ChainOrchestratorV7.restart_all uploads it) and steps from restart["step"] to total_steps; on_sample gets the
absolute step.

Why not the single-command-buffer step of _run_v7_single_bench.py: it waits on a fence after every step, and
on a card shared with another (pipelined) job that wait costs a scheduling slice per step (measured: 170
steps/s against ~1900 for the pipelined path on the same shared card). The kernels and pipelines are the same.
"""

from __future__ import annotations

import pathlib
import sys
import time
from typing import Callable, Optional

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def run_chain(case, device_map: list, total_steps: int, sample_steps: int, on_sample: Callable,
              expected_uuids: Optional[list] = None, pool_safety: float = 1.2,
              application_name: str = "wall_crossing", depth: int = 2, on_ready: Optional[Callable] = None,
              restart: Optional[dict] = None) -> dict:
    """Run ``case`` for total_steps steps on the devices of device_map (one v7 device index per slab).

    total_steps and sample_steps must be multiples of the case's defrag cadence. Returns
    {"device_uuids", "steps_per_second", "elapsed_seconds", "cuts"}. expected_uuids (hex, no dashes), when
    given, must equal the devices' UUIDs. on_ready(simulators, slab_cases), when given, is called after the
    bootstrap and before the first step."""
    from experiment.v7.utils.orchestrator_v7 import ChainOrchestratorV7
    from experiment.v7.utils.partition_v7 import compute_chain_partition
    from experiment.v7.utils.simulator_v7 import SphSimulatorV7
    from experiment.v7.utils.vulkan_context_v7 import VulkanContextV7
    from experiment.wall_crossing.run_log import vulkan_device_uuid

    cadence = int(case.numerics.defrag_cadence)
    start_step = int(restart["step"]) if restart is not None else 0
    if start_step % cadence or start_step >= total_steps:
        raise ValueError(f"restart step {start_step} must be a multiple of the defrag cadence below {total_steps}")
    if total_steps % cadence or sample_steps % cadence:
        raise ValueError(f"total_steps {total_steps} and sample_steps {sample_steps} must be multiples of the "
                         f"defrag cadence {cadence}")
    slab_count = len(device_map)
    chain = compute_chain_partition(case, [1.0] * slab_count, pool_safety)
    contexts, simulators = [], []
    orchestrator = None
    try:
        for device_index in device_map:
            contexts.append(VulkanContextV7.create(device_index=device_index, enable_validation=False,
                                                   application_name=application_name))
        uuids = [vulkan_device_uuid(context) for context in contexts]
        if expected_uuids is not None:
            wanted = [uuid.replace("-", "").lower() for uuid in expected_uuids]
            if wanted != uuids:
                raise RuntimeError(f"device uuids {uuids} differ from the expected {wanted}")
        for context, slab in zip(contexts, chain.slabs):
            simulators.append(SphSimulatorV7(context, slab, sync_scheme="per-direction"))
        orchestrator = ChainOrchestratorV7(simulators, defrag_cadence=cadence)
        if restart is None:
            orchestrator.bootstrap_all()
        else:
            import numpy as np

            from experiment.v7.utils.partition_v7 import restart_slab_rows
            state = restart["state"]
            rows_per_slab = restart_slab_rows(case, chain, state["position_voxel_id"][:, 0])
            orchestrator.restart_all([{name: np.ascontiguousarray(values[rows]) for name, values in state.items()}
                                      for rows in rows_per_slab], float(restart["stored_offset"]),
                                     volume_constraint=restart.get("volume_constraint"))
        if on_ready is not None:
            on_ready(simulators, chain.slabs)

        def on_defrag(frame: int, report: list) -> None:
            step = start_step + frame
            if step % sample_steps == 0 or step == total_steps:
                on_sample(step, simulators, chain.slabs)

        started = time.perf_counter()
        orchestrator.run_pipelined(total_steps - start_step, depth=depth, on_defrag=on_defrag)
        elapsed = time.perf_counter() - started
        return {"device_uuids": uuids, "steps_per_second": (total_steps - start_step) / elapsed,
                "elapsed_seconds": elapsed, "start_step": start_step,
                "cuts": [int(cut) for cut in getattr(chain, "cuts", [])]}
    finally:
        if orchestrator is not None:
            orchestrator.destroy()
        for simulator in simulators:
            simulator.destroy()
        for context in contexts:
            context.destroy()
