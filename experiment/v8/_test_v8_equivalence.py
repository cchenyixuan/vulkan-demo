"""
_test_v8_equivalence.py — with the periodic flags off, experiment/v8's kernels are experiment/v7's: run the same
cavity case on v7_single (experiment/v7's SPIR-V) and on v8 (experiment/v8's SPIR-V) in one process, one GPU,
and compare every per-particle field bit for bit after `--steps` steps.

The protocol is canonical_dump's --canonical-lists (experiment/seam_audit/canonical_dump.py): a global id per
particle in extension_fields, the voxel lists sorted by pid after the initial voxelization (their atomic-append
order is the one run-to-run variable of a single GPU), no bootstrap defrag, and the run ends before the first
defrag (steps < the case's cadence). A second v7_single run is the control: it must be bitwise equal to the
first, or the GPU itself is not reproducible and the comparison means nothing.

    .venv/Scripts/python.exe experiment/v8/_test_v8_equivalence.py --case logs/v8/smoke/cavity_250/case.yaml --steps 999
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

_REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

HEADLESS_5090_UUID = "ae137c668a40f5acaab90a83f7cda175"
FIELDS = {"position_voxel_id": 4, "velocity_mass": 4, "density_pressure": 2, "acceleration": 4, "shift": 4,
          "correction_inverse": 8, "density_gradient_kernel_sum": 4, "extension_fields": 4}


def with_global_ids(sim) -> None:
    """Upload the particle's initial row as a global id in extension_fields.w (canonical_dump's scheme, low word)."""
    original = sim._build_initial_data

    def build() -> dict[str, bytes]:
        data = original()
        count = int(sim.case.initial.positions.shape[0])
        fields = np.zeros((sim.case.capacities.total_pool_capacity(), 4), dtype=np.float32)
        first = sim.own_first_pid()
        fields[first:first + count, 3] = np.arange(count, dtype=np.float32)
        data["extension_fields"] = fields.tobytes()
        return data
    sim._build_initial_data = build


def sort_voxel_lists(sim) -> None:
    """canonical_dump's sort_voxel_lists: every inside list sorted by pid, re-uploaded."""
    slots = sim.case.capacities.max_particles_per_voxel
    raw = sim.readback_buffers_batch(["inside_particle_count", "inside_particle_index"])
    counts = np.frombuffer(raw["inside_particle_count"], np.uint32)
    index = np.frombuffer(raw["inside_particle_index"], np.uint32).copy()
    rows = index[:counts.size * slots].reshape(counts.size, slots)
    for voxel in np.flatnonzero(counts):
        rows[voxel, :counts[voxel]] = np.sort(rows[voxel, :counts[voxel]])
    sim._staging_upload(sim.buffers["inside_particle_index"], index.tobytes())


def canonical_run(sim, steps: int) -> dict[str, np.ndarray]:
    """Canonical bootstrap (sorted lists, no bootstrap defrag), `steps` steps with one fence, fields by global id."""
    with_global_ids(sim)
    sim._upload_initial_state()
    sim._submit_recording(sim.record_bootstrap_init)
    sort_voxel_lists(sim)
    sim.bootstrap_compute()
    sim.prepare_step(in_flight=1)
    for _ in range(steps):
        sim.submit_step()
    sim.drain()
    raw = sim.readback_buffers_batch(list(FIELDS))
    pool = sim.case.capacities.total_pool_capacity()
    state = {name: np.frombuffer(raw[name], dtype=np.float32)[:pool * components].reshape(pool, components).copy()
             for name, components in FIELDS.items()}
    alive = (state["position_voxel_id"][:, 3] > 0.5) & (state["velocity_mass"][:, 3] > 0.0)
    ids = np.rint(state["extension_fields"][alive, 3]).astype(np.int64)
    if np.unique(ids).size != ids.size:
        raise RuntimeError("global ids are not unique after the run")
    order = np.argsort(ids)
    state = {name: array[alive][order] for name, array in state.items()}
    state["global_id"] = ids[order]
    state["status"] = sim.readback_global_status()
    return state


def compare(first: dict, second: dict) -> dict:
    result = {"alive": [int(first["global_id"].size), int(second["global_id"].size)],
              "same_ids": bool(np.array_equal(first["global_id"], second["global_id"]))}
    for name in FIELDS:
        a, b = first[name], second[name]
        if a.shape != b.shape:
            result[name] = {"bitwise": False, "shape": [list(a.shape), list(b.shape)]}
            continue
        equal = bool(np.array_equal(a.view(np.uint32), b.view(np.uint32)))
        finite = np.isfinite(a) & np.isfinite(b)
        difference = float(np.abs(a[finite].astype(np.float64) - b[finite].astype(np.float64)).max()) if finite.any() else 0.0
        result[name] = {"bitwise": equal, "max_abs_difference": difference,
                        "differing_rows": int((a.view(np.uint32) != b.view(np.uint32)).any(axis=1).sum())}
    result["bitwise"] = result["same_ids"] and all(result[name]["bitwise"] for name in FIELDS)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", required=True)
    parser.add_argument("--gpu-uuid", default=HEADLESS_5090_UUID)
    parser.add_argument("--steps", type=int, default=999, help="< the case's defrag cadence")
    parser.add_argument("--out", default=None)
    arguments = parser.parse_args()

    from experiment.v7.utils.case_loader_v7 import load_case_v7
    from experiment.v7_single._run_single_bench import resolve_device_index
    from experiment.v7_single.utils.simulator_single import SphSimulatorSingle
    from experiment.v7_single.utils.vulkan_context_single import VulkanContextSingle
    from experiment.v8.utils.case_loader_v8 import load_case_v8
    from experiment.v8.utils.simulator_v8 import SphSimulatorV8

    device_index = resolve_device_index(arguments.gpu_uuid)
    ctx = VulkanContextSingle.create(device_index=device_index, enable_validation=False, application_name="v8_equivalence")
    report: dict = {"case": arguments.case, "steps": arguments.steps, "protocol": "canonical lists, no bootstrap defrag"}
    try:
        case7 = load_case_v7(arguments.case)
        case8 = load_case_v8(arguments.case)
        assert case8.periodic == (False, False)
        if arguments.steps >= int(case7.numerics.defrag_cadence):
            sys.exit(f"--steps must be < the defrag cadence {case7.numerics.defrag_cadence}")
        builds = {"v7_single": lambda: SphSimulatorSingle(ctx, case7), "v8": lambda: SphSimulatorV8(ctx, case8),
                  "v7_single_control": lambda: SphSimulatorSingle(ctx, case7)}
        states: dict[str, dict] = {}
        for name, make in builds.items():
            sim = make()
            states[name] = canonical_run(sim, arguments.steps)
            status = states[name]["status"]
            report[f"status_{name}"] = {"alive": status["alive_particle_count"],
                                        "overflow": {k: v for k, v in status.items() if k.startswith("overflow_") and v}}
            print(f"[equivalence] {name}: {arguments.steps} steps, alive {status['alive_particle_count']}, "
                  f"spv {sim.spirv_directory}")
            sim.destroy()
        report["v7_vs_v8"] = compare(states["v7_single"], states["v8"])
        report["v7_vs_v7"] = compare(states["v7_single"], states["v7_single_control"])
    finally:
        ctx.destroy()
    for pair in ("v7_vs_v8", "v7_vs_v7"):
        result = report[pair]
        worst = max(result[name].get("max_abs_difference", float("inf")) for name in FIELDS)
        print(f"[equivalence] {pair}: {'BITWISE' if result['bitwise'] else 'differs'} (max |diff| {worst:.3e}, "
              f"alive {result['alive']})")
    verdict = ("PASS" if report["v7_vs_v8"]["bitwise"] and report["v7_vs_v7"]["bitwise"]
               else "INCONCLUSIVE" if not report["v7_vs_v7"]["bitwise"] else "FAIL")
    report["verdict"] = verdict
    if arguments.out:
        out = pathlib.Path(arguments.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"[equivalence] {verdict}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
