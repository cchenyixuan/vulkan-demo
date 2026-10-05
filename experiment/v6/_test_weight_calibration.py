"""
_test_weight_calibration.py — E32 unit tests of the pilot weight calibration
(experiment/v6/weight_calibration.py). Zero-dependency runnable script (assert + non-zero exit on
failure), CPU only, no Vulkan, no case files.

  1  phase_intervals: complete ticks, the configuration-dependent last tick of a phase (no peer, no
     cascade force), incomplete frames.
  2  interval_union_length: disjoint, overlapping, nested, touching, unsorted.
  3  busy_statistics: one sim per device (median of T_A + T_B + T_C); two sims on one device run one
     after the other (union = sum) or interleaved (union = elapsed span); timestamps without a common
     clock (union longer than the span: largest median).
  4  calibration arithmetic on a synthetic histogram: device-map 0,0,1 at equal speed gives
     omega = (1, 1, 2) and fluid shares 1/4, 1/4, 1/2 within a column; a fixed per-step cost makes
     omega under-correct, and a second round moves closer to balance; equal slabs keep their cuts.
  5  weights file: write / read round trip (cuts recomputed), and the refusals (another particle set,
     another device map, weights that no longer give the stored cuts).

Usage:
    .venv/Scripts/python.exe experiment/v6/_test_weight_calibration.py
"""
from __future__ import annotations

import pathlib
import sys
import tempfile
import types

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiment.v6 import weight_calibration as calibration  # noqa: E402
from experiment.v6.utils.partition_v6 import (  # noqa: E402
    KIND_FLUID,
    MINIMUM_OWN_COLUMNS_HARD,
    chain_cuts_from_counts,
)

_passed = 0
_failed: list[str] = []


def check(condition: bool, message: str) -> None:
    global _passed
    if condition:
        _passed += 1
    else:
        _failed.append(message)
        print(f"  FAIL: {message}")


def frame_ticks(start: int, a: int, b: int, c: int, gap: int = 4) -> dict:
    """Ticks of one frame with phase lengths a, b, c, a gap between the phases."""
    ticks = {"a_start": start, "a_voxel_end": start + a // 2, "a_ghost_leading_end": start + a - 1,
             "a_ghost_trailing_end": start + a}
    b_start = start + a + gap
    ticks.update({"b_start": b_start, "b_density_deep_interior_end": b_start + b // 2,
                  "b_force_deep_interior_end": b_start + b})
    c_start = b_start + b + gap
    ticks.update({"c_start": c_start, "c_force_end": c_start + c})
    return ticks


def test_phase_intervals() -> None:
    print("[1] phase_intervals")
    ticks = frame_ticks(1_000, 300, 500, 200)
    intervals = calibration.phase_intervals(ticks)
    check(intervals == {"A": (1_000, 1_300), "B": (1_304, 1_804), "C": (1_808, 2_008)},
          f"complete frame: {intervals}")
    no_peer = {key: value for key, value in ticks.items() if not key.startswith("a_ghost")}
    check(calibration.phase_intervals(no_peer)["A"] == (1_000, 1_150), "slab without peers ends A at a_voxel_end")
    no_cascade = {key: value for key, value in ticks.items() if key != "b_force_deep_interior_end"}
    check(calibration.phase_intervals(no_cascade)["B"] == (1_304, 1_554),
          "cascade force off ends B at b_density_deep_interior_end")
    for missing in ("a_start", "b_start", "c_start", "c_force_end"):
        incomplete = {key: value for key, value in ticks.items() if key != missing}
        check(calibration.phase_intervals(incomplete) is None, f"frame without {missing} is incomplete")
    only_starts = {key: value for key, value in ticks.items() if key.endswith("start")}
    check(calibration.phase_intervals(only_starts) is None, "frame without any end tick is incomplete")


def test_union() -> None:
    print("[2] interval_union_length")
    cases = (([], 0), ([(0, 10)], 10), ([(0, 10), (20, 25)], 15), ([(0, 10), (5, 15)], 15),
             ([(0, 20), (5, 10)], 20), ([(0, 10), (10, 20)], 20), ([(20, 25), (0, 10), (5, 12)], 17))
    for intervals, expected in cases:
        check(calibration.interval_union_length(intervals) == expected,
              f"union of {intervals}: {calibration.interval_union_length(intervals)} != {expected}")


def frames_of(starts, a: int, b: int, c: int) -> list:
    return [calibration.phase_intervals(frame_ticks(start, a, b, c)) for start in starts]


def test_busy_statistics() -> None:
    print("[3] busy_statistics")
    period, frame_count = 2_000, 50
    single = frames_of([index * period for index in range(frame_count)], 300, 500, 200)
    single[7] = None                                       # one incomplete frame is skipped
    statistics = calibration.busy_statistics([single], [0], [1.0])
    check(statistics["per_sim"][0]["complete_frames"] == frame_count - 1, "incomplete frame not counted")
    check(abs(statistics["per_device"][0]["busy_us"] - 1.0) < 1e-12,
          f"one sim per device: busy {statistics['per_device'][0]['busy_us']} us != 1.0 us (1000 ticks)")
    check(abs(statistics["per_sim"][0]["b_p50_us"] - 0.5) < 1e-12, "phase B median")

    # device 0 hosts sims 0 and 1 run one after the other (every frame: s0 then s1), device 1 hosts sim 2
    serial_period = 2_100
    first = frames_of([index * serial_period for index in range(frame_count)], 300, 500, 200)
    second = frames_of([index * serial_period + 1_050 for index in range(frame_count)], 300, 500, 200)
    third = frames_of([index * serial_period for index in range(frame_count)], 300, 500, 200)
    statistics = calibration.busy_statistics([first, second, third], [0, 0, 1], [1.0, 1.0, 1.0])
    device = statistics["per_device"][0]
    check(device["method"].startswith("union") and abs(device["busy_us"] - 2.0) < 1e-9,
          f"serialized sharing: union {device['busy_us']} us != 2.0 us ({device['method']})")
    omega = calibration.normalized(calibration.omega_from_busy(
        [1_000, 1_000, 1_000], [statistics["per_device"][d]["busy_us"] for d in (0, 0, 1)]))
    check(np.allclose(np.array(omega) / omega[0], [1.0, 1.0, 2.0]), f"serialized sharing: omega {omega}")

    # interleaved: both sims' phases stretched to twice the length, overlapping in time
    first = frames_of([index * serial_period for index in range(frame_count)], 600, 1_000, 400)
    second = frames_of([index * serial_period + 10 for index in range(frame_count)], 600, 1_000, 400)
    statistics = calibration.busy_statistics([first, second, third], [0, 0, 1], [1.0, 1.0, 1.0])
    device = statistics["per_device"][0]
    # union per frame = [0, 610] + [604, 1614] + [1608, 2018] = 2018 ticks
    check(device["method"].startswith("union") and abs(device["busy_us"] - 2.018) < 1e-9,
          f"interleaved sharing: union {device['busy_us']} us != 2.018 us ({device['method']})")

    # same stretched intervals, but sim 1's ticks on another clock (offset by far more than the run)
    second_offset = frames_of([index * serial_period + 10**9 for index in range(frame_count)], 600, 1_000, 400)
    statistics = calibration.busy_statistics([first, second_offset, third], [0, 0, 1], [1.0, 1.0, 1.0])
    device = statistics["per_device"][0]
    check(device["method"].startswith("largest") and abs(device["busy_us"] - 2.0) < 1e-9,
          f"no common clock: busy {device['busy_us']} us via {device['method']}")


def uniform_histogram(fluid_columns: int, wall_columns: int, per_column: int) -> np.ndarray:
    return np.array([0] * wall_columns + [per_column] * fluid_columns + [0] * wall_columns, dtype=np.int64)


def test_calibration_arithmetic() -> None:
    print("[4] omega -> cuts")
    histogram = uniform_histogram(400, 3, 1_000)
    total = int(histogram.sum())
    cuts = chain_cuts_from_counts(histogram, [1.0, 1.0, 1.0], MINIMUM_OWN_COLUMNS_HARD)
    fluid = calibration.slab_fluid_counts(histogram, cuts)
    # device-map 0,0,1, equal per-particle cost: device 0 does sims 0 and 1 one after the other
    cost = 1e-3
    busy_device = {0: cost * (fluid[0] + fluid[1]), 1: cost * fluid[2]}
    omega = calibration.normalized(calibration.omega_from_busy(fluid, [busy_device[d] for d in (0, 0, 1)]))
    check(np.allclose(np.array(omega) / omega[0], [1.0, 1.0, 2.0], rtol=0.01), f"omega {omega}")
    new_cuts = chain_cuts_from_counts(histogram, omega, MINIMUM_OWN_COLUMNS_HARD)
    shares = np.array(calibration.slab_fluid_counts(histogram, new_cuts)) / total
    check(np.all(np.abs(shares - [0.25, 0.25, 0.5]) <= 1_000 / total + 1e-12),
          f"fluid shares {shares} != 1/4, 1/4, 1/2 within one column")

    # K = 2, device 1 is 25 % slower per particle and both pay a fixed cost per step: omega under-corrects,
    # the second round gets closer (the step time is the slower device's)
    histogram = uniform_histogram(200, 3, 1_000)
    fixed, per_particle = 30.0, (1e-3, 1.25e-3)

    def step_imbalance(weights):
        cuts = chain_cuts_from_counts(histogram, weights, MINIMUM_OWN_COLUMNS_HARD)
        fluid = calibration.slab_fluid_counts(histogram, cuts)
        busy = [fixed + rate * count for rate, count in zip(per_particle, fluid)]
        return max(busy) / min(busy) - 1.0, fluid, busy, cuts
    imbalance_equal, fluid, busy, cuts_equal = step_imbalance([1.0, 1.0])
    first_round = calibration.normalized(calibration.omega_from_busy(fluid, busy))
    imbalance_first, fluid, busy, cuts_first = step_imbalance(first_round)
    second_round = calibration.normalized(calibration.omega_from_busy(fluid, busy))
    imbalance_second, _, _, _ = step_imbalance(second_round)
    ideal_share = (1 / per_particle[0]) / (1 / per_particle[0] + 1 / per_particle[1])
    check(cuts_first != cuts_equal, "the first round moves the cut")
    check(imbalance_first < imbalance_equal, f"round 1 reduces the imbalance ({imbalance_equal:.3f} -> "
                                             f"{imbalance_first:.3f})")
    check(fluid[0] / sum(fluid) < ideal_share + 1e-9,
          f"round 1 under-corrects with a fixed cost (share {fluid[0] / sum(fluid):.4f} <= {ideal_share:.4f})")
    check(imbalance_second <= imbalance_first, f"round 2 does not move away ({imbalance_first:.4f} -> "
                                               f"{imbalance_second:.4f})")

    # symmetric slabs and devices: omega equal, cuts unchanged
    histogram = uniform_histogram(206, 0, 1_000)
    cuts = chain_cuts_from_counts(histogram, [1.0, 1.0], MINIMUM_OWN_COLUMNS_HARD)
    fluid = calibration.slab_fluid_counts(histogram, cuts)
    omega = calibration.normalized(calibration.omega_from_busy(fluid, [1e-3 * value for value in fluid]))
    check(chain_cuts_from_counts(histogram, omega, MINIMUM_OWN_COLUMNS_HARD) == cuts, "equal slabs keep their cuts")
    check(abs(sum(calibration.normalized([2.0, 6.0])) - 2.0) < 1e-12, "normalized has mean 1")


def fake_case(histogram: np.ndarray, smoothing_length: float = 0.01):
    """The fields _bin_fluid_counts reads: one fluid particle per histogram count at its column centre."""
    columns = np.repeat(np.arange(len(histogram)), histogram)
    positions = np.zeros((columns.size, 3), dtype=np.float32)
    positions[:, 0] = (columns + 0.5) * smoothing_length
    return types.SimpleNamespace(
        grid=types.SimpleNamespace(grid_dimension_x=len(histogram), origin_x=0.0),
        physics=types.SimpleNamespace(smoothing_length=smoothing_length),
        initial=types.SimpleNamespace(positions=positions, material_group=np.zeros(columns.size, dtype=np.uint32)),
        materials=[types.SimpleNamespace(kind=KIND_FLUID)])


def test_weights_file() -> None:
    print("[5] weights file")
    histogram = uniform_histogram(60, 2, 50)
    case = fake_case(histogram)
    weights = calibration.normalized([1.0, 1.0, 1.7])
    cuts = chain_cuts_from_counts(histogram, weights, MINIMUM_OWN_COLUMNS_HARD)
    with tempfile.TemporaryDirectory() as directory:
        case_path = pathlib.Path(directory) / "case.yaml"
        case_path.write_text("name: synthetic\n", encoding="utf-8")
        record = {"format": calibration.WEIGHTS_FILE_FORMAT, "case": str(case_path),
                  "case_sha256": calibration.file_digest(case_path), "slab_count": 3, "device_map": [0, 0, 1],
                  "fluid_histogram_sha256": calibration.histogram_digest(histogram), "host": "elsewhere",
                  "weights": [np.float64(value) for value in weights], "cuts": cuts, "rounds": []}
        path = pathlib.Path(directory) / "weights" / "w.json"
        digest = calibration.write_weights_file(path, record)
        check(digest == calibration.file_digest(path), "write returns the file's sha256")
        messages = []
        loaded = calibration.read_weights_file(path, case, case_path, [0, 0, 1], log=messages.append)
        check(loaded["weights"] == weights and loaded["cuts"] == cuts, "round trip keeps weights and cuts")
        check(any("node" in message for message in messages), "another node is reported")
        check(calibration.read_weights_file(path, case, case_path, None, log=messages.append)["device_map"]
              == [0, 0, 1], "no device map given: the file's")

        def refused(case_object, device_map, label) -> None:
            try:
                calibration.read_weights_file(path, case_object, case_path, device_map, log=messages.append)
            except ValueError:
                check(True, label)
                return
            check(False, f"not refused: {label}")
        refused(case, [0, 1, 0], "another device map")
        refused(fake_case(uniform_histogram(60, 2, 51)), [0, 0, 1], "another particle set")
        record["cuts"] = [cut + 1 for cut in cuts]
        calibration.write_weights_file(path, record)
        refused(case, [0, 0, 1], "stored cuts the weights do not give")


def main() -> int:
    test_phase_intervals()
    test_union()
    test_busy_statistics()
    test_calibration_arithmetic()
    test_weights_file()
    print(f"\n{_passed} checks passed, {len(_failed)} failed")
    if _failed:
        for message in _failed[:20]:
            print(f"  FAILED: {message}")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
