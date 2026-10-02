"""
window_analysis.py — column-0 "crossing within h this step" statistics pooled
over the crossing-capture window (see crossing_window.py).

For every frame f of the window and every test-run (B1) particle i in a seam
column 0 (unsigned column distance 0 to the nearest B cut):
  flagged      another particle that crossed a seam during frame f lies within
               smoothing_length of i (B1 positions after frame f)
  departed     ... and that crossing particle now belongs to the OTHER slab, i.e.
               it left i's slab this frame (the V5 defect (2): i's GPU dropped it
               from both its own and its ghost lists for the rest of the step)
  arrived      flagged but every crossing particle within h arrived INTO i's slab
  control      column-0 captured particles at distance (h, capture radius] from
               the nearest crossing: same place and time, but no crossing
               neighbour (the "unflagged" control of the window)
Each such particle is matched to the reference runs A1 and A2 captured at the
SAME frame (KD-tree on positions, tolerance 0.05 h, or by global id), and
  test  = |B1_i - A1_a(i)|      noise = |A1_a(i) - A2_b(a(i))|
on the same particles, for acceleration (vector norm), shift (vector norm),
density, pressure, kernel_sum. Ratios test/noise of mean, rms, p99, max.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

FIELDS = ("acceleration", "shift", "density", "pressure", "kernel_sum")
GROUPS = ("flagged", "departed", "arrived", "control")


def load_window(path) -> dict:
    path = pathlib.Path(path)
    with np.load(path) as archive:
        return {key: archive[key] for key in archive.files}


def unsigned_column_distance(x: np.ndarray, origin_x: float, smoothing_length: float,
                             cuts: np.ndarray, column_count: int) -> np.ndarray:
    column = np.floor((x - origin_x) / smoothing_length).astype(np.int64)
    np.clip(column, 0, column_count - 1, out=column)
    right = column[:, None] - cuts[None, :]
    distance = np.where(right >= 0, right, -right - 1)
    return distance.min(axis=1)


def _difference(first: dict, first_rows: np.ndarray, second: dict, second_rows: np.ndarray,
                field: str) -> np.ndarray:
    a = first[field][first_rows].astype(np.float64)
    b = second[field][second_rows].astype(np.float64)
    difference = a - b
    if difference.ndim == 2:
        return np.linalg.norm(difference, axis=1)
    return np.abs(difference)


def _describe(values: np.ndarray) -> dict:
    if values.size == 0:
        return {"count": 0}
    return {"count": int(values.size), "mean": float(values.mean()),
            "rms": float(np.sqrt((values ** 2).mean())),
            "p50": float(np.percentile(values, 50)), "p99": float(np.percentile(values, 99)),
            "max": float(values.max())}


def _ratio(test: dict, noise: dict, key: str):
    if test.get("count", 0) == 0 or noise.get("count", 0) == 0:
        return None
    if noise[key] == 0.0:
        return float("inf") if test[key] > 0.0 else 1.0
    return test[key] / noise[key]


def _match_rows(source: dict, source_rows: np.ndarray, target: dict, target_frame_rows: np.ndarray,
                tolerance: float, method: str) -> tuple[np.ndarray, np.ndarray]:
    """For each source row: (found mask over source_rows, matching target row
    indices for the found ones) among target_frame_rows."""
    found = np.zeros(source_rows.size, dtype=bool)
    if source_rows.size == 0 or target_frame_rows.size == 0:
        return found, np.zeros(0, dtype=np.int64)
    if method == "id":
        target_ids = target["id"][target_frame_rows]
        order = np.argsort(target_ids)
        sorted_ids = target_ids[order]
        positions = np.clip(np.searchsorted(sorted_ids, source["id"][source_rows]), 0, sorted_ids.size - 1)
        found = sorted_ids[positions] == source["id"][source_rows]
        return found, target_frame_rows[order[positions[found]]]
    from scipy.spatial import cKDTree
    tree = cKDTree(target["position"][target_frame_rows])
    distance, index = tree.query(source["position"][source_rows], k=1)
    found = distance <= tolerance
    return found, target_frame_rows[index[found]]


def analyze_window(reference_paths, test_paths, *, cuts, origin_x: float,
                   smoothing_length: float, column_count: int, match: str = "kdtree",
                   capture_radius_factor: float = 1.5, fluid_material_groups=None) -> dict:
    """fluid_material_groups: material group ids of FLUID kind; when given, only
    fluid column-0 particles are compared (walls get no force in any run)."""
    from scipy.spatial import cKDTree
    references = [load_window(path) for path in reference_paths]
    tests = [load_window(path) for path in test_paths]
    if len(references) < 2 or not tests:
        raise ValueError("need two reference windows and at least one test window")
    cuts = np.asarray(sorted(cuts), dtype=np.int64)
    tolerance = 0.05 * smoothing_length
    report = {"match": match, "tolerance": tolerance, "pairs": []}
    pairs = [(tests[0], references[0], references[1], "B1 vs A1 (noise A1 vs A2)")]
    if len(tests) > 1:
        pairs.append((tests[1], references[1], references[0], "B2 vs A2 (noise A2 vs A1)"))
    for test, first_reference, second_reference, label in pairs:
        per_group = {group: {"test": {field: [] for field in FIELDS},
                             "noise": {field: [] for field in FIELDS},
                             "signed": {field: [] for field in ("kernel_sum", "density")}}
                     for group in GROUPS}
        counts = {"frames": 0, "frames_with_crossings": 0, "crossing_particles": 0,
                  "candidates": {group: 0 for group in GROUPS},
                  "unmatched": {group: 0 for group in GROUPS}}
        if not test or not first_reference or not second_reference:
            report["pairs"].append({"label": label, "counts": counts, "empty": True})
            continue
        frames = np.unique(test["frame"])
        counts["frames"] = int(frames.size)
        for frame in frames:
            rows = np.nonzero(test["frame"] == frame)[0]
            crossing_rows = rows[test["crossed_this_frame"][rows]]
            if crossing_rows.size == 0:
                continue
            counts["frames_with_crossings"] += 1
            counts["crossing_particles"] += int(crossing_rows.size)
            x = test["position"][rows, 0].astype(np.float64)
            distance = unsigned_column_distance(x, origin_x, smoothing_length, cuts, column_count)
            column0_rows = rows[distance == 0]
            if fluid_material_groups is not None and "material" in test:
                column0_rows = column0_rows[np.isin(test["material"][column0_rows],
                                                    np.asarray(list(fluid_material_groups)))]
            crossing_tree = cKDTree(test["position"][crossing_rows])
            group_rows = {group: [] for group in GROUPS}
            for row in column0_rows:
                neighbours = crossing_tree.query_ball_point(test["position"][row], r=smoothing_length)
                neighbours = [crossing_rows[k] for k in neighbours if crossing_rows[k] != row]
                if neighbours:
                    group_rows["flagged"].append(row)
                    if any(test["slab"][k] != test["slab"][row] for k in neighbours):
                        group_rows["departed"].append(row)
                    else:
                        group_rows["arrived"].append(row)
                    continue
                if test["crossed_this_frame"][row]:
                    continue            # a crossing particle with no OTHER crossing within h
                nearest_distance, _ = crossing_tree.query(test["position"][row], k=1)
                if nearest_distance <= capture_radius_factor * smoothing_length:
                    group_rows["control"].append(row)
            first_frame_rows = np.nonzero(first_reference["frame"] == frame)[0]
            second_frame_rows = np.nonzero(second_reference["frame"] == frame)[0]
            for group, selected in group_rows.items():
                if not selected:
                    continue
                source_rows = np.array(selected, dtype=np.int64)
                counts["candidates"][group] += int(source_rows.size)
                found_first, first_rows = _match_rows(test, source_rows, first_reference,
                                                      first_frame_rows, tolerance, match)
                test_rows = source_rows[found_first]
                found_second, second_rows = _match_rows(first_reference, first_rows, second_reference,
                                                        second_frame_rows, tolerance, match)
                test_rows, first_rows = test_rows[found_second], first_rows[found_second]
                counts["unmatched"][group] += int(source_rows.size - test_rows.size)
                if test_rows.size == 0:
                    continue
                for field in ("kernel_sum", "density"):
                    if field in test:
                        per_group[group]["signed"][field].append(
                            test[field][test_rows].astype(np.float64)
                            - first_reference[field][first_rows].astype(np.float64))
                for field in FIELDS:
                    if field not in test:
                        continue
                    per_group[group]["test"][field].append(
                        _difference(test, test_rows, first_reference, first_rows, field))
                    per_group[group]["noise"][field].append(
                        _difference(first_reference, first_rows, second_reference, second_rows, field))
        statistics = {}
        for group in GROUPS:
            statistics[group] = {}
            for field in FIELDS:
                test_values = np.concatenate(per_group[group]["test"][field])                     if per_group[group]["test"][field] else np.zeros(0)
                noise_values = np.concatenate(per_group[group]["noise"][field])                     if per_group[group]["noise"][field] else np.zeros(0)
                test_stats, noise_stats = _describe(test_values), _describe(noise_values)
                statistics[group][field] = {
                    "test": test_stats, "noise": noise_stats,
                    "ratio": {key: _ratio(test_stats, noise_stats, key) for key in ("mean", "rms", "p99", "max")}}
                if field in per_group[group]["signed"] and per_group[group]["signed"][field]:
                    signed = np.concatenate(per_group[group]["signed"][field])
                    statistics[group][field]["signed_test_minus_reference"] = {
                        "mean": float(signed.mean()), "median": float(np.median(signed)),
                        "count": int(signed.size)}
        report["pairs"].append({"label": label, "counts": counts, "statistics": statistics})
    return report


def render_markdown(report: dict, title: str) -> str:
    lines = [f"### {title}", ""]
    for pair in report["pairs"]:
        counts = pair["counts"]
        lines.append(f"*{pair['label']}* — window frames {counts['frames']}, with crossings "
                     f"{counts['frames_with_crossings']}, crossing particles {counts['crossing_particles']}, "
                     f"column-0 candidates {counts['candidates']}, unmatched {counts['unmatched']}")
        lines.append("")
        if pair.get("empty"):
            continue
        lines.append("| group | n | accel rms ratio | accel max ratio | shift rms ratio | density rms ratio | "
                     "kernel_sum rms ratio |")
        lines.append("|---|---|---|---|---|---|---|")
        for group in GROUPS:
            statistics = pair["statistics"][group]
            count = statistics["acceleration"]["test"].get("count", 0)

            def ratio(field, key):
                value = statistics[field]["ratio"][key]
                return "—" if value is None else f"{value:.2f}"
            lines.append(f"| {group} | {count} | {ratio('acceleration', 'rms')} | {ratio('acceleration', 'max')} | "
                         f"{ratio('shift', 'rms')} | {ratio('density', 'rms')} | {ratio('kernel_sum', 'rms')} |")
        lines.append("")
    return "\n".join(lines)


def write_report(report: dict, out_dir, title: str) -> None:
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "window_report.json").write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    (out_dir / "window_report.md").write_text(render_markdown(report, title), encoding="utf-8")


# --------------------------------------------------------------------------- self-test
def _synthetic_window(random_generator, base_positions, frames, crossing_by_frame, cut_x,
                      noise_scale, perturb_rows_by_frame=None, perturb=0.0):
    records = []
    for frame in frames:
        count = base_positions.shape[0]
        positions = base_positions + random_generator.normal(0.0, 1e-7, base_positions.shape)
        acceleration = np.ones((count, 2)) + random_generator.normal(0.0, noise_scale, (count, 2))
        if perturb_rows_by_frame and frame in perturb_rows_by_frame:
            acceleration[perturb_rows_by_frame[frame]] += perturb
        crossed = np.zeros(count, dtype=bool)
        crossed[crossing_by_frame[frame]] = True
        slab = (positions[:, 0] >= cut_x).astype(np.uint8)
        records.append({"id": np.arange(count, dtype=np.uint32), "slab": slab,
                        "position": positions.astype(np.float32),
                        "acceleration": acceleration.astype(np.float32),
                        "shift": np.zeros((count, 2), np.float32),
                        "density": np.full(count, 1000.0, np.float32),
                        "pressure": np.zeros(count, np.float32),
                        "kernel_sum": np.ones(count, np.float32),
                        "crossed_this_frame": crossed,
                        "frame": np.full(count, frame, np.int32)})
    return {key: np.concatenate([record[key] for record in records]) for key in records[0]}


def run_self_test() -> int:
    import tempfile
    random_generator = np.random.default_rng(7)
    smoothing_length, origin_x, cut = 0.01, 0.0, 10
    cut_x = origin_x + cut * smoothing_length
    spacing = smoothing_length / 4
    xs = np.arange(cut_x - 2 * smoothing_length + spacing / 2, cut_x + 2 * smoothing_length, spacing)
    ys = np.arange(0.0, 0.2, spacing)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")
    base_positions = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
    frames = list(range(5))
    # one crossing particle per frame just right of the cut (it left the left slab)
    crossing_by_frame, perturb_rows = {}, {}
    for frame in frames:
        row = int(np.argmin(np.abs(base_positions[:, 0] - (cut_x + spacing / 2))
                            + np.abs(base_positions[:, 1] - (0.03 + 0.03 * frame))))
        crossing_by_frame[frame] = [row]
        distances = np.linalg.norm(base_positions - base_positions[row], axis=1)
        left_column0 = (base_positions[:, 0] < cut_x) & (base_positions[:, 0] >= cut_x - smoothing_length)
        perturb_rows[frame] = np.nonzero(left_column0 & (distances < smoothing_length))[0]
    reference_one = _synthetic_window(random_generator, base_positions, frames, crossing_by_frame, cut_x, 1e-6)
    reference_two = _synthetic_window(random_generator, base_positions, frames, crossing_by_frame, cut_x, 1e-6)
    test = _synthetic_window(random_generator, base_positions, frames, crossing_by_frame, cut_x, 1e-6,
                             perturb_rows, perturb=1e-3)
    with tempfile.TemporaryDirectory() as directory:
        paths = []
        for name, arrays in (("a1", reference_one), ("a2", reference_two), ("b1", test)):
            path = pathlib.Path(directory) / f"{name}.npz"
            np.savez(path, **arrays)
            paths.append(path)
        report = analyze_window(paths[:2], paths[2:], cuts=[cut], origin_x=origin_x,
                                smoothing_length=smoothing_length, column_count=40)
    statistics = report["pairs"][0]["statistics"]
    departed_ratio = statistics["departed"]["acceleration"]["ratio"]["rms"]
    control_ratio = statistics["control"]["acceleration"]["ratio"]["rms"]
    arrived_count = statistics["arrived"]["acceleration"]["test"].get("count", 0)
    print(f"[window self-test] departed rms ratio {departed_ratio:.1f}, control rms ratio "
          f"{control_ratio:.2f}, arrived n={arrived_count}, counts {report['pairs'][0]['counts']}")
    ok = departed_ratio is not None and departed_ratio > 50 and control_ratio is not None and control_ratio < 3
    print("[window self-test] " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(run_self_test())
