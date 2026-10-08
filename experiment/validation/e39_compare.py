"""e39_compare.py - E39: a v7 cavity run against a v6 run of the same case (Re = 1000; CPU only, no GPU).

Both directories are cavity_runner.py runs (meta.json, samples.jsonl + samples/, snapshots/, segments.jsonl). Each is
analysed with the v6 report's own code (cavity_analysis.load_run + analyse_run with the defaults of
cavity_analysis.main, docs/validation/cavity_re1000.md): the time average over the run's last --average-span (20)
time units, linear MLS in the wall-row frame, against Marchi 2021 Tc (Tables 19-21):
  psi_min       minimum of the stream function of the time-averaged MLS u grid (513^2, psi integrated upward from the
                bottom wall row; snapshots every time unit inside the averaging window);
  u_min         minimum of u(0.5, y) on the time-averaged 1001-point vertical centre line, y in [0.05, 0.45];
  v_max, v_min  maximum / minimum of v(x, 0.5) on the horizontal centre line, x in [0.05, 0.45] / [0.60, 0.99];
  u, v L2       relative L2 error at the 28 + 28 Table 20 / 21 points, rms(mean - Tc) / rms(Tc).
Every difference (v7 - v6) is divided by its noise scale, the two runs' scales in quadrature. A run's scale is the
time-average standard error of the v6 analysis, std sqrt(tau_int / n) with the integrated autocorrelation time
(cavity_analysis.window_statistics), of
  the extrema   the dense-line point nearest the extremum (the 'sem' of cavity_analysis' extrema);
  the L2 errors the in-window samples projected on the gradient of the L2 error at the mean, (mean - Tc) / (28 L2)
                (first order; the covariance between the 28 points counts);
  psi_min       the per-snapshot minima psi_min(t_k) of the window's snapshots. E36 section 8.2's scale, half the
                difference of the minima of the two half windows (used for E37's comparison in
                docs/seam_audit/v6_opt.md), is listed next to it.
Context rows: the 56-point profile distance |P7 - P6| / |P6| against long_run_convergence's window noise floor
sqrt(sum sem6^2 + sem7^2) / |P6| (two independent runs of the same flow give a ratio near 1), the largest pointwise
|difference| / combined sem (cavity_tables' K = 1 vs K = 2 measure) and the mean fluid kinetic energy. Positions
(y(u_min), x(v_max), x(v_min), the vortex centre) are listed with their differences in units of dx, without a noise
scale.

Window: --window-end own (default) = each run's last sample (the v6 analysis); common = the earlier of the two last
samples for both; a number T = the span ending at T for both (E36 / E37 compared (80, 100]: --window-end 100).
Checks: the two runs must be the same case (meta spacing, support radius, dt, particle count, nominal Re; --force
overrides); different numerics (xi, epsilon_squared_factor, density storage), wall model, K, solver, windows, an
incomplete run and non-zero invariants (drift, overflow, far migration, frame stamps; summed over segments) are
listed as notes. The wall model is meta.json's wall_boundary (cavity_runner since E37), else E36's wall_bc (the
v7-wall-bc runner: 3 = adami_rho0, bit-identical to the release's adami), else the run's case.yaml copy.

Outputs: <out>.json (everything, with the per-snapshot psi_min series) and <out>.md (the tables, in the report's
language); the markdown also goes to stdout. psi_min costs about 5 s per snapshot and pass and takes three passes (the
window, every snapshot, the two half windows): about 5 min per run at ~20 snapshots; --no-stream-function skips it.

    # E39 accuracy regression (v7 run first, then the existing v6 run); <main> = the main checkout, where the v6 runs
    # live (C:/Users/cchen/PycharmProjects/vulkan-demo); 500^2 K = 2 likewise with n500_k2_float32_xi0p001_eps0p0025
    .venv/Scripts/python.exe -m experiment.validation.e39_compare \
        --v7 logs/validation/cavity_re1000_v7/n250_k2_float32_xi0p001_eps0p0025 \
        --v6 <main>/logs/validation/cavity_re1000/n250_k2_float32_xi0p001_eps0p0025 --out logs/e39/cavity/n250_k2_simple
    # adami, K = 1: 250^2 against E37's v6-rc2 release run, 500^2 against E36's adami_rho0 (v7-wall-bc WALL_BC = 3)
    .venv/Scripts/python.exe -m experiment.validation.e39_compare \
        --v7 logs/validation/cavity_re1000_v7/n250_k1_float32_xi0p001_eps0p0025_adami \
        --v6 <main>/logs/e37/A/runs/n250_k1_release_adami --out logs/e39/cavity/n250_k1_adami
    .venv/Scripts/python.exe -m experiment.validation.e39_compare \
        --v7 logs/validation/cavity_re1000_v7/n500_k1_float32_xi0p001_eps0p0025_adami \
        --v6 <main>/logs/e36/runs/n500_k1_v7_diag3_rho0 --out logs/e39/cavity/n500_k1_adami
    # plumbing checks on v6 data: a run against itself (every difference 0) and against another numerics variant
    .venv/Scripts/python.exe -m experiment.validation.e39_compare \
        --v7 <main>/logs/validation/cavity_re1000/n250_k2_float32_xi0p001_eps0p0025 \
        --v6 <main>/logs/validation/cavity_re1000/n250_k2_float32_xi0p001 --out logs/e39/cavity/plumbing_variant
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

import numpy as np
import yaml

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiment.validation import cavity_reference  # noqa: E402
from experiment.validation.cavity_analysis import (analyse_run, load_run, stream_function_minimum,  # noqa: E402
                                                   window_statistics)
from experiment.validation.cavity_tables import REFERENCE_EXTREMA, invariants, md_table, variant_label  # noqa: E402

SIDES = ("v6", "v7")
# the case's identity in meta.json: a comparison across different values is refused without --force
CASE_KEYS = ("spacing", "support_radius", "dt", "expected_total", "reynolds_nominal")
INVARIANT_KEYS = ("drift", "overflow", "far_migration", "stamp_gpu", "stamp_host")
# (key, markdown label, kind): 'extremum' = a raw value with a Marchi 2021 Tc, 'percent' = a relative L2 error
QUANTITIES = (("psi_min", "ψ_min", "extremum"), ("u_min", "u_min", "extremum"), ("v_max", "v_max", "extremum"),
              ("v_min", "v_min", "extremum"), ("u_L2_relative", "u 相对 L2(28 点)", "percent"),
              ("v_L2_relative", "v 相对 L2(28 点)", "percent"))
POSITION_LABELS = {"u_min": "y(u_min)", "v_max": "x(v_max)", "v_min": "x(v_min)"}
# wall_bc of E36's runner (branch v7-wall-bc): 0 = the v6 wall, 3 = adami_rho0 (= the release's adami since E37)
E36_WALL_MODELS = {0: "simple", 3: "adami"}


def log(message: str) -> None:
    print(f"[e39_compare {time.strftime('%H:%M:%S')}] {message}", flush=True)


# ----------------------------------------------------------------------------------------------- one run
def wall_model(run: dict) -> str:
    """simple | adami (| E36's diagnostic wall_bc modes), see the module doc."""
    meta = run["meta"]
    if "wall_boundary" in meta:
        return str(meta["wall_boundary"])
    if "wall_bc" in meta:
        return E36_WALL_MODELS.get(int(meta["wall_bc"]), f"wall_bc {meta['wall_bc']} (E36 diagnostic)")
    copy = run["dir"] / "case.yaml"
    if copy.exists():
        return str((yaml.safe_load(copy.read_text(encoding="utf-8")).get("numerics") or {}).get("wall_boundary",
                                                                                               "simple"))
    return "simple"


def solver_of(run: dict) -> str:
    """meta.json's solver (v6 before E39); E36's runner (branch v7-wall-bc, its own unrelated experiment/v7, recorded
    as solver v7 with a wall_bc) is v7-wall-bc."""
    meta = run["meta"]
    return "v7-wall-bc" if "wall_bc" in meta else str(meta.get("solver", "v6"))


def build_label(run: dict) -> str:
    """Solver and commit of a run (E36's runs with their WALL_BC mode)."""
    meta = run["meta"]
    solver = solver_of(run) + (f" WALL_BC={meta['wall_bc']}" if "wall_bc" in meta else "")
    git = meta.get("git") or {}
    return f"{solver} {git.get('head', '?')}{' (dirty)' if git.get('dirty') else ''}"


def truncated(run: dict, t_stop: float) -> dict:
    """The run's samples up to t_stop (--window-end common / <number>): analyse_run then averages the span ending at
    the last of them, and stream_function_minimum takes the snapshots inside that window."""
    keep = run["times"] <= t_stop + 1e-9
    copy = dict(run)
    copy["rows"] = [row for row, kept in zip(run["rows"], keep) if kept]
    copy["times"] = run["times"][keep]
    copy["kinetic_energy"] = run["kinetic_energy"][keep]
    copy["arrays"] = {name: values[keep] for name, values in run["arrays"].items()}
    return copy


def window_mask(run: dict, average_span: float) -> np.ndarray:
    """The samples analyse_run averages (its expression)."""
    return run["times"] > float(run["times"].max()) - average_span - 1e-9


def snapshot_times(run: dict, window: tuple[float, float]) -> list[float]:
    """Times of the snapshots stream_function_minimum averages over ``window`` (its selection)."""
    dt = float(run["meta"]["dt"])
    times = sorted(int(path.stem[1:]) * dt for path in (run["dir"] / "snapshots").glob("t*.npz"))
    return [value for value in times if window[0] - 1e-9 <= value <= window[1] + 1e-9]


def stream_noise(run: dict, window: tuple[float, float], label: str) -> dict:
    """psi_min of every snapshot in the window (with the series' time-average standard error) and of the two half
    windows (E36 section 8.2: half the difference of their minima), all by stream_function_minimum."""
    times = snapshot_times(run, window)
    log(f"{label}: psi_min of each of {len(times)} snapshots")
    series = []
    for value in times:
        stream = stream_function_minimum(run, (value, value))
        series.append({"time": value, "psi_min_wall": stream["psi_min_wall"], "x_wall": stream["x_wall"],
                       "y_wall": stream["y_wall"]})
    values = np.array([entry["psi_min_wall"] for entry in series])
    result = {"series": series, "n": int(values.size), "sem": None, "std": None, "tau": None, "half_windows": None}
    if values.size > 1:
        statistics = window_statistics(values[:, None])
        result.update(sem=float(statistics["sem"][0]), std=float(statistics["std"][0]),
                      tau=float(statistics["tau"][0]))
    middle = 0.5 * (window[0] + window[1])
    parts = ([value for value in times if value <= middle], [value for value in times if value > middle])
    if all(parts):
        log(f"{label}: psi_min of the two half windows ({len(parts[0])} + {len(parts[1])} snapshots)")
        halves = [{"window": [part[0], part[-1]], "snapshots": len(part),
                   "psi_min_wall": stream_function_minimum(run, (part[0], part[-1]))["psi_min_wall"]} for part in parts]
        result["half_windows"] = {"first": halves[0], "second": halves[1],
                                  "scale": 0.5 * abs(halves[0]["psi_min_wall"] - halves[1]["psi_min_wall"])}
    return result


def run_quantities(run: dict, analysis: dict, mask: np.ndarray, reference: dict, label: str,
                   with_stream: bool) -> dict:
    """{quantity: {'value', 'noise', ...}} of one run (see the module doc)."""
    quantities = {}
    stream = analysis.get("stream")
    if with_stream and stream:
        noise = stream_noise(run, analysis["window"], label)
        quantities["psi_min"] = {"value": stream["psi_min_wall"], "noise": noise["sem"],
                                 "noise_half_windows": (noise["half_windows"] or {}).get("scale"),
                                 "snapshots": stream["snapshots"], "per_snapshot": noise}
    for name in ("u_min", "v_max", "v_min"):
        extremum = analysis["extrema"][name]
        quantities[name] = {"value": extremum["value"], "noise": extremum["sem"]}
    for line in ("u", "v"):
        error = analysis["errors"][("wall", "mls", line)]
        gradient = error["pointwise"] / (error["pointwise"].size * error["L2"])
        series = run["arrays"][f"{line}_marchi_wall_mls"][mask] @ gradient
        reference_rms = float(np.sqrt(np.mean(reference[line][1] ** 2)))
        noise = float(window_statistics(series[:, None])["sem"][0]) / reference_rms
        quantities[f"{line}_L2_relative"] = {"value": error["L2_relative"], "L2": error["L2"], "Linf": error["Linf"],
                                             "noise": noise}
    energy = analysis["kinetic_energy_window"]
    quantities["kinetic_energy"] = {"value": float(energy["mean"][0]), "noise": float(energy["sem"][0])}
    return quantities


def describe(run: dict, analysis: dict, label: str) -> dict:
    meta = run["meta"]
    dt = float(meta["dt"])
    result = run["result"]
    totals = invariants({"segments": run["segments"]})
    stream = analysis.get("stream") or {}
    return {"label": label, "directory": str(run["dir"]), "build": build_label(run), "solver": solver_of(run),
            "git": meta.get("git"), "code_hashes": meta.get("code_hashes"), "case": meta["case"],
            "slabs": run["slabs"], "device_map": meta.get("device_map"), "wall_boundary": wall_model(run),
            "variant": run["variant"], "resolution": run["resolution"], "spacing": run["spacing"], "dt": dt,
            "expected_total": meta.get("expected_total"), "complete": result.get("status") == "complete",
            "stop_time": result["stop_step"] * dt if "stop_step" in result else None,
            "t_last": analysis["t_last"], "window": list(analysis["window"]),
            "samples_in_window": analysis["samples_in_window"], "snapshots_in_window": stream.get("snapshots"),
            "sample_interval": float(np.median(np.diff(run["times"]))),
            "t_steady_online": analysis["t_steady_online"], "t_steady_sliding": analysis["t_steady_sliding"],
            "invariants": {key: totals[key] for key in INVARIANT_KEYS}, "segments": totals["segments"],
            "environment": meta.get("environment")}


# ----------------------------------------------------------------------------------------------- the pair
def case_differences(runs: dict) -> list[str]:
    """The CASE_KEYS on which the two runs differ (a different case: refused without --force)."""
    differences = []
    for key in CASE_KEYS:
        values = [runs[side]["meta"].get(key) for side in SIDES]
        if None in values or not math.isclose(float(values[0]), float(values[1]), rel_tol=1e-12):
            differences.append(f"{key}:v6 {values[0]},v7 {values[1]}")
    return differences


def comparability_notes(runs: dict, descriptions: dict) -> list[str]:
    """Everything else that differs between the runs or is not clean."""
    six, seven = descriptions["v6"], descriptions["v7"]
    notes = []
    if seven["solver"] != "v7":
        notes.append(f"--v7 的运行不是 E39 的 v7(meta.json:{seven['build']})")
    if six["solver"] == "v7-wall-bc" and runs["v6"]["meta"].get("wall_bc") == 3:
        notes.append(f"--v6 的运行是 E36 的 adami_rho0({six['build']}):发布版(E37 起)的 adami 与它逐位相同"
                     "(docs/seam_audit/v6_opt.md,壁面选项一节)")
    elif six["solver"] != "v6":
        notes.append(f"--v6 的运行不是 v6(meta.json:{six['build']})")
    for key, name in (("variant", "数值设置"), ("wall_boundary", "壁面"), ("slabs", "K")):
        if six[key] != seven[key]:
            shown = {side: variant_label(descriptions[side][key]) if key == "variant" else descriptions[side][key]
                     for side in SIDES}
            notes.append(f"{name}不同:v6 {shown['v6']},v7 {shown['v7']}")
    if six["case"] != seven["case"]:
        notes.append(f"算例文件不同:v6 {six['case']},v7 {seven['case']}")
    tolerance = 0.5 * min(six["sample_interval"], seven["sample_interval"])
    if any(abs(six["window"][index] - seven["window"][index]) > tolerance for index in range(2)):
        notes.append(f"平均窗口不同:v6 {six['window'][0]:.2f}–{six['window'][1]:.2f},"
                     f"v7 {seven['window'][0]:.2f}–{seven['window'][1]:.2f}(--window-end 可取同一窗口)")
    for side in SIDES:
        description = descriptions[side]
        if not description["complete"]:
            notes.append(f"{side} 的运行没有完成(没有 result.json 或状态不是 complete)")
        if any(description["invariants"].values()):
            notes.append(f"{side} 的不变量不为 0:{description['invariants']}")
        if description["snapshots_in_window"] is not None and description["snapshots_in_window"] < 4:
            notes.append(f"{side} 的平均窗口里只有 {description['snapshots_in_window']} 个快照,ψ_min 的噪声尺度不可靠")
    return notes


def ratio(difference: float, noise: float | None) -> float | None:
    if noise is None or not math.isfinite(noise):
        return None
    if noise == 0.0:
        return 0.0 if difference == 0.0 else math.inf
    return abs(difference) / noise


def quantity_rows(quantities: dict, reference: dict) -> list[dict]:
    extrema_reference = reference["extrema"]
    rows = []
    for key, label, kind in QUANTITIES:
        if any(key not in quantities[side] for side in SIDES):
            continue
        values = {side: float(quantities[side][key]["value"]) for side in SIDES}
        noises = {side: quantities[side][key]["noise"] for side in SIDES}
        combined = (math.hypot(noises["v6"], noises["v7"])
                    if all(value is not None for value in noises.values()) else None)
        difference = values["v7"] - values["v6"]
        row = {"quantity": key, "label": label, "kind": kind, "v6": values["v6"], "v7": values["v7"],
               "difference": difference, "noise_v6": noises["v6"], "noise_v7": noises["v7"], "noise": combined,
               "ratio": ratio(difference, combined)}
        if kind == "extremum":
            target = extrema_reference[key][0]
            deviations = {side: 100.0 * (abs(values[side]) - abs(target)) / abs(target) for side in SIDES}
            row.update(reference=target, deviation_percent_v6=deviations["v6"], deviation_percent_v7=deviations["v7"],
                       deviation_difference_pp=deviations["v7"] - deviations["v6"],
                       noise_pp=None if combined is None else 100.0 * combined / abs(target))
        if key == "psi_min":
            halves = {side: quantities[side][key]["noise_half_windows"] for side in SIDES}
            half_combined = (math.hypot(halves["v6"], halves["v7"])
                             if all(value is not None for value in halves.values()) else None)
            row.update(noise_half_windows_v6=halves["v6"], noise_half_windows_v7=halves["v7"],
                       noise_half_windows=half_combined, ratio_half_windows=ratio(difference, half_combined))
        rows.append(row)
    return rows


def profile_comparison(analyses: dict, reference: dict) -> dict:
    """56-point mean profiles (u then v, wall frame, MLS): distance against the window noise floor, pointwise ratios."""
    means, sems = {}, {}
    for side in SIDES:
        statistics = analyses[side]["statistics"]
        means[side] = np.concatenate([statistics[f"{line}_marchi_wall_mls"]["mean"] for line in ("u", "v")])
        sems[side] = np.concatenate([statistics[f"{line}_marchi_wall_mls"]["sem"] for line in ("u", "v")])
    difference = means["v7"] - means["v6"]
    norm = float(np.linalg.norm(means["v6"]))
    combined = np.sqrt(sems["v6"] ** 2 + sems["v7"] ** 2)
    distance = float(np.linalg.norm(difference)) / norm
    floor = float(np.sqrt(np.sum(combined ** 2))) / norm
    pointwise = np.abs(difference) / combined
    index = int(np.argmax(pointwise))
    count = reference["u"][0].size
    line = "u" if index < count else "v"
    return {"distance": distance, "noise_floor": floor, "ratio": ratio(distance, floor),
            "max_pointwise_ratio": float(pointwise[index]), "max_pointwise_line": line,
            "max_pointwise_coordinate": float(reference[line][0][index % count]),
            "max_abs_difference": float(np.abs(difference).max()), "median_combined_sem": float(np.median(combined))}


def position_rows(analyses: dict, reference: dict, spacing: float) -> list[dict]:
    extrema_reference = reference["extrema"]
    rows = []
    for name, (_, position_key) in REFERENCE_EXTREMA.items():
        values = {side: analyses[side]["extrema"][name]["position_wall"] for side in SIDES}
        rows.append({"position": POSITION_LABELS[name], "reference": extrema_reference[position_key][0], **values})
    if all(analyses[side].get("stream") for side in SIDES):
        for label, field, position_key in (("x(ψ_min)", "x_wall", "x_at_psi_min"),
                                           ("y(ψ_min)", "y_wall", "y_at_psi_min")):
            values = {side: analyses[side]["stream"][field] for side in SIDES}
            rows.append({"position": label, "reference": extrema_reference[position_key][0], **values})
    for row in rows:
        row["difference"] = row["v7"] - row["v6"]
        row["difference_over_spacing"] = row["difference"] / spacing
    return rows


# ----------------------------------------------------------------------------------------------- output
def plain(value):
    """A JSON-ready copy: numpy scalars / arrays as Python numbers / lists, non-finite floats as None."""
    if isinstance(value, dict):
        return {(" ".join(map(str, key)) if isinstance(key, tuple) else str(key)): plain(item)
                for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, pathlib.Path):
        return str(value)
    return value


def number(value: float | None, pattern: str) -> str:
    return "–" if value is None or not math.isfinite(value) else format(value, pattern)


def markdown(result: dict) -> str:
    descriptions = result["runs"]
    six, seven = descriptions["v6"], descriptions["v7"]
    lines = [f"# E39 方腔对照:{seven['label']}(v7)对 {six['label']}(v6)", "",
             f"窗口:{result['window_end_mode']}。差值一律为 v7 − v6,噪声尺度为两次运行各自的尺度按平方和合成。", ""]

    def both(function) -> list[str]:
        return [function(descriptions[side]) for side in SIDES]

    def steady(description: dict) -> str:
        return (f"{number(description['t_steady_online'], '.2f')} / {number(description['t_steady_sliding'], '.2f')}")

    def window(description: dict) -> str:
        snapshots = description["snapshots_in_window"]
        return (f"{description['window'][0]:.2f}–{description['window'][1]:.2f}(样本 {description['samples_in_window']}"
                f"{'' if snapshots is None else f',快照 {snapshots}'})")

    def invariant_text(description: dict) -> str:
        return "0" if not any(description["invariants"].values()) else json.dumps(description["invariants"])

    rows = [["运行目录"] + both(lambda description: f"`{description['directory']}`"),
            ["构建"] + both(lambda description: description["build"]),
            ["算例"] + both(lambda description: f"`{description['case']}`"),
            ["K / 壁面 / 设置"] + both(lambda description: f"{description['slabs']} / {description['wall_boundary']} / "
                                                         f"{variant_label(description['variant'])}"),
            ["停步 t / 最后样本 t"] + both(lambda description: f"{number(description['stop_time'], '.2f')} / "
                                                           f"{description['t_last']:.2f}"),
            ["稳态 t_s(窗口判据 / 滑动重算)"] + both(steady),
            ["平均窗口"] + both(window),
            ["不变量(drift、overflow、far migration、帧戳)"] + both(invariant_text),
            ["完成"] + both(lambda description: "是" if description["complete"] else "否")]
    lines += [md_table(["", f"v6:{six['label']}", f"v7:{seven['label']}"], rows), ""]

    def noises(row: dict, scale: float, pattern: str) -> str:
        return " / ".join(number(None if row[key] is None else scale * row[key], pattern)
                          for key in ("noise_v6", "noise_v7", "noise"))

    table = []
    for row in result["quantities"]:
        if row["kind"] == "extremum":
            cells = [f"{row['v6']:.6f}({row['deviation_percent_v6']:+.2f} %)",
                     f"{row['v7']:.6f}({row['deviation_percent_v7']:+.2f} %)",
                     f"{row['difference']:+.1e}({row['deviation_difference_pp']:+.3f} pp)", noises(row, 1.0, ".1e")]
        else:
            cells = [f"{100 * row['v6']:.3f} %", f"{100 * row['v7']:.3f} %", f"{100 * row['difference']:+.3f} pp",
                     noises(row, 100.0, ".3f") + " pp"]
        table.append([row["label"]] + cells + [number(row["ratio"], ".1f")])
    profile = result["profile"]
    table.append(["56 点剖面距离 ‖P₇ − P₆‖ / ‖P₆‖(附加)", "", "", f"{profile['distance']:.1e}",
                  f"噪声底 {profile['noise_floor']:.1e}", number(profile["ratio"], ".2f")])
    table.append([f"56 点逐点 max \\|Δ\\| / 合成标准误差(附加;{profile['max_pointwise_line']} 线 "
                  f"{profile['max_pointwise_coordinate']:.4f})", "", "", f"{profile['max_abs_difference']:.1e}",
                  f"中位数 {profile['median_combined_sem']:.1e}", f"{profile['max_pointwise_ratio']:.1f}"])
    energy = result["kinetic_energy"]
    table.append(["流体动能(窗口均值,附加)", f"{energy['v6']:.5f}", f"{energy['v7']:.5f}",
                  f"{energy['difference']:+.1e}", noises(energy, 1.0, ".1e"), number(energy["ratio"], ".1f")])
    lines += [md_table(["量", "v6", "v7", "v7 − v6", "噪声尺度:v6 / v7 / 合成", "\\|差\\| / 噪声"], table), ""]

    psi = next((row for row in result["quantities"] if row["quantity"] == "psi_min"), None)
    if psi is not None and psi.get("noise_half_windows") is not None:
        lines += [f"ψ_min 的 E36 §8.2 尺度(两个半窗口平均场的 ψ_min 之差的一半):v6 {psi['noise_half_windows_v6']:.1e},"
                  f"v7 {psi['noise_half_windows_v7']:.1e},合成 {psi['noise_half_windows']:.1e};"
                  f"\\|差\\| / 该尺度 = {number(psi['ratio_half_windows'], '.1f')}。", ""]

    position_table = [[row["position"], f"{row['reference']:.4f}", f"{row['v6']:.4f}", f"{row['v7']:.4f}",
                       f"{row['difference']:+.5f}", f"{row['difference_over_spacing']:+.2f}"]
                      for row in result["positions"]]
    lines += ["位置(壁面行框架;没有噪声尺度,差按 Δx 计):", "",
              md_table(["位置", "Marchi 2021 T_c", "v6", "v7", "v7 − v6", "差 / Δx"], position_table), ""]

    lines += ["噪声尺度(每次运行;两次按平方和合成):",
              "- 极值:平均窗口内稠密中线上离极值最近一点的时间平均标准误差 std·√(τ_int/n)(cavity_analysis 的 sem);",
              "- 相对 L2:窗口内每个样本的 28 点剖面投影到 L2 误差在平均剖面处的梯度 (ū − T_c)/(28·L2) 上,"
              "取这个序列的时间平均标准误差,再除以 rms(T_c)(一阶近似,点间协方差计入);",
              "- ψ_min:窗口内每个快照(每个时间单位一个)单独求 ψ_min,取这个序列的时间平均标准误差;"
              "另列 E36 §8.2 的半窗口尺度;",
              "- 56 点剖面距离的噪声底:√Σ(sem₆² + sem₇²) / ‖P₆‖(long_run_convergence 的窗口噪声底),"
              "同一流动的两次独立运行比值约为 1。", ""]
    if result["notes"]:
        lines += ["注意:"] + [f"- {note}" for note in result["notes"]] + [""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--v7", required=True, help="the v7 run directory (cavity_runner --solver v7)")
    parser.add_argument("--v6", required=True, help="the v6 run directory of the same case")
    parser.add_argument("--v7-label", default=None, help="default: the directory name")
    parser.add_argument("--v6-label", default=None, help="default: the directory name")
    parser.add_argument("--out", default=None, help="output prefix: <out>.json and <out>.md (default: stdout only)")
    parser.add_argument("--average-span", type=float, default=20.0, help="averaging window (cavity_analysis default)")
    parser.add_argument("--window", type=float, default=10.0, help="steady-test window of the sliding recomputation")
    parser.add_argument("--steady-tol", type=float, default=1.0e-3)
    parser.add_argument("--window-end", default="own",
                        help="own (each run's last sample: the v6 analysis), common (the earlier one) or a time")
    parser.add_argument("--no-stream-function", action="store_true", help="skip psi_min (about 5 min per run)")
    parser.add_argument("--force", action="store_true", help="compare runs of different cases")
    arguments = parser.parse_args()
    reference = cavity_reference.marchi2021()
    directories = {"v6": pathlib.Path(arguments.v6).resolve(), "v7": pathlib.Path(arguments.v7).resolve()}
    labels = {"v6": arguments.v6_label or directories["v6"].name, "v7": arguments.v7_label or directories["v7"].name}
    runs = {}
    for side in SIDES:
        run = load_run(directories[side].name, directories[side].parent)
        if run is None:
            sys.exit(f"[e39_compare] {directories[side]}: no samples.jsonl")
        runs[side] = run
    if arguments.window_end == "own":
        t_stop = None
    elif arguments.window_end == "common":
        t_stop = min(float(runs[side]["times"].max()) for side in SIDES)
    else:
        try:
            t_stop = float(arguments.window_end)
        except ValueError:
            parser.error(f"--window-end takes own, common or a time, got {arguments.window_end!r}")
    if t_stop is not None:
        runs = {side: truncated(run, t_stop) for side, run in runs.items()}
    window_end_mode = ("各自最后 --average-span 个时间单位(cavity_analysis)" if t_stop is None
                       else f"两次运行都取截止 t = {t_stop:g} 的最后 {arguments.average_span:g} 个时间单位")
    differences = case_differences(runs)
    if differences and not arguments.force:
        sys.exit(f"[e39_compare] not the same case: {'; '.join(differences)} (--force compares them anyway)")
    analyses, descriptions, quantities = {}, {}, {}
    for side in SIDES:
        log(f"{side}: analysing {directories[side]} ({runs[side]['times'].size} samples"
            f"{'' if arguments.no_stream_function else ', with the stream function'})")
        analyses[side] = analyse_run(runs[side], arguments.average_span, arguments.steady_tol, arguments.window,
                                     not arguments.no_stream_function)
        descriptions[side] = describe(runs[side], analyses[side], labels[side])
    notes = ([f"不是同一算例(--force):{'; '.join(differences)}"] if differences else []) + \
        comparability_notes(runs, descriptions)
    if not arguments.no_stream_function:
        notes += [f"{side} 的平均窗口里没有快照,ψ_min 没有比较" for side in SIDES if not analyses[side].get("stream")]
    for side in SIDES:
        mask = window_mask(runs[side], arguments.average_span)
        if int(mask.sum()) != analyses[side]["samples_in_window"]:
            raise RuntimeError(f"{side}: window mask {int(mask.sum())} != analyse_run's "
                               f"{analyses[side]['samples_in_window']} samples")
        quantities[side] = run_quantities(runs[side], analyses[side], mask, reference, side,
                                          not arguments.no_stream_function)
    energy = {side: quantities[side]["kinetic_energy"] for side in SIDES}
    energy_noise = math.hypot(energy["v6"]["noise"], energy["v7"]["noise"])
    energy_difference = energy["v7"]["value"] - energy["v6"]["value"]
    result = {"tool": "experiment/validation/e39_compare.py", "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
              "command": sys.argv, "window_end": arguments.window_end, "window_end_mode": window_end_mode,
              "average_span": arguments.average_span, "runs": descriptions, "notes": notes,
              "quantities": quantity_rows(quantities, reference),
              "profile": profile_comparison(analyses, reference),
              "kinetic_energy": {"v6": energy["v6"]["value"], "v7": energy["v7"]["value"],
                                 "difference": energy_difference, "noise_v6": energy["v6"]["noise"],
                                 "noise_v7": energy["v7"]["noise"], "noise": energy_noise,
                                 "ratio": ratio(energy_difference, energy_noise)},
              "positions": position_rows(analyses, reference, descriptions["v6"]["spacing"]),
              "per_run": quantities}
    text = markdown(result)
    print(text, flush=True)
    if arguments.out:
        prefix = pathlib.Path(arguments.out)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        json_path, markdown_path = prefix.with_name(prefix.name + ".json"), prefix.with_name(prefix.name + ".md")
        json_path.write_text(json.dumps(plain(result), indent=1, ensure_ascii=False), encoding="utf-8")
        markdown_path.write_text(text, encoding="utf-8")
        log(f"wrote {json_path} and {markdown_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
