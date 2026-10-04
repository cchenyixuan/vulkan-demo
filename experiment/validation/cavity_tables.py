"""cavity_tables.py - summary CSVs and markdown tables of the cavity validation (called by cavity_analysis)."""
from __future__ import annotations

import json
import math

import numpy as np

from experiment.validation import cavity_reference
from experiment.validation.cavity_analysis import DATA, LOGS, write_csv

REFERENCE_EXTREMA = {"u_min": ("u_min", "y_at_u_min"), "v_max": ("v_max", "x_at_v_max"), "v_min": ("v_min", "x_at_v_min")}


def variant_label(variant: str) -> str:
    """float32 -> 'float32 ρ', delta -> 'δρ', float32_xi0.001_eps0.0025 -> 'float32 ρ, ξ = 0.001, ε² = 0.0025 h²'."""
    storage, *rest = variant.split("_")
    parts = ["float32 ρ" if storage == "float32" else "δρ"]
    for token in rest:
        if token.startswith("xi"):
            parts.append(f"ξ = {token[2:]}")
        elif token.startswith("eps"):
            parts.append(f"ε² = {token[3:]} h²")
    return ", ".join(parts)


def variant_order(variant: str) -> tuple:
    return (variant != "float32", variant != "delta", variant)


def ordered_variants(analyses: list[dict]) -> list[str]:
    return sorted({item["variant"] for item in analyses}, key=variant_order)


def run_label(item: dict) -> str:
    return f"{item['resolution']}² K={item['slabs']} {variant_label(item['variant'])}"


def invariants(item: dict) -> dict:
    totals = {"drift": 0, "overflow": 0, "far_migration": 0, "stamp_gpu": 0, "stamp_host": 0}
    for segment in item["segments"]:
        for key in totals:
            totals[key] += int(segment["totals"].get(key, 0))
    totals["segments"] = len(item["segments"])
    totals["complete"] = all(segment["status"] == "complete" for segment in item["segments"])
    return totals


def wall_hours(item: dict) -> float:
    return sum(segment["wall_s"] for segment in item["segments"]) / 3600.0


def estimates() -> dict:
    path = LOGS / "campaign.jsonl"
    if not path.exists():
        return {}
    entries = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            entry = json.loads(line)
            entries.setdefault(entry["run"], entry)
    return entries


def order(errors: list[float], spacings: list[float]) -> list[float]:
    return [math.log(errors[index] / errors[index + 1]) / math.log(spacings[index] / spacings[index + 1])
            if errors[index] > 0 and errors[index + 1] > 0 else float("nan") for index in range(len(errors) - 1)]


def fit_order(errors: list[float], spacings: list[float]) -> float:
    if len(errors) < 2:
        return float("nan")
    return float(np.polyfit(np.log(spacings), np.log(errors), 1)[0])


def md_table(header: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def write_all(analyses: list[dict]) -> None:
    reference = cavity_reference.marchi2021()
    extrema_reference = reference["extrema"]
    analyses = sorted(analyses, key=lambda item: (item["resolution"], -item["slabs"], variant_order(item["variant"])))
    planned = estimates()
    sections = {}

    # ---- run summary ----------------------------------------------------------------------------------
    rows, md_rows = [], []
    for item in analyses:
        inv = invariants(item)
        error_u, error_v = item["errors"][("wall", "mls", "u")], item["errors"][("wall", "mls", "v")]
        stream = item.get("stream") or {}
        estimate = planned.get(item["id"], {})
        rows.append([item["id"], item["resolution"], item["slabs"], item["variant"], f"{item['spacing']:.6f}",
                     item["t_steady_online"], item["t_steady_sliding"], f"{item['window'][0]:.2f}", f"{item['window'][1]:.2f}",
                     item["samples_in_window"], f"{error_u['L2']:.5f}", f"{error_u['Linf']:.5f}", f"{error_u['Linf_at']:.7f}",
                     f"{error_v['L2']:.5f}", f"{error_v['Linf']:.5f}", f"{error_v['Linf_at']:.7f}",
                     inv["drift"], inv["overflow"], inv["far_migration"], inv["stamp_gpu"], inv["stamp_host"],
                     inv["segments"], f"{wall_hours(item):.2f}",
                     f"{estimate.get('hours', float('nan')):.2f}" if estimate.get("hours") else ""])
        md_rows.append([run_label(item), f"{item['t_steady_online']:.1f}" if item["t_steady_online"] else "未达到",
                        f"{item['t_steady_sliding']:.1f}" if item["t_steady_sliding"] else "未达到",
                        f"{item['window'][0]:.1f}–{item['window'][1]:.1f}({item['samples_in_window']})",
                        f"{error_u['L2']:.4f} / {error_u['Linf']:.4f}", f"{error_v['L2']:.4f} / {error_v['Linf']:.4f}",
                        "0" if all(inv[key] == 0 for key in ("drift", "overflow", "far_migration", "stamp_gpu", "stamp_host"))
                        else str({key: inv[key] for key in ("drift", "overflow", "far_migration", "stamp_gpu", "stamp_host")}),
                        f"{wall_hours(item):.2f}", f"{estimate['hours']:.2f}" if estimate.get("hours") else "–"])
    write_csv(DATA / "summary.csv",
              ["run", "resolution", "K", "variant", "dx", "t_steady_online", "t_steady_sliding", "window_start",
               "window_end", "samples", "u_L2", "u_Linf", "u_Linf_at_y", "v_L2", "v_Linf", "v_Linf_at_x", "drift",
               "overflow_total", "far_migration", "stamp_errors_gpu", "stamp_errors_host", "segments", "wall_hours",
               "estimated_hours"], rows,
              "Re = 1000 cavity validation runs (v6 release switches). Errors: time-averaged linear-MLS velocity at the Marchi 2021\n"
              "Table 20/21 points minus Tc, wall-row frame; L2 = rms over the 28 points, Linf = max. Invariants summed over segments.")
    sections["RUNS"] = md_table(["run", "稳态 t_s(窗口判据)", "稳态 t_s(滑动重算)", "平均窗口(样本数)",
                                 "u:L2 / L∞", "v:L2 / L∞", "不变量", "实际 h", "估算 h"], md_rows)

    # ---- steady state: the deciding window, the sliding rule, stationarity inside the averaging window --------
    rows, md_rows = [], []
    for item in analyses:
        meta = json.loads((LOGS / item["id"] / "meta.json").read_text(encoding="utf-8"))
        window_length = meta["window_steps"] * meta["dt"]
        windows_path = LOGS / item["id"] / "windows.jsonl"
        windows = [json.loads(line) for line in windows_path.read_text(encoding="utf-8").splitlines() if line] \
            if windows_path.exists() else []
        deciding = next((window for window in windows if item["t_steady_online"] is not None
                         and abs(window["time"] - item["t_steady_online"]) < 1e-6), None)
        before = next((window for window in reversed(windows) if deciding and window["time"] < deciding["time"] - 1e-6), None)
        finite = np.isfinite(item["profile_changes"])
        half_profile = float(item["profile_changes"][finite][-1]) if finite.any() else float("nan")
        half_energy = float(item["ke_changes"][finite][-1]) if finite.any() else float("nan")
        energy = item["kinetic_energy_window"]
        relative_energy_std = float(energy["std"][0] / energy["mean"][0])
        sem = np.concatenate([item["statistics"]["u_marchi_wall_mls"]["sem"], item["statistics"]["v_marchi_wall_mls"]["sem"]])
        rows.append([item["id"], f"{window_length:.4f}", item["t_steady_online"],
                     f"{deciding['profile_change']:.2e}" if deciding else "", f"{deciding['kinetic_energy_change']:.2e}" if deciding else "",
                     f"{before['profile_change']:.2e}" if before else "", f"{before['kinetic_energy_change']:.2e}" if before else "",
                     item["t_steady_sliding"], f"{half_profile:.2e}", f"{half_energy:.2e}", f"{energy['mean'][0]:.4f}",
                     f"{relative_energy_std:.2e}", f"{np.median(sem):.2e}", f"{sem.max():.2e}"])
        md_rows.append([run_label(item), f"{window_length:.3f}",
                        f"{before['profile_change']:.1e} / {before['kinetic_energy_change']:.1e}" if before else "–",
                        (f"{item['t_steady_online']:.2f}: {deciding['profile_change']:.1e} / {deciding['kinetic_energy_change']:.1e}"
                         if deciding else "未达到"),
                        f"{item['t_steady_sliding']:.1f}" if item["t_steady_sliding"] else "未达到",
                        f"{half_profile:.1e} / {half_energy:.1e}", f"{energy['mean'][0]:.2f} (± {relative_energy_std:.0e})",
                        f"{np.median(sem):.1e} / {sem.max():.1e}"])
    write_csv(DATA / "steady.csv",
              ["run", "window_length", "t_steady_online", "deciding_window_profile_change", "deciding_window_ke_change",
               "previous_window_profile_change", "previous_window_ke_change", "t_steady_sliding",
               "averaging_halves_profile_change", "averaging_halves_ke_change", "ke_mean", "ke_relative_std",
               "sem_median_56pt", "sem_max_56pt"], rows,
              "Steady state. Window rule (runner): relative L2 change of the window-mean profile at the 56 Marchi points and\n"
              "relative change of the window-mean fluid kinetic energy between consecutive windows, both < tolerance; the\n"
              "deciding window and the one before it are listed. Sliding: the same rule at every sample time (analysis).\n"
              "Averaging halves: the same two changes between the second and first halves of the averaging window.\n"
              "ke = 1/2 sum m |v|^2 over fluid particles (m = rho0 * 1.1293 dx^2); sem over the 56 time-averaged points.")
    sections["STEADY"] = md_table(["run", "窗口长度", "前一窗口:剖面 / 动能变化",
                                   "t_s(窗口判据):剖面 / 动能变化", "t_s(滑动重算)",
                                   "平均窗口后半段对前半段:剖面 / 动能", "动能均值(± 相对标准差)",
                                   "标准误差:中位数 / 最大"], md_rows)

    # ---- extrema and vortex centre -------------------------------------------------------------------------
    rows, md_rows = [], []
    for item in analyses:
        cells = [run_label(item)]
        record = [item["id"]]
        for name, (value_key, position_key) in REFERENCE_EXTREMA.items():
            entry = item["extrema"][name]
            ref_value, ref_position = extrema_reference[value_key][0], extrema_reference[position_key][0]
            record += [f"{entry['value']:.5f}", f"{entry['value'] - ref_value:+.5f}", f"{entry['position_wall']:.4f}",
                       f"{entry['position_wall'] - ref_position:+.4f}", f"{entry['position_fluid']:.4f}"]
            cells.append(f"{entry['value']:.4f} ({100 * (abs(entry['value']) - abs(ref_value)) / abs(ref_value):+.2f} %) @ "
                         f"{entry['position_wall']:.4f} ({entry['position_wall'] - ref_position:+.4f})")
        stream = item.get("stream") or {}
        if stream:
            ref_psi = extrema_reference["psi_min"][0]
            ref_x, ref_y = extrema_reference["x_at_psi_min"][0], extrema_reference["y_at_psi_min"][0]
            record += [f"{stream['psi_min_wall']:.5f}", f"{stream['psi_min_wall'] - ref_psi:+.5f}", f"{stream['x_wall']:.4f}",
                       f"{stream['y_wall']:.4f}", f"{stream['x_wall'] - ref_x:+.4f}", f"{stream['y_wall'] - ref_y:+.4f}",
                       stream["snapshots"]]
            cells.append(f"{stream['psi_min_wall']:.4f} ({100 * (abs(stream['psi_min_wall']) - abs(ref_psi)) / abs(ref_psi):+.2f} %) @ "
                         f"({stream['x_wall']:.4f}, {stream['y_wall']:.4f}) (Δ {stream['x_wall'] - ref_x:+.4f}, "
                         f"{stream['y_wall'] - ref_y:+.4f})")
        else:
            record += [""] * 7
            cells.append("–")
        rows.append(record)
        md_rows.append(cells)
    reference_row = ["Marchi 2021 T_c(Table 19)"] + [
        f"{extrema_reference[value_key][0]:.4f} @ {extrema_reference[position_key][0]:.4f}"
        for value_key, position_key in REFERENCE_EXTREMA.values()] + [
        f"{extrema_reference['psi_min'][0]:.4f} @ ({extrema_reference['x_at_psi_min'][0]:.4f}, "
        f"{extrema_reference['y_at_psi_min'][0]:.4f})"]
    write_csv(DATA / "extrema.csv",
              ["run"] + [f"{name}_{column}" for name in REFERENCE_EXTREMA for column in
                         ("value", "error", "position_wall", "position_error", "position_fluid")]
              + ["psi_min", "psi_min_error", "x_psi_min", "y_psi_min", "x_error", "y_error", "snapshots"], rows,
              "Extrema of the time-averaged dense MLS centre lines (parabolic refinement) and the primary vortex centre (minimum of\n"
              "psi = integral of u upward from the bottom wall, time-averaged MLS grid). Positions in the wall-row frame unless noted.\n"
              "Reference: Marchi 2021 Table 19 Tc.")
    sections["EXTREMA"] = md_table(["run", "u_min(幅值误差)@ y(位置差)", "v_max @ x", "v_min @ x",
                                    "ψ_min @ (x, y)"],
                                   [reference_row] + md_rows)

    # ---- convergence ------------------------------------------------------------------------------------------
    rows, md_rows = [], []
    for variant in ordered_variants(analyses):
        selected = [item for item in analyses if item["variant"] == variant and item["slabs"] == 2]
        if len(selected) < 2:
            continue
        spacings = [1.0 / item["resolution"] for item in selected]
        for line in ("u", "v"):
            for norm in ("L2", "Linf"):
                values = [item["errors"][("wall", "mls", line)][norm] for item in selected]
                orders = order(values, spacings)
                rows.append([variant, line, norm] + [f"{value:.5f}" for value in values]
                            + [f"{value:.2f}" for value in orders] + [f"{fit_order(values, spacings):.2f}"])
                md_rows.append([variant_label(variant), line, norm.replace("inf", "∞"),
                                " → ".join(f"{value:.4f}" for value in values),
                                ", ".join(f"{value:.2f}" for value in orders), f"{fit_order(values, spacings):.2f}"])
        for name, (value_key, _) in REFERENCE_EXTREMA.items():
            values = [abs(item["extrema"][name]["value"] - extrema_reference[value_key][0]) for item in selected]
            orders = order(values, spacings)
            rows.append([variant, name, "abs"] + [f"{value:.5f}" for value in values]
                        + [f"{value:.2f}" for value in orders] + [f"{fit_order(values, spacings):.2f}"])
            md_rows.append([variant_label(variant), name, "\\|误差\\|",
                            " → ".join(f"{value:.4f}" for value in values), ", ".join(f"{value:.2f}" for value in orders),
                            f"{fit_order(values, spacings):.2f}"])
    resolutions = sorted({item["resolution"] for item in analyses if item["slabs"] == 2})
    write_csv(DATA / "convergence.csv",
              ["variant", "quantity", "norm"] + [f"error_dx_1_{value}" for value in resolutions]
              + [f"order_{resolutions[index]}_{resolutions[index + 1]}" for index in range(len(resolutions) - 1)]
              + ["fitted_order"], rows,
              "Errors at the Marchi 2021 points (wall-row frame, MLS) and of the extrema vs resolution (K = 2); observed orders\n"
              "p = log(e1 / e2) / log(dx1 / dx2) between consecutive resolutions and a least-squares fit over all.")
    sections["CONVERGENCE"] = md_table(["变体", "量", "范数",
                                        "误差,Δx = " + " → ".join(f"1/{value}" for value in resolutions),
                                        "观测阶(相邻分辨率)", "拟合阶"], md_rows)

    # ---- three-grid apparent order and Richardson limit of the extrema (from the SPH values alone) -----------
    rows, md_rows = [], []
    quantities = [("u_min", "value", "u_min"), ("y(u_min)", "position_wall", "y_at_u_min"),
                  ("v_max", "value", "v_max"), ("x(v_max)", "position_wall", "x_at_v_max"),
                  ("v_min", "value", "v_min"), ("x(v_min)", "position_wall", "x_at_v_min")]
    stream_quantities = [("ψ_min", "psi_min_wall", "psi_min"), ("x(ψ_min)", "x_wall", "x_at_psi_min"),
                         ("y(ψ_min)", "y_wall", "y_at_psi_min")]
    for variant in ordered_variants(analyses):
        selected = [item for item in analyses if item["variant"] == variant and item["slabs"] == 2]
        if [item["resolution"] for item in selected] != [250, 500, 1000]:
            continue
        series = []
        for label, field, reference_key in quantities:
            extremum = {"u_min": "u_min", "y(u_min)": "u_min", "v_max": "v_max", "x(v_max)": "v_max",
                        "v_min": "v_min", "x(v_min)": "v_min"}[label]
            series.append((label, [item["extrema"][extremum][field] for item in selected], reference_key))
        if all(item.get("stream") for item in selected):
            for label, field, reference_key in stream_quantities:
                series.append((label, [item["stream"][field] for item in selected], reference_key))
        for label, values, reference_key in series:
            coarse, medium, fine = values
            reference_value = extrema_reference[reference_key][0]
            first, second = medium - coarse, fine - medium
            apparent, limit = float("nan"), float("nan")
            if first * second > 0 and abs(second) < abs(first):
                apparent = math.log(first / second) / math.log(2.0)
                if 0.5 <= apparent <= 4.0:
                    limit = fine + second / (2.0 ** apparent - 1.0)
                else:
                    apparent = float("nan")
            rows.append([variant, label, f"{coarse:.6f}", f"{medium:.6f}", f"{fine:.6f}",
                         "" if math.isnan(apparent) else f"{apparent:.3f}", "" if math.isnan(limit) else f"{limit:.6f}",
                         f"{reference_value:.10g}", f"{fine - reference_value:+.6f}",
                         "" if math.isnan(limit) else f"{limit - reference_value:+.6f}"])
            relative = (lambda value: f" ({100 * (abs(value) - abs(reference_value)) / abs(reference_value):+.1f} %)"
                        if label in ("u_min", "v_max", "v_min", "ψ_min") else "")
            md_rows.append([variant_label(variant), label,
                            " → ".join(f"{value:.4f}" for value in values),
                            "–" if math.isnan(apparent) else f"{apparent:.2f}",
                            "–" if math.isnan(limit) else f"{limit:.4f}{relative(limit)}",
                            f"{reference_value:.4f}", f"{fine - reference_value:+.4f}",
                            "–" if math.isnan(limit) else f"{limit - reference_value:+.4f}"])
    if rows:
        write_csv(DATA / "richardson.csv",
                  ["variant", "quantity", "value_250", "value_500", "value_1000", "apparent_order", "richardson_limit",
                   "marchi2021_Tc", "value_1000_minus_Tc", "limit_minus_Tc"], rows,
                  "Three-grid apparent order p = log((f500 - f250) / (f1000 - f500)) / log 2 and Richardson limit\n"
                  "f1000 + (f1000 - f500) / (2^p - 1) of the SPH extrema (K = 2, wall-row frame), computed from the SPH values\n"
                  "alone; left empty when the sequence is not monotonically converging or p is outside [0.5, 4].")
        sections["RICHARDSON"] = md_table(["变体", "量", "250² → 500² → 1000²", "表观阶 p",
                                           "Richardson 外推值", "Marchi 2021 T_c", "1000² − T_c", "外推值 − T_c"], md_rows)

    # ---- numerics variants (xi, epsilon) side by side: K = 2 float32 runs, by resolution -------------------------
    selected = [item for item in analyses if item["slabs"] == 2 and item["variant"].split("_")[0] == "float32"]
    if len({item["variant"] for item in selected}) > 1:
        rows, md_rows = [], []
        for item in selected:
            error_u, error_v = item["errors"][("wall", "mls", "u")], item["errors"][("wall", "mls", "v")]
            record = [item["id"], item["resolution"], item.get("xi", 0.1), item.get("epsilon_factor", 0.01),
                      f"{error_u['L2']:.5f}", f"{error_u['Linf']:.5f}", f"{error_v['L2']:.5f}", f"{error_v['Linf']:.5f}"]
            cells = [f"{item['resolution']}²", variant_label(item["variant"]),
                     f"{error_u['L2']:.4f} / {error_v['L2']:.4f}", f"{error_u['Linf']:.4f} / {error_v['Linf']:.4f}"]
            for name, (value_key, position_key) in REFERENCE_EXTREMA.items():
                entry = item["extrema"][name]
                reference_value = extrema_reference[value_key][0]
                relative = 100 * (abs(entry["value"]) - abs(reference_value)) / abs(reference_value)
                offset = entry["position_wall"] - extrema_reference[position_key][0]
                record += [f"{entry['value']:.5f}", f"{relative:+.3f}", f"{offset:+.5f}"]
                cells.append(f"{relative:+.2f} % @ {offset:+.4f}")
            stream = item.get("stream") or {}
            if stream:
                reference_psi = extrema_reference["psi_min"][0]
                relative = 100 * (abs(stream["psi_min_wall"]) - abs(reference_psi)) / abs(reference_psi)
                x_offset = stream["x_wall"] - extrema_reference["x_at_psi_min"][0]
                y_offset = stream["y_wall"] - extrema_reference["y_at_psi_min"][0]
                record += [f"{stream['psi_min_wall']:.5f}", f"{relative:+.3f}", f"{x_offset:+.5f}", f"{y_offset:+.5f}"]
                cells.append(f"{relative:+.2f} % @ ({x_offset:+.4f}, {y_offset:+.4f})")
            else:
                record += [""] * 4
                cells.append("–")
            rows.append(record)
            md_rows.append(cells)
        write_csv(DATA / "numerics_variants.csv",
                  ["run", "resolution", "xi", "epsilon_squared_factor", "u_L2", "u_Linf", "v_L2", "v_Linf"]
                  + [f"{name}_{column}" for name in REFERENCE_EXTREMA for column in
                     ("value", "magnitude_error_percent", "position_error")]
                  + ["psi_min", "psi_min_magnitude_error_percent", "x_psi_min_error", "y_psi_min_error"], rows,
                  "K = 2 float32 runs with the release numerics (xi = 0.1, eps^2 = 0.01 h^2) and the variants, by resolution:\n"
                  "errors at the Marchi points and the extrema relative to Marchi 2021 Tc (wall-row frame, MLS).")
        sections["VARIANTS"] = md_table(["分辨率", "设置", "L2:u / v", "L∞:u / v", "u_min 幅值误差 @ y 位置差",
                                         "v_max 幅值误差 @ x 位置差", "v_min 幅值误差 @ x 位置差",
                                         "ψ_min 幅值误差 @ (x, y) 位置差"], md_rows)

    # ---- K = 1 vs K = 2 -------------------------------------------------------------------------------------------
    pairs = [(item, other) for item in analyses for other in analyses
             if item["slabs"] == 2 and other["slabs"] == 1 and item["resolution"] == other["resolution"]
             and item["variant"] == other["variant"]]
    rows, md_rows = [], []
    for k2, k1 in pairs:
        for line in ("u", "v"):
            name = f"{line}_marchi_wall_mls"
            difference = k2["statistics"][name]["mean"] - k1["statistics"][name]["mean"]
            combined = np.sqrt(k2["statistics"][name]["sem"] ** 2 + k1["statistics"][name]["sem"] ** 2)
            dense = k2["statistics"][f"{line}_dense_mls"]["mean"] - k1["statistics"][f"{line}_dense_mls"]["mean"]
            error = k2["errors"][("wall", "mls", line)]
            rows.append([k2["resolution"], line, f"{np.abs(difference).max():.2e}", f"{np.sqrt(np.mean(difference ** 2)):.2e}",
                         f"{np.abs(dense).max():.2e}", f"{np.median(combined):.2e}", f"{np.abs(difference / combined).max():.2f}",
                         f"{error['L2']:.4f}", f"{error['Linf']:.4f}",
                         f"{k1['errors'][('wall', 'mls', line)]['L2']:.4f}"])
            md_rows.append([f"{k2['resolution']}²", line, f"{np.abs(difference).max():.1e}", f"{np.sqrt(np.mean(difference ** 2)):.1e}",
                            f"{np.abs(dense).max():.1e}", f"{np.median(combined):.1e}",
                            f"{np.abs(difference / combined).max():.1f}",
                            f"{error['L2']:.4f} / {k1['errors'][('wall', 'mls', line)]['L2']:.4f}",
                            f"{np.sqrt(np.mean(difference ** 2)) / error['L2']:.3f}"])
    if rows:
        write_csv(DATA / "k1_vs_k2.csv",
                  ["resolution", "line", "max_abs_diff_28pt", "rms_diff_28pt", "max_abs_diff_dense", "median_combined_sem",
                   "max_diff_over_sem", "K2_L2_error", "K2_Linf_error", "K1_L2_error"], rows,
                  "K = 2 minus K = 1 time-averaged profiles (MLS, wall-row frame) at the Marchi points and on the dense lines.")
        sections["K1K2"] = md_table(["分辨率", "中线", "max \\|Δ\\|(28 点)", "rms Δ(28 点)", "max \\|Δ\\|(稠密线)",
                                     "合成标准误差(中位数)", "max(\\|Δ\\| / 合成标准误差)", "L2 误差 K=2 / K=1",
                                     "rms Δ / L2 误差"], md_rows)

    # ---- float32 vs delta density ----------------------------------------------------------------------------------
    rows, md_rows = [], []
    for item in analyses:
        if item["variant"] != "float32":
            continue
        other = next((candidate for candidate in analyses if candidate["variant"] == "delta"
                      and candidate["resolution"] == item["resolution"] and candidate["slabs"] == item["slabs"]), None)
        if other is None:
            continue
        for line in ("u", "v"):
            name = f"{line}_marchi_wall_mls"
            difference = other["statistics"][name]["mean"] - item["statistics"][name]["mean"]
            combined = np.sqrt(other["statistics"][name]["sem"] ** 2 + item["statistics"][name]["sem"] ** 2)
            error_f, error_d = item["errors"][("wall", "mls", line)], other["errors"][("wall", "mls", line)]
            rows.append([item["resolution"], item["slabs"], line, f"{np.abs(difference).max():.2e}",
                         f"{np.sqrt(np.mean(difference ** 2)):.2e}", f"{np.median(combined):.2e}",
                         f"{error_f['L2']:.5f}", f"{error_d['L2']:.5f}", f"{error_f['Linf']:.5f}", f"{error_d['Linf']:.5f}"])
            md_rows.append([f"{item['resolution']}²", line, f"{np.abs(difference).max():.1e}",
                            f"{np.sqrt(np.mean(difference ** 2)):.1e}", f"{np.median(combined):.1e}",
                            f"{error_f['L2']:.4f} → {error_d['L2']:.4f}", f"{error_f['Linf']:.4f} → {error_d['Linf']:.4f}"])
    if rows:
        write_csv(DATA / "float32_vs_delta.csv",
                  ["resolution", "K", "line", "max_abs_diff", "rms_diff", "median_combined_sem", "L2_float32", "L2_delta",
                   "Linf_float32", "Linf_delta"], rows,
                  "delta-density minus float32 time-averaged profiles (MLS, wall-row frame) at the Marchi points, and the errors of each.")
        sections["DELTA"] = md_table(["分辨率", "中线", "max \\|Δ\\|", "rms Δ", "合成标准误差(中位数)", "L2:float32 → δρ",
                                      "L∞:float32 → δρ"], md_rows)

    # ---- sensitivity: frame and interpolation -----------------------------------------------------------------------
    md_rows, rows = [], []
    for item in analyses:
        cells = [run_label(item)]
        record = [item["id"]]
        for frame, method in (("wall", "mls"), ("wall", "dense"), ("wall", "shepard"), ("mid", "dense"),
                              ("fluid", "mls")):
            cells.append(" / ".join(f"{item['errors'][(frame, method, line)]['L2']:.4f}" for line in ("u", "v")))
            record += [f"{item['errors'][(frame, method, line)][norm]:.5f}" for line in ("u", "v") for norm in ("L2", "Linf")]
        cells.append(" / ".join(f"{item['frame_reynolds'][frame]:.0f}" for frame in ("wall", "mid", "fluid")))
        md_rows.append(cells)
        rows.append(record)
    write_csv(DATA / "frame_sensitivity.csv",
              ["run"] + [f"{frame}_{method}_{line}_{norm}" for frame, method in
                         (("wall", "mls"), ("wall", "dense"), ("wall", "shepard"), ("mid", "dense"), ("fluid", "mls"))
                         for line in ("u", "v") for norm in ("L2", "Linf")], rows,
              "Errors at the Marchi 2021 points in three frames (unit square = wall rows, half way to the fluid, fluid box)\n"
              "and two interpolations; 'dense' = cubic spline of the time-averaged 1001-point MLS line.")
    sections["SENSITIVITY"] = md_table(["run", "L2 u / v:壁面行框架,MLS(主结果)", "壁面行框架,稠密线样条",
                                        "壁面行框架,Shepard", "半间距框架(壁在 ±(0.5+Δx/2)),稠密线样条",
                                        "流体框框架,MLS", "Re:壁面行 / 半间距 / 流体框"], md_rows)

    # ---- markdown -------------------------------------------------------------------------------------------------------
    text = "\n\n".join(f"<!-- {name} -->\n{table}" for name, table in sections.items()) + "\n"
    (DATA / "tables.md").write_text(text, encoding="utf-8")
    print("wrote", DATA / "tables.md")
