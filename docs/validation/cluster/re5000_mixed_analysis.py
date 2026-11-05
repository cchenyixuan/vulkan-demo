"""Re 5000, 2500^2: the 5090 + 7900 XTX (mixed) run against the 2x5090 (boost) run it continues, for sec:crossvendor.

Run from the rc3 worktree root:  python docs/validation/cluster/re5000_mixed_analysis.py [--out DIR]

Only samples after the mixed run's start step count (the samples at or before it were copied in from the boost run for
the window test). Every mixed sample lies on the boost run's 67000-step sample grid, so each one has a boost sample at
the same global step while the boost run reached it.

  1. run facts      devices, pilot weights, cut, slab particles, start / end, steps/s, the runner's invariant totals
                    (drift, overflow_*, far_migration, stamp errors) per segment
  2. same step      mixed - boost on the 28 + 28 Marchi points (relative L2 in the Marchi norm, max |du|, |dv|), on the
                    1001-point dense lines (max |du|, |dv|) and kinetic energy; for scale, the 2x5090 - K = 1 differences
                    the observer logged at common steps (l2_newest.csv)
  3. Marchi         relative L2 / Linf of the newest mixed sample and of the mean over all mixed samples, and the boost
                    run's mean over the same steps
  4. extrema        u_min, v_max, v_min and their positions from the window-mean dense lines (parabola refinement, as in
                    long_run_convergence.py) against Marchi 2021 Table 27
  5. window tests   the mixed run's own window records after the start
Writes report.md, summary.json, same_step.csv and same_step.png into --out.
"""
from __future__ import annotations

import argparse, csv, json, pathlib, sys

import numpy as np

_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiment.validation import cavity_reference, cavity_sampling as sampling  # noqa: E402
from experiment.validation.cavity_analysis import EXTREMA_SEARCH  # noqa: E402

RUNS = _ROOT / "logs" / "validation" / "cavity_re5000_v7"
MIXED_RUN = RUNS / "n2500_k2xv_float32_xi0p001_eps0p0025_adami_re5000"
BOOST_RUN = RUNS / "n2500_k2_float32_xi0p001_eps0p0025_adami_re5000_from_k1"
STATE = RUNS / "pipeline_state"
EXTREMA_REFERENCE = {"u_min": ("u_min", "y_at_u_min"), "v_max": ("v_max", "x_at_v_max"), "v_min": ("v_min", "x_at_v_min")}
PROFILE_KEYS = ("u_marchi_wall_mls", "v_marchi_wall_mls", "u_dense_mls", "v_dense_mls")


def read_jsonl(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:      # a line cut by a shutdown; the next write starts a new line
                continue
    return rows


def load_sample(run_dir: pathlib.Path, step: int) -> dict | None:
    path = run_dir / "samples" / f"s{step:010d}.npz"
    if not path.exists():
        return None
    with np.load(path) as archive:
        return {key: archive[key].astype(np.float64) for key in PROFILE_KEYS}


def relative(values: np.ndarray, reference: np.ndarray) -> float:
    """Relative L2 in percent of the reference norm (the Marchi norm for both errors and run-to-run differences)."""
    return float(100 * np.sqrt((values ** 2).sum() / (reference ** 2).sum()))


def extrema(dense_coordinates: np.ndarray, u_dense: np.ndarray, v_dense: np.ndarray) -> dict:
    lines = {"u": u_dense, "v": v_dense}
    result = {}
    for name, (line, kind, search) in EXTREMA_SEARCH.items():
        position, value = sampling.refine_extremum(dense_coordinates, lines[line], kind, search=search)
        result[name] = float(value)
        result[f"{name}_position"] = float(position)
    return result


def observer_reference() -> dict | None:
    """2x5090 - K = 1 at common steps from the observer's trace: the run-to-run scale of a same-vendor change."""
    path = RUNS / "l2_newest.csv"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["run"] == "boost_minus_k1" and row["diff_u_pct"]]
    if not rows:
        return None
    steps, unique = set(), []
    for row in rows:
        if row["step"] not in steps:
            steps.add(row["step"]); unique.append(row)
    du = np.array([float(row["diff_u_pct"]) for row in unique]); dv = np.array([float(row["diff_v_pct"]) for row in unique])
    return {"count": len(unique), "t_range": [float(unique[0]["t"]), float(unique[-1]["t"])],
            "u_mean": float(du.mean()), "u_min": float(du.min()), "u_max": float(du.max()),
            "v_mean": float(dv.mean()), "v_min": float(dv.min()), "v_max": float(dv.max())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mixed-run", type=pathlib.Path, default=MIXED_RUN)
    parser.add_argument("--boost-run", type=pathlib.Path, default=BOOST_RUN)
    parser.add_argument("--out", type=pathlib.Path, default=_ROOT / "docs" / "validation" / "cluster" / "re5000_mixed")
    arguments = parser.parse_args()
    mixed_dir, boost_dir, out = arguments.mixed_run, arguments.boost_run, arguments.out
    out.mkdir(parents=True, exist_ok=True)

    meta = json.loads((mixed_dir / "meta.json").read_text(encoding="utf-8"))
    start_step = int(meta["start_from"]["source_step"])
    start_time = float(meta["start_from"]["source_time"])
    dt = float(meta["dt"])
    expected_total = int(meta["expected_total"])
    reference = cavity_reference.marchi2021(reynolds=5000)
    u_reference, v_reference = reference["u"][1], reference["v"][1]
    with np.load(mixed_dir / "points.npz") as points:
        dense_coordinates = points["dense_reference"].astype(np.float64)

    mixed_rows = [row for row in read_jsonl(mixed_dir / "samples.jsonl") if row["step"] > start_step]
    boost_rows = {row["step"]: row for row in read_jsonl(boost_dir / "samples.jsonl")}
    if not mixed_rows:
        raise SystemExit(f"{mixed_dir.name}: no sample after the start step {start_step} yet")

    # ---------------------------------------------------------------- 1. run facts
    weights_record = json.loads((STATE / "mixed_weights.json").read_text(encoding="utf-8")) if (STATE / "mixed_weights.json").exists() else {}
    devices = [device.get("name") for device in weights_record.get("devices", [])]
    segments = read_jsonl(mixed_dir / "segments.jsonl")
    result = json.loads((mixed_dir / "result.json").read_text(encoding="utf-8")) if (mixed_dir / "result.json").exists() else None
    steady = json.loads((mixed_dir / "steady.json").read_text(encoding="utf-8")) if (mixed_dir / "steady.json").exists() else None
    alive = sorted({row["alive"] for row in mixed_rows})
    rates = np.array([row["fps"] for row in mixed_rows])
    facts = {"devices": devices, "weights": meta.get("weights"), "cuts": meta.get("cuts"), "slab_particles": meta.get("slab_particles"),
             "pilot_rounds": len(weights_record.get("rounds", [])), "start_step": start_step, "start_time": start_time,
             "start_checkpoint": meta["start_from"].get("file"), "last_step": mixed_rows[-1]["step"], "last_time": mixed_rows[-1]["time"],
             "samples": len(mixed_rows), "expected_total": expected_total, "alive_values": alive,
             "mls_fallbacks_max": max(row["mls_fallbacks"] for row in mixed_rows),
             "min_neighbours_min": min(row["min_neighbours"] for row in mixed_rows),
             "density_range": [min(row["density_min"] for row in mixed_rows), max(row["density_max"] for row in mixed_rows)],
             "steps_per_second_median": float(np.median(rates)), "steps_per_second_range": [float(rates.min()), float(rates.max())],
             "segments": [{key: segment.get(key) for key in ("segment", "start_step", "status", "totals", "error", "device_uuids")}
                          for segment in segments],
             "result": result, "steady": steady}

    # ---------------------------------------------------------------- 2. same step
    table, mixed_stack, boost_stack = [], [], []
    for row in mixed_rows:
        step = row["step"]
        mixed_sample = load_sample(mixed_dir, step)
        boost_sample = load_sample(boost_dir, step) if step in boost_rows else None
        entry = {"step": step, "t": round(row["time"], 4),
                 "mixed_L2u_pct": relative(mixed_sample["u_marchi_wall_mls"] - u_reference, u_reference),
                 "mixed_L2v_pct": relative(mixed_sample["v_marchi_wall_mls"] - v_reference, v_reference),
                 "mixed_KE": row["kinetic_energy"], "alive": row["alive"], "fps": row["fps"]}
        if boost_sample is not None:
            du = mixed_sample["u_marchi_wall_mls"] - boost_sample["u_marchi_wall_mls"]
            dv = mixed_sample["v_marchi_wall_mls"] - boost_sample["v_marchi_wall_mls"]
            entry.update({"boost_L2u_pct": relative(boost_sample["u_marchi_wall_mls"] - u_reference, u_reference),
                          "boost_L2v_pct": relative(boost_sample["v_marchi_wall_mls"] - v_reference, v_reference),
                          "diff_u_pct": relative(du, u_reference), "diff_v_pct": relative(dv, v_reference),
                          "max_abs_du": float(np.abs(du).max()), "max_abs_dv": float(np.abs(dv).max()),
                          "dense_max_abs_du": float(np.abs(mixed_sample["u_dense_mls"] - boost_sample["u_dense_mls"]).max()),
                          "dense_max_abs_dv": float(np.abs(mixed_sample["v_dense_mls"] - boost_sample["v_dense_mls"]).max()),
                          "KE_relative_diff": (row["kinetic_energy"] - boost_rows[step]["kinetic_energy"]) / boost_rows[step]["kinetic_energy"]})
            boost_stack.append(boost_sample)
            mixed_stack.append(mixed_sample)
        table.append(entry)
    common = [entry for entry in table if "diff_u_pct" in entry]

    def summary(key: str) -> dict:
        values = np.array([entry[key] for entry in common])
        return {"mean": float(values.mean()), "max": float(np.abs(values).max()), "last": float(values[-1])} if values.size else {}

    same_step = {"common_samples": len(common), "t_range": [common[0]["t"], common[-1]["t"]] if common else None,
                 **{key: summary(key) for key in ("diff_u_pct", "diff_v_pct", "max_abs_du", "max_abs_dv", "dense_max_abs_du",
                                                  "dense_max_abs_dv", "KE_relative_diff")},
                 "same_vendor_scale_boost_minus_k1": observer_reference()}

    # ---------------------------------------------------------------- 3. Marchi + 4. extrema
    def window(samples: list[dict]) -> dict:
        mean = {key: np.mean([sample[key] for sample in samples], axis=0) for key in PROFILE_KEYS}
        errors_u, errors_v = mean["u_marchi_wall_mls"] - u_reference, mean["v_marchi_wall_mls"] - v_reference
        values = {"samples": len(samples), "L2u_pct": relative(errors_u, u_reference), "L2v_pct": relative(errors_v, v_reference),
                  "Linf_u": float(np.abs(errors_u).max()), "Linf_v": float(np.abs(errors_v).max())}
        values.update(extrema(dense_coordinates, mean["u_dense_mls"], mean["v_dense_mls"]))
        return values

    all_mixed = [load_sample(mixed_dir, row["step"]) for row in mixed_rows]
    marchi = {"newest_mixed": {"t": mixed_rows[-1]["time"], "L2u_pct": table[-1]["mixed_L2u_pct"], "L2v_pct": table[-1]["mixed_L2v_pct"]},
              "window": {"t_range": [start_time, mixed_rows[-1]["time"]], "mixed": window(all_mixed)}}
    if mixed_stack:
        marchi["window_common"] = {"t_range": [common[0]["t"], common[-1]["t"]], "mixed": window(mixed_stack), "boost": window(boost_stack)}
    extrema_reference = {name: {"value": reference["extrema"][value_key][0], "position": reference["extrema"][position_key][0]}
                         for name, (value_key, position_key) in EXTREMA_REFERENCE.items() if value_key in reference["extrema"]}

    # ---------------------------------------------------------------- 5. window tests
    windows = [row for row in read_jsonl(mixed_dir / "windows.jsonl") if row["step"] > start_step]

    summary_record = {"facts": facts, "same_step": same_step, "marchi": marchi, "extrema_reference": extrema_reference,
                      "windows_after_start": windows, "dt": dt}
    (out / "summary.json").write_text(json.dumps(summary_record, indent=1, default=float), encoding="utf-8")
    columns = list(table[0].keys()) + [key for key in common[0].keys() if key not in table[0]] if common else list(table[0].keys())
    with open(out / "same_step.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader(); writer.writerows(table)
    write_report(out, summary_record, mixed_dir, boost_dir)
    plot(out, table)
    sys.stdout.buffer.write((out / "report.md").read_bytes())
    return 0


def write_report(out: pathlib.Path, record: dict, mixed_dir: pathlib.Path, boost_dir: pathlib.Path) -> None:
    facts, same, marchi = record["facts"], record["same_step"], record["marchi"]
    lines = ["# Re 5000, 2500²: RTX 5090 + RX 7900 XTX against 2× RTX 5090", "",
             f"Mixed run `{mixed_dir.name}`, continued from `{boost_dir.name}` at step {facts['start_step']} (t = {facts['start_time']:.2f}).", "",
             "## 1. Run", "",
             f"- devices (pilot record): {facts['devices']}; weights {facts['weights']} (5090 : XTX = "
             f"{facts['weights'][0] / facts['weights'][1]:.3f}), cut {facts['cuts']}, slab particles {facts['slab_particles']}, "
             f"{facts['pilot_rounds']} pilot round(s)",
             f"- samples after the start: {facts['samples']}, t {facts['start_time']:.3f} → {facts['last_time']:.3f} "
             f"(step {facts['start_step']} → {facts['last_step']})",
             f"- steps/s median {facts['steps_per_second_median']:.1f} (range {facts['steps_per_second_range'][0]:.1f}–{facts['steps_per_second_range'][1]:.1f})",
             f"- alive at every sample: {facts['alive_values']} (expected {facts['expected_total']}); MLS fallbacks max "
             f"{facts['mls_fallbacks_max']}; min neighbours {facts['min_neighbours_min']}; density {facts['density_range'][0]:.3f}–{facts['density_range'][1]:.3f}"]
    for segment in facts["segments"]:
        lines.append(f"- segment {segment['segment']} from step {segment['start_step']}: {segment['status']}, totals {segment['totals']}"
                     + (f", error {segment['error']}" if segment.get("error") else ""))
    if not facts["segments"]:
        lines.append("- no segment record yet (the runner writes it at the end of the segment)")
    lines.append(f"- result.json: {facts['result']['status'] if facts['result'] else 'not written yet'}; steady.json: {facts['steady']}")
    lines += ["", "## 2. Same global step, mixed − 2×5090", ""]
    if same["common_samples"]:
        lines += [f"{same['common_samples']} common samples, t {same['t_range'][0]:.3f}–{same['t_range'][1]:.3f}. Relative L2 in the Marchi norm.", "",
                  "| quantity | mean | max | last |", "|---|---|---|---|"]
        for key, label in (("diff_u_pct", "u, 28 points (%)"), ("diff_v_pct", "v, 28 points (%)"), ("max_abs_du", "max |Δu|, 28 points"),
                           ("max_abs_dv", "max |Δv|, 28 points"), ("dense_max_abs_du", "max |Δu|, 1001 points"),
                           ("dense_max_abs_dv", "max |Δv|, 1001 points"), ("KE_relative_diff", "ΔKE / KE")):
            values = same[key]
            lines.append(f"| {label} | {values['mean']:.3g} | {values['max']:.3g} | {values['last']:.3g} |")
        scale = same["same_vendor_scale_boost_minus_k1"]
        if scale:
            lines += ["", f"Same-vendor scale: 2×5090 − K = 1 at {scale['count']} common steps (t {scale['t_range'][0]:.1f}–{scale['t_range'][1]:.1f}): "
                          f"u {scale['u_min']:.4f}–{scale['u_max']:.4f} % (mean {scale['u_mean']:.4f}), v {scale['v_min']:.4f}–{scale['v_max']:.4f} % "
                          f"(mean {scale['v_mean']:.4f})."]
    else:
        lines.append("No common sample (the boost run did not reach the mixed steps).")
    lines += ["", "## 3. Against Marchi 2021 (relative L2 %, Linf in lid units)", "", "| | t | samples | L2 u | L2 v | Linf u | Linf v |", "|---|---|---|---|---|---|---|"]
    newest = marchi["newest_mixed"]
    lines.append(f"| mixed, newest sample | {newest['t']:.3f} | 1 | {newest['L2u_pct']:.3f} | {newest['L2v_pct']:.3f} | | |")
    rows = [("mixed, all samples", marchi["window"]["t_range"], marchi["window"]["mixed"])]
    if "window_common" in marchi:
        rows += [("mixed, common steps", marchi["window_common"]["t_range"], marchi["window_common"]["mixed"]),
                 ("2×5090, same steps", marchi["window_common"]["t_range"], marchi["window_common"]["boost"])]
    for label, t_range, values in rows:
        lines.append(f"| {label} | {t_range[0]:.2f}–{t_range[1]:.2f} | {values['samples']} | {values['L2u_pct']:.3f} | {values['L2v_pct']:.3f} | "
                     f"{values['Linf_u']:.2e} | {values['Linf_v']:.2e} |")
    lines += ["", "## 4. Extrema of the window-mean dense lines vs Marchi 2021 Table 27", "",
              "| | value (Marchi) | position (Marchi) | " + " | ".join(f"{label} value | {label} position" for label, _, _ in rows) + " |",
              "|---|---|---|" + "---|---|" * len(rows)]
    for name, values in record["extrema_reference"].items():
        cells = [f"{data[name]:.5f} ({100 * (data[name] - values['value']) / abs(values['value']):+.2f} %) | {data[f'{name}_position']:.4f} "
                 f"({data[f'{name}_position'] - values['position']:+.4f})" for _, _, data in rows]
        lines.append(f"| {name} | {values['value']:.5f} | {values['position']:.4f} | " + " | ".join(cells) + " |")
    lines += ["", "## 5. Window tests of the mixed run after the start", ""]
    lines += [f"- t {row['time']:.2f}: profile {row['profile_change']:.3e}, KE {row['kinetic_energy_change']:.3e}" for row in record["windows_after_start"]] or ["- none yet"]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot(out: pathlib.Path, table: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    blue, orange, ink, muted, grid, surface = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    common = [entry for entry in table if "diff_u_pct" in entry]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150, facecolor=surface)
    t = [entry["t"] for entry in table]
    axes[0].plot(t, [entry["mixed_L2u_pct"] for entry in table], "-o", color=blue, markersize=3, linewidth=1.6, label="u, 5090 + XTX")
    axes[0].plot(t, [entry["mixed_L2v_pct"] for entry in table], "-o", color=orange, markersize=3, linewidth=1.6, label="v, 5090 + XTX")
    if common:
        tc = [entry["t"] for entry in common]
        axes[0].plot(tc, [entry["boost_L2u_pct"] for entry in common], "--", color=blue, linewidth=1.2, label="u, 2×5090")
        axes[0].plot(tc, [entry["boost_L2v_pct"] for entry in common], "--", color=orange, linewidth=1.2, label="v, 2×5090")
        axes[1].plot(tc, [entry["diff_u_pct"] for entry in common], "-o", color=blue, markersize=3, linewidth=1.6, label="u")
        axes[1].plot(tc, [entry["diff_v_pct"] for entry in common], "-o", color=orange, markersize=3, linewidth=1.6, label="v")
    axes[0].set_title("relative L2 against Marchi 2021 (%)", fontsize=10, color=ink, loc="left")
    axes[1].set_title("same step: (5090 + XTX) − 2×5090, Marchi norm (%)", fontsize=10, color=ink, loc="left")
    for axis in axes:
        axis.set_facecolor(surface); axis.grid(True, color=grid, linewidth=0.6); axis.set_xlabel("t", color=ink)
        axis.tick_params(colors=muted); axis.legend(frameon=False, fontsize=8)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color(muted)
    axes[1].set_ylim(bottom=0)
    figure.tight_layout()
    figure.savefig(out / "same_step.png", facecolor=surface)
    plt.close(figure)


if __name__ == "__main__":
    sys.exit(main())
