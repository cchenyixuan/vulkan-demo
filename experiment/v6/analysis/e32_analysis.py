"""
e32_analysis.py — E32 tables and figures (docs/perf_model/E32.md).

  nowait     part 1: the gate rows of logs/e32/nowait_audit/summary.json (analyze.py, one =0 pair as the floor)
             and the ensemble view (ensemble.json): per configuration and horizon the largest cross / within-=0
             median ratio over fields and bins, the smallest permutation p, the gate's fail share for =0
             against =0 (null) and =1 against =0 (test)
  weights    part 2: every calibration record of the campaign (rounds: weights, cuts, fluid per slab, per-sim
             busy and per-device busy, omega, pilot fps)
  eta        part 2 (a): per case and trial K = 1 GPU 0 / 1, K = 2 equal and calibrated weights (full precision);
             eta = fps_K2 / (2 mean K1), eta_min = fps_K2 / (2 min K1); mean +- sample std; the paired
             calibrated - equal difference per trial
  wait       part 2 (a): per case and weighting, per sim the own columns, n_B, T_B p50 and the B -> C wait from
             the traced runs (e31_analysis.sim_waits)
  k3         part 2 (b): K = 3, two sims on GPU 0: fps equal / calibrated per trial, omega
  tracecost  diagnosis: 3-D 8M K = 1 on both GPUs at once, step trace off / on (does the instrument slow a card?)
  matched    diagnosis: 3-D 8M calibrated again and compared equal / calibrated right after, in one desktop state
  gpu0       GPU 0 speed / GPU 1 speed over the campaign (K = 1 pairs, pilots, traced runs) against the desktop
             state at every idle check (GPU 0 drives the display)
  walls      part 2 (c): cut_balance.walls_table (n_B with walls per slab at K = 2 / 4 / 8, end / interior)
  aligned    part 3: cut_balance.aligned_table (old / aligned cases)
Figures: ensemble pair distances, eta equal vs calibrated, per-sim B -> C waits, old imbalance per case.

    .venv/Scripts/python.exe -m experiment.v6.analysis.e32_analysis --campaign logs/e32/campaign \\
        --audit logs/e32/nowait_audit --out docs/perf_model/e32
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiment.v6.analysis import step_trace_model as model   # noqa: E402
from experiment.v6.analysis.e31_analysis import sim_waits       # noqa: E402

AUDIT_FIELDS = ("velocity", "acceleration", "shift", "density", "pressure", "kernel_sum")


# ------------------------------------------------------------------ part 1

def nowait_tables(audit: pathlib.Path) -> dict:
    summary = json.loads((audit / "summary.json").read_text(encoding="utf-8"))
    ensemble = json.loads((audit / "ensemble.json").read_text(encoding="utf-8"))
    gate = [{"configuration": row["configuration"], "horizon": row["horizon"], "worst": row["worst"],
             "pass": row["pass"], "problems": row["problems"],
             "column0_rms_ratio": (row.get("test") or {}).get("column0_rms_ratio"),
             "second_pair_d0_rms_ratio": (row.get("test") or {}).get("second_pair_d0_rms_ratio"),
             "null_column0_rms_ratio": (row.get("null") or {}).get("column0_rms_ratio"),
             "invariants_valid": all(item.get("valid") and not item.get("far_migration_count")
                                     for item in row["invariants"].values())}
            for row in summary["rows"]]
    view = []
    for configuration, horizons in ensemble.items():
        for horizon, entry in horizons.items():
            items = [(field, label, item) for field, bins in entry["fields"].items() for label, item in bins.items()]
            ratios = [(item["median"]["cross"] / item["median"]["within_0"], field, label)
                      for field, label, item in items if item["median"].get("within_0")]
            view.append({
                "configuration": configuration, "horizon": int(horizon), "runs": len(entry["runs"]),
                "max_cross_over_within0": max(ratios)[0], "max_at": list(max(ratios)[1:]),
                "min_cross_over_within0": min(ratios)[0],
                "tests": len(items),
                "cross_p_below_005": sum(item["permutation"]["cross_p"] < 0.05 for _, _, item in items),
                "one_p_below_005": sum(item["permutation"]["one_p"] < 0.05 for _, _, item in items),
                "min_cross_p": min(item["permutation"]["cross_p"] for _, _, item in items),
                "min_one_p": min(item["permutation"]["one_p"] for _, _, item in items),
                "gate_fail_null_d0": max(item["gate_rates"]["null"]["fail_fraction"] for _, label, item in items
                                         if label == "d0"),
                "gate_fail_test_d0": max(item["gate_rates"]["test"]["fail_fraction"] for _, label, item in items
                                         if label == "d0"),
                "velocity_far": entry["fields"]["velocity"]["far"]["median"],
                "density_d0": entry["fields"]["density"]["d0"]["median"]})
    return {"gate": gate, "ensemble": view}


# ------------------------------------------------------------------ part 2

def calibration_records(campaign: pathlib.Path) -> dict:
    out = {}
    for path in sorted((campaign / "weights").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        rounds = []
        for item in record["rounds"]:
            per_sim = item["statistics"]["per_sim"]
            rounds.append({"round": item["round"], "weights": item["weights"], "cuts": item["cuts"],
                           "own_columns": item["own_columns"], "fluid": item["fluid"], "all": item["all"],
                           "busy_sim_p50_us": [sim["busy_p50_us"] for sim in per_sim],
                           "busy_sim_p95_us": [sim["busy_p95_us"] for sim in per_sim],
                           "phase_p50_us": [[sim["a_p50_us"], sim["b_p50_us"], sim["c_p50_us"]] for sim in per_sim],
                           "busy_device_us": {str(device): value["busy_us"] for device, value
                                              in item["statistics"]["per_device"].items()},
                           "device_method": {str(device): value["method"] for device, value
                                             in item["statistics"]["per_device"].items()},
                           "device_union": {str(device): {key: value.get(key) for key in
                                                          ("union_us", "span_per_frame_us", "sim_medians_us")}
                                            for device, value in item["statistics"]["per_device"].items()
                                            if "union_us" in value},
                           "omega": item["omega"], "next_cuts": item["next_cuts"], "changed": item["changed"],
                           "pilot_steady_fps": item["pilot_steady_fps"], "seconds": item["seconds"]})
        out[path.stem] = {"weights": record["weights"], "cuts": record["cuts"], "device_map": record["device_map"],
                          "seconds": record["seconds"], "rounds": rounds, "host": record["host"],
                          "devices": record["devices"], "commit": record["commit"],
                          "settings": record["settings"]}
    return out


def eta_tables(campaign: pathlib.Path) -> dict:
    rows = model.read_results(campaign)
    out = {}
    for name in dict.fromkeys(row["case"] for row in rows if row.get("mode") == "eta"):
        trials = []
        for trial in sorted({row["trial"] for row in rows if row.get("mode") == "eta" and row["case"] == name}):
            def pick(arm):
                return sorted((row for row in rows if row.get("mode") == "eta" and row["case"] == name
                               and row["trial"] == trial and row.get("arm") == arm), key=lambda row: row.get("gpu", 0))
            k1, equal, auto = pick("k1"), pick("equal"), pick("auto")
            if len(k1) != 2 or len(equal) != 1 or len(auto) != 1:
                continue
            k1_fps = [model.precise_fps(row, campaign) for row in k1]
            equal_fps, auto_fps = model.precise_fps(equal[0], campaign), model.precise_fps(auto[0], campaign)
            trials.append({"trial": trial, "k1": k1_fps, "equal": equal_fps, "auto": auto_fps,
                           "eta_equal": equal_fps / (2 * np.mean(k1_fps)), "eta_auto": auto_fps / (2 * np.mean(k1_fps)),
                           "eta_min_equal": equal_fps / (2 * min(k1_fps)), "eta_min_auto": auto_fps / (2 * min(k1_fps)),
                           "auto_over_equal": auto_fps / equal_fps,
                           "invariants": [{key: row.get(key) for key in ("drift", "overflow_total", "stamp_errors",
                                                                          "far_migration_total")}
                                          for row in k1 + equal + auto]})
        entry = {"trials": trials}
        for key in ("eta_equal", "eta_auto", "eta_min_equal", "eta_min_auto", "auto_over_equal", "equal", "auto"):
            values = np.array([trial[key] for trial in trials])
            entry[f"{key}_mean"] = float(values.mean())
            entry[f"{key}_std"] = float(values.std(ddof=1)) if len(values) > 1 else None
        entry["k1_mean"] = [float(np.mean([trial["k1"][gpu] for trial in trials])) for gpu in (0, 1)]
        out[name] = entry
    return out


def wait_tables(campaign: pathlib.Path) -> dict:
    out = {}
    for path in sorted((campaign / "runs").glob("trace__*")):
        if not path.is_dir():
            continue
        _, name, arm = path.name.split("__")
        run = model.load_run(path)
        summary = model.run_summary(run)
        out.setdefault(name, {})[arm] = {
            "sims": sim_waits(run), "steady_fps": run["meta"]["result"]["steady_fps"],
            "weights": run["meta"]["weights"],
            "cuts": [slab["own_global_first_column"] for slab in run["meta"]["partition"]][1:],
            "exposed_steps_fraction": summary["exposed_steps_fraction"],
            "exposure_mean_us": summary["exposure_mean_us"], "r_chain_worst_p50": summary["r_chain_worst_p50"]}
    return out


def k3_table(campaign: pathlib.Path) -> dict:
    rows = [row for row in model.read_results(campaign) if row.get("mode") == "k3"]
    out = {"trials": []}
    for trial in sorted({row["trial"] for row in rows}):
        entry = {"trial": trial}
        for arm in ("equal", "auto"):
            match = [row for row in rows if row["trial"] == trial and row["arm"] == arm]
            if match:
                entry[arm] = model.precise_fps(match[0], campaign)
                entry[f"{arm}_invariants"] = {key: match[0].get(key) for key in
                                              ("drift", "overflow_total", "stamp_errors", "far_migration_total")}
        out["trials"].append(entry)
    for arm in ("equal", "auto"):
        values = [trial[arm] for trial in out["trials"] if arm in trial]
        if values:
            out[f"{arm}_mean"] = float(np.mean(values))
    if "equal_mean" in out and "auto_mean" in out:
        out["auto_over_equal"] = out["auto_mean"] / out["equal_mean"]
    return out


# ------------------------------------------------------------------ GPU 0 and the desktop

def desktop_awake(gpus) -> bool:
    """GPU 0 drives the desktop: idle above 30 W or 300 MHz = the display is awake (asleep: ~18 W, ~30 MHz)."""
    return float(gpus[0][3]) > 30.0 or float(gpus[0][4]) > 300.0


def campaign_starts(campaign: pathlib.Path) -> list:
    """(time, run id, GPU state) of every start line of campaign.log."""
    import datetime
    out = []
    for line in (campaign / "campaign.log").read_text(encoding="utf-8").splitlines():
        if " start " not in line or "gpu check " not in line:
            continue
        stamp = datetime.datetime.strptime(line[:19], "%Y-%m-%d %H:%M:%S")
        run_id = line[20:].split(" ", 2)[1].rstrip(":")
        state = json.loads(line.split("gpu check ", 1)[1])
        out.append((stamp, run_id, state))
    return out


def tracecost_table(campaign: pathlib.Path) -> dict:
    rows = [row for row in model.read_results(campaign) if row.get("mode") == "tracecost"]
    out = {"trials": []}
    for trial in sorted({row["trial"] for row in rows}):
        entry = {"trial": trial}
        for arm in ("off", "on"):
            pair = sorted((row for row in rows if row["trial"] == trial and row["arm"] == arm), key=lambda row: row["gpu"])
            if len(pair) == 2:
                entry[arm] = [model.precise_fps(row, campaign) for row in pair]
                entry[f"{arm}_desktop_awake"] = desktop_awake(pair[0]["gpu_check"]["gpus"])
        out["trials"].append(entry)
    for gpu in (0, 1):
        on = [trial["on"][gpu] for trial in out["trials"] if "on" in trial]
        off = [trial["off"][gpu] for trial in out["trials"] if "off" in trial]
        if on and off:
            out[f"gpu{gpu}_on_over_off"] = float(np.mean(on) / np.mean(off))
    return out


def matched_table(campaign: pathlib.Path) -> dict:
    rows = [row for row in model.read_results(campaign) if row.get("mode") == "matched"]
    out = {"trials": []}
    record_path = campaign / "weights" / "3d_8m_k2_matched.json"
    if record_path.exists():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        out["weights"], out["cuts"] = record["weights"], record["cuts"]
        out["rounds"] = [{"cuts": item["cuts"], "busy_us": item["busy_us"], "fluid": item["fluid"], "all": item["all"],
                          "omega": item["omega"], "next_cuts": item["next_cuts"],
                          "pilot_steady_fps": item["pilot_steady_fps"]} for item in record["rounds"]]
    for trial in sorted({row["trial"] for row in rows if "trial" in row}):
        entry = {"trial": trial}
        for arm in ("equal", "auto"):
            match = [row for row in rows if row.get("trial") == trial and row.get("arm") == arm]
            if match:
                entry[arm] = model.precise_fps(match[0], campaign)
                entry[f"{arm}_desktop_awake"] = desktop_awake(match[0]["gpu_check"]["gpus"])
        if "equal" in entry and "auto" in entry:
            entry["auto_over_equal"] = entry["auto"] / entry["equal"]
        out["trials"].append(entry)
    ratios = [trial["auto_over_equal"] for trial in out["trials"] if "auto_over_equal" in trial]
    if ratios:
        out["auto_over_equal_mean"] = float(np.mean(ratios))
        out["auto_over_equal_std"] = float(np.std(ratios, ddof=1)) if len(ratios) > 1 else None
    return out


def gpu0_timeline(campaign: pathlib.Path) -> dict:
    """GPU 0 speed / GPU 1 speed over the campaign from every source, and the desktop state at every start:
    K = 1 pairs (fps ratio), pilot rounds (per-particle busy, inverted), traced K = 2 runs (steady per-particle
    busy, inverted)."""
    import datetime
    starts = campaign_starts(campaign)
    origin = starts[0][0] if starts else None

    def minutes(stamp):
        return (stamp - origin).total_seconds() / 60.0
    states = [{"minute": minutes(stamp), "awake": desktop_awake(state["gpus"]), "power_w": float(state["gpus"][0][3]),
               "clock_mhz": float(state["gpus"][0][4]), "run": run_id} for stamp, run_id, state in starts]
    points = []
    started = {run_id: stamp for stamp, run_id, _ in starts}
    rows = model.read_results(campaign)
    for row in rows:
        if row.get("kind") != "k1" or row.get("gpu") != 0:
            continue
        partner = next((other for other in rows if other["run_id"] == row["run_id"][:-1] + "1"), None)
        pair_id = row["run_id"].rsplit("/", 1)[0]
        if partner is None or pair_id not in started:
            continue
        points.append({"minute": minutes(started[pair_id]), "source": f"K=1 pair ({row['case']})",
                       "ratio": model.precise_fps(row, campaign) / model.precise_fps(partner, campaign), "run": pair_id})
    for path in sorted((campaign / "weights").glob("*_k2*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        stamp = datetime.datetime.strptime(record["timestamp"], "%Y-%m-%d %H:%M:%S")
        for item in record["rounds"]:
            sims = item["statistics"]["per_sim"]
            ratio = (sims[1]["busy_p50_us"] / item["all"][1]) / (sims[0]["busy_p50_us"] / item["all"][0])
            points.append({"minute": minutes(stamp), "source": "pilot (K=2)", "ratio": ratio,
                           "run": f"{path.stem} round {item['round']}"})
    for path in sorted((campaign / "runs").glob("trace__3d_*")):
        if not path.is_dir():
            continue
        run = model.load_run(path)
        device, meta = run["device"], run["meta"]
        own = [slab["own_particle_count"] for slab in meta["partition"]]
        per_particle = []
        for sim in (0, 1):
            rows_sim = model.select(device, (device["sim"] == sim) & (device["complete"] == 1)
                                    & (device["step"] >= meta["warmup"]))
            busy = ((rows_sim["a_end"] - rows_sim["a_start"]) + (rows_sim["b_end"] - rows_sim["b_start"])
                    + (rows_sim["c_end"] - rows_sim["c_start"]))
            per_particle.append(float(np.median(busy)) / own[sim])
        run_id = path.name.replace("__", "/")
        if run_id in started:
            points.append({"minute": minutes(started[run_id]), "source": "traced K=2 run",
                           "ratio": per_particle[1] / per_particle[0], "run": run_id})
    return {"origin": origin.strftime("%H:%M:%S") if origin else None, "states": states, "points": points}


# ------------------------------------------------------------------ figures

def figures(result: dict, audit: pathlib.Path, out: pathlib.Path) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    paths = []

    # 1: ensemble pair distances (velocity rms, every pair, by class)
    ensemble = json.loads((audit / "ensemble.json").read_text(encoding="utf-8"))
    panels = [(configuration, horizon) for configuration, horizons in ensemble.items() for horizon in horizons]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
    colors = {"within_0": "#1f77b4", "within_1": "#d62728", "cross": "#7f7f7f"}
    labels = {"within_0": "=0 vs =0", "within_1": "=1 vs =1", "cross": "=0 vs =1"}
    jitter = np.random.default_rng(32)
    for axis, bin_label in zip(axes, ("d0", "far")):
        labelled = set()
        for index, (configuration, horizon) in enumerate(panels):
            pairs = ensemble[configuration][horizon]["fields"]["velocity"][bin_label]["pairs"]
            for pair, value in pairs.items():
                first, second = pair.split("-")
                kind = ("within_0" if first[1] == second[1] == "0" else
                        "within_1" if first[1] == second[1] == "1" else "cross")
                offset = {"within_0": -0.22, "cross": 0.0, "within_1": 0.22}[kind]
                axis.scatter(index + offset + jitter.uniform(-0.07, 0.07), value, s=9, color=colors[kind], alpha=0.75,
                             label=None if kind in labelled else labels[kind])
                labelled.add(kind)
        axis.set_yscale("log")
        axis.set_xticks(range(len(panels)))
        axis.set_xticklabels([f"{c.replace('cavity', '').replace('_1m', ' 1M').replace('_k', ' K=')}\nN={h}"
                              for c, h in panels], fontsize=7)
        axis.set_ylabel("rms |Δv| between two runs (m/s)")
        axis.set_title({"d0": "seam columns (d = 0)", "far": "far from the seam (d ≥ 8)"}[bin_label], fontsize=10)
        axis.grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=8, loc="upper left")
    figure.suptitle("E32 part 1: depth-2 restarts, every pair of 12 runs (6 × NO_WAIT=0, 6 × =1)", fontsize=10)
    figure.tight_layout()
    path = out / "fig1_nowait_pairs.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    paths.append(path)

    # 2: eta equal vs calibrated (+ K = 3)
    eta = result["eta"]
    names = list(eta)
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [3, 1.3]})
    x = np.arange(len(names))
    for offset, arm, color in ((-0.18, "equal", "#7f7f7f"), (0.18, "auto", "#2ca02c")):
        means = [100 * eta[name][f"eta_{arm}_mean"] for name in names]
        errors = [100 * (eta[name][f"eta_{arm}_std"] or 0) for name in names]
        axes[0].bar(x + offset, means, 0.34, yerr=errors, capsize=3, color=color,
                    label={"equal": "equal weights", "auto": "calibrated (--weights-file)"}[arm])
        for position, value in zip(x + offset, means):
            axes[0].text(position, value + 0.6, f"{value:.1f}", ha="center", fontsize=7)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([name.replace("_", " ") for name in names])
    axes[0].set_ylabel("η (%)  = fps_K2 / (2 · mean fps_K1)")
    axes[0].set_ylim(60, 100)
    axes[0].legend(fontsize=8, loc="lower right")
    axes[0].set_title("K = 2, 3 alternating trials, mean ± std", fontsize=9)
    k3 = result["k3"]
    if "equal_mean" in k3:
        axes[1].bar([0, 1], [k3["equal_mean"], k3["auto_mean"]], color=["#7f7f7f", "#2ca02c"])
        for position, value in zip((0, 1), (k3["equal_mean"], k3["auto_mean"])):
            axes[1].text(position, value + 2, f"{value:.1f}", ha="center", fontsize=8)
        axes[1].set_xticks([0, 1])
        axes[1].set_xticklabels(["equal", "calibrated"])
        axes[1].set_ylabel("fps")
        axes[1].set_title("K = 3, map 0,0,1 (2-D 4M)", fontsize=9)
    figure.tight_layout()
    path = out / "fig2_eta.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    paths.append(path)

    # 3: per-sim B -> C wait, equal vs calibrated (traced runs)
    waits = result["wait"]
    names = [name for name in result["eta"] if name in waits]
    figure, axes = plt.subplots(1, len(names), figsize=(3.6 * len(names), 3.4))
    for axis, name in zip(np.atleast_1d(axes), names):
        for offset, arm, color in ((-0.18, "equal", "#7f7f7f"), (0.18, "auto", "#2ca02c")):
            if arm not in waits[name]:
                continue
            sims = waits[name][arm]["sims"]
            axis.bar(np.arange(len(sims)) + offset, [sim["b_to_c_mean_us"] for sim in sims], 0.34, color=color,
                     label={"equal": "equal", "auto": "calibrated"}[arm])
        axis.set_xticks(range(2))
        axis.set_xticklabels(["s0 (GPU 0)", "s1 (GPU 1)"])
        axis.set_title(name.replace("_", " "), fontsize=9)
        axis.set_ylabel("B → C wait, mean (µs)")
        axis.grid(alpha=0.3, axis="y")
    np.atleast_1d(axes)[0].legend(fontsize=8)
    figure.tight_layout()
    path = out / "fig3_waits.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    paths.append(path)

    # 4: old imbalance per aligned case at K = 2 / 4 / 8 (new: 0 by construction where feasible)
    aligned = result["aligned"]
    names = list(aligned)
    figure, axis = plt.subplots(figsize=(12, 3.8))
    x = np.arange(len(names))
    for offset, slab_count, color in ((-0.27, 2, "#1f77b4"), (0.0, 4, "#ff7f0e"), (0.27, 8, "#d62728")):
        values = [100 * (aligned[name]["old"][f"K{slab_count}"] or 0) for name in names]
        axis.bar(x + offset, values, 0.26, color=color, label=f"old lattice, K = {slab_count}")
    axis.set_xticks(x)
    axis.set_xticklabels([name.replace("cavity", "") for name in names], rotation=70, fontsize=6.5)
    axis.set_ylabel("fluid max / mean − 1 (%)")
    axis.set_title("Equal-weight fluid imbalance of the old node lattices (E31 cut rule); every aligned case: 0.000 % "
                   "where the K is feasible", fontsize=9)
    axis.legend(fontsize=8)
    axis.grid(alpha=0.3, axis="y")
    figure.tight_layout()
    path = out / "fig4_old_imbalance.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    paths.append(path)

    # 5: GPU 0 speed relative to GPU 1 over the campaign, with the desktop state (GPU 0 drives the display)
    timeline = result["gpu0_timeline"]
    states = sorted(timeline["states"], key=lambda state: state["minute"])
    figure, axis = plt.subplots(figsize=(11, 3.9))
    for current, following in zip(states, states[1:] + [None]):
        if not current["awake"]:
            end = following["minute"] if following else current["minute"] + 1.0
            axis.axvspan(current["minute"], end, color="#dddddd", lw=0)
    markers = {"pilot (K=2)": ("D", "#d62728"), "traced K=2 run": ("s", "#2ca02c")}
    labelled = set()
    for point in timeline["points"]:
        source = point["source"]
        marker, color = markers.get(source, ("o", "#1f77b4" if "3d" in source else "#9467bd"))
        axis.scatter(point["minute"], point["ratio"], marker=marker, color=color, s=28,
                     label=None if source in labelled else source, zorder=3)
        labelled.add(source)
    axis.axhline(1.0, color="black", lw=0.8)
    axis.set_xlabel(f"minutes after {timeline['origin']} (grey: GPU 0 idle at ~18 W / ~30 MHz = display asleep)")
    axis.set_ylabel("GPU 0 speed / GPU 1 speed")
    axis.set_title("GPU 0 drives the desktop: its speed relative to GPU 1 follows the display state, not the "
                   "instrumentation", fontsize=9)
    axis.legend(fontsize=7, loc="lower left", ncol=3)
    axis.grid(alpha=0.3)
    figure.tight_layout()
    path = out / "fig5_gpu0_desktop.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    paths.append(path)
    return paths


def percent(value, digits=1) -> str:
    return "—" if value is None else f"{100 * value:.{digits}f}"


def markdown_tables(result: dict) -> str:
    """The tables of docs/perf_model/E32.md (pasted from here, so the numbers are the analysis' own)."""
    lines = ["## part 1: gate rows", "",
             "| 配置 | N | d = 0 比值 a / shift / v / ρ / p / ksum(测试) | 第二对 a / shift / ρ | =0 对 =0(w0_t3)a / shift / ρ | 最大 | 门 |",
             "|---|---|---|---|---|---|---|"]
    for row in result["nowait"]["gate"]:
        column0, second, null = row["column0_rms_ratio"], row["second_pair_d0_rms_ratio"], row["null_column0_rms_ratio"]
        lines.append(f"| {row['configuration']} | {row['horizon']} | "
                     + " / ".join(f"{column0[field]:.2f}" for field in ("acceleration", "shift", "velocity", "density",
                                                                      "pressure", "kernel_sum"))
                     + " | " + " / ".join(f"{second[field]:.2f}" for field in ("acceleration", "shift", "density"))
                     + " | " + " / ".join(f"{null[field]:.2f}" for field in ("acceleration", "shift", "density"))
                     + f" | {row['worst']:.2f} | {'通过' if row['pass'] else '不过'} |")
    lines += ["", "## part 1: ensemble", "",
              "| 配置 | N | 交叉 / =0 内(中位数比,全部场与分箱) | p < 0.05 交叉 / 含 =1 | 门不过的比例:=0 对 =0 / =1 对 =0(d = 0,最差的场) | 远处速度差中位数 =0 内 / =1 内 / 交叉 |",
              "|---|---|---|---|---|---|"]
    for row in result["nowait"]["ensemble"]:
        far = row["velocity_far"]
        lines.append(f"| {row['configuration']} | {row['horizon']} | {row['min_cross_over_within0']:.2f}–"
                     f"{row['max_cross_over_within0']:.2f} | {row['cross_p_below_005']} / {row['one_p_below_005']}"
                     f"(共 {row['tests']}) | {percent(row['gate_fail_null_d0'], 0)} % / {percent(row['gate_fail_test_d0'], 0)} % | "
                     f"{far['within_0']:.2g} / {far['within_1']:.2g} / {far['cross']:.2g} |")
    lines += ["", "## part 2: calibration", "",
              "| 算例 | 轮 | 权重 | 切点 | 流体 / slab | 忙时 / 卡 (µs) | ω | 下一切点 | pilot fps |", "|---|---|---|---|---|---|---|---|---|"]
    for name, record in result["weights"].items():
        for item in record["rounds"]:
            lines.append(f"| {name} | {item['round']} | {', '.join(f'{w:.4f}' for w in item['weights'])} | {item['cuts']} | "
                         f"{', '.join(f'{f:,}' for f in item['fluid'])} | "
                         f"{', '.join(f'{b:,.0f}' for b in item['busy_device_us'].values())} | "
                         f"{', '.join(f'{w:.4f}' for w in item['omega'])} | {item['next_cuts']} | {item['pilot_steady_fps']:.1f} |")
    lines += ["", "## part 2: eta", "",
              "| 算例 | 试验 | K = 1 GPU 0 / 1 | K = 2 等权重 | K = 2 校准 | η 等权重 | η 校准 | η_min 等权重 / 校准 | 校准 / 等权重 |",
              "|---|---|---|---|---|---|---|---|---|"]
    for name, entry in result["eta"].items():
        for trial in entry["trials"]:
            lines.append(f"| {name} | {trial['trial']} | {trial['k1'][0]:.2f} / {trial['k1'][1]:.2f} | {trial['equal']:.2f} | "
                         f"{trial['auto']:.2f} | {percent(trial['eta_equal'])} | {percent(trial['eta_auto'])} | "
                         f"{percent(trial['eta_min_equal'])} / {percent(trial['eta_min_auto'])} | "
                         f"{trial['auto_over_equal']:.4f} |")
        lines.append(f"| {name} | 均值 ± std | {entry['k1_mean'][0]:.2f} / {entry['k1_mean'][1]:.2f} | "
                     f"{entry['equal_mean']:.2f} ± {entry['equal_std']:.2f} | {entry['auto_mean']:.2f} ± {entry['auto_std']:.2f} | "
                     f"**{percent(entry['eta_equal_mean'])} ± {percent(entry['eta_equal_std'])}** | "
                     f"**{percent(entry['eta_auto_mean'])} ± {percent(entry['eta_auto_std'])}** | "
                     f"{percent(entry['eta_min_equal_mean'])} / {percent(entry['eta_min_auto_mean'])} | "
                     f"{entry['auto_over_equal_mean']:.4f} ± {entry['auto_over_equal_std']:.4f} |")
    lines += ["", "## part 2: waits", "",
              "| 算例 | 权重 | 切点 | sim | own 列 | n_B | T_B p50 (µs) | B → C 等待 p50 / 平均 / p95 (µs) | 周期平均 (µs) | fps |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for name, arms in result["wait"].items():
        for arm, entry in arms.items():
            for sim in entry["sims"]:
                lines.append(f"| {name} | {arm} | {entry['cuts']} | s{int(sim['sim'])} | {sim['own_columns']} | {sim['n_B']:,} | "
                             f"{sim['T_B_p50_us']:.0f} | {sim['b_to_c_p50_us']:.1f} / {sim['b_to_c_mean_us']:.1f} / "
                             f"{sim['b_to_c_p95_us']:.1f} | {sim['period_mean_us']:.0f} | {entry['steady_fps']:.2f} |")
    k3 = result["k3"]
    lines += ["", "## part 2: K = 3", "", "| 试验 | 等权重 fps | 校准 fps |", "|---|---|---|"]
    for trial in k3["trials"]:
        lines.append(f"| {trial['trial']} | {trial.get('equal', float('nan')):.2f} | {trial.get('auto', float('nan')):.2f} |")
    tracecost = result["tracecost"]
    lines += ["", "## diagnosis: step trace cost, 3-D 8M K = 1 (both GPUs at once)", "",
              "| 试验 | 关:GPU 0 / GPU 1 | 开:GPU 0 / GPU 1 | 桌面(关 / 开) |", "|---|---|---|---|"]
    for trial in tracecost["trials"]:
        lines.append(f"| {trial['trial']} | {trial['off'][0]:.3f} / {trial['off'][1]:.3f} | {trial['on'][0]:.3f} / "
                     f"{trial['on'][1]:.3f} | {'醒' if trial['off_desktop_awake'] else '睡'} / "
                     f"{'醒' if trial['on_desktop_awake'] else '睡'} |")
    matched = result["matched"]
    if matched.get("trials"):
        lines += ["", "## diagnosis: matched calibration, 3-D 8M K = 2", "",
                  f"weights {matched.get('weights')} cuts {matched.get('cuts')}; rounds "
                  + "; ".join(f"cuts {item['cuts']} busy {[round(value) for value in item['busy_us']]} -> "
                              f"{item['next_cuts']} (pilot {item['pilot_steady_fps']:.2f} fps)"
                              for item in matched.get("rounds", [])), "",
                  "| 试验 | 等权重 fps | 校准 fps | 校准 / 等权重 | 桌面(等 / 校) |", "|---|---|---|---|---|"]
        for trial in matched["trials"]:
            lines.append(f"| {trial['trial']} | {trial['equal']:.3f} | {trial['auto']:.3f} | {trial['auto_over_equal']:.4f} | "
                         f"{'醒' if trial['equal_desktop_awake'] else '睡'} / {'醒' if trial['auto_desktop_awake'] else '睡'} |")
        if "auto_over_equal_mean" in matched:
            lines.append(f"| 均值 ± std | | | {matched['auto_over_equal_mean']:.4f} ± {matched['auto_over_equal_std']:.4f} | |")
    lines += ["", "## GPU 0 / GPU 1 speed timeline", "", "| 分钟 | 来源 | GPU 0 / GPU 1 | 运行 |", "|---|---|---|---|"]
    for point in sorted(result["gpu0_timeline"]["points"], key=lambda item: item["minute"]):
        lines.append(f"| {point['minute']:.1f} | {point['source']} | {point['ratio']:.3f} | {point['run']} |")
    lines += ["", "## part 2 (c): walls (K = 8)", "",
              "| 算例 | n_B / slab(百万) | own 列 | 端 / 中间 n_B | 流体偏差 | n_B 偏差 |", "|---|---|---|---|---|---|"]
    for name, entry in result["walls"].items():
        item = entry["8"] if "8" in entry else entry[8]
        fluid = item["fluid"]
        lines.append(f"| {name} | {', '.join(f'{value / 1e6:.2f}' for value in item['n_B'])} | "
                     f"{', '.join(str(value) for value in item['own_columns'])} | {item['end_over_interior']:.3f} | "
                     f"{100 * (max(fluid) / np.mean(fluid) - 1):.2f} % | {100 * item['n_B_imbalance']:.2f} % |")
    lines += ["", "## part 3: aligned cases", "",
              "| 算例 | 用途 | 旧:流体 / 流体列 | 新:流体 / 总数 | 流体列 + 每侧墙列 | 流体变化 | 旧 K=2 / 4 / 8 偏差 (%) | 新 K=2 / 4 / 8 (%) | K = 1 显存 |",
              "|---|---|---|---|---|---|---|---|---|"]
    for name, entry in result["aligned"].items():
        old, new = entry["old"], entry["new"]
        lines.append(f"| `{name}` | {entry['role']} | {old['fluid']:,} / {old['fluid_columns']} | {new['fluid']:,} / "
                     f"{new['total']:,} | {new['fluid_columns']} + {new['wall_columns']} | {100 * entry['fluid_change']:+.1f} % | "
                     + " / ".join(percent(old[f'K{k}'], 2) for k in (2, 4, 8)) + " | "
                     + " / ".join(percent(new[f'K{k}'], 3) for k in (2, 4, 8)) + " | "
                     f"{entry['k1_gib']:.1f} GiB{'' if entry['k1_fits'] else ' **装不下**'} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", default="logs/e32/campaign")
    parser.add_argument("--audit", default="logs/e32/nowait_audit")
    parser.add_argument("--out", default="docs/perf_model/e32")
    parser.add_argument("--no-figures", action="store_true")
    arguments = parser.parse_args()
    from experiment.v6.analysis import cut_balance
    campaign, audit = _REPO_ROOT / arguments.campaign, _REPO_ROOT / arguments.audit
    out = _REPO_ROOT / arguments.out
    out.mkdir(parents=True, exist_ok=True)
    result = {"nowait": nowait_tables(audit), "weights": calibration_records(campaign), "eta": eta_tables(campaign),
              "wait": wait_tables(campaign), "k3": k3_table(campaign), "tracecost": tracecost_table(campaign),
              "matched": matched_table(campaign), "gpu0_timeline": gpu0_timeline(campaign),
              "walls": cut_balance.walls_table(), "aligned": cut_balance.aligned_table()}
    (out / "e32_summary.json").write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")
    (out / "e32_tables.md").write_text(markdown_tables(json.loads(json.dumps(result, default=float))), encoding="utf-8")
    print(f"wrote {out / 'e32_summary.json'} and e32_tables.md")
    if not arguments.no_figures:
        for path in figures(result, audit, out):
            print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
