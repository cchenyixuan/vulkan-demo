"""
e31_analysis.py — E31 tables and figures (docs/perf_model/E31.md) from logs/e31/campaign (e31_campaign.py),
logs/e31/cut_balance (cut_balance.py) and the E29 scan (old cuts).

  eta      per case and trial: K = 1 GPU 0 / 1 and K = 2 fps (full precision), eta = fps_K2 / (2 mean K1),
           eta_min = fps_K2 / (2 min K1); mean +- sample std over the trials; E29's single-trial eta (old cuts,
           trace off and on) next to it
  wait     3-D narrow, 3-D 8M, 2-D 1M with the E31 cuts (trace/<case>/k2) against E29 (old cuts): per sim own
           columns, n_B, T_B p50, B -> C wait p50 / mean, period mean; worse-link exposed steps
  phasec   per sim, the phase C segments in queue order (tick to tick: barrier + launch + kernel or copy) p50 /
           p95 with the dispatch size of the kernel (groups, threads: the simulator's own dispatch-size methods
           on the same chain partition) and its active work (particles in the band, from the warmup voxel counts);
           the sum of the segment p50s against the phase C p50; phase B interior sweeps against the band sweeps,
           per particle
  nowait   K = 2 fps with V7_PHASE_A_NO_WAIT = 0 / 1 per trial, mean +- std, ratio; C -> A gap p50 / p95 from the
           traced runs

    .venv/Scripts/python.exe -m experiment.v7.analysis.e31_analysis --campaign logs/e31/campaign \\
        --out docs/perf_model/e31
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiment.v7.analysis import step_trace_model as model   # noqa: E402
from experiment.v7.analysis.e31_campaign import CASES           # noqa: E402

E29_SCAN = _REPO_ROOT / "logs/e29_step_trace/scan"
E29_SUMMARY = _REPO_ROOT / "docs/perf_model/e29/e29_summary.json"
PHASE_C_ORDER = ("c_start", "c_expand_end", "c_install_leading_end", "c_install_trailing_end",
                 "c_append_departed_end", "c_band_compact_end", "c_correction_boundary_end",
                 "c_density_boundary_end", "c_density_end", "c_force_end")
PHASE_B_ORDER = ("b_start", "b_correction_interior_end", "b_density_deep_interior_end", "b_force_deep_interior_end")
SEGMENT_NAMES = {"c_expand_end": "expand_ghost_lists", "c_install_leading_end": "install_migrations leading",
                 "c_install_trailing_end": "install_migrations trailing", "c_append_departed_end": "append_departed",
                 "c_band_compact_end": "band_compact", "c_correction_boundary_end": "correction_boundary",
                 "c_density_boundary_end": "density_boundary", "c_density_end": "density copy (scratch -> primary)",
                 "c_force_end": "force_boundary",
                 "b_correction_interior_end": "correction_interior", "b_density_deep_interior_end": "density_deep_interior",
                 "b_force_deep_interior_end": "force_deep_interior"}


def records(campaign: pathlib.Path) -> list:
    return model.read_results(campaign)


def fps(record: dict, campaign: pathlib.Path) -> float:
    return model.precise_fps(record, campaign)


# ------------------------------------------------------------------ eta

def eta_table(campaign: pathlib.Path) -> dict:
    rows = records(campaign)
    e29 = {case["case"]: case for case in json.loads(E29_SUMMARY.read_text(encoding="utf-8"))["cases"]}
    out = {}
    for name in dict.fromkeys(r["case"] for r in rows if r.get("mode") == "eta"):
        trials = []
        for trial in sorted({r["trial"] for r in rows if r.get("mode") == "eta" and r["case"] == name}):
            def pick(kind):
                return sorted((r for r in rows if r.get("mode") == "eta" and r["case"] == name
                               and r["trial"] == trial and r["kind"] == kind), key=lambda r: r.get("gpu", 0))
            k1, k2 = pick("k1"), pick("k2")
            if len(k1) != 2 or len(k2) != 1:
                continue
            k1_fps, k2_fps = [fps(r, campaign) for r in k1], fps(k2[0], campaign)
            trials.append({"trial": trial, "k1": k1_fps, "k2": k2_fps,
                           "eta": k2_fps / (2 * np.mean(k1_fps)), "eta_min": k2_fps / (2 * min(k1_fps)),
                           "k1_spread": (max(k1_fps) - min(k1_fps)) / np.mean(k1_fps),
                           "drift": [r.get("drift") for r in k1 + k2], "overflow": [r.get("overflow_total") for r in k1 + k2]})
        etas = np.array([t["eta"] for t in trials])
        mins = np.array([t["eta_min"] for t in trials])
        k2s = np.array([t["k2"] for t in trials])
        out[name] = {"trials": trials, "eta_mean": float(etas.mean()), "eta_std": float(etas.std(ddof=1)),
                     "eta_min_mean": float(mins.mean()), "eta_min_std": float(mins.std(ddof=1)),
                     "k2_mean": float(k2s.mean()), "k2_std": float(k2s.std(ddof=1)),
                     "k1_mean": [float(np.mean([t["k1"][g] for t in trials])) for g in (0, 1)],
                     "e29": {key: e29[name][key] for key in ("eta", "eta_min", "eta_traced", "eta_min_traced", "fps_k1",
                                                             "fps_k2", "fps_k1_traced", "fps_k2_traced")}
                     if name in e29 else None}
    return out


# ------------------------------------------------------------------ waits (old vs new cuts)

def sim_waits(run: dict) -> list:
    meta, device = run["meta"], run["device"]
    out = []
    cycles = {item["sim"]: item for item in model.cycle_components(run)}
    for sim in sorted(set(device["sim"].astype(int))):
        rows = model.select(device, (device["sim"] == sim) & (device["step"] >= meta["warmup"]) & (device["complete"] == 1))
        wait = (rows["c_start"] - rows["b_end"]) / 1e3
        slab = meta["slabs"]["per_sim"][sim]
        out.append({"sim": sim, "own_columns": slab["own_columns"], "n_B": slab["n_B"],
                    "own_particles": slab["own_particles"],
                    "T_B_p50_us": float(np.median((rows["b_end"] - rows["b_start"]) / 1e3)),
                    "b_to_c_p50_us": float(np.median(wait)), "b_to_c_mean_us": float(np.mean(wait)),
                    "b_to_c_p95_us": float(np.percentile(wait, 95)),
                    "period_mean_us": cycles[sim]["period_mean"], "period_p50_us": cycles[sim]["period"]})
    return out


def wait_table(campaign: pathlib.Path) -> dict:
    out = {}
    for name in ("3d_narrow", "3d_8m", "2d_1m", "2d_10k"):
        new_dir = campaign / "runs" / f"trace__{name}__k2"
        old_dir = E29_SCAN / "runs" / f"{name}__k2__on"
        if not new_dir.exists():
            continue
        entry = {}
        for label, path in (("e31", new_dir), ("e29", old_dir)):
            if not path.exists():
                continue
            run = model.load_run(path)
            summary = model.run_summary(run)
            entry[label] = {"sims": sim_waits(run), "exposed_steps_fraction": summary["exposed_steps_fraction"],
                            "exposure_mean_us": summary["exposure_mean_us"], "r_chain_worst_p50": summary["r_chain_worst_p50"],
                            "steady_fps": run["meta"]["result"]["steady_fps"], "steps": run["meta"]["max_steps"],
                            "abs_phase_offset_p50": [link["phase_offset_abs_ratio_p50"] for link in summary["links"].values()]}
        out[name] = entry
    return out


# ------------------------------------------------------------------ phase C breakdown

def segments(run: dict, order) -> dict:
    """Per sim: {end label: (p50, p95) us} of tick-to-tick segments in queue order, plus the phase total."""
    meta, device = run["meta"], run["device"]
    out = {}
    for sim in sorted(set(device["sim"].astype(int))):
        rows = model.select(device, (device["sim"] == sim) & (device["step"] >= meta["warmup"]) & (device["complete"] == 1))
        present = [label for label in order if label in rows and np.isfinite(rows[label]).all()]
        entry = {"labels": present, "segments": {}}
        for previous, label in zip(present, present[1:]):
            values = (rows[label] - rows[previous]) / 1e3
            entry["segments"][label] = [float(np.median(values)), float(np.percentile(values, 95))]
        total = (rows[present[-1]] - rows[present[0]]) / 1e3
        entry["total"] = [float(np.median(total)), float(np.percentile(total, 95))]
        entry["sum_of_p50"] = float(sum(value[0] for value in entry["segments"].values()))
        out[sim] = entry
    return out


def dispatch_sizes(case_path: str, slab_count: int) -> list:
    """Per slab: groups and threads of every phase C dispatch, from SphSimulatorV7's own dispatch-size methods
    on the chain partition the bench builds (pool safety 1.2), without a device."""
    from experiment.v7.utils.case_loader_v7 import load_case_v7
    from experiment.v7.utils.partition_v7 import compute_chain_partition, configured_band_widths
    from experiment.v7.utils.simulator_v7 import _BAND_SLOT_LANES, SphSimulatorV7
    global_case = load_case_v7(str(_REPO_ROOT / case_path))
    chain = compute_chain_partition(global_case, [1.0] * slab_count, 1.2)
    correction_band, density_band, force_band = configured_band_widths()
    out = []
    for index, slab in enumerate(chain.slabs):
        sim = SphSimulatorV7.__new__(SphSimulatorV7)
        sim.case = slab
        sim._transport_segments = {direction: None for direction in ("leading", "trailing")
                                   if getattr(slab.transport, direction) is not None}
        workgroup = slab.capacities.workgroup_size
        groups = {
            "c_expand_end": sim._per_expand_dispatch_count(),
            "c_install_leading_end": sim._per_ghost_pid_dispatch_count("leading") if "leading" in sim._transport_segments else 0,
            "c_install_trailing_end": sim._per_ghost_pid_dispatch_count("trailing") if "trailing" in sim._transport_segments else 0,
            "c_append_departed_end": (sim._per_departed_dispatch_count()
                                      if slab.capacities.departed_pool_size > 0 and sim._transport_segments else 0),
            "c_correction_boundary_end": sim._per_band_dispatch_count(correction_band, sim._ghost_self_layer(2, 1, "correction")),
            "c_density_boundary_end": sim._per_band_dispatch_count(density_band, sim._ghost_self_layer(2, 1, "density")),
            "c_force_end": sim._per_band_dispatch_count(force_band),
        }
        out.append({"slab": index, "workgroup": workgroup, "lanes": _BAND_SLOT_LANES,
                    "max_particles_per_voxel": slab.capacities.max_particles_per_voxel,
                    "face": slab.grid.grid_dimension_y * slab.grid.grid_dimension_z,
                    "own_pool_size": slab.capacities.own_pool_size,
                    "ghost_pool": [slab.capacities.leading_ghost_pool_size, slab.capacities.trailing_ghost_pool_size],
                    "replica_region_size": slab.capacities.replica_region_size,
                    "departed_pool_size": slab.capacities.departed_pool_size,
                    "density_copy_bytes": slab.capacities.own_pool_size * 8,
                    "groups": groups, "threads": {key: value * workgroup for key, value in groups.items()}})
    return out


def band_work(meta: dict, sim: int, band: int, ghost_self: int) -> int:
    """Own particles in the `band` columns next to every seam of `sim` (warmup voxel counts), plus the peer's
    adjacent own column (the inner ghost column) when ghost_self = 1."""
    slabs = meta["slabs"]["per_sim"]
    columns = slabs[sim]["own_column_particles"]
    total = 0
    if sim > 0:                                   # leading seam
        total += sum(columns[:band]) + (slabs[sim - 1]["own_column_particles"][-1] if ghost_self else 0)
    if sim < len(slabs) - 1:                      # trailing seam
        total += sum(columns[-band:]) + (slabs[sim + 1]["own_column_particles"][0] if ghost_self else 0)
    return int(total)


def phasec_table(campaign: pathlib.Path, load_dispatch: bool = True) -> dict:
    from experiment.v7.utils.partition_v7 import configured_band_widths
    correction_band, density_band, force_band = configured_band_widths()
    out = {}
    for name in ("2d_10k", "2d_1m", "3d_narrow", "3d_8m"):
        entry = {}
        for kind, directory in (("k2", campaign / "runs" / f"trace__{name}__k2"),
                                ("k1", campaign / "runs" / f"phasec__{name}__k1")):
            if not directory.exists():
                continue
            run = model.load_run(directory)
            meta = run["meta"]
            item = {"phase_c": segments(run, PHASE_C_ORDER), "phase_b": segments(run, PHASE_B_ORDER),
                    "slabs": meta["slabs"]["per_sim"] if meta.get("slabs") else None}
            if kind == "k2":
                work = {}
                for sim in range(len(meta["slabs"]["per_sim"])):
                    slab = meta["slabs"]["per_sim"][sim]
                    work[sim] = {"correction_boundary": band_work(meta, sim, correction_band, 1),
                                 "density_boundary": band_work(meta, sim, density_band, 1),
                                 "force_boundary": band_work(meta, sim, force_band, 0),
                                 "correction_interior": slab["n_correction_interior"],
                                 "density_deep_interior": slab["n_density_deep_interior"],
                                 "force_deep_interior": slab["n_B"]}
                item["work"] = work
                if load_dispatch:
                    item["dispatch"] = dispatch_sizes(CASES[name][0], 2)
            else:
                slab = meta["slabs"]["per_sim"][0]
                item["work"] = {0: {"correction_interior": slab["n_correction_interior"],
                                    "density_deep_interior": slab["n_density_deep_interior"],
                                    "force_deep_interior": slab["n_B"]}}
            entry[kind] = item
        out[name] = entry
    return out


# ------------------------------------------------------------------ nowait

def nowait_table(campaign: pathlib.Path) -> dict:
    rows = [r for r in records(campaign) if r.get("mode") == "nowait"]
    out = {}
    for name in dict.fromkeys(r["case"] for r in rows):
        untraced = [r for r in rows if r["case"] == name and not r.get("trace")]
        per = {}
        for setting in (0, 1):
            values = [fps(r, campaign) for r in sorted(untraced, key=lambda r: r["trial"]) if r["phase_a_no_wait"] == setting]
            per[setting] = {"fps": values, "mean": float(np.mean(values)), "std": float(np.std(values, ddof=1))
                            if len(values) > 1 else float("nan")}
        pairs = []
        for trial in sorted({r["trial"] for r in untraced}):
            values = {r["phase_a_no_wait"]: fps(r, campaign) for r in untraced if r["trial"] == trial}
            if 0 in values and 1 in values:
                pairs.append(values[1] / values[0] - 1.0)
        traced = {}
        for setting in (0, 1):
            directory = campaign / "runs" / f"nowait__{name}__trace__w{setting}"
            if directory.exists():
                run = model.load_run(directory)
                gaps = model.c_to_a_gap(run)
                cycles = {item["sim"]: item for item in model.cycle_components(run)}
                traced[setting] = {"c_to_a": {sim: [float(np.median(v)), float(np.percentile(v, 95)), float(np.mean(v))]
                                              for sim, v in gaps.items()},
                                   "a_to_b": {sim: cycles[sim]["a_to_b"] for sim in cycles},
                                   "period_mean": {sim: cycles[sim]["period_mean"] for sim in cycles},
                                   "steady_fps": run["meta"]["result"]["steady_fps"]}
        out[name] = {"settings": per, "pair_ratios": pairs,
                     "ratio_mean": float(np.mean(pairs)) if pairs else float("nan"),
                     "ratio_std": float(np.std(pairs, ddof=1)) if len(pairs) > 1 else float("nan"), "traced": traced}
    return out


# ------------------------------------------------------------------ figures

LABELS = {"2d_10k": "2-D 10k", "2d_1m": "2-D 1M", "3d_narrow": "3-D narrow", "3d_8m": "3-D 8M"}


def figures(result: dict, out: pathlib.Path) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    paths = []

    # 1. eta: E29 (pre-E31 cuts, one trial, trace off and on) against E31 (3 trials)
    eta = result["eta"]
    figure, axis = plt.subplots(figsize=(6.6, 4.2))
    for position, name in enumerate(eta):
        item = eta[name]
        values = [100 * t["eta"] for t in item["trials"]]
        axis.plot([position + 0.12] * len(values), values, "o", color="tab:blue", ms=5, alpha=0.7)
        axis.errorbar(position + 0.12, 100 * item["eta_mean"], yerr=100 * item["eta_std"], fmt="_", color="tab:blue",
                      ms=16, capsize=4, label="E31 cuts: trials, mean +- std" if position == 0 else None)
        axis.plot(position + 0.12, 100 * item["eta_min_mean"], "v", color="tab:blue", mfc="none", ms=6,
                  label="E31 eta_min (mean)" if position == 0 else None)
        if item["e29"]:
            axis.plot([position - 0.12] * 2, [100 * item["e29"]["eta"], 100 * item["e29"]["eta_traced"]], "s",
                      color="tab:orange", ms=5, label="E29 (old cuts): trace off / on" if position == 0 else None)
    axis.set_xticks(range(len(eta)), [LABELS.get(name, name) for name in eta])
    axis.set_ylabel("eta = fps_K2 / (2 mean fps_K1)  (%)")
    axis.grid(alpha=0.3, axis="y")
    axis.legend(fontsize=8)
    axis.set_title("K = 2 efficiency before and after the cut fix")
    figure.tight_layout()
    paths.append(out / "e31_eta.png")
    figure.savefig(paths[-1], dpi=150)
    plt.close(figure)

    # 2. per sim: n_B and the B -> C wait, E29 against E31
    wait = result["wait"]
    names = [name for name in ("3d_narrow", "3d_8m", "2d_1m") if name in wait and "e29" in wait[name]]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    for position, name in enumerate(names):
        for offset, label, color in ((-0.18, "e29", "tab:orange"), (0.18, "e31", "tab:blue")):
            sims = wait[name][label]["sims"]
            for sim in sims:
                shift = position + offset + (sim["sim"] - 0.5) * 0.14
                axes[0].bar(shift, sim["n_B"] / 1e6, width=0.13, color=color, alpha=0.5 + 0.4 * sim["sim"])
                axes[1].bar(shift, sim["b_to_c_mean_us"] / 1e3, width=0.13, color=color, alpha=0.5 + 0.4 * sim["sim"])
    for axis, ylabel in ((axes[0], "n_B per sim (millions)"), (axes[1], "B -> C wait per step, mean (ms)")):
        axis.set_xticks(range(len(names)), [LABELS[name] for name in names])
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.3, axis="y")
    axes[0].set_title("orange: E29 (old cuts), blue: E31; light s0, dark s1")
    axes[1].set_title("the faster side's wait before phase C")
    figure.tight_layout()
    paths.append(out / "e31_wait.png")
    figure.savefig(paths[-1], dpi=150)
    plt.close(figure)

    # 3. phase C segments (K = 2 s0, and the K = 1 run), stacked
    phasec = result["phasec"]
    segment_order = ["c_expand_end", "c_install_leading_end", "c_install_trailing_end", "c_append_departed_end",
                     "c_correction_boundary_end", "c_density_boundary_end", "c_density_end", "c_force_end"]
    colors = dict(zip(segment_order, ["#9e9e9e", "#bdbdbd", "#bdbdbd", "#e0e0e0", "tab:green", "tab:olive",
                                      "tab:purple", "tab:red"]))
    bars = []
    for name in ("2d_10k", "2d_1m", "3d_narrow", "3d_8m"):
        for kind in ("k1", "k2"):
            if name in phasec and kind in phasec[name]:
                sims = phasec[name][kind]["phase_c"]
                sim = "0" if "0" in sims else 0
                bars.append((f"{LABELS[name]}\n{'K = 1' if kind == 'k1' else 'K = 2 s0'}", sims[sim]["segments"]))
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1, 1]})
    for axis, subset in ((axes[0], [b for b in bars if b[0].startswith("2-D")]),
                         (axes[1], [b for b in bars if b[0].startswith("3-D")])):
        for position, (label, values) in enumerate(subset):
            bottom = 0.0
            for key in segment_order:
                if key in values:
                    scale = 1.0 if label.startswith("2-D") else 1e-3
                    height = values[key][0] * scale
                    axis.bar(position, height, bottom=bottom, color=colors[key], width=0.6,
                             label=SEGMENT_NAMES[key] if position == 0 or key not in [k for _, v in subset[:position] for k in v] else None)
                    bottom += height
        axis.set_xticks(range(len(subset)), [label for label, _ in subset], fontsize=8)
        axis.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("phase C segment p50 (us)")
    axes[1].set_ylabel("phase C segment p50 (ms)")
    handles, labels = [], []
    for axis in axes:
        for handle, label in zip(*axis.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    axes[1].legend(handles, labels, fontsize=7, loc="upper center")
    axes[0].set_title("2-D: phase C per step, tick to tick")
    axes[1].set_title("3-D")
    figure.tight_layout()
    paths.append(out / "e31_phasec.png")
    figure.savefig(paths[-1], dpi=150)
    plt.close(figure)

    # 4. NO_WAIT: fps per trial and the C -> A gap
    nowait = result["nowait"]
    if nowait:
        figure, axes = plt.subplots(1, 2, figsize=(11, 4.0))
        for position, name in enumerate(nowait):
            settings = nowait[name]["settings"]
            reference = float(np.mean(settings["0"]["fps"]))       # the default (with the wait)
            for setting, color in ((0, "tab:orange"), (1, "tab:blue")):
                values = settings[str(setting)]["fps"]
                axes[0].plot([position + (setting - 0.5) * 0.3] * len(values), [v / reference * 100 for v in values], "o",
                             color=color, label=f"V7_PHASE_A_NO_WAIT={setting}" if position == 0 else None)
            traced = nowait[name]["traced"]
            for setting, color in ((0, "tab:orange"), (1, "tab:blue")):
                key = str(setting) if str(setting) in traced else setting
                if key in traced:
                    gaps = traced[key]["c_to_a"]
                    for sim_key, values in gaps.items():
                        axes[1].bar(position + (setting - 0.5) * 0.3 + (int(sim_key) - 0.5) * 0.12, values[0], width=0.11,
                                    color=color, alpha=0.5 + 0.4 * int(sim_key))
        for axis in axes:
            axis.set_xticks(range(len(nowait)), [LABELS[name] for name in nowait])
            axis.grid(alpha=0.3, axis="y")
        axes[0].set_ylabel("K = 2 fps / mean with the wait (%)")
        axes[0].legend(fontsize=8)
        axes[1].set_ylabel("C -> A gap p50 (us); light s0, dark s1")
        axes[0].set_title("fps per trial (3 alternating pairs)")
        axes[1].set_title("C(n-1) end -> A(n) start, traced runs")
        figure.tight_layout()
        paths.append(out / "e31_nowait.png")
        figure.savefig(paths[-1], dpi=150)
        plt.close(figure)
    return paths


# ------------------------------------------------------------------ main

def jsonable(value):
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", default="logs/e31/campaign")
    parser.add_argument("--out", default="docs/perf_model/e31")
    parser.add_argument("--no-dispatch", action="store_true", help="skip the case loads for the dispatch sizes")
    arguments = parser.parse_args()
    campaign = _REPO_ROOT / arguments.campaign
    out = _REPO_ROOT / arguments.out
    out.mkdir(parents=True, exist_ok=True)
    result = {"eta": eta_table(campaign), "wait": wait_table(campaign),
              "phasec": phasec_table(campaign, not arguments.no_dispatch), "nowait": nowait_table(campaign)}
    cut_balance = _REPO_ROOT / "logs/e31/cut_balance/cut_balance.json"
    if cut_balance.exists():
        result["cut_balance"] = json.loads(cut_balance.read_text(encoding="utf-8"))
    result = jsonable(result)
    (out / "e31_summary.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"wrote {out / 'e31_summary.json'}")
    for path in figures(result, out):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
