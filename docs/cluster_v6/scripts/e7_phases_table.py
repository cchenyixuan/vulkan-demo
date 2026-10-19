"""e7_phases_table.py: the paper's tab:phases from the E7 batch-2 step traces (user, 2026-10-19).

For each case the K = 1 reference side (eight simultaneous K = 1 runs of the case behind one start barrier, step-traced,
labels reftrace_<short>_K1_g<i>: revision 8 job b2Z or revision 9 job b2AA) and the K = 8 chain side (the point's traced
run) under one rule set:
  - steady steps = the trace's steps >= warmup whose next step is present (500 .. 1498 of a 1500-step run), minus
    the defrag step (the defrag event at frame f sits in the period of step f - 1);
  - phase A / B / C and the step period: each GPU's median over its kept steps, then the mean of the GPUs
    (min-max alongside); T_A = a_end - a_start, T_B = b_end - b_start, T_C = c_end - c_start, period =
    a_start(n + 1) - a_start(n), GPU times on the host clock (step_trace_model);
  - the step mean two ways: "window" = (sum of the 999 steady periods - the in-window defrag wall time) / 999, the
    fps-without-defrag convention of the batch-2 tables; "skip" = the mean of the kept steps (the defrag step
    dropped);
  - a strong case's reference values are also given divided by K = 8 (the reference ran the whole case on one
    card); the weak per-card case (cavity2d_n1440 for 2d_n1440_k8) is not divided.
The chain side: with --chain-job (revision 9, job b2AA: the four K = 8 points traced again in the reference job's own
node and job, trace dirs <short>_K8_C / 2d_n1440_k8_K8_C) that job's own traces are the chain side and the original g2
chain traces (jobs b2B / b2V / b2S / b2Y) are listed as a second chain row; without it the g2 traces are the chain side
(revision 8, the reference job on g2 itself).
Ratio: reference step mean (strong / 8, weak not divided) / chain step mean, the window - defrag means (the median and
skip means alongside); it is compared with the point's eta on g2 from the batch-2 table (--eta-csv: eta without defrag,
mean over the trials, and eta with defrag): ratio - eta is printed, nothing else is done with it.
The reference groups' fps without defrag (the log's defrag_events line) is compared with the original untraced
reference runs of the same case on g2 (results.jsonl rows of the reference job, steady_fps_nodefrag): the group mean
against the original mean over its trials and cards, and card by card; a relative difference beyond --tolerance
(default 1 %) is flagged, nothing else is done with it.

Usage:
    python docs/cluster_v6/scripts/e7_phases_table.py --logs ../vulkan-demo/logs/n56 --reftrace-job e7_b2AA_<id>
        --chain-job e7_b2AA_<id> --out docs/cluster_v6/data/e7/batch2 [--tolerance 1.0] [--eta-csv <csv>]
Writes <out>/phases_table.md, phases_table.json and phases_table.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re
import statistics
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from experiment.v7.analysis import step_trace_model  # noqa: E402

# case, the paper's row, strong (divide the reference by K) or weak, the g2 chain trace (job, trace dir, log), the
# trace dir + log of the same point in a --chain-job, the original untraced K = 1 references on g2 (job, label pattern
# of the members) and the point's row of the batch-2 eta table (job, point)
CASES = (
    {"case": "cavity2d_n8000", "short": "2d_n8000", "paper": "strong 8000² (64M)", "kind": "strong", "slabs": 8,
     "chain": ("e7_b2B_1694328", "x_2d_n8000_K8_C", "x_2d_n8000_K8_trace_C.log"),
     "own_chain": ("2d_n8000_K8_C", "2d_n8000_K8_trace_C.log"),
     "references": ("e7_b2B_1694328", r"x_2d_n8000_K8_t\d+_R1_2d_n8000_g(\d+)"),
     "eta": ("e7_b2B_1694328", "x_2d_n8000_K8")},
    {"case": "cavity2d_n5640", "short": "2d_n5640", "paper": "strong 5640² (31.8M)", "kind": "strong", "slabs": 8,
     "chain": ("e7_b2V_1703604", "2d_n5640_K8_C", "2d_n5640_K8_trace_C.log"),
     "own_chain": ("2d_n5640_K8_C", "2d_n5640_K8_trace_C.log"),
     "references": ("e7_b2V_1703604", r"2d_n5640_K8_t\d+_R1_2d_n5640_g(\d+)"),
     "eta": ("e7_b2V_1703604", "2d_n5640_K8")},
    {"case": "cavity2d_n4000", "short": "2d_n4000", "paper": "strong 4000² (16M)", "kind": "strong", "slabs": 8,
     "chain": ("e7_b2S_1695411", "2d_n4000_K8_C", "2d_n4000_K8_trace_C.log"),
     "own_chain": ("2d_n4000_K8_C", "2d_n4000_K8_trace_C.log"),
     "references": ("e7_b2S_1695411", r"2d_n4000_K8_t\d+_R1_2d_n4000_g(\d+)"),
     "eta": ("e7_b2S_1695411", "2d_n4000_K8")},
    {"case": "cavity2d_n1440", "short": "2d_n1440", "paper": "weak 1440 × 1440 (2.14M per card)", "kind": "weak",
     "slabs": 8, "chain": ("e7_b2Y_1704741", "2d_n1440_k8_K8_C", "2d_n1440_k8_K8_trace_C.log"),
     "own_chain": ("2d_n1440_k8_K8_C", "2d_n1440_k8_K8_trace_C.log"),
     "references": ("e7_b2S_1695411", r"2d_n1440_k8_K8_t\d+_R1_2d_n1440_g(\d+)"),
     "eta": ("e7_b2S_1695411", "2d_n1440_k8_K8")},
)
DEFRAG_LINE = re.compile(r"defrag_event call=\d+ f(\d+) .*?wall_ms=([\d.]+)")
NODEFRAG_LINE = re.compile(r"defrag_events .*steady_fps=([\d.]+) steady_fps_nodefrag=([\d.]+)")
STATS = ("a", "b", "c", "period_median", "period_mean_window", "period_mean_skip")
RATIO_STATS = ("period_mean_window", "period_median", "period_mean_skip")
ETA_CSV = REPO / "docs" / "cluster_v6" / "data" / "e7" / "batch2" / "eta_r_b2_k8.csv"


def job_dir(logs: pathlib.Path, job: str) -> pathlib.Path:
    found = sorted(logs.glob(f"*/*_{job}"))
    if len(found) != 1:
        raise SystemExit(f"{job}: {len(found)} directories under {logs}")
    return found[0]


def job_node(job_directory: pathlib.Path):
    """NodeList of the pulled sacct.txt (None without one)."""
    sacct = job_directory / "sacct.txt"
    if not sacct.exists():
        return None
    lines = sacct.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        return None
    header, first = lines[0].split("|"), lines[1].split("|")
    return dict(zip(header, first)).get("NodeList")


def defrag_events(log: pathlib.Path, warmup: int, steps: int) -> list:
    """(frame, wall_ms) of the defrag events inside the steady window."""
    events = []
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        match = DEFRAG_LINE.search(line)
        if match and warmup < int(match.group(1)) < steps:
            events.append((int(match.group(1)), float(match.group(2))))
    return events


def fps_nodefrag(log: pathlib.Path) -> tuple:
    match = NODEFRAG_LINE.search(log.read_text(encoding="utf-8", errors="replace"))
    return (float(match.group(1)), float(match.group(2))) if match else (None, None)


def per_sim(run: dict, events: list) -> list:
    """Per sim (ms): the medians of A, B, C and the period over the kept steady steps, the two step means, the
    counts; every sim of the run (one for a K = 1 reference, K for a chain)."""
    warmup = run["meta"]["warmup"]
    device = run["device"]
    skip = {frame - 1 for frame, _ in events}
    defrag_ms = sum(wall for _, wall in events)
    out = []
    for sim in sorted(set(device["sim"].astype(int))):
        rows = step_trace_model.select(device, (device["sim"] == sim) & (device["complete"] == 1))
        order = np.argsort(rows["step"])
        rows = {key: value[order] for key, value in rows.items()}
        steps = rows["step"].astype(int)
        consecutive = np.concatenate([steps[1:] == steps[:-1] + 1, [False]])
        steady = (steps >= warmup) & consecutive
        kept = steady & ~np.isin(steps, list(skip))
        next_a = np.concatenate([rows["a_start"][1:], [np.nan]])
        period = (next_a - rows["a_start"]) / 1e6          # the trace's times are ns on the host clock: ms
        parts = {"a": (rows["a_end"] - rows["a_start"]) / 1e6, "b": (rows["b_end"] - rows["b_start"]) / 1e6,
                 "c": (rows["c_end"] - rows["c_start"]) / 1e6}
        entry = {"sim": int(sim), "steady_steps": int(steady.sum()), "kept_steps": int(kept.sum()),
                 "skipped_steps": sorted(int(step) for step in steps[steady & ~kept]),
                 "a": float(np.median(parts["a"][kept])), "b": float(np.median(parts["b"][kept])),
                 "c": float(np.median(parts["c"][kept])), "period_median": float(np.median(period[kept])),
                 "period_mean_window": float((period[steady].sum() - defrag_ms) / steady.sum()),
                 "period_mean_skip": float(np.mean(period[kept]))}
        out.append(entry)
    return out


def aggregate(entries: list, divide: int = 1) -> dict:
    """Mean over the GPUs of each per-GPU statistic (min-max alongside), divided by DIVIDE."""
    out = {"gpus": len(entries)}
    for name in STATS:
        values = [entry[name] / divide for entry in entries]
        out[name] = {"mean": statistics.mean(values), "min": min(values), "max": max(values)}
    return out


def reference_group(job_directory: pathlib.Path, short: str) -> dict:
    """The traced reference group reftrace_<short>_K1: one trace + log per member."""
    members = []
    for trace in sorted(job_directory.glob(f"traces/reftrace_{short}_K1_g*")):
        member = int(trace.name.rsplit("_g", 1)[1])
        log = job_directory / f"{trace.name}.log"
        run = step_trace_model.load_run(trace)
        events = defrag_events(log, run["meta"]["warmup"], run["meta"]["max_steps"])
        (entry,) = per_sim(run, events)
        fps, nodefrag = fps_nodefrag(log)
        members.append({"gpu": member, "trace": trace.name, "defrag_events": events, "fps": fps,
                        "fps_nodefrag": nodefrag, **entry})
    if not members:
        raise SystemExit(f"no traces reftrace_{short}_K1_g* under {job_directory / 'traces'}")
    return {"members": members, "raw": aggregate(members)}


def chain(job: str, job_directory: pathlib.Path, trace_name: str, log_name: str) -> dict:
    run = step_trace_model.load_run(job_directory / "traces" / trace_name)
    events = defrag_events(job_directory / log_name, run["meta"]["warmup"], run["meta"]["max_steps"])
    entries = per_sim(run, events)
    fps, nodefrag = fps_nodefrag(job_directory / log_name)
    return {"job": job, "node": job_node(job_directory), "trace": trace_name, "defrag_events": events, "fps": fps,
            "fps_nodefrag": nodefrag, "sims": entries, "values": aggregate(entries)}


def original_references(job_directory: pathlib.Path, pattern: str) -> dict:
    """steady_fps_nodefrag of the original untraced K = 1 reference runs: by card and all."""
    regex = re.compile(pattern)
    by_card: dict = {}
    for line in (job_directory / "results.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        match = regex.fullmatch(row["label"])
        if match and row.get("status") == "pass" and row.get("steady_fps_nodefrag") is not None:
            by_card.setdefault(int(match.group(1)), []).append((row["label"], float(row["steady_fps_nodefrag"])))
    values = [fps for entries in by_card.values() for _, fps in entries]
    return {"job": job_directory.name, "pattern": pattern, "runs": len(values),
            "mean": statistics.mean(values) if values else None,
            "by_card": {str(card): {"runs": len(entries), "mean": statistics.mean(fps for _, fps in entries)}
                        for card, entries in sorted(by_card.items())}}


def compare(group: dict, original: dict, tolerance: float) -> dict:
    new_values = [member["fps_nodefrag"] for member in group["members"]]
    new_mean = statistics.mean(new_values)
    out = {"new_mean": new_mean, "new_min": min(new_values), "new_max": max(new_values),
           "original_mean": original["mean"], "original_runs": original["runs"],
           "relative_difference_pct": 100.0 * (new_mean - original["mean"]) / original["mean"],
           "tolerance_pct": tolerance, "cards": []}
    out["beyond_tolerance"] = abs(out["relative_difference_pct"]) > tolerance
    for member in group["members"]:
        card = original["by_card"].get(str(member["gpu"]))
        difference = 100.0 * (member["fps_nodefrag"] - card["mean"]) / card["mean"] if card else None
        out["cards"].append({"gpu": member["gpu"], "new": member["fps_nodefrag"],
                             "original_mean": card["mean"] if card else None, "original_runs": card["runs"] if card else 0,
                             "relative_difference_pct": difference,
                             "beyond_tolerance": difference is not None and abs(difference) > tolerance})
    return out


def eta_rows(path: pathlib.Path) -> dict:
    """(job, point) -> the batch-2 eta row (eta with and without defrag, mean over the trials)."""
    rows = {}
    if not path.exists():
        return rows
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows[(row["job"], row["point"])] = {"job": row["job"], "point": row["point"], "trials": int(row["trials"]),
                                                "eta": float(row["eta_mean"]), "eta_std": float(row["eta_std"]),
                                                "eta_nodefrag": float(row["eta_nodefrag_mean"]),
                                                "eta_nodefrag_std": float(row["eta_nodefrag_std"])}
    return rows


def ratios(reference: dict, chain_values: dict, eta) -> dict:
    """reference step mean / chain step mean per statistic, and ratio (window) - the g2 eta."""
    out = {name: reference[name]["mean"] / chain_values[name]["mean"] for name in RATIO_STATS}
    if eta:
        out["eta_g2"] = eta
        out["difference_nodefrag"] = out["period_mean_window"] - eta["eta_nodefrag"]
        out["difference_defrag"] = out["period_mean_window"] - eta["eta"]
    return out


def ms(value: float) -> str:
    return f"{value:.3f}"


def spread(stat: dict) -> str:
    return f"{stat['mean']:.3f} ({stat['min']:.3f}–{stat['max']:.3f})"


def chain_label(side: dict, slabs: int) -> str:
    return f"chain K = {slabs} ({side['job']}, {side['node'] or 'node ?'})"


def markdown(result: dict) -> str:
    lines = [f"# tab:phases from the step traces (reference groups: {result['reftrace_job']}, node {result['node']}; "
             f"chain side: {result['chain_job'] or 'the g2 chain traces'})", "",
             "Rules: steady steps 500–1498 minus the defrag step; A / B / C and the step median = each GPU's median, then the "
             "mean of the 8 GPUs (min–max in brackets); step mean (window) = (Σ of the 999 steady periods − in-window defrag "
             "wall time) / 999; step mean (skip) = mean of the kept steps. ms. Strong references also ÷ 8; the weak reference "
             "(2.14M per card) is not divided.", "",
             "| case | row | A | B | C | step median | step mean (window − defrag) | step mean (skip defrag step) |",
             "|---|---|---|---|---|---|---|---|"]
    for item in result["cases"]:
        rows = [("reference, raw (8 × K = 1)", item["reference"]["raw"])]
        if item["kind"] == "strong":
            rows.append((f"reference ÷ {item['slabs']}", item["reference"]["divided"]))
        rows.append((chain_label(item["chain"], item["slabs"]), item["chain"]["values"]))
        if item["chain_g2"]:
            rows.append((chain_label(item["chain_g2"], item["slabs"]), item["chain_g2"]["values"]))
        for name, values in rows:
            lines.append(f"| {item['paper']} | {name} | {spread(values['a'])} | {spread(values['b'])} | {spread(values['c'])} | "
                         f"{spread(values['period_median'])} | {spread(values['period_mean_window'])} | "
                         f"{spread(values['period_mean_skip'])} |")
    lines += ["", "## reference / chain step-mean ratio against the g2 η", "",
              "Ratio = reference step mean (strong ÷ 8, weak not divided) / chain step mean, window − defrag means (median "
              "and skip-mean ratios alongside); η on g2 = the batch-2 table's mean over the point's trials, without defrag "
              "(with defrag in brackets); difference = ratio (window) − η.", "",
              "| case | chain | reference mean | chain mean | ratio (window) | ratio (median) | ratio (skip) | η g2 no defrag (with) | "
              "ratio − η (no defrag) | ratio − η (with defrag) | η row |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for item in result["cases"]:
        reference = item["reference"]["divided"] if item["kind"] == "strong" else item["reference"]["raw"]
        sides = [item["chain"]] + ([item["chain_g2"]] if item["chain_g2"] else [])
        for side in sides:
            ratio = side["ratio"]
            eta = ratio.get("eta_g2")
            lines.append(f"| {item['paper']} | {side['job']}, {side['node'] or 'node ?'} | "
                         f"{ms(reference['period_mean_window']['mean'])} | {ms(side['values']['period_mean_window']['mean'])} | "
                         f"{ratio['period_mean_window']:.4f} | "
                         f"{ratio['period_median']:.4f} | {ratio['period_mean_skip']:.4f} | "
                         + (f"{eta['eta_nodefrag']:.4f} ({eta['eta']:.4f}) | {ratio['difference_nodefrag']:+.4f} | "
                            f"{ratio['difference_defrag']:+.4f} | {eta['job']} {eta['point']}, {eta['trials']} trials |"
                            if eta else "– | – | – | no row |"))
    lines += ["", "## fps without defrag: the traced reference groups against the original untraced references on g2", "",
              "| case | group mean (min–max) | original mean (runs, job) | relative difference | cards beyond tolerance |",
              "|---|---|---|---|---|"]
    for item in result["cases"]:
        check = item["fps_check"]
        beyond = [f"g{card['gpu']} {card['relative_difference_pct']:+.2f} %" for card in check["cards"] if card["beyond_tolerance"]]
        lines.append(f"| {item['paper']} | {check['new_mean']:.3f} ({check['new_min']:.3f}–{check['new_max']:.3f}) | "
                     f"{check['original_mean']:.3f} ({check['original_runs']}, {item['original_references']['job']}) | "
                     f"{check['relative_difference_pct']:+.2f} %{' **beyond ' + str(check['tolerance_pct']) + ' %**' if check['beyond_tolerance'] else ''} | "
                     f"{', '.join(beyond) if beyond else 'none'} |")
    lines += ["", "Per card (fps without defrag, new / original mean / relative difference):", ""]
    for item in result["cases"]:
        cards = ", ".join(f"g{card['gpu']} {card['new']:.3f} / {card['original_mean']:.3f} / {card['relative_difference_pct']:+.2f} %"
                          for card in item["fps_check"]["cards"])
        lines.append(f"- {item['paper']}: {cards}")
    lines += ["", "Chain fps without defrag (with defrag): " + "; ".join(
        f"{item['paper']} {side['job']} {side['fps_nodefrag']:.3f} ({side['fps']:.3f})"
        for item in result["cases"] for side in [item["chain"]] + ([item["chain_g2"]] if item["chain_g2"] else [])), ""]
    lines += ["## per GPU (ms, raw)", ""]
    for item in result["cases"]:
        lines += [f"### {item['paper']}", "", "| side | GPU | kept steps | skipped | A | B | C | step median | mean (window) | mean (skip) | fps no defrag |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for member in item["reference"]["members"]:
            lines.append(f"| reference | {member['gpu']} | {member['kept_steps']} | {member['skipped_steps']} | {ms(member['a'])} | "
                         f"{ms(member['b'])} | {ms(member['c'])} | {ms(member['period_median'])} | {ms(member['period_mean_window'])} | "
                         f"{ms(member['period_mean_skip'])} | {member['fps_nodefrag']:.3f} |")
        for side_name, side in [("chain", item["chain"])] + ([("chain g2", item["chain_g2"])] if item["chain_g2"] else []):
            for entry in side["sims"]:
                lines.append(f"| {side_name} ({side['job']}) | {entry['sim']} | {entry['kept_steps']} | {entry['skipped_steps']} | {ms(entry['a'])} | "
                             f"{ms(entry['b'])} | {ms(entry['c'])} | {ms(entry['period_median'])} | {ms(entry['period_mean_window'])} | "
                             f"{ms(entry['period_mean_skip'])} | {side['fps_nodefrag']:.3f} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def write_csv(result: dict, path: pathlib.Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["case", "row", "job", "gpus"] + [f"{name}_{part}" for name in STATS for part in ("mean", "min", "max")]
                        + ["ratio_window", "ratio_median", "ratio_skip", "eta_g2_nodefrag", "eta_g2", "ratio_minus_eta_nodefrag"])
        for item in result["cases"]:
            rows = [("reference_raw", result["reftrace_job"], item["reference"]["raw"], None),
                    ("chain", item["chain"]["job"], item["chain"]["values"], item["chain"]["ratio"])]
            if item["kind"] == "strong":
                rows.insert(1, ("reference_divided", result["reftrace_job"], item["reference"]["divided"], None))
            if item["chain_g2"]:
                rows.append(("chain_g2", item["chain_g2"]["job"], item["chain_g2"]["values"], item["chain_g2"]["ratio"]))
            for name, job, values, ratio in rows:
                eta = ratio.get("eta_g2") if ratio else None
                writer.writerow([item["case"], name, job, values["gpus"]]
                                + [f"{values[stat][part]:.4f}" for stat in STATS for part in ("mean", "min", "max")]
                                + ([f"{ratio[stat]:.5f}" for stat in RATIO_STATS] if ratio else ["", "", ""])
                                + ([f"{eta['eta_nodefrag']:.5f}", f"{eta['eta']:.5f}", f"{ratio['difference_nodefrag']:+.5f}"]
                                   if eta else ["", "", ""]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--logs", required=True, help="the logs/n56 mirror")
    parser.add_argument("--reftrace-job", required=True, help="the job with the reftrace groups, e.g. e7_b2AA_1716356")
    parser.add_argument("--chain-job", default=None,
                        help="the job whose own K = 8 traces are the chain side (e.g. e7_b2AA_1716356); default: the g2 "
                             "chain traces of batch 2")
    parser.add_argument("--eta-csv", default=str(ETA_CSV), help="the batch-2 eta table (g2) for the ratio comparison")
    parser.add_argument("--out", required=True)
    parser.add_argument("--tolerance", type=float, default=1.0, help="relative fps difference flagged, percent")
    parser.add_argument("--cases", nargs="*", default=None, metavar="SHORT",
                        help="only these cases (2d_n8000 ...; default all four)")
    arguments = parser.parse_args()
    logs = pathlib.Path(arguments.logs)
    reftrace_dir = job_dir(logs, arguments.reftrace_job)
    chain_dir = job_dir(logs, arguments.chain_job) if arguments.chain_job else None
    etas = eta_rows(pathlib.Path(arguments.eta_csv))
    result = {"reftrace_job": arguments.reftrace_job, "node": job_node(reftrace_dir), "chain_job": arguments.chain_job,
              "eta_csv": arguments.eta_csv if etas else None, "tolerance_pct": arguments.tolerance, "cases": []}
    for spec in CASES:
        if arguments.cases and spec["short"] not in arguments.cases:
            continue
        group = reference_group(reftrace_dir, spec["short"])
        if spec["kind"] == "strong":
            group["divided"] = aggregate(group["members"], spec["slabs"])
        reference_values = group["divided"] if spec["kind"] == "strong" else group["raw"]
        eta = etas.get(spec["eta"])
        g2_job, g2_trace, g2_log = spec["chain"]
        g2_chain = chain(g2_job, job_dir(logs, g2_job), g2_trace, g2_log)
        g2_chain["ratio"] = ratios(reference_values, g2_chain["values"], eta)
        if chain_dir is not None:
            own_trace, own_log = spec["own_chain"]
            primary = chain(arguments.chain_job, chain_dir, own_trace, own_log)
            primary["ratio"] = ratios(reference_values, primary["values"], eta)
            secondary = g2_chain
        else:
            primary, secondary = g2_chain, None
        reference_job, pattern = spec["references"]
        original = original_references(job_dir(logs, reference_job), pattern)
        result["cases"].append({"case": spec["case"], "paper": spec["paper"], "kind": spec["kind"], "slabs": spec["slabs"],
                                "reference": group, "chain": primary, "chain_g2": secondary,
                                "original_references": original, "fps_check": compare(group, original, arguments.tolerance)})
    out = pathlib.Path(arguments.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "phases_table.md").write_text(markdown(result), encoding="utf-8")
    (out / "phases_table.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    write_csv(result, out / "phases_table.csv")
    print(markdown(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
