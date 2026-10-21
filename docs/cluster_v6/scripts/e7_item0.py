"""
e7_item0.py — E7 item 0 (user, 2026-10-12, before batches 2 and 4): tables only, from the existing data, on why the
3-D strong-scaling eta (0.873) sits ~11 % under the weak eta (0.984) at the same load per card.

  1. Every K = 1 reference run of the given jobs (index.tsv role "reference", K = 1; the batch-1, batch-3 and
     validation jobs): per case the total particles, the fluid particles (the run log's domain.obj count), the wall
     fraction, fps (steady_steps / steady_seconds), particle-steps per second (total and fluid), the nodes and the
     loop-window SM clock and power (results telemetry_loop of the run's GPU); then the same per case and node.
  2. Per slab of the given K = 8 traced runs (arm C): columns, particles (fluid / wall from the lattice: wall layers
     at both x ends, side walls along every column, verified against the partition's counts), medians over the
     steady steps of T_A, T_B, T_C, the A->B gap, the B->C wait (c_start - b_end), the C->A gap (a_start(n + 1) -
     c_end(n)) and the step period; busy = T_A + T_B + T_C per step (median); busy and period per particle (total
     and fluid); end slabs and middle slabs apart (medians of the per-slab values); beside them the K = 1 runs of the
     same case and of the per-card case (period per particle = 1 / (fps x particles), nothing else is measured
     at K = 1 here), from the jobs given.
  3. Observations: numbers only (how the K = 1 throughput per particle moves with the size and the wall fraction).

Usage: python docs/cluster_v6/scripts/e7_item0.py --job DIR [--job DIR ...] --trace DIR [--trace DIR ...]
           [--per-card CASE=CASE ...] [--out item0.md] [--json item0.json]
  DIR of --job: a pulled job directory (results.jsonl, index.tsv, the run logs); DIR of --trace: <job>/traces/<label>
  (its job directory is the grandparent). --per-card maps a traced case to its per-card K = 1 case (default
  cavity3d_n200_k8=cavity3d_n200, cavity2d_n8000=cavity2d_n2840).
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
import statistics
import sys

import numpy as np

_REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[3]
PER_CARD_DEFAULT = {"cavity3d_n200_k8": "cavity3d_n200", "cavity2d_n8000": "cavity2d_n2840"}
OBJ_LINE = re.compile(r"^\s+- (\S+\.obj): ([\d,]+) particles \((\w+)\)", re.MULTILINE)


def number(value, digits=1, unit=""):
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "-"
    return f"{value:,.{digits}f}{unit}"


def read_index(directory: pathlib.Path) -> dict:
    """label -> (role, family, case, K, trial, reference kind, reference case, point, line, host[, tag])."""
    out = {}
    path = directory / "index.tsv"
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.rstrip("\n").split("\t")
        if len(fields) >= 9:
            out[fields[0]] = fields
    return out


def read_results(directory: pathlib.Path) -> dict:
    out = {}
    for line in (directory / "results.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            out[row["label"]] = row
    return out


def case_particles(directory: pathlib.Path, label: str) -> dict:
    """{'fluid': n, 'wall': n, 'lid': n, 'total': n} from the run log's case_loader lines."""
    path = directory / f"{label}.log"
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    parts = {}
    for name, count, material in OBJ_LINE.findall(text):
        parts[{"stick_water": "fluid", "stick_wall": "wall", "stick_lid": "lid"}.get(material, material)] = int(count.replace(",", ""))
    total = re.search(r"total loaded: ([\d,]+) particles", text)
    if total:
        parts["total"] = int(total.group(1).replace(",", ""))
    return parts


def precise_fps(row: dict):
    steps, seconds = row.get("steady_steps"), row.get("steady_seconds")
    if steps and seconds:
        return steps / seconds
    return row.get("steady_fps")


def run_clock(row: dict) -> tuple:
    """(SM MHz median, W median) of the run's (first) GPU in the loop window."""
    loop = row.get("telemetry_loop") or {}
    devices = row.get("device_map") or []
    entry = loop.get(str(devices[0])) if devices else None
    if not entry:
        return None, None
    return entry.get("sm_clock_median"), entry.get("power_median")


# ---------------------------------------------------------------- 1. K = 1 references
def reference_runs(jobs: list) -> list:
    """Every K = 1 reference run of the jobs: dicts with case, job, node, fps, clock, power, particles."""
    runs = []
    cache = {}
    for directory in jobs:
        index = read_index(directory)
        results = read_results(directory)
        for label, fields in index.items():
            role, case, slabs = fields[1], fields[3], int(fields[4])
            if role != "reference" or slabs != 1:
                continue
            row = results.get(label)
            if row is None or row.get("status") != "pass":
                continue
            if case not in cache:
                cache[case] = case_particles(directory, label)
            particles = cache[case]
            fps = precise_fps(row)
            clock, power = run_clock(row)
            runs.append({"case": case, "job": directory.name, "node": row.get("node"), "label": label, "fps": fps,
                         "sm_mhz": clock, "power_w": power, "total": row.get("loaded_particles") or particles.get("total"),
                         "fluid": particles.get("fluid"), "wall": (particles.get("wall") or 0) + (particles.get("lid") or 0),
                         "solver": fields[11] if len(fields) > 11 else (row.get("solver_tag") or "")})
    return runs


def mean_sd(values: list) -> tuple:
    values = [value for value in values if value is not None]
    if not values:
        return None, None
    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def reference_tables(runs: list) -> tuple:
    """(per-case rows, per-case-and-node rows) sorted by dimension and size."""
    def key(case):
        return (case.startswith("cavity3d"), -1 if case not in sizes else sizes[case])
    sizes = {run["case"]: run["total"] for run in runs}
    by_case: dict = {}
    for run in runs:
        by_case.setdefault(run["case"], []).append(run)
    rows, node_rows = [], []
    for case in sorted(by_case, key=lambda case: (case.startswith("cavity3d"), sizes[case])):
        group = by_case[case]
        total, fluid = group[0]["total"], group[0]["fluid"]
        fps, fps_sd = mean_sd([run["fps"] for run in group])
        clock, _ = mean_sd([run["sm_mhz"] for run in group])
        power, _ = mean_sd([run["power_w"] for run in group])
        nodes = {}
        for run in group:
            nodes[run["node"]] = nodes.get(run["node"], 0) + 1
        rows.append({"case": case, "dimension": 3 if case.startswith("cavity3d") else 2, "total": total, "fluid": fluid,
                     "wall_fraction": (total - fluid) / total if total and fluid else None, "runs": len(group),
                     "fps": fps, "fps_sd": fps_sd, "steps_total_per_s": fps * total if fps and total else None,
                     "steps_fluid_per_s": fps * fluid if fps and fluid else None,
                     "nodes": nodes, "sm_mhz": clock, "power_w": power,
                     "jobs": sorted({run["job"] for run in group})})
        by_node: dict = {}
        for run in group:
            by_node.setdefault((run["node"], run["job"]), []).append(run)
        for (node, job), members in sorted(by_node.items()):
            fps_n, fps_n_sd = mean_sd([run["fps"] for run in members])
            clock_n, _ = mean_sd([run["sm_mhz"] for run in members])
            power_n, _ = mean_sd([run["power_w"] for run in members])
            node_rows.append({"case": case, "node": node, "job": job, "runs": len(members), "fps": fps_n, "fps_sd": fps_n_sd,
                              "fps_min": min(run["fps"] for run in members), "fps_max": max(run["fps"] for run in members),
                              "steps_total_per_s": fps_n * total if fps_n and total else None,
                              "steps_fluid_per_s": fps_n * fluid if fps_n and fluid else None,
                              "sm_mhz": clock_n, "power_w": power_n, "total": total, "fluid": fluid})
    return rows, node_rows


# ---------------------------------------------------------------- 2. per slab
def lattice(case: str, total: int, fluid: int, columns: int) -> dict:
    """The aligned case's lattice: fluid extents (x and the other axes), wall layers, slots per column, the empty
    slots before the first lattice x-index. 2-D cases are squares, 3-D ones cubes or x-stretched (name _k<m>)."""
    dimension = 3 if case.startswith("cavity3d") else 2
    stretch = re.search(r"_k(\d+)$", case)
    stretch = int(stretch.group(1)) if stretch else 1
    side = round((fluid / stretch) ** (1.0 / dimension))          # fluid extent along y (and z)
    fluid_x = fluid // side ** (dimension - 1)
    # wall layers: the same count on every face, (fluid_x + 2w) (side + 2w)^(d-1) = total
    wall = next((w for w in range(0, 64) if (fluid_x + 2 * w) * (side + 2 * w) ** (dimension - 1) == total), None)
    if wall is None or fluid_x * side ** (dimension - 1) != fluid:
        raise SystemExit(f"{case}: no lattice (fluid {fluid_x} x {side}^{dimension - 1} + walls) gives {total} / {fluid}")
    lattice_side = side + 2 * wall
    lattice_x = fluid_x + 2 * wall
    slots = round(lattice_x / columns)
    empty = columns * slots - lattice_x
    if empty < 0 or empty % 2:
        raise SystemExit(f"{case}: {columns} columns x {slots} slots against {lattice_x} lattice rows (empty {empty})")
    return {"dimension": dimension, "fluid_x": fluid_x, "side": side, "wall": wall, "lattice_x": lattice_x,
            "lattice_side": lattice_side, "slots": slots, "offset": empty // 2, "columns": columns}


def slab_particles(geometry: dict, first_column: int, last_column: int) -> dict:
    """Particles of the columns [first, last): total, fluid, wall, x-indices."""
    d, side, lattice_side = geometry["dimension"], geometry["side"], geometry["lattice_side"]
    per_fluid_x = side ** (d - 1)
    per_x_all = lattice_side ** (d - 1)
    fluid = wall = x_count = 0
    for slot in range(first_column * geometry["slots"], last_column * geometry["slots"]):
        x_index = slot - geometry["offset"]
        if x_index < 0 or x_index >= geometry["lattice_x"]:
            continue
        x_count += 1
        if x_index < geometry["wall"] or x_index >= geometry["lattice_x"] - geometry["wall"]:
            wall += per_x_all
        else:
            fluid += per_fluid_x
            wall += per_x_all - per_fluid_x
    return {"total": fluid + wall, "fluid": fluid, "wall": wall, "x_indices": x_count}


def slab_timings(run: dict) -> list:
    """Per sim over the steady steps (step >= warmup, consecutive complete rows): medians (us) of T_A, the A->B gap,
    T_B, the B->C wait, T_C, the C->A gap, the period and busy = T_A + T_B + T_C of the same step."""
    warmup = run["meta"]["warmup"]
    device = run["device"]
    out = []
    for sim in sorted(set(device["sim"].astype(int))):
        rows = {key: value[(device["sim"] == sim) & (device["complete"] == 1)] for key, value in device.items()}
        order = np.argsort(rows["step"])
        rows = {key: value[order] for key, value in rows.items()}
        steps = rows["step"]
        consecutive = np.concatenate([steps[1:] == steps[:-1] + 1, [False]])
        steady = (steps >= warmup) & consecutive
        next_a = np.concatenate([rows["a_start"][1:], [np.nan]])
        parts = {"T_A": rows["a_end"] - rows["a_start"], "A_to_B": rows["b_start"] - rows["a_end"],
                 "T_B": rows["b_end"] - rows["b_start"], "B_to_C": rows["c_start"] - rows["b_end"],
                 "T_C": rows["c_end"] - rows["c_start"], "C_to_A": next_a - rows["c_end"],
                 "period": next_a - rows["a_start"]}
        parts["busy"] = parts["T_A"] + parts["T_B"] + parts["T_C"]
        out.append({"sim": sim, "steps": int(steady.sum()),
                    **{key: float(np.median(value[steady]) / 1e3) for key, value in parts.items()}})
    return out


def slab_tables(trace: pathlib.Path, jobs: list, per_card: dict) -> dict:
    if str(_REPOSITORY_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPOSITORY_ROOT))
    from experiment.v7.analysis import step_trace_model
    run = step_trace_model.load_run(trace)
    meta = run["meta"]
    case = pathlib.Path(meta["case"]).parent.name
    job_directory = trace.parent.parent
    label = trace.name.replace("_E", "_trace_E").replace("_C", "_trace_C") if trace.name.endswith(("_E", "_C")) else trace.name
    results = read_results(job_directory)
    row = results.get(label) or next((value for key, value in results.items() if key.startswith(trace.name)), None)
    if row is None:
        raise SystemExit(f"{trace}: no results row {label} in {job_directory}")
    particles = case_particles(job_directory, label)
    total, fluid = row["loaded_particles"], particles["fluid"]
    geometry = lattice(case, total, fluid, row["grid_columns"])
    cuts = [0] + list(row["cuts"]) + [row["grid_columns"]] if row.get("cuts") else None
    own = row["own_particles"]
    columns = row["own_columns"]
    if cuts is None:
        cuts = [0]
        for width in columns:
            cuts.append(cuts[-1] + width)
    timings = {entry["sim"]: entry for entry in slab_timings(run)}
    slabs = []
    for sim in range(len(own)):
        counts = slab_particles(geometry, cuts[sim], cuts[sim + 1])
        if counts["total"] != own[sim]:
            raise SystemExit(f"{case} slab {sim}: lattice gives {counts['total']} particles, the partition {own[sim]}")
        timing = timings[sim]
        slabs.append({"sim": sim, "columns": columns[sim], "cut": [cuts[sim], cuts[sim + 1]], **counts, **timing,
                      "busy_per_total_ns": timing["busy"] * 1e3 / counts["total"],
                      "busy_per_fluid_ns": timing["busy"] * 1e3 / counts["fluid"],
                      "period_per_total_ns": timing["period"] * 1e3 / counts["total"],
                      "end": sim in (0, len(own) - 1)})
    references = {}
    for reference_case in (case, per_card.get(case)):
        if not reference_case:
            continue
        runs = [run for run in reference_runs(jobs) if run["case"] == reference_case]
        if not runs:
            continue
        same_job = [run for run in runs if run["job"] == job_directory.name] or runs
        fps, fps_sd = mean_sd([run["fps"] for run in same_job])
        references[reference_case] = {"runs": len(same_job), "jobs": sorted({run["job"] for run in same_job}),
                                      "nodes": sorted({run["node"] for run in same_job}), "fps": fps, "fps_sd": fps_sd,
                                      "total": same_job[0]["total"], "fluid": same_job[0]["fluid"],
                                      "period_us": 1e6 / fps, "period_per_total_ns": 1e9 / (fps * same_job[0]["total"]),
                                      "period_per_fluid_ns": 1e9 / (fps * same_job[0]["fluid"]),
                                      "sm_mhz": mean_sd([run["sm_mhz"] for run in same_job])[0],
                                      "power_w": mean_sd([run["power_w"] for run in same_job])[0]}
    return {"trace": str(trace), "label": label, "case": case, "job": job_directory.name, "node": row.get("node"),
            "solver": row.get("solver_tag") or "", "K": meta["K"], "steady_fps": precise_fps(row), "geometry": geometry,
            "weights": meta.get("weights"), "slabs": slabs, "references": references, "per_card_case": per_card.get(case)}


# ---------------------------------------------------------------- markdown
def group_median(slabs: list, key: str, end: bool):
    values = [slab[key] for slab in slabs if slab["end"] == end]
    return statistics.median(values) if values else None


def markdown(rows: list, node_rows: list, traces: list) -> str:
    out = ["## 1. K = 1 参照（全部作业）", "",
           "每次运行的 fps = steady_steps / steady_seconds；粒子·步/s = fps × 粒子数；时钟和功耗是该卡在计时窗内的中位数，再对运行取平均。"
           "墙占比 = (总数 − 流体) / 总数。", "",
           "| 算例 | 维 | 总粒子 | 流体粒子 | 墙占比 | 运行数 | fps（mean ± sd） | 总粒子·步/s（M） | 流体粒子·步/s（M） | 节点（次数） | SM MHz | W |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for row in rows:
        nodes = ", ".join(f"{node} ({count})" for node, count in sorted(row["nodes"].items()))
        out.append(f"| {row['case'].replace('cavity', '')} | {row['dimension']} | {row['total']:,} | {row['fluid']:,} | "
                   f"{100 * row['wall_fraction']:.1f} % | {row['runs']} | {row['fps']:.2f} ± {row['fps_sd']:.2f} | "
                   f"{row['steps_total_per_s'] / 1e6:.1f} | {row['steps_fluid_per_s'] / 1e6:.1f} | {nodes} | "
                   f"{number(row['sm_mhz'], 0)} | {number(row['power_w'], 0)} |")
    out += ["", "按算例、节点和作业：", "",
            "| 算例 | 节点 | 作业 | 运行数 | fps（mean ± sd；min–max） | 总粒子·步/s（M） | 流体粒子·步/s（M） | SM MHz | W |",
            "|---|---|---|---|---|---|---|---|---|"]
    for row in node_rows:
        out.append(f"| {row['case'].replace('cavity', '')} | {row['node']} | {row['job']} | {row['runs']} | "
                   f"{row['fps']:.2f} ± {row['fps_sd']:.2f}；{row['fps_min']:.2f}–{row['fps_max']:.2f} | "
                   f"{row['steps_total_per_s'] / 1e6:.1f} | {row['steps_fluid_per_s'] / 1e6:.1f} | "
                   f"{number(row['sm_mhz'], 0)} | {number(row['power_w'], 0)} |")
    out += ["", "## 2. K = 8 的 C 臂 trace，按 slab", ""]
    for trace in traces:
        geometry = trace["geometry"]
        out += [f"### {trace['case'].replace('cavity', '')} K = {trace['K']}，C 臂（{trace['job']}，{trace['node']}，"
                f"{trace['solver'] or 'rc1'}，{trace['steady_fps']:.2f} fps）", "",
                f"格子 {geometry['lattice_x']} × {geometry['lattice_side']}" + (f" × {geometry['lattice_side']}" if geometry['dimension'] == 3 else "")
                + f"，流体 {geometry['fluid_x']} × {geometry['side']}" + (f" × {geometry['side']}" if geometry['dimension'] == 3 else "")
                + f"，墙 {geometry['wall']} 层；每列 {geometry['slots']} 层粒子，{geometry['columns']} 列，首尾各空 {geometry['offset']} 层。"
                f"每个 slab 的粒子数由格子算出，与分区日志逐一相符。时间是稳态步（step ≥ warmup）的中位数，µs；busy = 同一步的 T_A + T_B + T_C。",
                "",
                "| slab | 列 | 粒子 | 流体 | 墙 | 墙占比 | T_A | A→B | T_B | B→C 等待 | T_C | C→A 间隙 | 步长 | busy | busy / 粒子（ns） | busy / 流体粒子（ns） | 步长 / 粒子（ns） |",
                "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for slab in trace["slabs"]:
            name = f"s{slab['sim']}" + ("（端）" if slab["end"] else "")
            out.append(f"| {name} | {slab['cut'][0]}–{slab['cut'][1]} ({slab['columns']}) | {slab['total']:,} | {slab['fluid']:,} | "
                       f"{slab['wall']:,} | {100 * slab['wall'] / slab['total']:.1f} % | {slab['T_A']:.0f} | {slab['A_to_B']:.0f} | "
                       f"{slab['T_B']:.0f} | {slab['B_to_C']:.0f} | {slab['T_C']:.0f} | {slab['C_to_A']:.0f} | {slab['period']:.0f} | "
                       f"{slab['busy']:.0f} | {slab['busy_per_total_ns']:.3f} | {slab['busy_per_fluid_ns']:.3f} | "
                       f"{slab['period_per_total_ns']:.3f} |")
        for end, title in ((True, "端 slab（中位数）"), (False, "中间 slab（中位数）")):
            out.append(f"| **{title}** | | {group_median(trace['slabs'], 'total', end):,.0f} | {group_median(trace['slabs'], 'fluid', end):,.0f} | "
                       f"{group_median(trace['slabs'], 'wall', end):,.0f} | | {group_median(trace['slabs'], 'T_A', end):.0f} | "
                       f"{group_median(trace['slabs'], 'A_to_B', end):.0f} | {group_median(trace['slabs'], 'T_B', end):.0f} | "
                       f"{group_median(trace['slabs'], 'B_to_C', end):.0f} | {group_median(trace['slabs'], 'T_C', end):.0f} | "
                       f"{group_median(trace['slabs'], 'C_to_A', end):.0f} | {group_median(trace['slabs'], 'period', end):.0f} | "
                       f"{group_median(trace['slabs'], 'busy', end):.0f} | {group_median(trace['slabs'], 'busy_per_total_ns', end):.3f} | "
                       f"{group_median(trace['slabs'], 'busy_per_fluid_ns', end):.3f} | {group_median(trace['slabs'], 'period_per_total_ns', end):.3f} |")
        out += ["", "并列的 K = 1（同一作业的参照运行；K = 1 只有步长，没有 phase 的分解）：", "",
                "| K = 1 | 粒子 | 流体 | 运行数 | 节点 | fps | 步长（µs） | 步长 / 粒子（ns） | 步长 / 流体粒子（ns） | SM MHz | W |",
                "|---|---|---|---|---|---|---|---|---|---|---|"]
        for reference_case, entry in trace["references"].items():
            kind = "同一算例" if reference_case == trace["case"] else "每卡那一份的算例"
            out.append(f"| {reference_case.replace('cavity', '')}（{kind}） | {entry['total']:,} | {entry['fluid']:,} | {entry['runs']} | "
                       f"{', '.join(entry['nodes'])} | {entry['fps']:.2f} ± {entry['fps_sd']:.2f} | {entry['period_us']:.0f} | "
                       f"{entry['period_per_total_ns']:.3f} | {entry['period_per_fluid_ns']:.3f} | {number(entry['sm_mhz'], 0)} | "
                       f"{number(entry['power_w'], 0)} |")
        out.append("")
    return "\n".join(out) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--job", action="append", default=[], required=True)
    parser.add_argument("--trace", action="append", default=[])
    parser.add_argument("--per-card", action="append", default=[], metavar="CASE=CASE")
    parser.add_argument("--out", default=None)
    parser.add_argument("--json", default=None)
    arguments = parser.parse_args()
    jobs = [pathlib.Path(job) for job in arguments.job]
    per_card = dict(PER_CARD_DEFAULT)
    for item in arguments.per_card:
        case, reference = item.split("=", 1)
        per_card[case] = reference
    runs = reference_runs(jobs)
    rows, node_rows = reference_tables(runs)
    traces = [slab_tables(pathlib.Path(trace), jobs, per_card) for trace in arguments.trace]
    text = markdown(rows, node_rows, traces)
    if arguments.out:
        pathlib.Path(arguments.out).write_text(text, encoding="utf-8")
    else:
        print(text)
    if arguments.json:
        pathlib.Path(arguments.json).write_text(json.dumps({"references": rows, "references_by_node": node_rows,
                                                            "traces": traces}, indent=1, default=str), encoding="utf-8")
    print(f"{len(runs)} K = 1 reference runs, {len(rows)} cases, {len(traces)} traces", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
