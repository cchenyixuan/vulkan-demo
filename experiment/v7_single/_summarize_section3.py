"""
_summarize_section3.py — tables of "§3 补测" (docs/seam_audit/v7_single.md) from _run_section3_campaign.py's
results.jsonl, stats/*.json and telemetry.csv, measurement A's throughput summary (the wall-clock step at 1 and
2 frames in flight) and measurement B's fused records (the interleaved per-kernel numbers to compare with).

Writes <dir>/summary.json, table2.md (item 2: per-kernel cost of 2-D 8M and cavity3d_8m_b4, fused, frames
1001-3000, 3 runs; LaTeX rows; candidates and neighbours of the end state; the difference to B), sizes.md
(item 3: per size the candidates / neighbours at the initial and the end state, grid and occupancy, the GPU
step, ps per candidate, the GPU idle fraction = measurement A's wall step / GPU step - 1 at 1 and 2 in flight,
and the 4M dip analysis) and summary.csv (the per-size table).

    python experiment/v7_single/_summarize_section3.py --dir logs/v7_single/section3
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import pathlib
import statistics
import sys

_REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from _summarize_kernel_breakdown import DISPATCH, STAGE_LABEL  # noqa: E402

SIZE_ORDER = ["1m", "2m", "4m", "4m_gen", "8m", "16m", "32m", "3d_8m"]
SIZE_LABEL = {"1m": "1M", "2m": "2M", "4m": "4M(旧手工)", "4m_gen": "4M gen", "8m": "8M", "16m": "16M", "32m": "32M",
              "3d_8m": "3-D 8M"}
TABLE2_SIZES = ["8m", "3d_8m"]
B_TAG_OF_SIZE = {"8m": "cavity2d_8m", "3d_8m": "cavity3d_8m_b4"}
NEIGHBOR_KERNELS = ("correction_density", "correction", "density_kernel", "force", "wall_extrapolate")


def mean_std(values: list[float]) -> tuple[float, float]:
    return (statistics.fmean(values), statistics.stdev(values) if len(values) > 1 else 0.0) if values else (None, None)


def load_jsonl(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_telemetry(path: pathlib.Path, index: int = 1) -> list[tuple[float, float, float, float, str, str]]:
    """(epoch, sm_mhz, power_w, temp_c, pcie_gen, pcie_width) of the measured card."""
    samples = []
    if not path.exists():
        return samples
    with open(path, newline="", encoding="utf-8", errors="replace") as handle:
        for row in csv.reader(handle):
            if len(row) < 8 or not row[1].strip().isdigit() or int(row[1]) != index:
                continue
            try:
                stamp = datetime.datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp()
                sm = float(row[2].split()[0]); power = float(row[4].split()[0]); temp = float(row[5].strip())
            except (ValueError, IndexError):
                continue
            gen = row[8].strip() if len(row) > 8 else ""
            width = row[9].strip() if len(row) > 9 else ""
            samples.append((stamp, sm, power, temp, gen, width))
    return samples


def covariates(samples, start: float | None, end: float | None) -> dict:
    inside = [s for s in samples if start is not None and end is not None and start <= s[0] <= end]
    if not inside:
        return {"samples": 0, "sm_mhz": None, "power_w": None, "temp_c_max": None, "pcie": None}
    from collections import Counter
    return {"samples": len(inside), "sm_mhz": statistics.fmean(s[1] for s in inside),
            "power_w": statistics.fmean(s[2] for s in inside), "temp_c_max": max(s[3] for s in inside),
            "pcie": Counter(f"Gen {s[4]} x{s[5]}" for s in inside).most_common(1)[0][0]}


def load_stats(directory: pathlib.Path, size: str, runs: list[int]) -> dict:
    out = {"initial": None, "end": []}
    initial = directory / "stats" / f"{size}_initial.json"
    if initial.exists():
        out["initial"] = json.loads(initial.read_text(encoding="utf-8"))
    for run in runs:
        end = directory / "stats" / f"{size}_run{run}_end.json"
        if end.exists():
            out["end"].append(json.loads(end.read_text(encoding="utf-8")))
    return out


def summarize_size(size: str, records: list[dict], stats: dict, samples, a_rows: dict, b_records: list[dict],
                   memory: dict | None) -> dict:
    runs = sorted(records, key=lambda r: r["run"])
    particles = runs[0]["particles"]
    stages = runs[0]["stages"]
    frame_values = [r["mean_us"]["gpu_frame"] for r in runs]
    frame_mean, frame_std = mean_std(frame_values)
    amortized_mean, amortized_std = mean_std([r["gpu_step_amortized_us"] for r in runs if r.get("gpu_step_amortized_us")])
    pairs_mean = statistics.fmean(r["candidates"]["candidate_pairs"] for r in runs)
    out = {"size": size, "label": SIZE_LABEL.get(size, size), "case": runs[0]["case"], "particles": particles,
           "dimension": runs[0]["dimension"], "runs": len(runs), "commit": runs[0].get("commit"),
           "drift_ok": all(r["drift"] == 0 and r["overflow"] == 0 for r in runs),
           "stages": {}, "stage_names": stages,
           "gpu_frame_us": frame_mean, "gpu_frame_std_us": frame_std,
           "gpu_frame_spread_pct": 100.0 * frame_std / frame_mean if frame_mean else None,
           "gpu_frame_values": frame_values,
           "gpu_step_amortized_us": amortized_mean, "gpu_step_amortized_std_us": amortized_std,
           "defrag_wall_us": statistics.fmean(v for r in runs for v in r.get("defrag_wall_us", [])) if any(r.get("defrag_wall_us") for r in runs) else None,
           "timed_wall_us": statistics.fmean(r["timed_us_per_step"] for r in runs),
           "candidate_pairs": pairs_mean,
           "candidates_per_particle": statistics.fmean(r["candidates"]["candidates_per_particle"] for r in runs),
           "candidates_per_fluid_particle": statistics.fmean(r["candidates"]["candidates_per_fluid_particle"] for r in runs),
           "candidates_per_wall_particle": statistics.fmean(r["candidates"]["candidates_per_wall_particle"] for r in runs),
           "fluid_particles": runs[0]["candidates"]["fluid_particles"], "wall_particles": runs[0]["candidates"]["wall_particles"],
           "grid_dimensions": runs[0]["candidates"]["grid_dimensions"], "total_voxels": runs[0]["candidates"]["total_voxels"],
           "end_occupied_voxels": statistics.fmean(r["candidates"]["occupied_voxels"] for r in runs),
           "end_mean_per_voxel": statistics.fmean(r["candidates"]["mean_particles_per_voxel"] for r in runs),
           "end_max_per_voxel": max(r["candidates"]["max_particles_per_voxel"] for r in runs),
           "gpu_throughput_mps": particles / frame_mean if frame_mean else None,
           "covariates": [covariates(samples, r.get("epoch_start"), r.get("epoch_end")) for r in runs],
           "switches": runs[0].get("switches")}
    out["host_round_trip_us"] = out["timed_wall_us"] - frame_mean
    for stage in stages + ["gpu_frame"]:
        values = [r["mean_us"][stage] for r in runs]
        mean, std = mean_std(values)
        entry = {"mean_us": mean, "std_us": std, "p50_us": statistics.fmean(r["p50_us"][stage] for r in runs),
                 "ns_per_particle": mean * 1000.0 / particles, "share_pct": 100.0 * mean / frame_mean}
        if stage in NEIGHBOR_KERNELS:
            entry["ps_per_candidate"] = statistics.fmean(1e6 * r["mean_us"][stage] / r["candidates"]["candidate_pairs"] for r in runs)
        out["stages"][stage] = entry
    out["step_ps_per_candidate"] = statistics.fmean(1e6 * r["mean_us"]["gpu_frame"] / r["candidates"]["candidate_pairs"] for r in runs)
    # CPU statistics: the initial state and the end states
    initial = stats.get("initial")
    if initial:
        c = initial["candidates"]
        out["initial"] = {"candidates_per_particle": c["candidates_per_particle"],
                          "candidates_per_fluid_particle": c["candidates_per_fluid_particle"],
                          "occupied_voxels": c["occupied_voxels"], "mean_per_voxel": c["mean_particles_per_voxel"],
                          "max_per_voxel": c["max_particles_per_voxel"], "grid_dimensions": c["grid_dimensions"],
                          "h": initial["h"], "dx": initial["dx"], "h_over_dx": initial["h_over_dx"]}
        if "neighbors" in initial:
            n = initial["neighbors"]
            out["initial"].update({"neighbors_f32_all": n["neighbors_f32_all"], "neighbors_f32_fluid": n["neighbors_f32_fluid"],
                                   "neighbors_f32_wall": n["neighbors_f32_wall"],
                                   "neighbors_strict64_all": n["neighbors_strict_all"],
                                   "neighbors_lo_all": n["neighbors_lo_all"], "neighbors_hi_all": n["neighbors_hi_all"],
                                   "neighbors_lo_fluid": n["neighbors_lo_fluid"], "neighbors_hi_fluid": n["neighbors_hi_fluid"],
                                   "ambiguous_pairs_2ulp": n["ambiguous_pairs_2ulp"],
                                   "ratio_candidates_to_neighbors": initial.get("ratio_candidates_to_neighbors_f32_all"),
                                   "ratio_candidates_to_neighbors_fluid": initial.get("ratio_candidates_to_neighbors_f32_fluid")})
    ends = stats.get("end") or []
    if ends:
        def mean_of(path):
            values = []
            for e in ends:
                value = e
                for key in path:
                    value = value.get(key) if isinstance(value, dict) else None
                    if value is None:
                        break
                if value is not None:
                    values.append(value)
            return statistics.fmean(values) if values else None
        out["end"] = {"runs": len(ends), "step": ends[0].get("step"),
                      "candidates_per_particle": mean_of(("candidates", "candidates_per_particle")),
                      "candidates_per_fluid_particle": mean_of(("candidates", "candidates_per_fluid_particle")),
                      "occupied_voxels": mean_of(("candidates", "occupied_voxels")),
                      "mean_per_voxel": mean_of(("candidates", "mean_particles_per_voxel")),
                      "max_per_voxel": max(e["candidates"]["max_particles_per_voxel"] for e in ends),
                      "cpu_binning_vs_gpu_mismatch": max(e.get("cpu_binning_vs_gpu_voxel_ids_mismatch", 0) for e in ends),
                      "neighbors_f32_all": mean_of(("neighbors", "neighbors_f32_all")),
                      "neighbors_f32_fluid": mean_of(("neighbors", "neighbors_f32_fluid")),
                      "neighbors_f32_wall": mean_of(("neighbors", "neighbors_f32_wall")),
                      "neighbors_strict64_all": mean_of(("neighbors", "neighbors_strict_all")),
                      "neighbors_lo_all": mean_of(("neighbors", "neighbors_lo_all")),
                      "neighbors_hi_all": mean_of(("neighbors", "neighbors_hi_all")),
                      "neighbors_lo_fluid": mean_of(("neighbors", "neighbors_lo_fluid")),
                      "neighbors_hi_fluid": mean_of(("neighbors", "neighbors_hi_fluid")),
                      "ambiguous_pairs_2ulp": mean_of(("neighbors", "ambiguous_pairs_2ulp")),
                      "ratio_candidates_to_neighbors": mean_of(("ratio_candidates_to_neighbors_f32_all",)),
                      "ratio_candidates_to_neighbors_fluid": mean_of(("ratio_candidates_to_neighbors_f32_fluid",))}
    # measurement A: the wall-clock step at 1 and 2 in flight -> GPU idle fraction
    out["a"] = {}
    for depth in (1, 2):
        row = a_rows.get(f"{size}/v7_single/{depth}")
        if row:
            wall = 1e6 / row["fps_mean"]
            out["a"][str(depth)] = {"fps": row["fps_mean"], "fps_std": row["fps_std"], "wall_us": wall,
                                    "mparticle_steps_per_s": row["mparticle_steps_per_s"],
                                    "idle_vs_amortized": wall / amortized_mean - 1 if amortized_mean else None,
                                    "idle_vs_frame": wall / frame_mean - 1}
    # measurement B (interleaved, frames 1001-5000): the fused records of this case
    b = [r for r in b_records if r.get("setting", "fused") == "fused" and r.get("tag") == B_TAG_OF_SIZE.get(size)]
    if b:
        out["b"] = {"runs": len(b), "gpu_frame_us": statistics.fmean(r["mean_us"]["gpu_frame"] for r in b),
                    "candidates_per_particle": statistics.fmean(r["candidates"]["candidates_per_particle"] for r in b),
                    "stages": {stage: statistics.fmean(r["mean_us"][stage] for r in b) for stage in stages if stage in b[0]["mean_us"]}}
        out["b"]["delta_pct"] = {stage: 100.0 * (out["stages"][stage]["mean_us"] / out["b"]["stages"][stage] - 1)
                                 for stage in out["b"]["stages"]}
        out["b"]["delta_pct"]["gpu_frame"] = 100.0 * (frame_mean / out["b"]["gpu_frame_us"] - 1)
    if memory:
        out["memory"] = {key: memory[key] for key in ("device_local_bytes", "device_local_bytes_per_particle", "slot_bytes",
                                                      "voxel_list_bytes_per_particle", "pool_capacity")}
    return out


def table2_markdown(summary: dict) -> str:
    sizes = [s for s in TABLE2_SIZES if s in summary]
    lines = ["# §3 补测 表 2:逐 kernel 成本(v7_single,融合核,第 1001–3000 步连续计时)", "",
             "每个算例 3 次独立运行;1 在飞(每步一次提交 + fence),生产录制里的 tick 回调写 BOTTOM_OF_PIPE 时间戳(屏障结构与生产步相同),"
             "defrag 帧(第 2000、3000 步)剔除;值 = 3 次运行各自均值的平均 ± 3 次的标准差;ns/粒子按全部粒子(含壁);"
             "ps/候选 = µs/步 × 10⁶ / 候选对数(第 3000 步结束时的体素列表)。", ""]
    head = "| kernel | 派发 |" + "".join(f" {summary[s]['label']} µs/步 | ns/粒子 | % | ps/候选 |" for s in sizes)
    lines += [head, "|" + "---|" * (2 + 4 * len(sizes))]
    stage_names = []
    for s in sizes:
        for stage in summary[s]["stage_names"]:
            if stage not in stage_names:
                stage_names.append(stage)
    for stage in stage_names + ["gpu_frame"]:
        label = "**步长**" if stage == "gpu_frame" else stage
        cells = []
        for s in sizes:
            e = summary[s]["stages"].get(stage)
            if e is None:
                cells.append(" – | – | – | – |")
                continue
            ps = f"{e['ps_per_candidate']:.2f}" if "ps_per_candidate" in e else "–"
            cells.append(f" {e['mean_us']:.0f} ± {e['std_us']:.0f} | {e['ns_per_particle']:.2f} | {e['share_pct']:.1f} | {ps} |")
        lines.append(f"| {label} | {DISPATCH.get(stage, '')} |" + "".join(cells))
    lines += ["", "## 第 3000 步结束时的状态", "",
              "| 项 |" + "".join(f" {summary[s]['label']} |" for s in sizes), "|---|" + "---|" * len(sizes)]

    def row(label, getter, fmt="{:.2f}"):
        cells = []
        for s in sizes:
            try:
                value = getter(summary[s])
            except (KeyError, TypeError):
                value = None
            cells.append(" – |" if value is None else f" {fmt.format(value)} |")
        lines.append(f"| {label} |" + "".join(cells))
    row("粒子数(流体 + 壁)", lambda d: d["particles"], "{:,}")
    row("流体粒子", lambda d: d["fluid_particles"], "{:,}")
    row("h / Δx", lambda d: d["initial"]["h_over_dx"], "{:.3f}")
    row("网格 Nx × Ny × Nz", lambda d: " × ".join(str(v) for v in d["grid_dimensions"]), "{}")
    row("每粒子候选数(全部)", lambda d: d["candidates_per_particle"], "{:.1f}")
    row("每粒子候选数(流体)", lambda d: d["candidates_per_fluid_particle"], "{:.1f}")
    row("候选对数 / 步", lambda d: d["candidate_pairs"], "{:,.0f}")
    row("每粒子邻居数 r < h,float32(全部)", lambda d: d["end"]["neighbors_f32_all"], "{:.2f}")
    row("每粒子邻居数 r < h,float32(流体)", lambda d: d["end"]["neighbors_f32_fluid"], "{:.2f}")
    row("邻居数 double 区间 [h(1−1e−5), h(1+1e−5)](全部)", lambda d: f"{d['end']['neighbors_lo_all']:.2f} – {d['end']['neighbors_hi_all']:.2f}", "{}")
    row("候选 / 邻居(全部)", lambda d: d["end"]["ratio_candidates_to_neighbors"], "{:.2f}")
    row("候选 / 邻居(流体)", lambda d: d["end"]["ratio_candidates_to_neighbors_fluid"], "{:.2f}")
    row("每体素粒子数 均值 / 最大(占用体素)", lambda d: f"{d['end_mean_per_voxel']:.1f} / {d['end_max_per_voxel']}", "{}")
    row("占用体素 / 总体素", lambda d: f"{d['end_occupied_voxels']:,.0f} / {d['total_voxels']:,}", "{}")
    row("步长 3 次运行的散布(标准差 / 均值)", lambda d: d["gpu_frame_spread_pct"], "{:.2f} %")
    row("含 defrag 帧与 defrag 命令摊到每步的 GPU 步长 µs", lambda d: d["gpu_step_amortized_us"], "{:.0f}")
    row("defrag 命令 µs(每次)", lambda d: d["defrag_wall_us"], "{:.0f}")
    row("计时帧墙钟 − GPU 帧(host 往返)µs", lambda d: d["host_round_trip_us"], "{:.0f}")
    lines += ["", "## 与测量 B(交错,第 1001–5000 步)的差", ""]
    for s in sizes:
        b = summary[s].get("b")
        if not b:
            continue
        parts = ", ".join(f"{stage} {delta:+.2f} %" for stage, delta in b["delta_pct"].items())
        lines.append(f"- {summary[s]['label']}:步长 {summary[s]['gpu_frame_us']:.0f} 对 B 的 {b['gpu_frame_us']:.0f} µs"
                     f"({b['delta_pct']['gpu_frame']:+.2f} %);逐 kernel:{parts};候选数 / 粒子 "
                     f"{summary[s]['candidates_per_particle']:.1f} 对 {b['candidates_per_particle']:.1f}。")
    lines += ["", "## LaTeX 行", "", "```latex",
              "% columns: kernel & dispatch & " + " & ".join(f"{summary[s]['label']} $\\mu$s/step & ns/particle & \\%" for s in sizes) + " \\\\",
              "% " + "; ".join(f"{summary[s]['label']}: {summary[s]['particles']:,} particles ({summary[s]['fluid_particles']:,} fluid), "
                               f"h/dx = {summary[s]['initial']['h_over_dx']:.3g}, {summary[s]['candidates_per_particle']:.0f} candidates "
                               f"and {summary[s]['end']['neighbors_f32_all']:.1f} neighbours per particle"
                               for s in sizes if summary[s].get("initial") and summary[s].get("end")),
              "% " + "; ".join(f"{summary[s]['label']}: 3 runs, step spread {summary[s]['gpu_frame_spread_pct']:.2f} %" for s in sizes)]
    for stage in stage_names + ["gpu_frame"]:
        label = "\\textbf{step total}" if stage == "gpu_frame" else STAGE_LABEL.get(stage, stage)
        cells = []
        for s in sizes:
            e = summary[s]["stages"].get(stage)
            cells.append("-- & -- & --" if e is None else f"{e['mean_us']:.0f} & {e['ns_per_particle']:.2f} & {e['share_pct']:.1f}")
        if stage == "gpu_frame":
            lines.append("\\midrule")
        lines.append(f"{label} & {'' if stage == 'gpu_frame' else DISPATCH.get(stage, '')} & " + " & ".join(cells) + " \\\\")
    lines += ["% ps per candidate: " + "; ".join(
        f"{summary[s]['label']} " + ", ".join(f"{stage} {e['ps_per_candidate']:.2f}" for stage, e in summary[s]["stages"].items()
                                              if "ps_per_candidate" in e) for s in sizes), "```", ""]
    lines += ["## 协变量(每次运行窗口内的 nvidia-smi 1 Hz 样本)", "", "| 算例 | 运行 | 样本 | SM MHz | W | 最高 °C | PCIe |",
              "|---|---|---|---|---|---|---|"]
    for s in sizes:
        for run, c in enumerate(summary[s]["covariates"], start=1):
            lines.append(f"| {summary[s]['label']} | {run} | {c['samples']} | {c['sm_mhz'] and f'{c['sm_mhz']:.0f}'} | "
                         f"{c['power_w'] and f'{c['power_w']:.0f}'} | {c['temp_c_max'] and f'{c['temp_c_max']:.0f}'} | {c['pcie']} |")
    return "\n".join(lines) + "\n"


def sizes_markdown(summary: dict, dip: dict) -> str:
    sizes = [s for s in SIZE_ORDER if s in summary]
    lines = ["# §3 补测 第 3 项:各尺寸的候选数、邻居数、网格与 GPU 步长", "",
             "初始态:加载器的初始粒子按网格原点与尺寸在 CPU 上分箱(float32,与 initialize_voxelization 相同);"
             "窗末:第 3000 步结束时的 GPU 读回。邻居 = |{j ≠ i : r_ij < h}|(float32,与 kernel 的判定相同;double 区间见 JSON)。"
             "GPU 步长:1 在飞、时间戳、预热 1000 + 计时 2000(8M 与 3-D 8M 为表 2 三次运行的均值,其余 1 次);"
             "摊销步长 = 含 defrag 帧与 defrag 命令;GPU 空闲比例 = 测量 A 的墙钟步长 ÷ 摊销 GPU 步长 − 1。", "",
             "| 算例 | 粒子 | h/Δx | 网格 Nx × Ny × Nz | 占用体素 初始 / 窗末 | 每体素 均值 / 最大 初始 | 窗末 | 候选/粒子 初始 | 窗末 | "
             "邻居/粒子 初始(全部 / 流体) | 窗末(全部 / 流体) | 候选/邻居 窗末 | GPU 步长 µs(摊销) | ps/候选 步 | 融合核 | force | "
             "A 墙钟 d1 / d2 µs | 空闲 d1 | d2 |",
             "|" + "---|" * 19]
    for s in sizes:
        d = summary[s]
        ini = d.get("initial") or {}
        end = d.get("end") or {}
        a1 = d["a"].get("1") or {}
        a2 = d["a"].get("2") or {}
        f = lambda v, fmt="{:.2f}": "–" if v is None else fmt.format(v)
        lines.append(
            f"| {d['label']} | {d['particles']:,} | {f(ini.get('h_over_dx'), '{:.3g}')} | {' × '.join(str(v) for v in d['grid_dimensions'])} | "
            f"{f(ini.get('occupied_voxels'), '{:,.0f}')} / {d['end_occupied_voxels']:,.0f} | "
            f"{f(ini.get('mean_per_voxel'), '{:.1f}')} / {f(ini.get('max_per_voxel'), '{}')} | {d['end_mean_per_voxel']:.1f} / {d['end_max_per_voxel']} | "
            f"{f(ini.get('candidates_per_particle'), '{:.1f}')} | {d['candidates_per_particle']:.1f} | "
            f"{f(ini.get('neighbors_f32_all'))} / {f(ini.get('neighbors_f32_fluid'))} | "
            f"{f(end.get('neighbors_f32_all'))} / {f(end.get('neighbors_f32_fluid'))} | {f(end.get('ratio_candidates_to_neighbors'))} | "
            f"{d['gpu_frame_us']:.0f} ({f(d['gpu_step_amortized_us'], '{:.0f}')}) | {d['step_ps_per_candidate']:.2f} | "
            f"{f(d['stages'].get('correction_density', {}).get('ps_per_candidate'))} | {f(d['stages'].get('force', {}).get('ps_per_candidate'))} | "
            f"{f(a1.get('wall_us'), '{:.0f}')} / {f(a2.get('wall_us'), '{:.0f}')} | "
            f"{f(a1.get('idle_vs_amortized'), '{:+.2%}')} | {f(a2.get('idle_vs_amortized'), '{:+.2%}')} |")
    lines += ["", "## 4M 凹陷", ""]
    for key, value in dip.items():
        if isinstance(value, float):
            lines.append(f"- {key}: {value:.4g}")
        else:
            lines.append(f"- {key}: {value}")
    return "\n".join(lines) + "\n"


def dip_analysis(summary: dict) -> dict:
    """Is the 4M point's ps per candidate the same as its neighbours' (2M, 8M)? For the hand-made 4m and, when
    present, the generator-built 4m_gen; plus the candidate-count part of the dip and the grids."""
    if any(s not in summary for s in ("2m", "8m")):
        return {"status": "sizes missing"}
    two, eight = summary["2m"], summary["8m"]
    out = {"criterion": "the 4M point's ps/candidate (whole step, fused kernel, force) against the mean of 2M and 8M; "
                        "the run-to-run spread of the step is the yardstick",
           "step_spread_pct_8m_runs": eight["gpu_frame_spread_pct"]}
    for point in ("4m", "4m_gen"):
        if point not in summary:
            continue
        four = summary[point]
        for name, getter in (("step", lambda d: d["step_ps_per_candidate"]),
                             ("correction_density", lambda d: d["stages"]["correction_density"]["ps_per_candidate"]),
                             ("force", lambda d: d["stages"]["force"]["ps_per_candidate"])):
            reference = 0.5 * (getter(two) + getter(eight))
            out[f"ps_per_candidate_{name}_2m_{point}_8m"] = [getter(two), getter(four), getter(eight)]
            out[f"ps_per_candidate_{name}_{point}_vs_neighbours_pct"] = 100.0 * (getter(four) / reference - 1)
        cand = [two["candidates_per_particle"], four["candidates_per_particle"], eight["candidates_per_particle"]]
        out[f"candidates_per_particle_2m_{point}_8m"] = cand
        out[f"candidates_{point}_vs_neighbours_pct"] = 100.0 * (cand[1] / (0.5 * (cand[0] + cand[2])) - 1)
        gpu_tp = [two["gpu_throughput_mps"], four["gpu_throughput_mps"], eight["gpu_throughput_mps"]]
        out[f"gpu_throughput_mps_2m_{point}_8m"] = gpu_tp
        out[f"gpu_throughput_{point}_vs_neighbours_pct"] = 100.0 * (gpu_tp[1] / (0.5 * (gpu_tp[0] + gpu_tp[2])) - 1)
        a2 = [summary[s]["a"].get("2", {}).get("mparticle_steps_per_s") for s in ("2m", point, "8m")]
        if all(a2):
            out[f"a_throughput_d2_2m_{point}_8m"] = a2
            out[f"a_throughput_d2_{point}_vs_neighbours_pct"] = 100.0 * (a2[1] / (0.5 * (a2[0] + a2[2])) - 1)
    for s in ("2m", "4m", "4m_gen", "8m"):
        if s not in summary:
            continue
        d = summary[s]
        out[f"grid_{s}"] = {"dimensions": d["grid_dimensions"], "total_voxels": d["total_voxels"],
                            "end_occupied_voxels": d["end_occupied_voxels"], "end_mean_per_voxel": d["end_mean_per_voxel"],
                            "end_max_per_voxel": d["end_max_per_voxel"], "fluid_particles": d["fluid_particles"],
                            "wall_particles": d["wall_particles"], "h_over_dx": (d.get("initial") or {}).get("h_over_dx")}
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", default="logs/v7_single/section3")
    parser.add_argument("--throughput-summary", default="docs/seam_audit/v7_single/throughput_summary.json")
    parser.add_argument("--breakdown-b", default="docs/seam_audit/v7_single/kernel_breakdown_results.jsonl")
    parser.add_argument("--memory", default="docs/seam_audit/v7_single/memory.json")
    parser.add_argument("--telemetry-index", type=int, default=1)
    arguments = parser.parse_args()
    directory = _REPOSITORY_ROOT / arguments.dir
    records = [r for r in load_jsonl(directory / "results.jsonl") if r.get("drift") == 0 and r.get("overflow") == 0]
    samples = load_telemetry(directory / "telemetry.csv", arguments.telemetry_index)
    a_path = _REPOSITORY_ROOT / arguments.throughput_summary
    a_rows = json.loads(a_path.read_text(encoding="utf-8"))["rows"] if a_path.exists() else {}
    b_records = load_jsonl(_REPOSITORY_ROOT / arguments.breakdown_b)
    memory_path = _REPOSITORY_ROOT / arguments.memory
    memory = {m["size"]: m for m in json.loads(memory_path.read_text(encoding="utf-8"))} if memory_path.exists() else {}
    by_size: dict[str, list[dict]] = {}
    for record in records:
        by_size.setdefault(record["size"], []).append(record)
    summary = {}
    for size in SIZE_ORDER:
        if size not in by_size:
            continue
        runs = sorted({r["run"] for r in by_size[size]})
        summary[size] = summarize_size(size, by_size[size], load_stats(directory, size, runs), samples, a_rows, b_records,
                                       memory.get(size))
    dip = dip_analysis(summary)
    (directory / "summary.json").write_text(json.dumps({"sizes": summary, "dip_4m": dip, "runs": len(records),
                                                         "commit": records[0].get("commit") if records else None},
                                                        indent=1, ensure_ascii=False), encoding="utf-8")
    table2 = table2_markdown(summary)
    sizes = sizes_markdown(summary, dip)
    (directory / "table2.md").write_text(table2, encoding="utf-8")
    (directory / "sizes.md").write_text(sizes, encoding="utf-8")
    fields = ["size", "particles", "fluid_particles", "h_over_dx", "grid", "total_voxels", "initial_occupied", "end_occupied",
              "initial_mean_per_voxel", "initial_max_per_voxel", "end_mean_per_voxel", "end_max_per_voxel",
              "initial_candidates", "end_candidates", "initial_candidates_fluid", "end_candidates_fluid",
              "initial_neighbors_f32_all", "initial_neighbors_f32_fluid", "end_neighbors_f32_all", "end_neighbors_f32_fluid",
              "end_neighbors_lo_all", "end_neighbors_hi_all", "end_ratio_candidates_neighbors",
              "gpu_frame_us", "gpu_frame_std_us", "gpu_step_amortized_us", "step_ps_per_candidate",
              "correction_density_ps_per_candidate", "force_ps_per_candidate", "gpu_throughput_mps",
              "a_wall_d1_us", "a_wall_d2_us", "idle_d1", "idle_d2", "memory_bytes_per_particle"]
    with open(directory / "summary.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for size, d in summary.items():
            ini = d.get("initial") or {}
            end = d.get("end") or {}
            writer.writerow({
                "size": size, "particles": d["particles"], "fluid_particles": d["fluid_particles"], "h_over_dx": ini.get("h_over_dx"),
                "grid": "x".join(str(v) for v in d["grid_dimensions"]), "total_voxels": d["total_voxels"],
                "initial_occupied": ini.get("occupied_voxels"), "end_occupied": d["end_occupied_voxels"],
                "initial_mean_per_voxel": ini.get("mean_per_voxel"), "initial_max_per_voxel": ini.get("max_per_voxel"),
                "end_mean_per_voxel": d["end_mean_per_voxel"], "end_max_per_voxel": d["end_max_per_voxel"],
                "initial_candidates": ini.get("candidates_per_particle"), "end_candidates": d["candidates_per_particle"],
                "initial_candidates_fluid": ini.get("candidates_per_fluid_particle"), "end_candidates_fluid": d["candidates_per_fluid_particle"],
                "initial_neighbors_f32_all": ini.get("neighbors_f32_all"), "initial_neighbors_f32_fluid": ini.get("neighbors_f32_fluid"),
                "end_neighbors_f32_all": end.get("neighbors_f32_all"), "end_neighbors_f32_fluid": end.get("neighbors_f32_fluid"),
                "end_neighbors_lo_all": end.get("neighbors_lo_all"), "end_neighbors_hi_all": end.get("neighbors_hi_all"),
                "end_ratio_candidates_neighbors": end.get("ratio_candidates_to_neighbors"),
                "gpu_frame_us": d["gpu_frame_us"], "gpu_frame_std_us": d["gpu_frame_std_us"],
                "gpu_step_amortized_us": d["gpu_step_amortized_us"], "step_ps_per_candidate": d["step_ps_per_candidate"],
                "correction_density_ps_per_candidate": d["stages"].get("correction_density", {}).get("ps_per_candidate"),
                "force_ps_per_candidate": d["stages"].get("force", {}).get("ps_per_candidate"),
                "gpu_throughput_mps": d["gpu_throughput_mps"],
                "a_wall_d1_us": d["a"].get("1", {}).get("wall_us"), "a_wall_d2_us": d["a"].get("2", {}).get("wall_us"),
                "idle_d1": d["a"].get("1", {}).get("idle_vs_amortized"), "idle_d2": d["a"].get("2", {}).get("idle_vs_amortized"),
                "memory_bytes_per_particle": (d.get("memory") or {}).get("device_local_bytes_per_particle")})
    print(table2)
    print(sizes)
    print(f"[section3] {len(records)} runs, {len(summary)} sizes, telemetry samples {len(samples)} -> {directory}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
