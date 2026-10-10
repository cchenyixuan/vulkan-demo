"""
_run_stride_campaign.py — column-stride test of the 4M dip ("§3 补测" item 8 of docs/seam_audit/v7_single.md).

Hypothesis under test: with the x-slowest voxel encoding a voxel column is N_y voxels x (particles per voxel) x 16 B
long in the particle arrays; at 4M (N_y = 410) that stride is ~160 KiB = 5 x 32 KiB, and the two 4M cases that share
only the particle count and the 410^2 grid both run 6 % slower per candidate than 2M / 8M. Two generator cases near
4M with other grids (406^2, 414^2) separate "column length" from "particle count"; 6M (stride ~6 x 32 KiB) and 14M
(E5's second dip) are added, 2M and 8M are the controls.

The GPU runs are plain commands (documented in the doc's 复现 section), this script wraps the per-case breakdown
and the summary:

  breakdown   one _run_kernel_breakdown.py --solver v7_single --interleave 0 run per size (warmup 1000 + 2000 timed
              frames, no state dump), process-based idle check before every run, nvidia-smi 1 Hz telemetry;
              -> <out>/breakdown/results.jsonl
  summarize   per size: N_y, column stride (N_y x mean particles per voxel x 16 B, KiB and multiples of 32 KiB),
              candidates per particle, GPU step, ps per candidate (step / fused / force), measurement A's S d2
              throughput (from <out>/throughput_d2/summary.json) and the 4m_gen depth re-measurement
              (<out>/throughput_4m_gen_depth/summary.json); scatter of ps/candidate (step) against the stride
              -> <out>/summary.{json,md}, <out>/stride_scatter.png

    python experiment/v7_single/_run_stride_campaign.py breakdown --out logs/v7_single/stride
    python experiment/v7_single/_run_stride_campaign.py summarize --out logs/v7_single/stride
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import statistics
import subprocess
import sys
import time

_REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

PYTHON = sys.executable
HEADLESS_5090_UUID = "ae137c668a40f5acaab90a83f7cda175"
EXPECT_GPU = "RTX 5090"
TELEMETRY_INDEX = 1
STRIDE_SIZES = ["2m", "4m_gen_n406", "4m_gen", "4m_gen_n414", "6m", "8m", "10m", "14m"]
SIZE_LABEL = {"2m": "2M", "4m_gen_n406": "4M gen 406²", "4m_gen": "4M gen 410²", "4m_gen_n414": "4M gen 414²",
              "6m": "6M", "8m": "8M", "10m": "10M", "14m": "14M"}
KIB = 1024.0
STRIDE_UNIT = 32 * 1024.0


def case_of(size: str) -> str:
    from _run_single_baseline_campaign import SIZES
    for tag, case, _ in SIZES:
        if tag == size:
            return case
    raise KeyError(size)


def log_line(out: pathlib.Path, message: str) -> None:
    stamp = datetime.datetime.now().strftime("%H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    with open(out / "campaign.log", "a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def git_state() -> dict:
    def run(*arguments):
        return subprocess.run(["git", *arguments], cwd=_REPOSITORY_ROOT, capture_output=True, text=True).stdout.strip()
    tracked = [line for line in run("status", "--porcelain", "--untracked-files=no").splitlines() if line.strip()]
    return {"commit": run("rev-parse", "HEAD"), "describe": run("describe", "--tags", "--always"), "tracked_dirty": tracked}


def wait_until_idle(out: pathlib.Path) -> None:
    """Block until no other solver / test process runs (two consecutive checks), this script's own pair ignored."""
    from _run_single_baseline_campaign import solver_processes
    quiet = waited = 0
    while quiet < 2:
        busy = solver_processes(os.getpid())
        if busy is None:
            log_line(out, "idle check skipped (process query failed)")
            return
        busy = [entry for entry in busy if "_run_stride_campaign" not in entry]
        if not busy:
            quiet += 1
        else:
            quiet = 0
            waited += 1
            if waited % 30 == 1:
                log_line(out, f"other processes still running, waiting: {busy}")
        time.sleep(1.0)


def command_environment() -> dict:
    env = {key: value for key, value in os.environ.items() if not key.startswith("V7_")}
    env["VK_LOADER_LAYERS_DISABLE"] = "VK_LAYER_KHRONOS_validation"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def run_breakdown(arguments, out: pathlib.Path) -> int:
    state = git_state()
    if state["tracked_dirty"] and not arguments.allow_dirty:
        log_line(out, f"tracked tree is dirty, refusing to measure: {state['tracked_dirty'][:5]}")
        return 2
    log_line(out, f"breakdown: commit {state['commit']} ({state['describe']})")
    directory = out / "breakdown"
    (directory / "runs").mkdir(parents=True, exist_ok=True)
    results_path = directory / "results.jsonl"
    done = set()
    if results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                if record.get("drift") == 0 and record.get("overflow") == 0:
                    done.add(record["size"])
    env = command_environment()
    telemetry = open(directory / "telemetry.csv", "a")
    sampler = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,index,clocks.sm,clocks.mem,power.draw,temperature.gpu,"
                                "utilization.gpu,memory.used,pcie.link.gen.current,pcie.link.width.current", "--format=csv", "-l", "1"],
                               stdout=telemetry, stderr=subprocess.STDOUT)
    stop = False
    try:
        for size in arguments.sizes.split(","):
            if size in done:
                continue
            wait_until_idle(out)
            command = [PYTHON, str(_REPOSITORY_ROOT / "_run_kernel_breakdown.py"), "--solver", "v7_single",
                       "--gpu-uuid", HEADLESS_5090_UUID, "--expect-gpu", EXPECT_GPU, "--case", case_of(size),
                       "--warmup", str(arguments.warmup), "--measure", str(arguments.measure), "--interleave", "0",
                       "--obj-cache", arguments.obj_cache, "--tag", size, "--run", "1"]
            log_line(out, f"breakdown {size}: {case_of(size)}")
            started = time.time()
            completed = subprocess.run(command, cwd=_REPOSITORY_ROOT, env=env, capture_output=True, text=True,
                                       encoding="utf-8", errors="replace")
            text = completed.stdout + "\n" + completed.stderr
            (directory / "runs" / f"{size}.log").write_text(text, encoding="utf-8")
            record = None
            for line in text.splitlines():
                if line.startswith("RESULT "):
                    record = json.loads(line[len("RESULT "):])
            if record is None:
                log_line(out, f"  FAILED rc {completed.returncode}")
                stop = True
                break
            record.update({"size": size, "commit": state["commit"], "wall_s": round(time.time() - started, 1)})
            with open(results_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            c = record["candidates"]
            log_line(out, f"  gpu frame {record['mean_us']['gpu_frame']:.0f} us, grid {c['grid_dimensions']}, "
                          f"candidates/particle {c['candidates_per_particle']:.1f}, drift {record['drift']} overflow "
                          f"{record['overflow']} ({record['wall_s']} s)")
            if record["drift"] != 0 or record["overflow"] != 0:
                log_line(out, "  STOP: drift or overflow != 0")
                stop = True
                break
    finally:
        sampler.terminate()
        try:
            sampler.wait(timeout=10)
        except subprocess.TimeoutExpired:
            sampler.kill()
        telemetry.close()
    return 3 if stop else 0


def load_jsonl(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def summarize(arguments, out: pathlib.Path) -> int:
    from experiment.v7_single._summarize_section3 import covariates, load_telemetry
    records = {r["size"]: r for r in load_jsonl(out / "breakdown" / "results.jsonl") if r.get("drift") == 0 and r.get("overflow") == 0}
    samples = load_telemetry(out / "breakdown" / "telemetry.csv", TELEMETRY_INDEX)
    d2_path = out / "throughput_d2" / "summary.json"
    d2 = json.loads(d2_path.read_text(encoding="utf-8")) if d2_path.exists() else {"rows": {}}
    depth_path = out / "throughput_4m_gen_depth" / "summary.json"
    depth = json.loads(depth_path.read_text(encoding="utf-8")) if depth_path.exists() else None
    sizes = [s for s in arguments.sizes.split(",") if s in records]
    rows = []
    for size in sizes:
        r = records[size]
        c = r["candidates"]
        ny = c["grid_dimensions"][1]
        stride = ny * c["mean_particles_per_voxel"] * 16.0
        pairs = c["candidate_pairs"]
        frame = r["mean_us"]["gpu_frame"]
        a = d2["rows"].get(f"{size}/v7_single/2")
        rows.append({"size": size, "label": SIZE_LABEL.get(size, size), "case": r["case"], "particles": r["particles"],
                     "commit": r.get("commit"), "grid": c["grid_dimensions"], "ny": ny,
                     "mean_particles_per_voxel": c["mean_particles_per_voxel"], "max_particles_per_voxel": c["max_particles_per_voxel"],
                     "stride_bytes": stride, "stride_kib": stride / KIB, "stride_over_32kib": stride / STRIDE_UNIT,
                     "candidates_per_particle": c["candidates_per_particle"], "candidate_pairs": pairs,
                     "gpu_frame_us": frame, "gpu_step_amortized_us": r.get("gpu_step_amortized_us"),
                     "gpu_throughput_mps": r["particles"] / frame,
                     "ps_step": 1e6 * frame / pairs,
                     "ps_correction_density": 1e6 * r["mean_us"]["correction_density"] / pairs,
                     "ps_force": 1e6 * r["mean_us"]["force"] / pairs,
                     "a_d2_fps": a["fps_mean"] if a else None, "a_d2_fps_std": a["fps_std"] if a else None,
                     "a_d2_mps": a["mparticle_steps_per_s"] if a else None, "a_d2_n": a["n"] if a else 0,
                     "covariates": covariates(samples, r.get("epoch_start"), r.get("epoch_end"))})
    controls = [row for row in rows if row["size"] in ("2m", "8m")]
    reference = statistics.fmean(row["ps_step"] for row in controls) if controls else None
    for row in rows:
        row["ps_step_vs_controls_pct"] = 100.0 * (row["ps_step"] / reference - 1) if reference else None
        row["ps_fused_vs_controls_pct"] = (100.0 * (row["ps_correction_density"] / statistics.fmean(c["ps_correction_density"] for c in controls) - 1)
                                           if controls else None)
        row["ps_force_vs_controls_pct"] = (100.0 * (row["ps_force"] / statistics.fmean(c["ps_force"] for c in controls) - 1)
                                           if controls else None)
    depth_result = None
    if depth:
        loss = depth.get("depth1_loss", {}).get("4m_gen|S")
        d1 = depth["rows"].get("4m_gen/v7_single/1")
        dd2 = depth["rows"].get("4m_gen/v7_single/2")
        depth_result = {"loss_mean": loss["mean"] if loss else None, "loss_std": loss["std"] if loss else None,
                        "n": loss["n"] if loss else 0, "d1_fps": d1["fps_mean"] if d1 else None, "d1_std": d1["fps_std"] if d1 else None,
                        "d2_fps": dd2["fps_mean"] if dd2 else None, "d2_std": dd2["fps_std"] if dd2 else None,
                        "d1_mps": d1["mparticle_steps_per_s"] if d1 else None, "d2_mps": dd2["mparticle_steps_per_s"] if dd2 else None,
                        "d1_sm_mhz": d1.get("sm_mhz") if d1 else None, "d2_sm_mhz": dd2.get("sm_mhz") if dd2 else None}
    summary = {"rows": rows, "reference_ps_step_2m_8m": reference, "depth_4m_gen": depth_result,
               "d2_ok_runs": d2.get("ok_runs"), "d2_failed_runs": d2.get("failed_runs"),
               "commit": rows[0]["commit"] if rows else None}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    # markdown
    lines = ["# §3 补测 第 8 项:4M 凹陷与体素列步长", "",
             "列步长 = N_y × 每体素平均粒子数(窗末,占用体素)× 16 B(position_voxel_id 一行);GPU 步长:1 在飞、时间戳、预热 1000 + "
             "计时 2000(1 次);ps/候选 = µs/步 × 10⁶ / 候选对数;'对 2M / 8M' = 相对两个对照算例均值的偏差;S d2 = 测量 A 协议(2 在飞,"
             "3 次交错试验)。", "",
             "| 算例 | 粒子 | 网格 | N_y | 每体素 均值 / 最大 | 列步长 KiB | / 32 KiB | 候选 / 粒子 | GPU 步长 µs | GPU 侧 M/s | "
             "ps/候选 步 / 融合核 / force | 对 2M / 8M(步 / 融合核 / force) | S d2 fps | S d2 M/s | SM MHz |",
             "|" + "---|" * 15]
    for row in rows:
        c = row["covariates"]
        lines.append(f"| {row['label']} | {row['particles']:,} | {row['grid'][0]} × {row['grid'][1]} | {row['ny']} | "
                     f"{row['mean_particles_per_voxel']:.2f} / {row['max_particles_per_voxel']} | {row['stride_kib']:.1f} | "
                     f"{row['stride_over_32kib']:.3f} | {row['candidates_per_particle']:.1f} | {row['gpu_frame_us']:.0f} | "
                     f"{row['gpu_throughput_mps']:.1f} | {row['ps_step']:.3f} / {row['ps_correction_density']:.3f} / {row['ps_force']:.3f} | "
                     f"{row['ps_step_vs_controls_pct']:+.2f} / {row['ps_fused_vs_controls_pct']:+.2f} / {row['ps_force_vs_controls_pct']:+.2f} % | "
                     + (f"{row['a_d2_fps']:.2f} ± {row['a_d2_fps_std']:.2f} (n={row['a_d2_n']})" if row["a_d2_fps"] else "–")
                     + f" | " + (f"{row['a_d2_mps']:.1f}" if row["a_d2_mps"] else "–") + f" | {c['sm_mhz'] and f'{c['sm_mhz']:.0f}'} |")
    if depth_result:
        lines += ["", "## 4m_gen 的 1 帧在飞重测(机器空闲,3 次交错试验)", "",
                  f"S d1 {depth_result['d1_fps']:.2f} ± {depth_result['d1_std']:.2f} fps({depth_result['d1_mps']:.1f} M/s,SM "
                  f"{depth_result['d1_sm_mhz'] and f'{depth_result['d1_sm_mhz']:.0f}'} MHz),S d2 {depth_result['d2_fps']:.2f} ± "
                  f"{depth_result['d2_std']:.2f}({depth_result['d2_mps']:.1f} M/s),depth-1 损失 "
                  f"{100 * depth_result['loss_mean']:+.2f} ± {100 * depth_result['loss_std']:.2f} %(n = {depth_result['n']})。"]
    text = "\n".join(lines) + "\n"
    (out / "summary.md").write_text(text, encoding="utf-8")
    print(text)
    # scatter
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    xs = [row["stride_kib"] for row in rows]
    ys = [row["ps_step"] for row in rows]
    for k in range(int(min(xs) / 32), int(max(xs) / 32) + 2):
        ax.axvline(32 * k, color="#cccccc", linewidth=0.6, linestyle=":")
    ax.scatter(xs, ys, s=28, color="#1f4e8c", zorder=3)
    for row, x, y in zip(rows, xs, ys):
        ax.annotate(row["label"], (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7)
    if reference:
        ax.axhline(reference, color="#888888", linewidth=0.8, linestyle="--", label="mean of 2M and 8M")
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_xlabel("column stride N_y × particles per voxel × 16 B (KiB); dotted lines every 32 KiB")
    ax.set_ylabel("ps per candidate (whole step)")
    ax.set_title("v7_single, one RTX 5090: per-candidate step cost against the voxel-column stride", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "stride_scatter.png", dpi=200)
    plt.close(fig)
    print(f"[stride] {len(rows)} sizes -> {out / 'summary.md'}, {out / 'stride_scatter.png'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("breakdown", "summarize"))
    parser.add_argument("--out", default="logs/v7_single/stride")
    parser.add_argument("--sizes", default=",".join(STRIDE_SIZES))
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--measure", type=int, default=2000)
    parser.add_argument("--obj-cache", default="logs/_obj_npy_cache")
    parser.add_argument("--allow-dirty", action="store_true")
    arguments = parser.parse_args()
    out = _REPOSITORY_ROOT / arguments.out
    out.mkdir(parents=True, exist_ok=True)
    if arguments.command == "breakdown":
        return run_breakdown(arguments, out)
    return summarize(arguments, out)


if __name__ == "__main__":
    sys.exit(main())
