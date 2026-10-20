"""xv_r_own.py — r_own of the E2 trace runs under the loss_sources.pdf definition (E7 report 4.4, e7_loss_figure.py):
per step the larger of the two links' t_own / T_B (T_B of the link's receiver), then the median over steps 500-1498;
beside it the value the tables print so far (the larger of the two links' own medians), and each link's median.

t_own = t_chain minus the two transfer queues' idle waits (e7_full_summary.transfer_queue_idle), as in xv_e0.trace_summary.

Usage:
    python docs/cross_vendor/scripts/xv_r_own.py --e12 docs/cross_vendor/data/e12 [--sizes 520,...] [--last-step 1498]
        [--json docs/cross_vendor/data/tables/r_own_worst_step.json]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve()
REPOSITORY = HERE.parents[3]
sys.path.insert(0, str(REPOSITORY))
sys.path.insert(0, str(REPOSITORY / "docs" / "cluster_v6" / "scripts"))
from experiment.v7.analysis import step_trace_model  # noqa: E402
import e7_full_summary  # noqa: E402


def r_own_of(trace_directory: pathlib.Path, first_step: int, last_step: int) -> dict:
    run = step_trace_model.load_run(trace_directory)
    metrics = step_trace_model.link_metrics(run)
    idle = e7_full_summary.transfer_queue_idle(run)
    per_link = {}
    for name, values in metrics.items():
        steps = values["step"].astype(int)
        keep = (steps >= first_step) & (steps <= last_step)
        readback_idle = np.array([idle.get((name, int(step)), (0.0, 0.0))[0] for step in steps])
        upload_idle = np.array([idle.get((name, int(step)), (0.0, 0.0))[1] for step in steps])
        t_own = values["t_chain"] - readback_idle - upload_idle
        per_link[name] = {"step": steps[keep], "r_own": (t_own / values["T_B"])[keep], "t_own": t_own[keep], "T_B": values["T_B"][keep]}
    common = sorted(set.intersection(*(set(v["step"].tolist()) for v in per_link.values())))
    stacked = []
    for name, values in per_link.items():
        index = {int(step): position for position, step in enumerate(values["step"])}
        stacked.append(values["r_own"][[index[step] for step in common]])
    worst = np.max(np.vstack(stacked), axis=0)
    return {"steps": len(common), "first_step": int(common[0]), "last_step": int(common[-1]),
            "r_own_worst_step_p50": float(np.median(worst)), "r_own_worst_step_p95": float(np.percentile(worst, 95)),
            "per_link_p50": {name: float(np.median(values["r_own"])) for name, values in per_link.items()},
            "max_of_link_medians": max(float(np.median(values["r_own"])) for values in per_link.values()),
            "share_of_steps_worst_is": {name: float(np.mean(np.argmax(np.vstack(stacked), axis=0) == position))
                                        for position, name in enumerate(per_link)}}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e12", default=str(REPOSITORY / "docs" / "cross_vendor" / "data" / "e12"))
    parser.add_argument("--sizes", default="520,1000,1440,2000,2840,4000,5640")
    parser.add_argument("--first-step", type=int, default=500)
    parser.add_argument("--last-step", type=int, default=1498)
    parser.add_argument("--json", default=None)
    arguments = parser.parse_args()
    out = {}
    lines = ["| 尺寸 | r_own（每步两链路取大，再取中位数） | p95 | 两链路各自中位数中较大的（现表） | 5090→XTX 中位 | XTX→5090 中位 | 每步取大时来自 XTX→5090 的份额 | 步数 |",
             "|---|---|---|---|---|---|---|---|"]
    for size in arguments.sizes.split(","):
        trace = pathlib.Path(arguments.e12) / "traces" / f"2d_n{size}_K2_trace"
        result = r_own_of(trace, arguments.first_step, arguments.last_step)
        out[size] = result
        links = result["per_link_p50"]
        lines.append(f"| {size}² | **{result['r_own_worst_step_p50']:.3f}** | {result['r_own_worst_step_p95']:.3f} | {result['max_of_link_medians']:.3f} | "
                     f"{links['s0_to_s1']:.3f} | {links['s1_to_s0']:.3f} | {result['share_of_steps_worst_is']['s1_to_s0'] * 100:.0f} % | "
                     f"{result['steps']}（{result['first_step']}–{result['last_step']}） |")
    text = "\n".join(lines)
    print(text)
    if arguments.json:
        pathlib.Path(arguments.json).write_text(json.dumps({"definition": "per step max over the two links of t_own / T_B(receiver), "
                                                            "median over steps first..last; t_own = t_chain - transfer queue idle",
                                                            "first_step": arguments.first_step, "last_step": arguments.last_step,
                                                            "sizes": out, "table": text}, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
