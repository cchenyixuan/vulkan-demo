"""
_build_tgv_artifact.py — assemble the Taylor–Green report page (docs/tgv/v8_tgv.md, the campaign summary, the figures,
the two acceptance tests) as one self-contained HTML fragment for the Artifact tool, with the renderer and the style
of experiment/v7_single/_build_artifact.py.

    .venv/Scripts/python.exe experiment/v8/_build_tgv_artifact.py --out logs/v8/report/v8_tgv_report.html
"""

from __future__ import annotations

import argparse
import html
import json
import pathlib
import re
import sys

_REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from experiment.v7_single._build_artifact import STYLE, figure_html, render_markdown, split_tldr   # noqa: E402


def chip(label: str, value: str, ok: bool = False) -> str:
    return f'<span class="chip{" ok" if ok else ""}"><span class="label">{html.escape(label)}</span><b>{html.escape(value)}</b></span>'


def chips_html(summary_path: pathlib.Path, periodic_test: pathlib.Path, equivalence_test: pathlib.Path) -> str:
    chips = []
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        runs = summary.get("runs", {})
        series = [runs[name] for name in summary.get("series", {}).get("re100", []) if name in runs]
        if series:
            ratios = [s["viscosity_ratio"] for s in series]
            chips.append(chip("ν_eff/ν, Re 100, 50²–800²", f"{min(ratios):.3f}–{max(ratios):.3f}"))
            errors = [s["errors"]["1"]["velocity_relative"] for s in series]
            chips.append(chip("速度相对 L2 误差 t=1", f"{100 * min(errors):.2f}–{100 * max(errors):.2f} %"))
            shapes = [s["snapshot_errors"].get("t1", {}).get("velocity_relative_effective") for s in series]
            shapes = [v for v in shapes if v is not None]
            if shapes:
                chips.append(chip("扣除 ν_eff 后的形状误差 t=1", f"{100 * max(shapes):.2f} → {100 * min(shapes):.3f} %"))
        conv = summary.get("convergence", {}).get("re100", {})
        if conv:
            chips.append(chip("收敛阶(形状误差 / 原始误差)", f"{conv['shape_t1']['order']:.2f} / {conv['velocity_relative_t1']['order']:.2f}"))
        bad = [name for name, s in runs.items() if s["drift"] or s["overflow"]]
        chips.append(chip("drift / overflow", f"{len(runs)} 个运行全 0" if not bad else f"非 0: {', '.join(bad)}", ok=not bad))
    if periodic_test.exists():
        report = json.loads(periodic_test.read_text(encoding="utf-8"))
        checks = report.get("checks", {})
        difference = checks.get("periodic_kernel_sum", {}).get("max_abs_difference")
        chips.append(chip("周期核和 GPU−CPU", f"{difference:.1e}, {report.get('verdict')}" if difference is not None else report.get("verdict", "?"),
                          ok=report.get("verdict") == "PASS"))
    if equivalence_test.exists():
        report = json.loads(equivalence_test.read_text(encoding="utf-8"))
        chips.append(chip("方腔 v7_single ≡ v8", f"{report.get('verdict')} ({report.get('steps')} 步逐位)", ok=report.get("verdict") == "PASS"))
    return '<div class="chips">' + "".join(chips) + "</div>" if chips else ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--doc", default="docs/tgv/v8_tgv.md")
    parser.add_argument("--data", default="docs/tgv/data")
    parser.add_argument("--figures", default="manuscripts/fig")
    parser.add_argument("--periodic-test", default="docs/tgv/data/periodic_test.json")
    parser.add_argument("--equivalence-test", default="docs/tgv/data/equivalence_test.json")
    arguments = parser.parse_args()
    root = _REPOSITORY_ROOT
    doc_path = root / arguments.doc
    before, tldr, after = split_tldr(doc_path.read_text(encoding="utf-8"))
    image_root = doc_path.parent
    data = root / arguments.data
    body: list[str] = []
    body.append("<h1>v8:周期域 + Taylor–Green 涡精度</h1>")
    body.append('<p class="meta">报告页由 experiment/v8/_build_tgv_artifact.py 从 docs/tgv/v8_tgv.md、docs/tgv/data 与 manuscripts/fig 生成;'
                "表与图都来自仓库里的数据文件。</p>")
    body.append(chips_html(data / "summary.json", root / arguments.periodic_test, root / arguments.equivalence_test))
    opening = re.sub(r"^# .*$", "", before, count=1, flags=re.MULTILINE)
    body.append(render_markdown(opening, image_root))
    if tldr.strip():
        body.append('<section class="verdict"><h2>结论(TL;DR)</h2>' + render_markdown(tldr, image_root) + "</section>")
    body.append(render_markdown(after, image_root))
    figures = root / arguments.figures
    body.append("<h2>附:图</h2>")
    for name, caption in (
            ("tgv_energy", "Re 100, h/dx 5, Ma 0.1: (a) kinetic energy decay E_k(t)/E_k(0) at 50²–800² against the exact "
                           "exp(−4νk²t); (b) relative L2 velocity error against the analytic field."),
            ("tgv_convergence", "Relative L2 errors at t = 1 against L/dx: velocity (raw), velocity against the field with the "
                                "fitted ν_eff (shape error) and pressure (centred, relative to the instantaneous amplitude); "
                                "Re 100 and Re 1000; guides of first and second order."),
            ("tgv_viscosity", "ν_eff/ν from the energy decay (least squares of ln E over t ∈ [0.1, 2]) against L/dx; Re 100 and "
                              "Re 1000 at h/dx 5, with the h/dx 4, 3.5 and Ma 0.01 variants at 200²."),
            ("tgv_ablation", "Ablations at Re 100, 200²: δ+ default, no PST, no δ diffusion, no KCG, vanilla WCSPH: (a) energy "
                             "decay, (b) relative L2 velocity error."),
            ("tgv_fields", "200² default run at t = 1: speed, pressure (mean removed), velocity error, and the particles "
                           "across the periodic edge x = L (the right part drawn at x + L), coloured by speed.")):
        body.append(figure_html(figures / f"{name}.png", caption))
    summary_md = data / "summary.md"
    if summary_md.exists():
        text = re.sub(r"^# .*$", "", summary_md.read_text(encoding="utf-8"), count=1, flags=re.MULTILINE)
        text = re.sub(r"^## ", "### ", text, flags=re.MULTILINE)
        body.append("<h2>附:活动汇总(原始,_analyze_tgv.py)</h2>" + render_markdown(text))
    for path, title in ((root / arguments.periodic_test, "附:周期域验收(原始 JSON)"),
                        (root / arguments.equivalence_test, "附:方腔等价验收(原始 JSON)")):
        if path.exists():
            body.append(f"<h2>{title}</h2><pre><code>" + html.escape(path.read_text(encoding="utf-8")) + "</code></pre>")
    readme = root / "experiment/v8/README.md"
    if readme.exists():
        text = re.sub(r"^# .*$", "", readme.read_text(encoding="utf-8"), count=1, flags=re.MULTILINE)
        text = re.sub(r"^## ", "### ", text, flags=re.MULTILINE)
        body.append("<h2>附:experiment/v8 README</h2>" + render_markdown(text))
    page = "<title>v8 Taylor–Green 报告</title>\n<style>" + STYLE + "</style>\n<main>\n" + "\n".join(body) + "\n</main>\n"
    out = pathlib.Path(arguments.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(f"[artifact] {len(page) / 1e6:.2f} MB -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
