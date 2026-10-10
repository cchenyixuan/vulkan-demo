"""
_build_artifact.py — assemble the v7_single report page (every table and figure of docs/seam_audit/v7_single.md)
as one self-contained HTML fragment for the Artifact tool: the document's sections (a small Markdown subset:
headers, paragraphs, bullet lists, tables, fenced code, images), the campaign summaries and the figures (PNG
embedded as data URIs). The page follows the artifact page contract (no document skeleton of its own, <title> and
<style> first, colour tokens on :root with a dark variant, phone-width layout).

    .venv/Scripts/python.exe experiment/v7_single/_build_artifact.py --out logs/v7_single/report/v7_single_report.html
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import pathlib
import re
import sys

_REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]

# layout: one reading column (max 1080px) with a verdict card on top; tables scroll inside their own box
STYLE = """
:root { --bg: #faf9f6; --fg: #1b1f24; --muted: #5c6370; --line: #d8d5cc; --accent: #1f4e8c; --accent-soft: #e6eef8;
  --code: #f1efe9; --ok: #2e7d4f; --ok-soft: #e3f1e8;
  --font-body: -apple-system, "Segoe UI", "PingFang SC", "Microsoft YaHei", "Noto Sans CJK SC", Roboto, sans-serif;
  --font-mono: Consolas, "SFMono-Regular", Menlo, "Cascadia Mono", monospace; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg: #15171a; --fg: #e8e6e1; --muted: #a2a69f;
  --line: #343842; --accent: #7fb0ea; --accent-soft: #1e2b3c; --code: #1e2126; --ok: #6fcf97; --ok-soft: #1d2f26; color-scheme: dark; } }
:root[data-theme="dark"] { --bg: #15171a; --fg: #e8e6e1; --muted: #a2a69f; --line: #343842; --accent: #7fb0ea;
  --accent-soft: #1e2b3c; --code: #1e2126; --ok: #6fcf97; --ok-soft: #1d2f26; color-scheme: dark; }
body { margin: 0; padding-inline: 16px; padding-block: 24px 64px; background: var(--bg); color: var(--fg);
  font: 15px/1.6 var(--font-body); font-variant-numeric: tabular-nums; }
main { max-width: 1080px; margin: 0 auto; min-width: 0; }
h1 { font-size: 1.55em; line-height: 1.25; margin: 0 0 6px; text-wrap: balance; }
h2 { font-size: 1.25em; margin: 40px 0 10px; padding-bottom: 4px; border-bottom: 1px solid var(--line); text-wrap: balance; }
h3 { font-size: 1.05em; margin: 26px 0 8px; } h4 { font-size: 1em; margin: 18px 0 6px; color: var(--muted); }
p { margin: 8px 0; } ul, ol { padding-left: 1.3em; } li { margin: 3px 0; }
a { color: var(--accent); }
code { font: 0.9em/1.4 var(--font-mono); background: var(--code); padding: 1px 4px; border-radius: 3px; }
pre { background: var(--code); padding: 10px 12px; border-radius: 6px; overflow-x: auto; font: 0.84em/1.45 var(--font-mono); }
pre code { background: none; padding: 0; }
.tablewrap { overflow-x: auto; margin: 10px 0 16px; max-width: 100%; }
table { border-collapse: collapse; font-size: 0.86em; white-space: nowrap; }
th, td { border: 1px solid var(--line); padding: 4px 8px; text-align: left; vertical-align: top; }
th { background: var(--code); font-weight: 600; }
figure { margin: 18px 0; } figure img { max-width: 100%; height: auto; border: 1px solid var(--line); background: #fff; }
figcaption { color: var(--muted); font-size: 0.9em; margin-top: 6px; }
.meta { color: var(--muted); font-size: 0.9em; }
.verdict { border: 1px solid var(--line); border-left: 4px solid var(--accent); background: var(--accent-soft);
  padding: 12px 16px; margin: 16px 0 8px; border-radius: 0 6px 6px 0; }
.verdict h2 { margin: 0 0 6px; border: 0; padding: 0; font-size: 1.1em; }
.chips { display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0 18px; }
.chip { display: inline-flex; align-items: baseline; gap: 6px; padding: 4px 10px; border-radius: 999px; font-size: 0.85em;
  background: var(--code); border: 1px solid var(--line); }
.chip.ok { background: var(--ok-soft); border-color: var(--ok); color: var(--fg); }
.chip b { font-weight: 600; }
.chip .label { color: var(--muted); letter-spacing: 0.02em; font-size: 0.92em; }
img.inline { max-width: 100%; height: auto; border: 1px solid var(--line); background: #fff; margin: 8px 0; }
:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
@media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }
"""


def embed_png(path: pathlib.Path) -> str | None:
    if not path.exists():
        return None
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def render_markdown(text: str, image_root: pathlib.Path | None = None) -> str:
    """Headers, paragraphs, bullet / numbered lists, tables, fenced code, images and inline code / bold."""
    lines = text.splitlines()
    out: list[str] = []
    index = 0
    paragraph: list[str] = []

    def inline(value: str) -> str:
        value = html.escape(value, quote=False)
        value = re.sub(r"`([^`]+)`", r"<code>\1</code>", value)
        value = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", value)
        value = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', value)
        return value

    def flush() -> None:
        if paragraph:
            out.append("<p>" + inline(" ".join(paragraph)) + "</p>")
            paragraph.clear()

    while index < len(lines):
        line = lines[index]
        if line.startswith("```"):
            flush()
            block = []
            index += 1
            while index < len(lines) and not lines[index].startswith("```"):
                block.append(lines[index])
                index += 1
            out.append("<pre><code>" + html.escape("\n".join(block)) + "</code></pre>")
            index += 1
            continue
        image = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)\s*$", line)
        if image:
            flush()
            source = embed_png((image_root / image.group(2)).resolve()) if image_root is not None else None
            if source:
                out.append(f'<img class="inline" src="{source}" alt="{html.escape(image.group(1))}">')
            else:
                out.append(f'<p class="meta">figure: {html.escape(image.group(2))}</p>')
            index += 1
            continue
        if line.startswith("|") and index + 1 < len(lines) and re.match(r"^\|?\s*-{3,}", lines[index + 1]):
            flush()
            header = [cell.strip() for cell in line.strip().strip("|").split("|")]
            rows = []
            index += 2
            while index < len(lines) and lines[index].startswith("|"):
                rows.append([cell.strip() for cell in lines[index].strip().strip("|").split("|")])
                index += 1
            out.append('<div class="tablewrap"><table><thead><tr>' + "".join(f"<th>{inline(cell)}</th>" for cell in header)
                       + "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{inline(cell)}</td>" for cell in row) + "</tr>"
                                                             for row in rows) + "</tbody></table></div>")
            continue
        match = re.match(r"^(#{1,4})\s+(.*)$", line)
        if match:
            flush()
            level = len(match.group(1))
            out.append(f"<h{level}>{inline(match.group(2))}</h{level}>")
            index += 1
            continue
        if re.match(r"^\s*[-*]\s+", line):
            flush()
            items = []
            while index < len(lines) and re.match(r"^\s*[-*]\s+", lines[index]):
                item = re.sub(r"^\s*[-*]\s+", "", lines[index])
                index += 1
                while index < len(lines) and lines[index].startswith("  ") and not re.match(r"^\s*[-*]\s+", lines[index]):
                    item += " " + lines[index].strip()
                    index += 1
                items.append(f"<li>{inline(item)}</li>")
            out.append("<ul>" + "".join(items) + "</ul>")
            continue
        if re.match(r"^\s*\d+\.\s+", line):
            flush()
            items = []
            while index < len(lines) and re.match(r"^\s*\d+\.\s+", lines[index]):
                item = re.sub(r"^\s*\d+\.\s+", "", lines[index])
                index += 1
                while index < len(lines) and lines[index].startswith("  ") and not re.match(r"^\s*(\d+\.|[-*])\s+", lines[index]):
                    item += " " + lines[index].strip()
                    index += 1
                items.append(f"<li>{inline(item)}</li>")
            out.append("<ol>" + "".join(items) + "</ol>")
            continue
        if not line.strip():
            flush()
            index += 1
            continue
        paragraph.append(line.strip())
        index += 1
    flush()
    return "\n".join(out)


def figure_html(path: pathlib.Path, caption: str) -> str:
    source = embed_png(path)
    if source is None:
        return f'<p class="meta">figure missing: {html.escape(str(path))}</p>'
    return (f'<figure><img src="{source}" alt="{html.escape(caption)}">'
            f"<figcaption>{html.escape(caption)}</figcaption></figure>")


def chips_html(root: pathlib.Path, cavity_summary: pathlib.Path, throughput_summary: pathlib.Path) -> str:
    """Verdict chips computed from the summary files (the cavity report JSON and the throughput campaign JSON)."""
    chips: list[str] = []
    try:
        cavity = json.loads((root / cavity_summary).read_text(encoding="utf-8"))
        u_l2 = 100 * cavity["errors"]["u"]["L2_relative"]
        v_l2 = 100 * cavity["errors"]["v"]["L2_relative"]
        ok = u_l2 < 1.0 and v_l2 < 1.0
        chips.append(f'<span class="chip{" ok" if ok else ""}"><span class="label">1000² adami 相对 L2</span>'
                     f"<b>u {u_l2:.2f} % · v {v_l2:.2f} %</b></span>")
        psi = cavity["extrema"]["psi_min"]["deviation_percent"]
        chips.append(f'<span class="chip"><span class="label">ψ_min 偏差</span><b>{psi:+.2f} %</b></span>')
        chips.append(f'<span class="chip"><span class="label">t_s</span><b>{cavity["t_steady_online"]:.2f}</b></span>')
        invariants = cavity.get("invariants", {})
        bad = [key for key, value in invariants.items()
               if isinstance(value, int) and not isinstance(value, bool) and key != "segments" and value]
        chips.append(f'<span class="chip{"" if bad else " ok"}"><span class="label">不变量</span>'
                     f'<b>{"全 0" if not bad else ", ".join(bad)}</b></span>')
    except (OSError, KeyError, ValueError, TypeError):
        chips.append('<span class="chip"><span class="label">1000² adami</span><b>无汇总</b></span>')
    try:
        throughput = json.loads((root / throughput_summary).read_text(encoding="utf-8"))
        loss = throughput["paper"]["depth1_loss_S"]
        chips.append(f'<span class="chip"><span class="label">depth-1 损失 S</span>'
                     f'<b>1M {100 * loss["1m"]["mean"]:+.1f} % · 32M {100 * loss["32m"]["mean"]:+.1f} %</b></span>')
        chips.append(f'<span class="chip ok"><span class="label">测量 A 运行</span><b>{throughput.get("ok_runs", "?")} 有效 · '
                     f'{throughput.get("failed_runs", 0)} 失败</b></span>')
    except (OSError, KeyError, ValueError, TypeError):
        pass
    return '<div class="chips">' + "".join(chips) + "</div>"


def section3_chips_html(root: pathlib.Path, section3_summary: pathlib.Path, memory_json: pathlib.Path) -> str:
    """Chips of "§3 补测": neighbours / candidates per particle, the GPU idle fraction at 2 in flight, memory."""
    chips: list[str] = []
    try:
        summary = json.loads((root / section3_summary).read_text(encoding="utf-8"))
        sizes = summary["sizes"]
        eight = sizes["8m"]
        chips.append(f'<span class="chip"><span class="label">2-D 8M 每粒子 邻居 / 候选</span>'
                     f'<b>{eight["end"]["neighbors_f32_all"]:.1f} / {eight["candidates_per_particle"]:.0f}</b></span>')
        if "3d_8m" in sizes:
            three = sizes["3d_8m"]
            chips.append(f'<span class="chip"><span class="label">3-D 8M 每粒子 邻居 / 候选</span>'
                         f'<b>{three["end"]["neighbors_f32_all"]:.1f} / {three["candidates_per_particle"]:.0f}</b></span>')
        idle = [sizes[s]["a"]["2"]["idle_vs_amortized"] for s in ("1m", "2m", "4m", "8m", "16m", "32m") if s in sizes]
        if idle:
            chips.append(f'<span class="chip"><span class="label">2 在飞 GPU 空闲(2-D)</span>'
                         f'<b>{100 * min(idle):+.2f} … {100 * max(idle):+.2f} %</b></span>')
        dip = summary.get("dip_4m", {})
        for point, label in (("4m", "4M(旧手工)每候选 ps 对 2M / 8M"), ("4m_gen", "4M gen 每候选 ps 对 2M / 8M")):
            key = f"ps_per_candidate_step_{point}_vs_neighbours_pct"
            if key in dip:
                chips.append(f'<span class="chip"><span class="label">{label}</span><b>{dip[key]:+.2f} %</b></span>')
    except (OSError, KeyError, ValueError, TypeError):
        chips.append('<span class="chip"><span class="label">§3 补测</span><b>无汇总</b></span>')
    try:
        memory = {m["size"]: m for m in json.loads((root / memory_json).read_text(encoding="utf-8"))}
        chips.append(f'<span class="chip"><span class="label">device-local B / 粒子</span>'
                     f'<b>2-D {memory["8m"]["device_local_bytes_per_particle"]:.0f} · 3-D {memory["3d_8m"]["device_local_bytes_per_particle"]:.0f}</b></span>')
    except (OSError, KeyError, ValueError, TypeError):
        pass
    return '<div class="chips">' + "".join(chips) + "</div>" if chips else ""


def split_tldr(text: str) -> tuple[str, str, str]:
    """(before, the TL;DR section without its header, after) of the report's Markdown."""
    match = re.search(r"^## 结论.*$", text, flags=re.MULTILINE)
    if not match:
        return text, "", ""
    rest = text[match.end():]
    following = re.search(r"^## ", rest, flags=re.MULTILINE)
    if following is None:
        return text[:match.start()], rest, ""
    return text[:match.start()], rest[:following.start()], rest[following.start():]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--doc", default="docs/seam_audit/v7_single.md")
    parser.add_argument("--throughput", default="logs/v7_single/throughput_r2")
    parser.add_argument("--throughput-summary", default="docs/seam_audit/v7_single/throughput_summary.json")
    parser.add_argument("--breakdown", default="logs/v7_single/kernel_breakdown_v2")
    parser.add_argument("--cavity-tables", default="docs/validation/data/cavity_re1000_tables_v7single.md")
    parser.add_argument("--cavity-summary", default="docs/validation/data/cavity_re1000_summary_v7single.json")
    parser.add_argument("--figures", default="manuscripts/fig")
    parser.add_argument("--section3", default="logs/v7_single/section3",
                        help="the §3 补测 campaign directory (summary.json, table2.md, sizes.md)")
    parser.add_argument("--memory-json", default="docs/seam_audit/v7_single/memory.json")
    parser.add_argument("--memory-md", default="docs/seam_audit/v7_single/memory.md")
    parser.add_argument("--platform-md", default="docs/seam_audit/v7_single/platform.md")
    arguments = parser.parse_args()
    root = _REPOSITORY_ROOT
    doc_path = root / arguments.doc
    before, tldr, after = split_tldr(doc_path.read_text(encoding="utf-8"))
    image_root = doc_path.parent
    body: list[str] = []
    body.append("<h1>v7_single:与 v7 同 kernel 的纯单卡构建</h1>")
    body.append('<p class="meta">报告页由 experiment/v7_single/_build_artifact.py 从 docs/seam_audit/v7_single.md 与各汇总文件生成;'
                "表与图都来自仓库里的数据文件。</p>")
    body.append(chips_html(root, pathlib.Path(arguments.cavity_summary), pathlib.Path(arguments.throughput_summary)))
    body.append(section3_chips_html(root, pathlib.Path(arguments.section3) / "summary.json", pathlib.Path(arguments.memory_json)))
    opening = re.sub(r"^# .*$", "", before, count=1, flags=re.MULTILINE)
    body.append(render_markdown(opening, image_root))
    if tldr.strip():
        body.append('<section class="verdict"><h2>结论(TL;DR)</h2>' + render_markdown(tldr, image_root) + "</section>")
    body.append(render_markdown(after, image_root))
    figures = root / arguments.figures
    body.append("<h2>附:图</h2>")
    body.append(figure_html(figures / "single_gpu_throughput.png",
                            "Figure 1 (single panel): v7_single throughput (M particle-steps/s), 1 and 2 frames in flight, 1M–32M, "
                            "one RTX 5090; the 4M point is the generator-built case lid_driven_cavity_2d_4m_gen."))
    body.append(figure_html(figures / "single_gpu_throughput_old_4m.png",
                            "The previous single-panel version with the hand-made 4M case (kept as _old_4m); the two curves are "
                            "indistinguishable at 4M (732.4 vs 731.7 M particle-steps/s at 2 in flight)."))
    body.append(figure_html(figures / "single_gpu_throughput_two_panel.png",
                            "The earlier two-panel version (measurement C): throughput and fps; kept under the _two_panel name."))
    for name, caption in (("cavity_re1000_profiles", "Centre-line profiles of the 1000² adami run against Marchi et al. 2021 Tc and Ghia 1982."),
                          ("cavity_re1000_field", "Time-averaged stream function (a) and vorticity (b) of the 1000² adami run."),
                          ("cavity_re1000_steady", "Supplementary: kinetic energy and the window criterion."),
                          ("cavity_re1000_errors", "Supplementary: pointwise errors at the reference points.")):
        body.append(figure_html(figures / f"{name}.png", caption))
    body.append("<h2>附:测量 A 汇总(原始)</h2>")
    summary = root / arguments.throughput / "summary.md"
    body.append(render_markdown(summary.read_text(encoding="utf-8")) if summary.exists() else "<p class='meta'>no summary</p>")
    body.append("<h2>附:测量 B 汇总(原始)</h2>")
    for setting in ("fused", "unfused"):
        path = root / arguments.breakdown / f"summary_{setting}.md"
        if path.exists():
            body.append(f"<h3>{setting}</h3>" + render_markdown(path.read_text(encoding="utf-8")))
    cavity = root / arguments.cavity_tables
    if cavity.exists():
        body.append("<h2>附:方腔表与图注(原始)</h2>" + render_markdown(cavity.read_text(encoding="utf-8")))
    section3 = root / arguments.section3
    for name, title in (("table2.md", "附:§3 补测 表 2 汇总(原始)"), ("sizes.md", "附:§3 补测 各尺寸汇总(原始)")):
        path = section3 / name
        if path.exists():
            body.append(f"<h2>{title}</h2>" + render_markdown(path.read_text(encoding="utf-8")))
    for path, title in ((root / arguments.memory_md, "附:内存清单(原始)"), (root / arguments.platform_md, "附:平台记录(原始)")):
        if path.exists():
            text = re.sub(r"^# .*$", "", path.read_text(encoding="utf-8"), count=1, flags=re.MULTILINE)
            text = re.sub(r"^## ", "### ", text, flags=re.MULTILINE)       # the file's own sections under this h2
            body.append(f"<h2>{title}</h2>" + render_markdown(text))
    page ="<title>v7_single 报告</title>\n<style>" + STYLE + "</style>\n<main>\n" + "\n".join(body) + "\n</main>\n"
    out = pathlib.Path(arguments.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(f"[artifact] {len(page) / 1e6:.2f} MB -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
