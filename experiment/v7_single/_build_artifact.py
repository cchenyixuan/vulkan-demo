"""
_build_artifact.py — assemble the v7_single report page (every table and figure of docs/seam_audit/v7_single.md)
as one self-contained HTML file: the document's sections (a small Markdown subset: headers, paragraphs, bullet
lists, tables, fenced code) plus the campaign summaries and the figures (PNG embedded as data URIs).

    .venv/Scripts/python.exe experiment/v7_single/_build_artifact.py --out logs/v7_single/report/v7_single_report.html
"""

from __future__ import annotations

import argparse
import base64
import html
import pathlib
import re
import sys

_REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]

STYLE = """
:root { --ink: #1a1a1a; --muted: #5a5a5a; --line: #d9d9d9; --accent: #08519c; --bg: #ffffff; --code: #f5f5f3; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --ink: #ececec; --muted: #a8a8a8; --line: #3a3a3a;
  --accent: #6baed6; --bg: #141414; --code: #1f1f1f; } }
:root[data-theme="dark"] { --ink: #ececec; --muted: #a8a8a8; --line: #3a3a3a; --accent: #6baed6; --bg: #141414; --code: #1f1f1f; }
body { margin: 0; padding: 24px 16px 64px; background: var(--bg); color: var(--ink);
  font: 15px/1.55 -apple-system, "Segoe UI", "PingFang SC", "Microsoft YaHei", Roboto, sans-serif; }
main { max-width: 1080px; margin: 0 auto; }
h1 { font-size: 1.6em; margin: 0 0 8px; } h2 { font-size: 1.25em; margin: 36px 0 10px; border-bottom: 1px solid var(--line); padding-bottom: 4px; }
h3 { font-size: 1.05em; margin: 24px 0 8px; } h4 { font-size: 1em; margin: 18px 0 6px; color: var(--muted); }
p, li { max-width: 100%; } code { font: 0.92em/1.4 Consolas, "SFMono-Regular", Menlo, monospace; background: var(--code); padding: 1px 4px; border-radius: 3px; }
pre { background: var(--code); padding: 10px 12px; border-radius: 6px; overflow-x: auto; font: 0.85em/1.45 Consolas, Menlo, monospace; }
pre code { background: none; padding: 0; }
.tablewrap { overflow-x: auto; margin: 10px 0 16px; }
table { border-collapse: collapse; font-size: 0.86em; white-space: nowrap; }
th, td { border: 1px solid var(--line); padding: 4px 8px; text-align: left; vertical-align: top; } th { background: var(--code); }
figure { margin: 18px 0; } figure img { max-width: 100%; height: auto; border: 1px solid var(--line); background: #fff; }
figcaption { color: var(--muted); font-size: 0.9em; margin-top: 6px; }
.meta { color: var(--muted); font-size: 0.9em; }
"""


def render_markdown(text: str) -> str:
    """Headers, paragraphs, bullet lists, tables, fenced code and inline code / bold of the report's Markdown."""
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
        if not line.strip():
            flush()
            index += 1
            continue
        paragraph.append(line.strip())
        index += 1
    flush()
    return "\n".join(out)


def figure_html(path: pathlib.Path, caption: str) -> str:
    if not path.exists():
        return f'<p class="meta">figure missing: {html.escape(str(path))}</p>'
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return (f'<figure><img src="data:image/png;base64,{payload}" alt="{html.escape(caption)}">'
            f"<figcaption>{html.escape(caption)}</figcaption></figure>")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--doc", default="docs/seam_audit/v7_single.md")
    parser.add_argument("--throughput", default="logs/v7_single/throughput_r2")
    parser.add_argument("--breakdown", default="logs/v7_single/kernel_breakdown")
    parser.add_argument("--cavity-tables", default="docs/validation/data/cavity_re1000_tables_v7single.md")
    parser.add_argument("--figures", default="manuscripts/fig")
    arguments = parser.parse_args()
    root = _REPOSITORY_ROOT
    sections = [("<h1>v7_single:与 v7 同 kernel 的纯单卡构建 — 报告</h1>", "")]
    body = [render_markdown((root / arguments.doc).read_text(encoding="utf-8"))]
    figures = root / arguments.figures
    body.append("<h2>附:图</h2>")
    body.append(figure_html(figures / "single_gpu_throughput.png", "Figure C: v7_single throughput (M particle-steps/s) and fps, 1 and 2 frames in flight, 1M–32M, one RTX 5090."))
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
    page = ("<!doctype html><html lang='zh'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>"
            "<title>v7_single report</title><meta name='description' content='Pure single-GPU build of the v7 SPH kernels: inventory, acceptance, throughput, kernel costs, cavity accuracy'>"
            f"<style>{STYLE}</style></head><body><main>" + sections[0][0] + "\n".join(body) + "</main></body></html>")
    out = pathlib.Path(arguments.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(f"[artifact] {len(page) / 1e6:.2f} MB -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
