"""
style.py - the one style module of the chapter-draft pack (data/micropolar/ch5_draft).

Sizes, fonts, line and marker conventions, the symbols of the axis labels, and the writers of figures (.pdf, .png)
and tables (.csv, .tex with booktabs). Every item script imports this module and nothing else for its looks, so a
symbol (the gap in particular) is changed here once.

Conventions of the pack:
  exact solutions      black solid lines
  runs                 markers (open = Newtonian control, filled = micropolar)
  the wall rule        dashed
  the film function    dash-dotted
  fields               viridis for magnitudes, RdBu_r centred on zero for signed quantities, equal aspect
  no titles; panel letters (a), (b), ...; every line plot readable in grey (marker shape and line style carry the
  series, colour only repeats them)
"""
import csv
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MILLIMETRE = 1.0 / 25.4
WIDE = 150.0 * MILLIMETRE
NARROW = 72.0 * MILLIMETRE
FONT_SIZE = 9.0
RASTER_DPI = 300

# ---- symbols (mathtext, without the dollar signs) -------------------------------------------------------------
GAP = "g"                                   # the symbol of the gap may still change
SPACING = r"\Delta x"
LENGTH = r"\ell"                            # the micropolar length l
RADIUS_IN_GAP = rf"(r - R_1)\,/\,{GAP}"
VELOCITY = r"v_\theta\,/\,U"
SPIN = r"w\,/\,\Omega"
PRESSURE = r"p\,/\,(\rho U^2)"
EXCESS_SPIN = r"(w - \frac{1}{2}\,\nabla\!\times\!\mathbf{v})\,/\,\Omega"
TORQUE_CHANGE = "D"
GAP_OVER_TWO_LENGTHS = rf"m = {GAP}\,/\,(2\ell)"
LENGTH_IN_SPACINGS = rf"\ell\,/\,{SPACING}"
CAVITY_U = r"u\,/\,U"
CAVITY_V = r"v\,/\,U"
CAVITY_SPIN = r"w\,L\,/\,U"
CAVITY_EXCESS_SPIN = r"(w - \frac{1}{2}\,\nabla\!\times\!\mathbf{v})\,L\,/\,U"


def math(symbol: str) -> str:
    return f"${symbol}$"


# ---- lines and markers ----------------------------------------------------------------------------------------
INK = "#000000"
MUTED = "#555555"
GRID = "#d9d9d9"
EXACT = {"color": INK, "linestyle": "-", "linewidth": 1.0}
RULE = {"linestyle": (0, (4, 2)), "linewidth": 1.0}
FILM = {"linestyle": (0, (6, 2, 1, 2)), "linewidth": 1.0}
FIRST_ORDER = {"linestyle": (0, (1, 1.5)), "linewidth": 1.0}
# series: colour (dark to light, so that grey keeps the order), marker
SERIES = [("#1b3a6b", "o"), ("#b5482a", "s"), ("#3f8f5f", "^"), ("#8a6fb3", "D"), ("#c79a2a", "v")]
MARKER_SIZE = 4.0
FIELD_MAGNITUDE = "viridis"
FIELD_SIGNED = "RdBu_r"


def series(index: int) -> dict:
    colour, marker = SERIES[index % len(SERIES)]
    return {"color": colour, "marker": marker}


def run_markers(index: int, filled: bool = True) -> dict:
    """Keyword arguments of a run drawn as markers only."""
    colour, marker = SERIES[index % len(SERIES)]
    return {"linestyle": "none", "marker": marker, "markersize": MARKER_SIZE, "markeredgecolor": colour,
            "markeredgewidth": 0.8, "markerfacecolor": colour if filled else "white", "color": colour}


def setup() -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["CMU Serif", "Computer Modern Roman", "Latin Modern Roman",
                                               "DejaVu Serif"],
        "mathtext.fontset": "cm", "font.size": FONT_SIZE, "axes.labelsize": FONT_SIZE, "axes.titlesize": FONT_SIZE,
        "legend.fontsize": FONT_SIZE - 1.0, "xtick.labelsize": FONT_SIZE - 1.0, "ytick.labelsize": FONT_SIZE - 1.0,
        "axes.linewidth": 0.6, "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
        "xtick.direction": "in", "ytick.direction": "in", "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4, "ytick.minor.width": 0.4, "xtick.top": True, "ytick.right": True,
        "axes.grid": False, "legend.frameon": False, "legend.handlelength": 2.4, "lines.linewidth": 1.0,
        "savefig.dpi": RASTER_DPI, "figure.dpi": 100, "pdf.fonttype": 42, "axes.unicode_minus": True,
    })


def panel_letter(axis, letter: str, x: float = 0.0, y: float = 1.03, **keywords) -> None:
    axis.text(x, y, f"({letter})", transform=axis.transAxes, ha="left", va="bottom", fontsize=FONT_SIZE, **keywords)


def save(figure, stem: str) -> None:
    for suffix in ("pdf", "png"):
        figure.savefig(HERE / f"{stem}.{suffix}", dpi=RASTER_DPI)
    plt.close(figure)
    print(f"[pack] {stem}.pdf, .png")


# ---- tables ---------------------------------------------------------------------------------------------------
_TEX_ESCAPES = {"%": r"\%", "_": r"\_", "&": r"\&", "#": r"\#"}


def tex_escape(text: str) -> str:
    for character, replacement in _TEX_ESCAPES.items():
        text = text.replace(character, replacement)
    return text


def write_table(stem: str, header: list, rows: list, tex_header: list | None = None, alignment: str | None = None,
                tex_rows: list | None = None, note: str | None = None) -> None:
    """rows: lists of strings (already formatted). The .csv carries header and rows as they are; the .tex is a
    booktabs tabular (no float, no caption: the chapter supplies them), with tex_header / tex_rows where the TeX
    text differs from the plain text (math, run names in \\texttt)."""
    with open(HERE / f"{stem}.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    tex_header = tex_header or [tex_escape(cell) for cell in header]
    tex_rows = tex_rows or [[tex_escape(cell) for cell in row] for row in rows]
    alignment = alignment or "l" + "r" * (len(header) - 1)
    lines = ["% generated by the script of the same name; do not edit", rf"\begin{{tabular}}{{{alignment}}}",
             r"\toprule", " & ".join(tex_header) + r" \\", r"\midrule"]
    for row in tex_rows:
        lines.append(r"\midrule" if row == "rule" else " & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    if note:
        lines.append("% " + note)
    (HERE / f"{stem}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"[pack] {stem}.csv, .tex ({len(rows)} rows)")


def texttt(name: str) -> str:
    return r"\texttt{" + name.replace("_", r"\_") + "}"


# ---- captions and the index -----------------------------------------------------------------------------------
def record(item: str, title: str, status: str, caption: str, files: list, sources: list, numbers: dict | None = None,
           tag: str = "") -> None:
    """Every item script ends with this: its caption, its status (final / provisional / not reached), the files it
    wrote, the runs or data files it read and the key numbers, kept as items/<item>.json; make_all.py assembles
    captions.md and the index of README.md from these."""
    (HERE / "items").mkdir(exist_ok=True)
    (HERE / "items" / f"{item}.json").write_text(json.dumps(
        {"item": item, "title": title, "status": status, "caption": caption, "files": files, "sources": sources,
         "numbers": numbers or {}, "tag": tag}, indent=1), encoding="utf-8", newline="\n")


def load_json(relative: str):
    return json.loads((REPO_ROOT / relative).read_text(encoding="utf-8"))


def exists(relative: str) -> bool:
    return (REPO_ROOT / relative).exists()
