"""make_reference_re5000.py (2026-11-02): docs/validation/data/marchi2021_re5000.csv and ghia1982_re5000.csv.

Marchi 2021 Tables 27 (extrema), 28 (u at x = 1/2), 29 (v at y = 1/2) are parsed from the text layer of the local PDF, not
from the repository's marchi2021.json (that dictionary had 4 transcription errors at Re 1000). The same parser runs first
on the Re 1000 and Re 3200 tables and must reproduce the checked rows of marchi2021_re1000.csv / _re3200.csv; only then is
the Re 5000 file written (regression's docstring has the one tolerance), and its 56 profile values are compared with the
json. Ghia 1982 Re 5000 comes from the checked
ghia1982.json, as the Re 3200 file did.

    python docs/validation/cluster/make_reference_re5000.py
"""
from __future__ import annotations

import csv
import json
import pathlib
import re
import sys

import pymupdf

_ROOT = pathlib.Path(__file__).resolve().parents[3]
_MAIN = _ROOT.parent / "vulkan-demo"
PDF = _MAIN / "logs/validation/reference/marchi2021_vvuq.pdf"
MARCHI_JSON = _MAIN / "docs/cavity_validation/reference/marchi2021.json"
GHIA_JSON = _MAIN / "docs/cavity_validation/reference/ghia1982.json"
DATA = _ROOT / "docs/validation/data"
COLUMNS = ("table", "journal_page", "pdf_page", "group", "quantity", "coordinate_name", "coordinate", "Tc_printed", "Tc", "Uc",
           "pU", "Tp_or_Th_printed")
MINUS, TIMES = "\x03", "\x02"
NUMBER = r"\x03?\d\.\d+ \x02 10\x03?\d+ \(\d+\)"
EXTREMA_LABELS = {"umin": "u_min", "y(umin)": "y_at_u_min", "vmin": "v_min", "x(vmin)": "x_at_v_min", "vmax": "v_max",
                  "x(vmax)": "x_at_v_max", "wmin": "psi_min", "x(wmin)": "x_at_psi_min", "y(wmin)": "y_at_psi_min"}
# Re -> (extrema table, u table, v table)
TABLES = {1000: (19, 20, 21), 3200: (23, 24, 25), 5000: (27, 28, 29)}


def parse_number(text: str) -> tuple[str, float, float]:
    """A printed '-4.168 x 10^-1 (13)' -> (the csv's printed form, value, uncertainty: the code in units of the last digit)."""
    match = re.fullmatch(r"(\x03?)(\d)\.(\d+) \x02 10(\x03?)(\d+) \((\d+)\)", text)
    sign, integer, decimals, exponent_sign, exponent, code = match.groups()
    exponent_value = -int(exponent) if exponent_sign else int(exponent)
    value_text = f"{'-' if sign else ''}{integer}.{decimals}e{exponent_value}"
    uncertainty = int(code) * 10.0 ** (exponent_value - len(decimals))
    printed = f"{'-' if sign else ''}{integer}.{decimals} × 10^{exponent_value} ({code})"
    return printed, float(value_text), uncertainty


def table_text(document, table: int) -> list[tuple[int, str]]:
    """(1-based page, text) of every page part holding the table, its continuation included."""
    parts = []
    start = re.compile(rf"(?m)^Table {table}( \(continued\))?\n")
    stop = re.compile(rf"(?m)^Table {table + 1}\n")
    active = False
    for index, page in enumerate(document):
        text = page.get_text()
        begin = 0
        found = start.search(text)
        if found:
            active, begin = True, found.start()
        if not active:
            continue
        ended = stop.search(text, begin)
        if ended:
            parts.append((index + 1, text[begin:ended.start()]))
            return parts
        parts.append((index + 1, text[begin:]))
    return parts


def parse_table(document, table: int, pattern: str) -> dict:
    """Rows keyed by the label group of pattern: label -> (pdf page, pU, Tp / Th printed, Tc printed, Tc, Uc)."""
    rows = {}
    row = re.compile(pattern + rf"\n(\d\.\d+)\n({NUMBER})\n({NUMBER})\n")
    for page, text in table_text(document, table):
        for label, order, approximate, converged in row.findall(text):
            printed, value, uncertainty = parse_number(converged)
            rows[label] = (page, order, parse_number(approximate)[0], printed, value, uncertainty)
    return rows


def parse_reynolds(document, reynolds: int) -> list[dict]:
    extrema_table, u_table, v_table = TABLES[reynolds]
    extrema = parse_table(document, extrema_table, r"(u_?min|y\(umin\)|vmin|x\(vmin\)|vmax|x\(vmax\)|wmin|x\(wmin\)|y\(wmin\))")
    u_rows = parse_table(document, u_table, r"u\((\d\.\d+|\d)\)")
    v_rows = parse_table(document, v_table, r"v\((\d\.\d+|\d)\)")
    if len(extrema) != 9 or len(u_rows) != 28 or len(v_rows) != 28:
        sys.exit(f"Re {reynolds}: parsed {len(extrema)} extrema, {len(u_rows)} u, {len(v_rows)} v rows")
    rows = []
    for table, group, quantity, name, parsed in ((u_table, "u_at_x_0.5", "u", "y", u_rows), (v_table, "v_at_y_0.5", "v", "x", v_rows)):
        for coordinate in sorted(parsed, key=float):
            page, order, approximate, printed, value, uncertainty = parsed[coordinate]
            rows.append({"table": table, "journal_page": f"041004-{page}", "pdf_page": page, "group": group, "quantity": quantity,
                         "coordinate_name": name, "coordinate": coordinate, "Tc_printed": printed, "Tc": repr(value),
                         "Uc": f"{uncertainty:.2e}", "pU": order, "Tp_or_Th_printed": approximate})
    for label, quantity in EXTREMA_LABELS.items():
        page, order, approximate, printed, value, uncertainty = extrema[label]
        rows.append({"table": extrema_table, "journal_page": f"041004-{page}", "pdf_page": page, "group": "extrema", "quantity": quantity,
                     "coordinate_name": "", "coordinate": "", "Tc_printed": printed, "Tc": repr(value), "Uc": f"{uncertainty:.2e}",
                     "pU": order, "Tp_or_Th_printed": approximate})
    return rows


def read_csv(path: pathlib.Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return list(csv.DictReader(line for line in handle if not line.startswith("#")))


def regression(document) -> None:
    """The parser against the checked files: every value of every row, and the extrema's uncertainties. Exact, except the
    Re 3200 profile rows: that file took them from the json, which rounded four of them in the last digits (at most 1e-9,
    each below the table's Uc); those may differ by less than Uc and are counted."""
    for reynolds in (1000, 3200):
        parsed = {(row["group"], row["quantity"], float(row["coordinate"] or 0.0)): row for row in parse_reynolds(document, reynolds)}
        checked = read_csv(DATA / f"marchi2021_re{reynolds}.csv")
        bad, rounded = [], 0
        for row in checked:
            key = (row["group"], row["quantity"], float(row["coordinate"] or 0.0))
            if key not in parsed:
                bad.append(("missing", key))
                continue
            difference = abs(float(parsed[key]["Tc"]) - float(row["Tc"]))
            exact = difference <= 1e-12 * max(1.0, abs(float(row["Tc"])))
            if not exact and reynolds == 3200 and row["group"] != "extrema" and difference < float(parsed[key]["Uc"]):
                rounded += 1
            elif not exact:
                bad.append((key, parsed[key]["Tc"], row["Tc"]))
            elif row["Uc"] and abs(float(parsed[key]["Uc"]) - float(row["Uc"])) > 1e-3 * float(row["Uc"]):
                bad.append((key, "Uc", parsed[key]["Uc"], row["Uc"]))
        print(f"parser vs marchi2021_re{reynolds}.csv: {len(checked)} rows, {len(bad)} differences, {rounded} rounded by the json")
        if bad or len(checked) != len(parsed):
            sys.exit(f"parser regression failed at Re {reynolds}: {bad[:6]}, {len(checked)} checked rows, {len(parsed)} parsed")


def write_marchi(rows: list[dict]) -> None:
    header = """# Marchi, Santiago, Carvalho Jr., Lid-Driven Square Cavity Flow: A Benchmark Solution With an 8192 x 8192 Grid,
# J. Verif. Valid. Uncert. 6(4) 041004 (2021), doi:10.1115/1.4052149. Re = 5000 convergent values Tc of Tables 27 (extrema
# and primary-vortex centre), 28 (u at x = 1/2, 28 points) and 29 (v at y = 1/2, 28 points), parsed 2026-11-02 from the
# text layer of the local PDF by docs/validation/cluster/make_reference_re5000.py; the same parser reproduces every
# row of marchi2021_re1000.csv and marchi2021_re3200.csv. Uc is decoded from the printed two-digit code. Same column layout
# as marchi2021_re1000.csv so that cavity_reference.marchi2021(reynolds=5000) reads it.
"""
    with open(DATA / "marchi2021_re5000.csv", "w", newline="\n", encoding="utf-8") as handle:
        handle.write(header)
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def compare_json(rows: list[dict]) -> int:
    """The Re 5000 profile values of the repository json against the PDF."""
    source = json.loads(MARCHI_JSON.read_text(encoding="utf-8"))
    tables = {"u": (source["u_vertical_centerline"]["y"], source["u_vertical_centerline"]["5000"]),
              "v": (source["v_horizontal_centerline"]["x"], source["v_horizontal_centerline"]["5000"])}
    differences = 0
    for row in rows:
        if row["group"] == "extrema":
            continue
        coordinates, values = tables[row["quantity"]]
        value = values[[f"{float(item):.10f}" for item in coordinates].index(f"{float(row['coordinate']):.10f}")]
        if abs(value - float(row["Tc"])) > 1e-12:
            differences += 1
            print(f"  json differs: {row['quantity']}({row['coordinate']}) json {value}, PDF {row['Tc']} (Uc {row['Uc']})")
    return differences


def write_ghia() -> None:
    source = json.loads(GHIA_JSON.read_text(encoding="utf-8"))
    lines = ["# Ghia, Ghia, Shin, J. Comput. Phys. 48 (1982) 387-411, Tables I and II, Re = 5000 (257^2 grid). Values from the",
             "# repository transcription docs/cavity_validation/reference/ghia1982.json (gist.github.com/ivan-pi, checked against the",
             "# paper). Shown as a familiar reference only (figures); errors are computed against Marchi 2021 (marchi2021_re5000.csv).",
             "# Added 2026-11-02 for the Re = 5000 run.", "line,coordinate,value"]
    for group, key, coordinate_name in (("u_at_x_0.5", "u_vertical_centerline", "y"), ("v_at_y_0.5", "v_horizontal_centerline", "x")):
        table = source[key]
        for coordinate, value in zip(table[coordinate_name], table["5000"]):
            lines.append(f"{group},{coordinate},{value}")
    (DATA / "ghia1982_re5000.csv").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main() -> int:
    document = pymupdf.open(PDF)
    regression(document)
    rows = parse_reynolds(document, 5000)
    differences = compare_json(rows)
    print(f"Re 5000: {sum(row['group'] != 'extrema' for row in rows)} profile rows, json differs at {differences}")
    write_marchi(rows)
    write_ghia()
    print(f"wrote {DATA / 'marchi2021_re5000.csv'} and ghia1982_re5000.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
