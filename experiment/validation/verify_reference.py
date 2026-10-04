"""verify_reference.py - re-check docs/validation/data/marchi2021_re1000.csv against the paper PDF.

Every CSV row is tied to its own table row in the PDF: within the region of its table (from the caption
'Table N' to the next caption) on the page the row names, the row label (u(0.0546875), umin, y(umin), ...,
'wmin' is the text layer's psi_min) is followed by p_U, the grid value T_h / T_p with its code and the
convergent value T_c with its code. Each is parsed (sign, mantissa, exponent, code) and compared with the CSV;
a swapped row, a wrong sign or exponent, or a wrong code fails. The PDF is not in the repository; pass the copy
downloaded from the authors' institution
(http://servidor.demec.ufpr.br/CFD/artigos_revistas/2021_Marchi_Santiago_Carvalho-Jr_VVUQ.pdf,
md5 68a6fe98736a1353ca12ccb990cccc92).

    .venv/Scripts/python.exe -m experiment.validation.verify_reference --pdf logs/validation/reference/marchi2021_vvuq.pdf
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys

from experiment.validation import cavity_reference

EXPECTED_MD5 = "68a6fe98736a1353ca12ccb990cccc92"
EXTREMA_LABELS = {"u_min": "umin", "y_at_u_min": "y(umin)", "v_min": "vmin", "x_at_v_min": "x(vmin)",
                  "v_max": "vmax", "x_at_v_max": "x(vmax)", "psi_min": "wmin", "x_at_psi_min": "x(wmin)",
                  "y_at_psi_min": "y(wmin)"}
NUMBER = re.compile(r"^\s*(-?)([0-9]+\.[0-9]+)\s*x\s*10(-?[0-9]+)\s*(\((\d+)\))?\s*$")


def parse(printed: str) -> tuple[float, str, str]:
    """'-3.885697938 x 10-1 (47)' -> (value, mantissa digits with sign and exponent, code)."""
    text = printed.replace("−", "-").replace("×", "x").replace("^", "")
    match = NUMBER.match(text)
    if not match:
        raise ValueError(f"cannot parse {printed!r}")
    sign, mantissa, exponent, _, code = match.groups()
    value = float(f"{sign}{mantissa}e{exponent}")
    return value, f"{sign}{mantissa}e{int(exponent)}", code or ""


def table_lines(document, page_number: int, table: int) -> list[str]:
    lines = [line.replace("\x03", "-").replace("\x02", "x").strip()
             for line in document[page_number - 1].get_text().splitlines()]
    starts = [index for index, line in enumerate(lines) if re.match(rf"^Table {table}\b", line)]
    if not starts:
        raise ValueError(f"Table {table} not found on PDF page {page_number}")
    start = starts[0] + 1
    stop = next((index for index in range(start, len(lines)) if re.match(r"^Table \d+\b", lines[index])), len(lines))
    return lines[start:stop]


def main() -> int:
    import pymupdf
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pdf", required=True)
    arguments = parser.parse_args()
    digest = hashlib.md5(open(arguments.pdf, "rb").read()).hexdigest()
    if digest != EXPECTED_MD5:
        print(f"WARNING: PDF md5 {digest} differs from the expected {EXPECTED_MD5}")
    document = pymupdf.open(arguments.pdf)
    rows = cavity_reference.marchi2021()["rows"]
    failures = 0
    for row in rows:
        table, page = int(row["table"]), int(row["pdf_page"])
        if row["group"] == "extrema":
            label = EXTREMA_LABELS[row["quantity"]]
        else:
            label = f"{row['quantity']}({row['coordinate']})"
        lines = table_lines(document, page, table)
        hits = [index for index, line in enumerate(lines) if line == label]
        problems = []
        if len(hits) != 1:
            problems.append(f"label {label!r} found {len(hits)} times in Table {table} on page {page}")
        else:
            index = hits[0]
            p_u, grid, convergent = lines[index + 1], lines[index + 2], lines[index + 3]
            pdf_value, pdf_tokens, pdf_code = parse(convergent)
            csv_value, csv_tokens, csv_code = parse(row["Tc_printed"])
            if (pdf_tokens, pdf_code) != (csv_tokens, csv_code) or float(row["Tc"]) != pdf_value:
                problems.append(f"Tc: PDF {convergent!r}, CSV {row['Tc_printed']!r} / {row['Tc']}")
            if parse(grid)[1:] != parse(row["Tp_or_Th_printed"])[1:]:
                problems.append(f"grid value: PDF {grid!r}, CSV {row['Tp_or_Th_printed']!r}")
            if p_u != row["pU"]:
                problems.append(f"pU: PDF {p_u!r}, CSV {row['pU']!r}")
        if problems:
            failures += 1
            print(f"FAIL table {table} {row['quantity']} {row['coordinate']}: " + "; ".join(problems))
    print(f"{len(rows) - failures} of {len(rows)} rows match their own table row in the PDF "
          "(label, p_U, grid value, T_c with sign, exponent and uncertainty code)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
