"""
e8_hops.py — E8 phase 2, deliverable 5: the three-hop split (readback / host (memcpy, rest) / upload, p50 / p95, us,
and the bytes per link) of every traced chain run of the jobs, through e7_hop_split.split.

Usage:
    python docs/cluster_v6/scripts/e8_hops.py --job DIR [--job DIR ...] --out-dir OUT
Writes OUT/hops.md, hops.json.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import e8_common as common  # noqa: E402
import e7_hop_split as hop_split  # noqa: E402

QUANTITIES = [name for name, _ in hop_split.QUANTITIES]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", action="append", required=True)
    parser.add_argument("--out-dir", required=True)
    arguments = parser.parse_args()
    out = pathlib.Path(arguments.out_dir)
    results = {}
    for directory in arguments.job:
        job = common.load_job(directory)
        for name, point in sorted(common.points_of(job).items(), key=lambda item: (item[1]["slabs"], item[1]["case"])):
            if not point.get("trace_dir") or point["slabs"] < 2:
                continue
            try:
                results[name] = {"job": job["name"], "case": point["case"], "K": point["slabs"], **hop_split.split(pathlib.Path(point["trace_dir"]))}
            except Exception as error:                                      # noqa: BLE001
                results[name] = {"job": job["name"], "case": point["case"], "K": point["slabs"], "error": repr(error)}
    header = "| point | K | links x steps | " + " | ".join(QUANTITIES) + " |\n|---|---|---|" + "---|" * len(QUANTITIES)
    lines = ["## Three hops (E8 phase 2): pooled over every link and steady step, p50 / p95 (us)", "", header]
    for name, r in results.items():
        if r.get("error"):
            lines.append(f"| {name} | {r['K']} | failed: {r['error']} |" + " |" * len(QUANTITIES))
            continue
        cells = " | ".join(f"{r['pooled'][q][0]:,.0f} / {r['pooled'][q][1]:,.0f}" for q in QUANTITIES)
        lines.append(f"| {name} | {r['K']} | {r['links']} x {'/'.join(map(str, r['steps_per_link']))} | {cells} |")
    lines += ["", "### Per link (p50 / p95, us; bytes p50)", "",
              "| point | link | host copy B | DMA B | " + " | ".join(QUANTITIES) + " |", "|---|---|---|---|" + "---|" * len(QUANTITIES)]
    for name, r in results.items():
        for link, values in (r.get("per_link") or {}).items():
            cells = " | ".join(f"{values[q][0]:,.0f} / {values[q][1]:,.0f}" for q in QUANTITIES)
            lines.append(f"| {name} | {link} ({values['sender']}→{values['receiver']}) | {values['host_copy_bytes']:,} | {values['dma_bytes']:,} | {cells} |")
    text = "\n".join(lines)
    print(text[:3000])
    common.write_text(out / "hops.md", text)
    common.write_json(out / "hops.json", results)
    print(f"[e8_hops] {len(results)} traces -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
