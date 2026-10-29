"""
_merge_throughput_results.py — assemble one measurement-A results directory out of several campaign runs.

The merged directory (results.jsonl + telemetry.csv) is what _summarize_v7_single_campaign.py and
_plot_single_baseline_paper.py read. Records are keyed by (size, solver, in_flight); every --add directory
replaces the base's records of the keys it contains (whole key, all trials) and appends the keys the base lacks;
run_log paths keep pointing into the directory the run came from. telemetry.csv files are concatenated in the same
order (the summariser selects samples by each run's epoch window, so overlapping files are harmless).

    .venv/Scripts/python.exe _merge_throughput_results.py --base logs/v7_single/throughput_r3
        --add logs/v7_single/throughput_rc4b --add logs/v7_single/throughput_rc4c
        --out logs/v7_single/throughput_rc4_merged

E9 (2026-10-27/29): throughput_rc4_merged = r3 (the 2026-10-10 atomic-order campaign) with the rc4b keys
(2M / 4M gen / 8M / 16M / 32M: S d1, S d2, chain d2) and the rc4c keys (1M S d2; 4M gen 414^2 S d1, S d2) measured
under v7-single-rc4.1; everything else in the table (1M S d1, the hand-made 4M, 3-D 8M, V, Vx) is still the old run.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys

_REPO = pathlib.Path(__file__).resolve().parent


def load(path: pathlib.Path) -> list[dict]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def key_of(record: dict) -> tuple:
    return (record["size"], record["solver"], record["in_flight"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base", required=True)
    parser.add_argument("--add", action="append", default=[], help="directory whose keys replace / extend the base (in order)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    base = _REPO / args.base
    out = _REPO / args.out
    out.mkdir(parents=True, exist_ok=True)
    records = load(base / "results.jsonl")
    telemetry = [base / "telemetry.csv"]
    report = [f"base {args.base}: {len(records)} records"]
    for directory in args.add:
        added = load(_REPO / directory / "results.jsonl")
        ok_keys = {key_of(r) for r in added if r.get("status") == "ok"}
        before = len(records)
        records = [r for r in records if key_of(r) not in ok_keys] + added
        report.append(f"add {directory}: {len(added)} records, keys {sorted('/'.join(map(str, k)) for k in ok_keys)} "
                      f"(replaced {before + len(added) - len(records)} base records)")
        telemetry.append(_REPO / directory / "telemetry.csv")
    with open(out / "results.jsonl", "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    with open(out / "telemetry.csv", "wb") as handle:
        for path in telemetry:
            if path.exists():
                with open(path, "rb") as source:
                    shutil.copyfileobj(source, handle)
    ok = [r for r in records if r.get("status") == "ok"]
    keys = sorted({key_of(r) for r in ok})
    report.append(f"out {args.out}: {len(records)} records ({len(ok)} ok), {len(keys)} keys")
    (out / "MERGE.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
