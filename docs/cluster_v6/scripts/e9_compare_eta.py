"""
e9_compare_eta.py — E8 (atomic defrag order, tag e8-3d-p2*) against E9 (prefix defrag order, tag v7-rc4) per point: eta
(mean, min-max over the trials), eta_min, the K-run and reference throughput, from the two e8_p2_summary.py tables.

    python docs/cluster_v6/scripts/e9_compare_eta.py --e8 docs/cluster_v6/data/e8/phase2/tables.json \
        --e9 docs/cluster_v6/data/e9/phase2/tables.json --md OUT.md --json OUT.json

Points are matched by their label (case + K, e.g. 3d_n392_K8); a point present on one side only is listed with blanks.
"""
from __future__ import annotations

import argparse
import json
import pathlib


def load_points(path: pathlib.Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {point["point"]: point for point in data["points"]}


def stat(point: dict, name: str, field: str = "mean"):
    """A trial statistic of the point (stats.<name>.<field>); the reference throughput is derived:
    reference_fps = fps_K / (K x eta) (the tables carry eta and the K run's fps, not the reference mean itself)."""
    stats = (point or {}).get("stats") or {}
    if name in ("reference_fps", "reference_fps_nodefrag"):
        fps = (stats.get("fps_nodefrag") or stats.get("fps") or {}).get(field)
        eta = (stats.get("eta") or {}).get(field)
        slabs = (point or {}).get("K")
        return fps / (slabs * eta) if fps and eta and slabs else None
    value = stats.get(name)
    if not value:
        return None
    return value.get(field)


def fmt(value, digits=3, sign=False):
    if value is None:
        return "-"
    return f"{value:+.{digits}f}" if sign else f"{value:.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--e8", required=True)
    parser.add_argument("--e9", required=True)
    parser.add_argument("--md", required=True)
    parser.add_argument("--json", default=None)
    parser.add_argument("--title", default="E8 (atomic defrag order) vs E9 (prefix defrag order): eta per point")
    arguments = parser.parse_args()
    e8, e9 = load_points(pathlib.Path(arguments.e8)), load_points(pathlib.Path(arguments.e9))
    order = list(e8) + [name for name in e9 if name not in e8]
    rows, records = [], []
    for name in order:
        a, b = e8.get(name), e9.get(name)
        ref = a or b
        eta8, eta9 = stat(a, "eta"), stat(b, "eta")
        row = {"point": name, "case": ref.get("case"), "K": ref.get("K"), "kind": ref.get("kind"), "particles": ref.get("particles"),
               "e8": {"eta": eta8, "eta_min_trials": stat(a, "eta", "min"), "eta_max_trials": stat(a, "eta", "max"), "n": stat(a, "eta", "n"),
                      "eta_min": stat(a, "eta_min"), "fps_K": stat(a, "fps_nodefrag") or stat(a, "fps"), "reference_fps": stat(a, "reference_fps_nodefrag") or stat(a, "reference_fps"),
                      "job": (a or {}).get("job")},
               "e9": {"eta": eta9, "eta_min_trials": stat(b, "eta", "min"), "eta_max_trials": stat(b, "eta", "max"), "n": stat(b, "eta", "n"),
                      "eta_min": stat(b, "eta_min"), "fps_K": stat(b, "fps_nodefrag") or stat(b, "fps"), "reference_fps": stat(b, "reference_fps_nodefrag") or stat(b, "reference_fps"),
                      "job": (b or {}).get("job")},
               "delta_eta": (eta9 - eta8) if (eta8 is not None and eta9 is not None) else None}
        records.append(row)
        rows.append(f"| {name} | {row['case']} | {row['K']} | {row['kind']} | {fmt(eta8)} ({fmt(row['e8']['eta_min_trials'])}-{fmt(row['e8']['eta_max_trials'])}, n={row['e8']['n'] or '-'}) | "
                    f"{fmt(eta9)} ({fmt(row['e9']['eta_min_trials'])}-{fmt(row['e9']['eta_max_trials'])}, n={row['e9']['n'] or '-'}) | **{fmt(row['delta_eta'], 3, True)}** | "
                    f"{fmt(row['e8']['eta_min'])} / {fmt(row['e9']['eta_min'])} | {fmt(row['e8']['fps_K'], 2)} / {fmt(row['e9']['fps_K'], 2)} | "
                    f"{fmt(row['e8']['reference_fps'], 2)} / {fmt(row['e9']['reference_fps'], 2)} |")
    lines = [f"### {arguments.title}", "",
             "eta = fps_K / (K x mean simultaneous K = 1 reference fps) without the defrag steps (the pair reference for 128M); mean (min-max) over the trials; "
             "eta_min = against the slowest reference member; fps columns = E8 / E9.", "",
             "| point | case | K | kind | E8 eta | E9 eta | E9 - E8 | eta_min E8 / E9 | fps_K E8 / E9 | reference fps E8 / E9 |",
             "|---|---|---|---|---|---|---|---|---|---|"] + rows
    pathlib.Path(arguments.md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    if arguments.json:
        pathlib.Path(arguments.json).write_text(json.dumps({"e8": arguments.e8, "e9": arguments.e9, "points": records}, indent=1), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
