"""e7_reference_spread.py RESULTS LABEL PARTIES PCT: the stop rule of the third batch-2 supplement (user 2026-10-13):
after a K = 1 reference group (LABEL_g0 .. LABEL_g<PARTIES-1>, one process per card) every member's steady fps is
compared with the mean of the other members; a member beyond PCT % of that mean fails the check (exit 1, the job stops
at once and reports). fps = steady_steps / steady_seconds of the member's results row when both exist (the precise
value), else its rounded steady_fps. Exit 2 when a member has no fps (the group check before this one already stops the
job on a failed member). Prints one line per member and the verdict; nothing is written.
"""
import json
import re
import sys


def main() -> int:
    results, label, parties, limit = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
    rows = {}
    with open(results, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[row["label"]] = row            # the last row of a label wins (labels are unique per job)
    members = []
    for member in range(parties):
        row = rows.get(f"{label}_g{member}")
        fps = None
        if row:
            steps, seconds = row.get("steady_steps"), row.get("steady_seconds")
            if steps and seconds:
                fps = steps / seconds
            elif row.get("steady_fps") is not None:
                fps = float(row["steady_fps"])
        members.append((member, fps))
    if any(fps is None for _, fps in members):
        missing = [f"g{member}" for member, fps in members if fps is None]
        print(f"REFERENCE SPREAD {label}: no fps for {', '.join(missing)} - cannot check")
        return 2
    if parties < 2:
        print(f"REFERENCE SPREAD {label}: one member ({members[0][1]:.3f} fps), nothing to compare")
        return 0
    worst, failed, parts = 0.0, [], []
    for member, fps in members:
        others = [value for index, value in members if index != member]
        mean_others = sum(others) / len(others)
        deviation = (fps - mean_others) / mean_others * 100.0
        worst = max(worst, abs(deviation))
        parts.append(f"g{member} {fps:.3f} ({deviation:+.2f} %)")
        if abs(deviation) > limit:
            failed.append(f"g{member} {fps:.3f} fps is {deviation:+.2f} % off the others' mean {mean_others:.3f}")
    verdict = "FAIL" if failed else "OK"
    print(f"REFERENCE SPREAD {label}: {', '.join(parts)}; max |deviation| {worst:.2f} % vs limit {limit:g} %: {verdict}")
    for text in failed:
        print(f"REFERENCE SPREAD {label}: {text}")
    return 1 if failed else 0


if __name__ == "__main__":
    if len(sys.argv) != 5 or not re.fullmatch(r"\d+", sys.argv[3]):
        print(__doc__)
        sys.exit(2)
    sys.exit(main())
