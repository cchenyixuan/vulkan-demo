"""Interleaved local timing rc1 vs rc2: steady fps per run, per solver mean/SD, paired differences per trial."""
import pathlib, re, statistics, sys
directory = pathlib.Path(sys.argv[1])
for size in ("2m", "1m"):
    runs = {}
    for path in sorted(directory.glob(f"time{size}_*.log")):
        _, solver, trial = path.stem.split("_")
        match = re.search(r"STEADY \(post-warmup \d+\): \d+ steps in [\d.]+s = ([\d.]+) fps", path.read_text(encoding="utf-8", errors="replace"))
        runs[(solver, trial)] = float(match.group(1)) if match else float("nan")
    trials = sorted({trial for _, trial in runs})
    print(f"2-D {size.upper()} K=2: " + "; ".join(f"t{trial} rc1 {runs.get(('rc1', trial), float('nan')):.1f} rc2 {runs.get(('rc2', trial), float('nan')):.1f}" for trial in trials))
    for solver in ("rc1", "rc2"):
        values = [runs[(solver, trial)] for trial in trials if (solver, trial) in runs]
        print(f"   {solver}: mean {statistics.mean(values):.1f} SD {statistics.stdev(values):.2f} fps over {len(values)} runs")
    deltas = [100 * (runs[("rc2", trial)] - runs[("rc1", trial)]) / runs[("rc1", trial)] for trial in trials]
    print(f"   rc2 vs rc1 per trial: {', '.join(f'{delta:+.2f} %' for delta in deltas)}; mean {statistics.mean(deltas):+.2f} % (SE {statistics.stdev(deltas) / len(deltas) ** 0.5:.2f})")
