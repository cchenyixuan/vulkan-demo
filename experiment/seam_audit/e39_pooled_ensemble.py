"""
e39_pooled_ensemble.py - E39 follow-up of the 2-D 1M ensemble (e39_ensemble): two independent 6 + 6 campaigns of the
same driver, snapshot and code (default logs/e39/ensemble and logs/e39/ensemble_rep) pooled into 12 + 12 runs at one
horizon. e39_ensemble's exact permutation tests enumerate C(2T, T) relabellings (T = 12: 2.7 million), so this one
draws RELABELLINGS random 12 / 12 relabellings with a fixed seed instead.

Per field (velocity, density; every fluid particle; e39_ensemble's pair_rms): all C(24, 2) = 276 pairwise rms
distances, then two statistics over the relabellings:
  spread = mean log d(within v7) - mean log d(within v6)   one-sided p (the v7 runs scatter more) and two-sided p
  cross  = mean log d(across) - mean log d(within)          one-sided p (a systematic v7 - v6 difference; it also
                                                            rises when one arm only scatters more)
plus the trajectory families: average-linkage clustering of log(d / 1e-6) cut into 2, 3 and 4 clusters, each
cluster's arm counts, its largest internal distance and its smallest distance to the other runs, and per run the
median and the nearest distance to the other runs.

The spread test of the pooled data was chosen after the first campaign showed within-v7 / within-v6 up to 3.9 at
N = 2000, so its p values are optimistic; the second campaign alone is the unbiased check (e39_ensemble analyze).

Usage (from the checkout root, CPU only):
  .venv/Scripts/python.exe -m experiment.seam_audit.e39_pooled_ensemble [--horizon 2000] \\
      [--campaigns logs/e39/ensemble,logs/e39/ensemble_rep] [--out logs/e39/ensemble_rep/pooled_N2000.json]
"""
from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import sys

import numpy as np

_REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from experiment.seam_audit.decomposition_audit import pair_rms  # noqa: E402
from experiment.seam_audit.dump_state import dump_file_paths  # noqa: E402

CASE_NAME = "cavity2d_1m"
FIELDS = ("velocity", "density")
RELABELLINGS = 20000
SEED = 39
DISTANCE_SCALE = 1.0e-6          # clustering on log(d / scale) >= 0
CLUSTER_COUNTS = (2, 3, 4)
DEFAULT_CAMPAIGNS = "logs/e39/ensemble,logs/e39/ensemble_rep"


def campaign_runs(dump_directory: pathlib.Path, horizon: int) -> list:
    """Run names with a dump at this horizon, in name order."""
    suffix = f"_N{horizon}.npz"
    return sorted(path.name[: -len(suffix)] for path in dump_directory.glob(f"*{suffix}"))


def pooled_runs(campaigns: list, horizon: int) -> tuple:
    """[(label, run name, campaign label, dump directory)] over every campaign; label = run@campaign."""
    runs = []
    for campaign in campaigns:
        dump_directory = campaign / "dumps" / CASE_NAME
        for run_name in campaign_runs(dump_directory, horizon):
            runs.append((f"{run_name}@{campaign.name}", run_name, campaign.name, dump_directory))
    if not runs:
        raise SystemExit(f"no {CASE_NAME} dumps at N = {horizon} in {[str(path) for path in campaigns]}")
    return runs


def mean_log_statistics(log_distance: np.ndarray, test_arm: np.ndarray, upper: np.ndarray) -> tuple:
    """(spread, cross) of one labelling: test_arm[i] = run i belongs to the tested arm (v7)."""
    both_test = np.outer(test_arm, test_arm) & upper
    both_reference = np.outer(~test_arm, ~test_arm) & upper
    across = upper & ~both_test & ~both_reference
    within = both_test | both_reference
    return (float(np.mean(log_distance[both_test]) - np.mean(log_distance[both_reference])),
            float(np.mean(log_distance[across]) - np.mean(log_distance[within])))


def trajectory_families(distance: np.ndarray, labels: list) -> dict:
    """Average-linkage clusters of log(d / DISTANCE_SCALE), cut into each of CLUSTER_COUNTS clusters."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    run_count = len(labels)
    log_distance = np.log(np.maximum(distance, DISTANCE_SCALE) / DISTANCE_SCALE) * (1.0 - np.eye(run_count))
    tree = linkage(squareform(log_distance, checks=False), method="average")
    families = {}
    for cluster_count in CLUSTER_COUNTS:
        assignment = fcluster(tree, cluster_count, criterion="maxclust")
        clusters = []
        for cluster in sorted(set(assignment), key=lambda value: -int(np.sum(assignment == value))):
            members = [index for index in range(run_count) if assignment[index] == cluster]
            inside = [distance[first, second] for first, second in itertools.combinations(members, 2)]
            outside = [distance[member, other] for member in members for other in range(run_count)
                       if assignment[other] != cluster]
            clusters.append({"runs": [labels[index] for index in members],
                             "test_arm_runs": sum(labels[index].startswith("v7") for index in members),
                             "largest_internal_distance": max(inside) if inside else 0.0,
                             "smallest_distance_to_others": min(outside) if outside else None})
        families[str(cluster_count)] = clusters
    return families


def analyze(campaigns: list, horizon: int) -> dict:
    runs = pooled_runs(campaigns, horizon)
    labels = [label for label, _, _, _ in runs]
    test_arm = np.array([run_name.startswith("v7_") for _, run_name, _, _ in runs])
    reference_label, reference_name, _, reference_directory = runs[0]
    reference_paths = dump_file_paths(reference_directory, reference_name, horizon)
    sidecar = json.loads(reference_paths["json"].read_text(encoding="utf-8"))
    fluid_groups = [index for index, kind in enumerate(sidecar["material_kinds"]) if int(kind) == 0]
    with np.load(reference_paths["npz"]) as archive:
        identifiers = archive["id"]
        fluid = np.isin(archive["material"], fluid_groups)
    masks = {"all": np.ones(int(fluid.sum()), dtype=bool)}
    random_generator = np.random.default_rng(SEED)
    run_count = len(runs)
    labellings = np.zeros((RELABELLINGS, run_count), dtype=bool)
    for row in range(RELABELLINGS):
        labellings[row, random_generator.choice(run_count, size=int(test_arm.sum()), replace=False)] = True
    upper = np.triu(np.ones((run_count, run_count), dtype=bool), 1)
    result = {"case": CASE_NAME, "horizon": horizon, "campaigns": [str(path) for path in campaigns], "runs": labels,
              "test_arm_runs": int(test_arm.sum()), "reference_arm_runs": int((~test_arm).sum()),
              "relabellings": RELABELLINGS, "seed": SEED, "fields": {}}
    for field in FIELDS:
        values = {}
        for label, run_name, _, dump_directory in runs:
            with np.load(dump_file_paths(dump_directory, run_name, horizon)["npz"]) as archive:
                if not np.array_equal(archive["id"], identifiers):
                    raise SystemExit(f"{label}: ids differ from {reference_label}")
                values[label] = archive[field][fluid].astype(np.float64)
        distance = np.zeros((run_count, run_count))
        for first, second in itertools.combinations(range(run_count), 2):
            value = pair_rms(values, labels[first], labels[second], masks)["all"]
            distance[first, second] = distance[second, first] = value
        if np.any(distance[upper] <= 0.0):
            raise SystemExit(f"{field}: a pair of runs is bit-identical at N = {horizon} (log distance undefined)")
        log_distance = np.log(np.where(upper | upper.T, distance, 1.0))
        observed = mean_log_statistics(log_distance, test_arm, upper)
        null = np.array([mean_log_statistics(log_distance, labelling, upper) for labelling in labellings])
        within_reference = distance[np.outer(~test_arm, ~test_arm) & upper]
        within_test = distance[np.outer(test_arm, test_arm) & upper]
        across = distance[upper & ~np.outer(test_arm, test_arm) & ~np.outer(~test_arm, ~test_arm)]
        per_run = []
        for index, label in enumerate(labels):
            others = np.delete(distance[index], index)
            per_run.append({"run": label, "median_distance": float(np.median(others)),
                            "nearest_distance": float(others.min())})
        result["fields"][field] = {
            "median_within_v6": float(np.median(within_reference)), "median_within_v7": float(np.median(within_test)),
            "median_across": float(np.median(across)),
            "within_v7_over_within_v6": float(np.median(within_test) / np.median(within_reference)),
            "spread_statistic": observed[0],
            "spread_p_one_sided": float(np.mean(null[:, 0] >= observed[0] - 1e-12)),
            "spread_p_two_sided": float(np.mean(np.abs(null[:, 0]) >= abs(observed[0]) - 1e-12)),
            "cross_statistic": observed[1], "cross_p": float(np.mean(null[:, 1] >= observed[1] - 1e-12)),
            "families": trajectory_families(distance, labels), "per_run": per_run, "distance": distance.tolist()}
    return result


def print_result(result: dict) -> None:
    print(f"{result['case']} N = {result['horizon']}: {result['reference_arm_runs']} v6 + {result['test_arm_runs']} v7 "
          f"runs from {len(result['campaigns'])} campaign(s); {result['relabellings']} random relabellings")
    for field, entry in result["fields"].items():
        print(f"\n{field}: median within v6 {entry['median_within_v6']:.3e}, within v7 {entry['median_within_v7']:.3e}, "
              f"across {entry['median_across']:.3e}; within v7 / within v6 {entry['within_v7_over_within_v6']:.2f}")
        print(f"  spread {entry['spread_statistic']:+.3f}: p one-sided {entry['spread_p_one_sided']:.3f}, two-sided "
              f"{entry['spread_p_two_sided']:.3f}; cross {entry['cross_statistic']:+.3f}: p {entry['cross_p']:.3f}")
        for cluster_count, clusters in entry["families"].items():
            print(f"  {cluster_count} families:")
            for cluster in clusters:
                runs = len(cluster["runs"])
                nearest_other = cluster["smallest_distance_to_others"]
                print(f"    {runs:2d} runs (v7 {cluster['test_arm_runs']}, v6 {runs - cluster['test_arm_runs']}), "
                      f"internal <= {cluster['largest_internal_distance']:.1e}, to the others >= "
                      f"{nearest_other if nearest_other is None else format(nearest_other, '.1e')}: "
                      f"{', '.join(sorted(cluster['runs']))}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--horizon", type=int, default=2000, help="steps after the restart (300 or 2000)")
    parser.add_argument("--campaigns", default=DEFAULT_CAMPAIGNS,
                        help=f"comma list of e39_ensemble --out directories (default {DEFAULT_CAMPAIGNS})")
    parser.add_argument("--out", default=None,
                        help="result JSON (default <last campaign>/pooled_N<horizon>.json)")
    arguments = parser.parse_args()
    campaigns = [(_REPOSITORY_ROOT / text.strip()).resolve() if not pathlib.Path(text.strip()).is_absolute()
                 else pathlib.Path(text.strip()) for text in arguments.campaigns.split(",") if text.strip()]
    result = analyze(campaigns, arguments.horizon)
    print_result(result)
    out = pathlib.Path(arguments.out) if arguments.out else campaigns[-1] / f"pooled_N{arguments.horizon}.json"
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
