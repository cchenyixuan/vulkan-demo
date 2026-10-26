#!/bin/bash
# m = 8 runs with per-piece GPU timestamps (user 2026-10-27 addendum): waits for run_series.sh to finish (same GPU), then
# 8M and 64M, each kernel split in 8 (the other unsplit), two runs each, through split_dispatch_run_ticks.py.
set -u
SP=/c/Users/cchen/AppData/Local/Temp/claude/C--Users-cchen-PycharmProjects-vulkan-demo/fd68689d-e4b4-4b3f-8636-fbe3d1d866de/scratchpad/e8
until grep -q "ALL DONE" "$SP/split/run_series.log" 2>/dev/null; do sleep 20; done
cd /c/Users/cchen/PycharmProjects/vulkan-demo-e8 || exit 2
for r in 1 2; do
  for cfg in "8m cases/aligned/cavity3d_n192/case.yaml 8000000" "64m cases/aligned/cavity3d_n392/case.yaml 64000000"; do
    set -- $cfg; L=$1; C=$2; N=$3
    for k in "cd8t 8 0" "fo8t 0 8"; do
      set -- $k
      echo "##### ${L}_$1_r$r ($(date +%T))"
      "$SP/split/run_split_ticks.sh" "${L}_$1_r$r" "$C" SPLIT_CD=$2 SPLIT_FO=$3 SPLIT_N=$N 2>&1 | grep -E "STEADY|final:|rc=|Traceback|Error|recorded" | cut -c1-160
    done
  done
done
echo "v7 tree after ticks: $(git rev-parse HEAD:experiment/v7) status: $(git status --short experiment/v7 | wc -l) changed"
echo "##### TICKS DONE $(date +%T)"
