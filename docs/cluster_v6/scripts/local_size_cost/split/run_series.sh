#!/bin/bash
# split-dispatch tail series (user 2026-10-27): per cube, two rounds of {m1_plain, m1_seg, cd m=2/4/8 (force plain), fo m=2/4/8
# (corr+dens plain)}; each run = run_item.sh protocol through the monkeypatch (300 steps, window 100-299, 575 W, full step trace).
set -u
SP=/c/Users/cchen/AppData/Local/Temp/claude/C--Users-cchen-PycharmProjects-vulkan-demo/fd68689d-e4b4-4b3f-8636-fbe3d1d866de/scratchpad/e8
cd /c/Users/cchen/PycharmProjects/vulkan-demo-e8 || exit 2
echo "v7 tree before: $(git rev-parse HEAD:experiment/v7) status: $(git status --short experiment/v7 | wc -l) changed"
series() {   # series LABEL CASE N
  local L=$1 C=$2 N=$3
  for r in 1 2; do
    for cfg in "m1_plain 0 0" "m1_seg 1 1" "cd2 2 0" "cd4 4 0" "cd8 8 0" "fo2 0 2" "fo4 0 4" "fo8 0 8"; do
      set -- $cfg
      echo "##### ${L}_$1_r$r ($(date +%T))"
      "$SP/split/run_split.sh" "${L}_$1_r$r" "$C" SPLIT_CD=$2 SPLIT_FO=$3 SPLIT_N=$N 2>&1 | grep -E "STEADY|final:|rc=|Traceback|Error" | cut -c1-160
    done
  done
}
series 8m  cases/aligned/cavity3d_n192/case.yaml 8000000
series 64m cases/aligned/cavity3d_n392/case.yaml 64000000
echo "v7 tree after: $(git rev-parse HEAD:experiment/v7) status: $(git status --short experiment/v7 | wc -l) changed"
echo "##### ALL DONE $(date +%T)"
