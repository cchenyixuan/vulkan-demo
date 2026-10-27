#!/bin/bash
# Prefix-sum defrag series (user 2026-10-27): after the 8M validation pair (8m_host_m1_r1 + 8m_off_m1_ctl, run by hand),
# the rest: 8M host r2, 64M host x2 + default control, then the m = 8 piece runs (each kernel split in 8, the other
# unsplit, per-piece ticks) with the prefix defrag, 8M and 64M, two runs each. GPU 1 (575 W), 300 steps, window 100-299.
set -u
SP=/c/Users/cchen/AppData/Local/Temp/claude/C--Users-cchen-PycharmProjects-vulkan-demo/fd68689d-e4b4-4b3f-8636-fbe3d1d866de/scratchpad/e8
HASH="$SP/prefix/stats_check.jsonl"
cd /c/Users/cchen/PycharmProjects/vulkan-demo-e8 || exit 2
echo "v7 tree before: $(git rev-parse HEAD:experiment/v7) status: $(git status --short experiment/v7 | wc -l) changed"
run() {  # run LABEL CASE N ENV...
  local L=$1 C=$2 N=$3; shift 3
  echo "##### $L ($(date +%T))"
  "$SP/prefix/run_prefix.sh" "$L" "$C" SPLIT_N="$N" LAYOUT_AUDIT=1 "$@" 2>&1 | grep -E "STEADY|final:|rc=|Traceback|Error|layout audit #0|defrag #0 done|recorded" | cut -c1-200
}
C8=cases/aligned/cavity3d_n192/case.yaml; C64=cases/aligned/cavity3d_n392/case.yaml
run 8m_host_m1_r2   $C8  8000000  PREFIX_MODE=host SPLIT_HASH_OUT="$HASH"
run 64m_host_m1_r1  $C64 64000000 PREFIX_MODE=host SPLIT_HASH_OUT="$HASH"
run 64m_off_m1_ctl  $C64 64000000 PREFIX_MODE=off  SPLIT_HASH_OUT="$HASH"
run 64m_host_m1_r2  $C64 64000000 PREFIX_MODE=host SPLIT_HASH_OUT="$HASH"
for r in 1 2; do
  run 8m_host_cd8t_r$r  $C8  8000000  PREFIX_MODE=host SPLIT_CD=8 SPLIT_FO=0
  run 8m_host_fo8t_r$r  $C8  8000000  PREFIX_MODE=host SPLIT_CD=0 SPLIT_FO=8
done
for r in 1 2; do
  run 64m_host_cd8t_r$r $C64 64000000 PREFIX_MODE=host SPLIT_CD=8 SPLIT_FO=0
  run 64m_host_fo8t_r$r $C64 64000000 PREFIX_MODE=host SPLIT_CD=0 SPLIT_FO=8
done
echo "v7 tree after: $(git rev-parse HEAD:experiment/v7) status: $(git status --short experiment/v7 | wc -l) changed"
echo "##### ALL DONE $(date +%T)"
