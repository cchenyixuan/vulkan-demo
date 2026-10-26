#!/bin/bash
# the whole local series (items 3-6), two runs each; 8M first, 64M last
set -u
SP=/c/Users/cchen/AppData/Local/Temp/claude/C--Users-cchen-PycharmProjects-vulkan-demo/fd68689d-e4b4-4b3f-8636-fbe3d1d866de/scratchpad/e8/local3d
C=cases/aligned
run() { echo "##### $1 ($(date +%T))"; "$SP/run_item.sh" "$@"; }
for r in 1 2; do
  run 8m_base_r$r        $C/cavity3d_n192/case.yaml
  run 8m_dwskip_off_r$r  $C/cavity3d_n192/case.yaml V7_DEEP_WALL_SKIP=0
  run 8m_border8_r$r     $C/cavity3d_n192_b8/case.yaml
  run 8m_lid0_r$r        $C/cavity3d_n192_lid0/case.yaml
done
for r in 1 2; do
  run 64m_base_r$r       $C/cavity3d_n392/case.yaml
  run 64m_dwskip_off_r$r $C/cavity3d_n392/case.yaml V7_DEEP_WALL_SKIP=0
done
echo "##### ALL DONE $(date +%T)"
