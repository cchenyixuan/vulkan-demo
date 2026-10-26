#!/bin/bash
# discriminating series (user 2026-10-26 evening): the three stretched boxes + the 16M / 32M cubes, two runs each, same protocol
set -u
SP=/c/Users/cchen/AppData/Local/Temp/claude/C--Users-cchen-PycharmProjects-vulkan-demo/fd68689d-e4b4-4b3f-8636-fbe3d1d866de/scratchpad/e8/local3d
C=cases/aligned
run() { echo "##### $1 ($(date +%T))"; "$SP/run_item.sh" "$@"; }
for r in 1 2; do
  run box_k2_r$r  $C/cavity3d_n192_k2/case.yaml
  run cube16m_r$r $C/cavity3d_n244/case.yaml
  run cube32m_r$r $C/cavity3d_n312/case.yaml
  run box_k4_r$r  $C/cavity3d_n192_k4/case.yaml
  run box_k8_r$r  $C/cavity3d_n192_k8/case.yaml
done
echo "##### ALL DONE $(date +%T)"
