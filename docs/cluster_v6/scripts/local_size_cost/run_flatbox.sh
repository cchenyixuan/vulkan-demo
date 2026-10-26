#!/bin/bash
# flat box (user 2026-10-27): cavity3d_n384_x96, K = 1, two runs, same protocol as run_boxes.sh
set -u
SP=/c/Users/cchen/AppData/Local/Temp/claude/C--Users-cchen-PycharmProjects-vulkan-demo/fd68689d-e4b4-4b3f-8636-fbe3d1d866de/scratchpad/e8/local3d
run() { echo "##### $1 ($(date +%T))"; "$SP/run_item.sh" "$@"; }
for r in 1 2; do
  run flatbox_r$r cases/aligned/cavity3d_n384_x96/case.yaml
done
echo "##### ALL DONE $(date +%T)"
