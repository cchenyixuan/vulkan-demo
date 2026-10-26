#!/bin/bash
# both GPU Trace captures in one (elevated) invocation: 8M cube then 64M cube, K = 1, steady steps; log capture_both.log
SPN=/c/Users/cchen/AppData/Local/Temp/claude/C--Users-cchen-PycharmProjects-vulkan-demo/fd68689d-e4b4-4b3f-8636-fbe3d1d866de/scratchpad/e8/nsight
{
  echo "##### cube8m $(date +%T)"; "$SPN/capture.sh" cube8m cases/aligned/cavity3d_n192/case.yaml 12000 400
  echo "##### cube64m $(date +%T)"; "$SPN/capture.sh" cube64m cases/aligned/cavity3d_n392/case.yaml 90000 1500
  echo "##### ALL DONE $(date +%T)"
} > "$SPN/capture_both.log" 2>&1
