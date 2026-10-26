#!/bin/bash
# third elevated batch (2026-10-27): GPU Trace with per-dispatch regimes (--time-every-action) for the 8M and 64M cubes, then two
# short nsys tests at 20 kHz (set gb20x and gb20x-top) to see whether the SM counters come alive at a lower rate. Log: elevated_run3.log
export PATH="/usr/bin:/bin:$PATH"
SPN=/c/Users/cchen/AppData/Local/Temp/claude/C--Users-cchen-PycharmProjects-vulkan-demo/fd68689d-e4b4-4b3f-8636-fbe3d1d866de/scratchpad/e8/nsight
{
  echo "##### ngfx cube8m_actions $(date +%T)";  "$SPN/capture.sh" cube8m_actions  cases/aligned/cavity3d_n192/case.yaml 12000 400 --time-every-action
  sleep 15
  echo "##### ngfx cube64m_actions $(date +%T)"; "$SPN/capture.sh" cube64m_actions cases/aligned/cavity3d_n392/case.yaml 90000 1000 --time-every-action
  sleep 15
  echo "##### nsys nsys8m_20k $(date +%T)";  "$SPN/nsys_capture.sh" nsys8m_20k cases/aligned/cavity3d_n192/case.yaml 12 1 20000 gb20x
  sleep 15
  echo "##### nsys nsys8m_top $(date +%T)";  "$SPN/nsys_capture.sh" nsys8m_top cases/aligned/cavity3d_n192/case.yaml 12 1 20000 gb20x-top
  echo "##### ALL DONE $(date +%T)"
} > "$SPN/elevated_run3.log" 2>&1
