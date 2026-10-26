#!/bin/bash
# fifth elevated batch (2026-10-27): 64M GPU Trace with the hardware event system off (the 64M activity start failed three times
# with "counters unavailable" while 8M works), then a 32M cube as the fallback control. Log: elevated_run5.log
export PATH="/usr/bin:/bin:$PATH"
SPN=/c/Users/cchen/AppData/Local/Temp/claude/C--Users-cchen-PycharmProjects-vulkan-demo/fd68689d-e4b4-4b3f-8636-fbe3d1d866de/scratchpad/e8/nsight
{
  echo "##### ngfx cube64m_hes0 $(date +%T)"; HES=0 "$SPN/capture.sh" cube64m_hes0 cases/aligned/cavity3d_n392/case.yaml 90000 1000
  sleep 20
  echo "##### ngfx cube32m $(date +%T)"; "$SPN/capture.sh" cube32m cases/aligned/cavity3d_n312/case.yaml 50000 600
  echo "##### ALL DONE $(date +%T)"
} > "$SPN/elevated_run5.log" 2>&1
