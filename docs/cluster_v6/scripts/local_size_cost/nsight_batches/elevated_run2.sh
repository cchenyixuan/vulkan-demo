#!/bin/bash
# second elevated batch (2026-10-27): nsys GPU-metrics sampling of the 8M and 64M cubes (time series the GPU Trace export lacks),
# then the Nsight Graphics GPU Trace of the 64M cube (the 8M one is already captured). Log: elevated_run2.log
export PATH="/usr/bin:/bin:$PATH"
SPN=/c/Users/cchen/AppData/Local/Temp/claude/C--Users-cchen-PycharmProjects-vulkan-demo/fd68689d-e4b4-4b3f-8636-fbe3d1d866de/scratchpad/e8/nsight
{
  echo "##### nsys cube8m $(date +%T)";  "$SPN/nsys_capture.sh" nsys8m  cases/aligned/cavity3d_n192/case.yaml 12 2
  echo "##### nsys cube64m $(date +%T)"; "$SPN/nsys_capture.sh" nsys64m cases/aligned/cavity3d_n392/case.yaml 120 3
  echo "##### ngfx cube64m $(date +%T)"; "$SPN/capture.sh" cube64m cases/aligned/cavity3d_n392/case.yaml 90000 1000
  echo "##### ALL DONE $(date +%T)"
} > "$SPN/elevated_run2.log" 2>&1
