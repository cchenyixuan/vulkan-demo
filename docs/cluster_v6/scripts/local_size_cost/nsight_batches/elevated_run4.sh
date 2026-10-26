#!/bin/bash
# fourth elevated batch (2026-10-27): the 64M cube GPU Trace alone (a second profiler run inside one batch always hits
# "close any existing GPU profiling sessions", so one run per batch). Log: elevated_run4.log
export PATH="/usr/bin:/bin:$PATH"
SPN=/c/Users/cchen/AppData/Local/Temp/claude/C--Users-cchen-PycharmProjects-vulkan-demo/fd68689d-e4b4-4b3f-8636-fbe3d1d866de/scratchpad/e8/nsight
{
  echo "##### ngfx cube64m $(date +%T)"; "$SPN/capture.sh" cube64m cases/aligned/cavity3d_n392/case.yaml 90000 1000
  echo "##### ALL DONE $(date +%T)"
} > "$SPN/elevated_run4.log" 2>&1
