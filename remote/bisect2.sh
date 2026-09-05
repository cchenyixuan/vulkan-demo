#!/usr/bin/env bash
# bisect2.sh <label> — residual-effect probes, logs to logs/bisect_<label>.log
set -euo pipefail
source ~/swq/env.sh
cd ~/swq/vulkan-demo
case "$1" in
single_g7c100_50k)
  python experiment/v5/_run_v5_single_bench.py       --case cases/lid_driven_cavity_2d/case.yaml --device 0       --max-steps 50000 --warmup 5000 --bench-window 45000       > logs/bisect_$1.log 2>&1 || true ;;
winphys_d1_*)
  python experiment/v5/_run_v5_chain_bench.py       --case cases/lid_driven_cavity_2d/case_windows.yaml       --weights 1,1 --device-map 0,1 --sync-scheme per-direction --depth 1       --max-steps 25000 --warmup 5000 > logs/bisect_$1.log 2>&1 || true ;;
esac
grep -E "STEADY|drift=|final:" "logs/bisect_$1.log" | tail -3
