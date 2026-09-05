#!/usr/bin/env bash
# ab_test.sh <label> <scheme> <depth> — one chain bench config, full log kept
set -euo pipefail
source ~/swq/env.sh
cd ~/swq/vulkan-demo
python experiment/v5/_run_v5_chain_bench.py     --case cases/lid_driven_cavity_2d/case.yaml     --weights 1,1 --device-map 0,1     --sync-scheme "$2" --depth "$3"     --max-steps 25000 --warmup 5000     > "logs/ab_$1.log" 2>&1 || true
grep -E "STEADY|drift=|VALIDATION" "logs/ab_$1.log" | tail -4
