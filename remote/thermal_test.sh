#!/usr/bin/env bash
# thermal_test.sh <label> <device_map> — K=3 anatomy chain + live GPU telemetry
set -u
source ~/swq/env.sh
cd ~/swq/vulkan-demo
LABEL="$1"; DMAP="$2"
rm -f "logs/campaign_a/${LABEL}_telemetry.csv"
( while true; do
    nvidia-smi --query-gpu=index,temperature.gpu,clocks.sm,clocks_event_reasons.active         --format=csv,noheader >> "logs/campaign_a/${LABEL}_telemetry.csv"
    sleep 5
  done ) &
SAMPLER=$!
python experiment/v5/_run_v5_chain_bench.py     --case cases/cavity_weak_k3_6m/case.yaml     --weights 1,1,1 --device-map "$DMAP" --sync-scheme per-direction     --depth 2 --anatomy --max-steps 6000 --warmup 2000     > "logs/campaign_a/${LABEL}.log" 2>&1
kill $SAMPLER 2>/dev/null
grep -aE "anatomy. f6000|STEADY" "logs/campaign_a/${LABEL}.log" | tail -4
echo "--- hottest clock samples ---"
sort -t, -k3 -n "logs/campaign_a/${LABEL}_telemetry.csv" | head -8
