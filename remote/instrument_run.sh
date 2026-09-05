#!/usr/bin/env bash
# instrument_run.sh <label> <case> <weights> <dmap> <depth> <steps>
set -u
source ~/swq/env.sh
cd ~/swq/vulkan-demo
LABEL="$1"
rm -f "logs/campaign_a/${LABEL}_clocks.csv"
( while true; do
    echo "$(date +%s),$(nvidia-smi --query-gpu=index,temperature.gpu,clocks.sm,clocks.mem,clocks_event_reasons.active --format=csv,noheader | tr '
' ';')"         >> "logs/campaign_a/${LABEL}_clocks.csv"
    sleep 2
  done ) &
SAMPLER=$!
python experiment/v5/_run_v5_chain_bench.py     --case "$2" --weights "$3" --device-map "$4" --sync-scheme per-direction     --depth "$5" --anatomy --defrag-cadence 250 --max-steps "$6" --warmup 2000     > "logs/campaign_a/${LABEL}.log" 2>&1
kill $SAMPLER 2>/dev/null
grep -aE "STEADY|drift=|worker .* p50" "logs/campaign_a/${LABEL}.log" | tail -12
