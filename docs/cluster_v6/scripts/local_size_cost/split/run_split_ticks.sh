#!/bin/bash
# run_split.sh (= run_item.sh through the split_dispatch_run_ticks.py monkeypatch; pass SPLIT_CD= SPLIT_FO= SPLIT_N= as ENV args)
# run_item.sh LABEL CASE_YAML [ENV=VALUE ...] — one timed + step-traced K = 1 run of the chain code on Vulkan device 1
# (= nvidia-smi GPU 1, the headless 5090), 300 steps, window 100-299, bootstrap defrag as usual, nvidia-smi at 1 Hz.
# Output: $OUT/<LABEL>/{run.log,telemetry.csv,trace/,env.txt,status.txt}
set -u
LABEL=$1; CASE=$2; shift 2
REPO=/c/Users/cchen/PycharmProjects/vulkan-demo-e8
OUT=${OUT:-/c/Users/cchen/AppData/Local/Temp/claude/C--Users-cchen-PycharmProjects-vulkan-demo/fd68689d-e4b4-4b3f-8636-fbe3d1d866de/scratchpad/e8/split/runs}
DIR=$OUT/$LABEL; rm -rf "$DIR"; mkdir -p "$DIR/trace"
cd "$REPO" || exit 2
printf '%s\n' "$@" > "$DIR/env.txt"; echo "case $CASE" >> "$DIR/env.txt"; nvidia-smi --query-gpu=index,power.limit,driver_version --format=csv,noheader >> "$DIR/env.txt"
nvidia-smi -i 1 --query-gpu=timestamp,utilization.gpu,memory.used,power.draw,clocks.sm,clocks.mem,temperature.gpu,clocks_throttle_reasons.active --format=csv,noheader -lms 1000 > "$DIR/telemetry.csv" 2>&1 &
SMI=$!
T0=$(date +%s)
env "$@" VK_LOADER_LAYERS_DISABLE=VK_LAYER_KHRONOS_validation PYTHONDONTWRITEBYTECODE=1 \
  /c/Users/cchen/PycharmProjects/vulkan-demo/.venv/Scripts/python.exe -u C:/Users/cchen/AppData/Local/Temp/claude/C--Users-cchen-PycharmProjects-vulkan-demo/fd68689d-e4b4-4b3f-8636-fbe3d1d866de/scratchpad/e8/split/split_dispatch_run_ticks.py --solver v7 --obj-cache logs/e8_local/obj_cache -- \
  --case "$CASE" --weights 1 --device-map 1 --sync-scheme per-direction --depth 2 --pool-safety 1.2 --max-steps 300 --warmup 100 --no-seam-check \
  --step-trace "$DIR/trace" --step-trace-detail full > "$DIR/run.log" 2>&1
RC=$?
sleep 1; kill $SMI 2>/dev/null
echo "rc=$RC seconds=$(( $(date +%s) - T0 ))" > "$DIR/status.txt"
grep -E "STEADY|TOTAL|final:|stage loop_start|stage loop_end|init complete|Traceback|Error|[split]" "$DIR/run.log" | cut -c1-160
cat "$DIR/status.txt"
