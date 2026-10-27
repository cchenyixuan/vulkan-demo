#!/bin/bash
# run_prefix.sh LABEL CASE_YAML [ENV=VALUE ...] - run_item.sh through prefix_defrag_run.py (PREFIX_MODE=off|raw|host,
# LAYOUT_AUDIT=1, SPLIT_CD= SPLIT_FO= SPLIT_N= SPLIT_HASH_OUT= as ENV args); STEPS / WARMUP override 300 / 100.
# One timed + step-traced K = 1 run of the chain code on Vulkan device 1 (= nvidia-smi GPU 1, the headless 5090),
# bootstrap defrag as usual, nvidia-smi at 1 Hz. Output: $OUT/<LABEL>/{run.log,telemetry.csv,trace/,env.txt,status.txt}
set -u
LABEL=$1; CASE=$2; shift 2
REPO=/c/Users/cchen/PycharmProjects/vulkan-demo-e8
SP=/c/Users/cchen/AppData/Local/Temp/claude/C--Users-cchen-PycharmProjects-vulkan-demo/fd68689d-e4b4-4b3f-8636-fbe3d1d866de/scratchpad/e8
SPW="C:/Users/cchen/AppData/Local/Temp/claude/C--Users-cchen-PycharmProjects-vulkan-demo/fd68689d-e4b4-4b3f-8636-fbe3d1d866de/scratchpad/e8"
OUT=${OUT:-$SP/prefix/runs}
STEPS=${STEPS:-300}; WARMUP=${WARMUP:-100}
DIR=$OUT/$LABEL; rm -rf "$DIR"; mkdir -p "$DIR/trace"
cd "$REPO" || exit 2
printf '%s\n' "$@" > "$DIR/env.txt"; echo "case $CASE steps $STEPS warmup $WARMUP" >> "$DIR/env.txt"
nvidia-smi --query-gpu=index,power.limit,driver_version --format=csv,noheader >> "$DIR/env.txt"
nvidia-smi -i 1 --query-gpu=timestamp,utilization.gpu,memory.used,power.draw,clocks.sm,clocks.mem,temperature.gpu,clocks_throttle_reasons.active --format=csv,noheader -lms 1000 > "$DIR/telemetry.csv" 2>&1 &
SMI=$!
T0=$(date +%s)
env "$@" PREFIX_AUDIT_OUT="$DIR/layout_audit.jsonl" VK_LOADER_LAYERS_DISABLE=VK_LAYER_KHRONOS_validation PYTHONDONTWRITEBYTECODE=1 \
  /c/Users/cchen/PycharmProjects/vulkan-demo/.venv/Scripts/python.exe -u "$SPW/prefix/prefix_defrag_run.py" --solver v7 --obj-cache logs/e8_local/obj_cache -- \
  --case "$CASE" --weights 1 --device-map 1 --sync-scheme per-direction --depth 2 --pool-safety 1.2 --max-steps "$STEPS" --warmup "$WARMUP" --no-seam-check \
  --step-trace "$DIR/trace" --step-trace-detail full > "$DIR/run.log" 2>&1
RC=$?
sleep 1; kill $SMI 2>/dev/null
echo "rc=$RC seconds=$(( $(date +%s) - T0 ))" > "$DIR/status.txt"
grep -E "STEADY|final:|stage loop_start|stage loop_end|init complete|Traceback|Error|\[prefix\]|\[split\] (pipelines|recorded|state)" "$DIR/run.log" | cut -c1-230
cat "$DIR/status.txt"
