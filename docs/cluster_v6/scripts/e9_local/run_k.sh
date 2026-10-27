#!/bin/bash
# run_k.sh LABEL CASE_YAML K [ENV=VALUE ...] - one timed + step-traced chain run of the e8 worktree (the E9 tree) on
# Vulkan device 1 (= nvidia-smi GPU 1, the headless 5090); K slabs all on device 1 (--device-map 1,1,...).
# STEPS / WARMUP (300 / 100), SEAM=1 keeps the seam check (default: on for K > 1, off for K = 1), EXTRA = more bench
# arguments (e.g. "--defrag-cadence 100 --defrag-log"). Output: $OUT/<LABEL>/{run.log,telemetry.csv,trace/,env.txt,status.txt}
set -u
LABEL=$1; CASE=$2; K=$3; shift 3
REPO=/c/Users/cchen/PycharmProjects/vulkan-demo-e8
SPE=/c/Users/cchen/AppData/Local/Temp/claude/C--Users-cchen-PycharmProjects-vulkan-demo/fd68689d-e4b4-4b3f-8636-fbe3d1d866de/scratchpad/e9
OUT=${OUT:-$SPE/runs}
STEPS=${STEPS:-300}; WARMUP=${WARMUP:-100}; EXTRA=${EXTRA:-}
WEIGHTS=$(printf '1%.0s,' $(seq 1 $K)); WEIGHTS=${WEIGHTS%,}
DMAP=$(printf '1%.0s,' $(seq 1 $K)); DMAP=${DMAP%,}
if [ "$K" = 1 ]; then SEAMFLAG=${SEAM:+}; [ "${SEAM:-0}" = 1 ] || SEAMFLAG="--no-seam-check"; else SEAMFLAG=""; fi
DIR=$OUT/$LABEL; rm -rf "$DIR"; mkdir -p "$DIR/trace"
cd "$REPO" || exit 2
printf '%s\n' "$@" > "$DIR/env.txt"; echo "case $CASE K $K steps $STEPS warmup $WARMUP extra [$EXTRA] seam [$SEAMFLAG]" >> "$DIR/env.txt"
nvidia-smi --query-gpu=index,power.limit,driver_version --format=csv,noheader >> "$DIR/env.txt"
nvidia-smi -i 1 --query-gpu=timestamp,utilization.gpu,memory.used,power.draw,clocks.sm,clocks.mem,temperature.gpu,clocks_throttle_reasons.active --format=csv,noheader -lms 1000 > "$DIR/telemetry.csv" 2>&1 &
SMI=$!
T0=$(date +%s)
env "$@" VK_LOADER_LAYERS_DISABLE=VK_LAYER_KHRONOS_validation PYTHONDONTWRITEBYTECODE=1 \
  /c/Users/cchen/PycharmProjects/vulkan-demo/.venv/Scripts/python.exe -u docs/cluster_v6/scripts/run_chain_v6.py --solver v7 --obj-cache logs/e8_local/obj_cache -- \
  --case "$CASE" --weights "$WEIGHTS" --device-map "$DMAP" --sync-scheme per-direction --depth 2 --pool-safety 1.2 --max-steps "$STEPS" --warmup "$WARMUP" $SEAMFLAG \
  --step-trace "$DIR/trace" --step-trace-detail full $EXTRA > "$DIR/run.log" 2>&1
RC=$?
sleep 1; kill $SMI 2>/dev/null
echo "rc=$RC seconds=$(( $(date +%s) - T0 ))" > "$DIR/status.txt"
grep -E "STEADY|final:|seam|SEAM|defrag_prefix\]|defrag_order|init complete|Traceback|Error|defrag_event|steady_fps" "$DIR/run.log" | cut -c1-230
cat "$DIR/status.txt"
