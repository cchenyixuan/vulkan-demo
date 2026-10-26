#!/bin/bash
# nsys_capture.sh LABEL CASE_YAML DELAY_S DURATION_S — Nsight Systems GPU-metrics sampling (200 kHz = 5 us, set gb20x: SM active,
# warps in flight, issue, DRAM ...) + Vulkan workload trace of the chain code, K = 1 on Vulkan device 1, same solver arguments
# as run_item.sh; collection starts DELAY_S after launch (inside the steady steps) and lasts DURATION_S; the app is then killed.
# Output: $SPN/<LABEL>/nsys_<LABEL>.nsys-rep (+ nsys.log, telemetry.csv). Needs GPU performance-counter access (admin).
set -u
export PATH="/usr/bin:/bin:$PATH"   # an elevated bash started by Start-Process has no MSYS tools on PATH
LABEL=$1; CASE=$2; DELAY=$3; DUR=$4; FREQ=${5:-200000}; SET=${6:-gb20x}
SPN=/c/Users/cchen/AppData/Local/Temp/claude/C--Users-cchen-PycharmProjects-vulkan-demo/fd68689d-e4b4-4b3f-8636-fbe3d1d866de/scratchpad/e8/nsight
NS="/c/Program Files/NVIDIA Corporation/Nsight Systems 2025.5.2/target-windows-x64/nsys.exe"
OUT=$SPN/$LABEL; mkdir -p "$OUT/trace"
WOUT="C:/Users/cchen/AppData/Local/Temp/claude/C--Users-cchen-PycharmProjects-vulkan-demo/fd68689d-e4b4-4b3f-8636-fbe3d1d866de/scratchpad/e8/nsight/$LABEL"
cd /c/Users/cchen/PycharmProjects/vulkan-demo-e8 || exit 2
export VK_LOADER_LAYERS_DISABLE=VK_LAYER_KHRONOS_validation PYTHONDONTWRITEBYTECODE=1
nvidia-smi -i 1 --query-gpu=timestamp,utilization.gpu,memory.used,power.draw,clocks.sm,clocks.mem,temperature.gpu,clocks_throttle_reasons.active --format=csv,noheader -lms 1000 > "$OUT/telemetry.csv" 2>&1 &
SMI=$!
T0=$(date +%s)
"$NS" profile --gpu-metrics-devices=help > "$OUT/nsys_devices.txt" 2>&1
MSYS_NO_PATHCONV=1 "$NS" profile --output="$WOUT/nsys_$LABEL" --force-overwrite=true \
  --trace=vulkan --vulkan-gpu-workload=individual \
  --gpu-metrics-devices=all --gpu-metrics-frequency="$FREQ" --gpu-metrics-set="$SET" \
  --sample=none --cpuctxsw=none --delay="$DELAY" --duration="$DUR" --kill=true --wait=primary --show-output=true \
  "C:/Users/cchen/PycharmProjects/vulkan-demo/.venv/Scripts/python.exe" -u docs/cluster_v6/scripts/run_chain_v6.py --solver v7 --obj-cache logs/e8_local/obj_cache -- \
  --case "$CASE" --weights 1 --device-map 1 --sync-scheme per-direction --depth 2 --pool-safety 1.2 --max-steps 300 --warmup 100 --no-seam-check \
  --step-trace "$WOUT/trace" --step-trace-detail full > "$OUT/nsys.log" 2>&1
RC=$?
sleep 1; kill $SMI 2>/dev/null
echo "rc=$RC seconds=$(( $(date +%s) - T0 ))" | tee "$OUT/status.txt"
ls -la "$OUT"
tail -15 "$OUT/nsys.log" | cut -c1-200
