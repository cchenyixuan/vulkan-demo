#!/bin/bash
# capture.sh LABEL CASE_YAML START_MS DURATION_MS — Nsight Graphics GPU Trace of the chain code, K = 1 on Vulkan device 1,
# same solver arguments as run_item.sh (300 steps, warmup 100, bootstrap defrag, full step trace); trace window starts
# START_MS after collection is ready (= inside the steady steps) and lasts DURATION_MS; metrics auto-exported; clocks unaltered.
set -u
LABEL=$1; CASE=$2; START=$3; DUR=$4
SPN=/c/Users/cchen/AppData/Local/Temp/claude/C--Users-cchen-PycharmProjects-vulkan-demo/fd68689d-e4b4-4b3f-8636-fbe3d1d866de/scratchpad/e8/nsight
NG="/c/Program Files/NVIDIA Corporation/Nsight Graphics 2026.3.1/host/windows-desktop-nomad-x64/ngfx.exe"
OUT=$SPN/$LABEL; rm -rf "$OUT"; mkdir -p "$OUT/trace"
WOUT="C:/Users/cchen/AppData/Local/Temp/claude/C--Users-cchen-PycharmProjects-vulkan-demo/fd68689d-e4b4-4b3f-8636-fbe3d1d866de/scratchpad/e8/nsight/$LABEL"
nvidia-smi -i 1 --query-gpu=timestamp,utilization.gpu,memory.used,power.draw,clocks.sm,clocks.mem,temperature.gpu,clocks_throttle_reasons.active --format=csv,noheader -lms 1000 > "$OUT/telemetry.csv" 2>&1 &
SMI=$!
T0=$(date +%s)
MSYS_NO_PATHCONV=1 "$NG" --activity="GPU Trace Profiler" --platform="Windows (x86_64)" \
  --exe "C:/Users/cchen/PycharmProjects/vulkan-demo/.venv/Scripts/python.exe" \
  --dir "C:/Users/cchen/PycharmProjects/vulkan-demo-e8" \
  --args "-u docs/cluster_v6/scripts/run_chain_v6.py --solver v7 --obj-cache logs/e8_local/obj_cache -- --case $CASE --weights 1 --device-map 1 --sync-scheme per-direction --depth 2 --pool-safety 1.2 --max-steps 300 --warmup 100 --no-seam-check --step-trace $WOUT/trace --step-trace-detail full" \
  --env "VK_LOADER_LAYERS_DISABLE=VK_LAYER_KHRONOS_validation; PYTHONDONTWRITEBYTECODE=1;" \
  --output-dir "$WOUT" --start-after-ms "$START" --max-duration-ms "$DUR" --auto-export --set-gpu-clocks unaltered \
  --architecture "Blackwell GB20x" --metric-set-id 0 --hes-enabled 1 --collect-screenshot 0 --no-timeout --verbose > "$OUT/ngfx.log" 2>&1
RC=$?
sleep 1; kill $SMI 2>/dev/null
echo "rc=$RC seconds=$(( $(date +%s) - T0 ))" | tee "$OUT/status.txt"
ls -la "$OUT"
tail -30 "$OUT/ngfx.log" | cut -c1-200
