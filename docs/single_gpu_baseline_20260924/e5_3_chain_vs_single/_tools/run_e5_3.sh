#!/usr/bin/env bash
# E5.3 — V5 chain mode K=1 vs V5 single-GPU mode, same card (V5 device index 1 = headless 5090),
# 2-D 8M, 3 interleaved trials, order rotated per trial.
cd /c/Users/cchen/PycharmProjects/vulkan-demo || exit 1
export VK_LOADER_LAYERS_DISABLE=VK_LAYER_KHRONOS_validation
export V5_CASCADE_FORCE=1
export V5_BAND_VOXEL_DISPATCH=1
PY=.venv/Scripts/python.exe
D=docs/single_gpu_baseline_20260924/e5_3_chain_vs_single
PARSE="C:/Users/cchen/AppData/Local/Temp/claude/C--Users-cchen-PycharmProjects-vulkan-demo/c99e4bc4-2b45-4fe7-8812-bbde3525ccab/scratchpad/e5_3_parse_chain.py"
CASE=cases/lid_driven_cavity_2d_8m/case.yaml
mkdir -p "$D"
rm -f "$D/telemetry.csv"
(nvidia-smi --query-gpu=timestamp,index,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,memory.used --format=csv -l 1 > "$D/telemetry.csv" 2>&1) &
SMI=$!
sleep 1

run_single() {   # $1 = trial
  $PY _run_single_baseline_bench.py --solver v5 --in-flight 2 --case $CASE --device 1 \
      --expect-gpu "RTX 5090" --warmup 1000 --measure 2000 --obj-cache logs/_obj_npy_cache \
      --tag e5.3_single --trial "$1" > "$D/single_t$1.log" 2>&1
  grep "^RESULT " "$D/single_t$1.log" | sed 's/^RESULT //' >> "$D/results.jsonl"
  grep "^RESULT " "$D/single_t$1.log" | sed 's/^RESULT //' | $PY -c "import json,sys; r=json.load(sys.stdin); print('trial %d single  fps=%8.3f drift=%d status=%s'%(r['trial'],r['fps'],r['drift'],r['status']))"
}
run_chain() {    # $1 = trial
  $PY experiment/v5/_run_v5_chain_bench.py --case $CASE --weights 1 --device-map 1 --depth 2 \
      --pool-safety 1.2 --max-steps 3000 --warmup 1000 > "$D/chain_k1_t$1.log" 2>&1
  $PY "$PARSE" "$D/chain_k1_t$1.log" "$1" e5.3_chain_k1 >> "$D/results.jsonl"
  tail -1 "$D/results.jsonl" | $PY -c "import json,sys; r=json.load(sys.stdin); print('trial %d chain   fps=%8.3f drift=%s status=%s'%(r['trial'],r.get('steady_fps',float('nan')),r.get('drift'),r['status']))"
}

for t in 1 2 3; do
  if [ $((t % 2)) -eq 1 ]; then run_single $t; sleep 2; run_chain $t; else run_chain $t; sleep 2; run_single $t; fi
  sleep 2
done

kill $SMI 2>/dev/null
taskkill //F //IM nvidia-smi.exe >/dev/null 2>&1
echo E5.3_DONE
