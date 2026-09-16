#!/bin/bash
#SBATCH -p hp_5090
#SBATCH --gpus=8
#SBATCH -J probe34_curve
#SBATCH -o /data/run01/scxm138/logs/probe34_curve_%j.out
# Standardized strong-scaling efficiency vs problem size, K=8, with error bars.
# Reference protocol: ALL 8 GPUs run K=1 SIMULTANEOUSLY (same power/thermal state as K=8), same physical window
# (chain bench 3000 steps, steady = last 2000); 128M: 4 GPU pairs run K=2 simultaneously (K=1 does not fit).
# 3 alternating trials of (reference, K=8) per size. Node-wide nvidia-smi sampler logs SM clock / power per GPU.
# $SIZES (env) selects the sizes; submit with: sbatch --time=HH:MM:SS --export=ALL,SIZES="64m" probe34_curve.sbatch
# v2 (2026-09-16): `wait` only on the bench PIDs (v1 waited on the telemetry sampler forever), logs copied to
# ~/run/logs after every trial so a timeout cannot lose data.
set -u
source ~/run/tools/env.sh
export MPLCONFIGDIR=/dev/shm/scxm138/mplconfig; mkdir -p "$MPLCONFIGDIR"
cd ~/run/vulkan-demo
SIZES=${SIZES:-"4m 16m 32m 64m 128m"}
TAG=$(echo "$SIZES" | tr -d ' ')
SHM=/dev/shm/scxm138/probe34_${TAG}_$SLURM_JOB_ID; mkdir -p "$SHM"; LOG=~/run/logs/probe34_${TAG}_$SLURM_JOB_ID; mkdir -p "$LOG"
T0=$(date +%s)
echo "=== node $(hostname) job $SLURM_JOB_ID sizes=[$SIZES] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-} $(date) ==="
nvidia-smi --query-gpu=index,name,pci.bus_id,power.limit --format=csv,noheader | head -8
CPUL=(); SYS=""; for bus in $(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader | sed 's/^0000//' | tr 'A-Z' 'a-z'); do p=/sys/bus/pci/devices/$bus; n=$(cat $p/numa_node 2>/dev/null || echo 0); [ "$n" -lt 0 ] && n=0; CPUL+=("$(cat /sys/devices/system/node/node$n/cpulist)"); [ -d "$p" ] && SYS="$SYS $p"; done
AFF=""; for p in $SYS; do n=$(cat $p/numa_node); AFF="$AFF$(cat /sys/devices/system/node/node$n/cpulist);"; done; export V5_WORKER_AFFINITY="${AFF%;}"
export V5_WORKER_COUNT_AWARE=1 V5_GHOST_POOL_FACTOR=0.25 V5_SWITCH_INTERVAL_MS=0.2 V5_SPLIT_TRANSFER_QUEUES=1 V5_CASCADE_FORCE=1 V5_BAND_VOXEL_DISPATCH=1
echo "fast submit default on: $(grep -c 'V5_FAST_SUBMIT\", \"1\"' experiment/v5/utils/simulator_v5.py)"
W8=1,1,1,1,1,1,1,1; D8=0,1,2,3,4,5,6,7
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,power.draw,temperature.gpu,clocks.current.sm --format=csv -l 5 > "$SHM/telemetry.csv" 2>/dev/null &
TELEPID=$!

bench() {  # label case weights devmap [cpulist]  -> prints RESULT
  local L=$1 C=$2 W=$3 D=$4 PREFIX=""
  [ -n "${5:-}" ] && PREFIX="taskset -c $5"
  local TA=$(date +%s)
  $PREFIX python -c "import sys; sys.setswitchinterval(0.0002); import runpy; sys.argv=['x','--case','$C','--weights','$W','--device-map','$D','--depth','2','--pool-safety','1.2','--max-steps','3000','--warmup','1000','--no-seam-check']; runpy.run_path('experiment/v5/_run_v5_chain_bench.py', run_name='__main__')" > "$SHM/$L.log" 2>&1
  local RC=$?
  local FPS=$(grep -a "STEADY" "$SHM/$L.log" | sed -E 's/.*= ([0-9.]+) fps.*/\1/' | tail -1)
  local DRIFT=$(grep -a "\[chain_v5\] final" "$SHM/$L.log" | sed -E 's/.*drift=([-0-9]+).*/\1/' | tail -1)
  echo "RESULT label=$L rc=$RC steady_fps=${FPS:-nan} drift=${DRIFT:-nan} wall=$(( $(date +%s) - TA ))s $(date +%T)"
  grep -aE "Traceback|Error" "$SHM/$L.log" | head -2
}
for SIZE in $SIZES; do
  CASE=cases/lid_driven_cavity_2d_$SIZE/case.yaml
  for T in 1 2 3; do
    echo; echo "=== $SIZE trial $T: simultaneous reference on all 8 GPUs, then K=8 ($(date +%T)) ==="
    PIDS=""
    if [ "$SIZE" = "128m" ]; then
      for P in 0 1 2 3; do bench ${SIZE}_ref_t${T}_g$((2*P))$((2*P+1)) $CASE 1,1 $((2*P)),$((2*P+1)) & PIDS="$PIDS $!"; done
    else
      for G in 0 1 2 3 4 5 6 7; do bench ${SIZE}_ref_t${T}_g$G $CASE 1 $G "${CPUL[$G]}" & PIDS="$PIDS $!"; done
    fi
    wait $PIDS
    bench ${SIZE}_k8_t$T $CASE $W8 $D8
    cp "$SHM"/*.log "$LOG"/ 2>/dev/null; cp "$SHM/telemetry.csv" "$LOG"/ 2>/dev/null
  done
done
kill $TELEPID 2>/dev/null
cp -r "$SHM"/. "$LOG"/ && rm -rf "$SHM"; ls "$LOG" | wc -l
echo "=== job done, total $(( $(date +%s) - T0 )) s, $(date) ==="
