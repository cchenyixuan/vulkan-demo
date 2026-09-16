#!/bin/bash
#SBATCH -p hp_5090
#SBATCH --gpus=8
#SBATCH -J probe35_weak
#SBATCH -o /data/run01/scxm138/logs/probe35_weak_%j.out
# Standardized WEAK-scaling efficiency with error bars: fixed particles per GPU, K = 1..8 on GPUs 0..K-1.
#   eta_weak(K) = fps_K(K x n) / mean_{i<K} fps_1(n; GPU i), reference = K=1 case (n per GPU) run on ALL 8 GPUs
#   simultaneously (same power/thermal state), same window (chain bench 3000 steps, steady = last 2000).
# $FAMILY selects the per-GPU load: weak (2M), weak4 (4M), weak8 (8M), weak16 (16M), weak32 (32M);
# case dirs are cases/cavity_${FAMILY}_k${K}_$((M*K))m.  $TRIALS (default "1 2 3") selects trial indices so a
# heavy family can be split into one-trial jobs (each trial carries its own reference, so eta stays same-node).
# Submit: sbatch --time=00:50:00 --export=ALL,FAMILY=weak8 probe35_weak.sbatch
set -u
source ~/run/tools/env.sh
export MPLCONFIGDIR=/dev/shm/scxm138/mplconfig; mkdir -p "$MPLCONFIGDIR"
cd ~/run/vulkan-demo
FAMILY=${FAMILY:-weak16}; TRIALS=${TRIALS:-"1 2 3"}
case $FAMILY in weak) M=2; KS="2 4 8";; weak4) M=4; KS="2 4 8";; weak8) M=8; KS="2 4 8";; weak16) M=16; KS="2 4 8";; weak32) M=32; KS="2 4 8";; *) echo "unknown FAMILY $FAMILY"; exit 1;; esac
casedir() { echo cases/cavity_${FAMILY}_k$1_$(( M * $1 ))m/case.yaml; }
TAG=${FAMILY}_t$(echo "$TRIALS" | tr -d ' ')
SHM=/dev/shm/scxm138/probe35_${TAG}_$SLURM_JOB_ID; mkdir -p "$SHM"; LOG=~/run/logs/probe35_${TAG}_$SLURM_JOB_ID; mkdir -p "$LOG"
T0=$(date +%s)
echo "=== node $(hostname) job $SLURM_JOB_ID family=$FAMILY M=${M}M/GPU Ks=[$KS] trials=[$TRIALS] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-} $(date) ==="
nvidia-smi --query-gpu=index,name,pci.bus_id,power.limit --format=csv,noheader | head -8
for K in 1 $KS; do [ -f "$(casedir $K)" ] || { echo "MISSING $(casedir $K)"; exit 1; }; done
CPUL=(); SYS=""; for bus in $(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader | sed 's/^0000//' | tr 'A-Z' 'a-z'); do p=/sys/bus/pci/devices/$bus; n=$(cat $p/numa_node 2>/dev/null || echo 0); [ "$n" -lt 0 ] && n=0; CPUL+=("$(cat /sys/devices/system/node/node$n/cpulist)"); [ -d "$p" ] && SYS="$SYS $p"; done
AFF=""; for p in $SYS; do n=$(cat $p/numa_node); AFF="$AFF$(cat /sys/devices/system/node/node$n/cpulist);"; done; export V5_WORKER_AFFINITY="${AFF%;}"
export V5_WORKER_COUNT_AWARE=1 V5_GHOST_POOL_FACTOR=0.25 V5_SWITCH_INTERVAL_MS=0.2 V5_SPLIT_TRANSFER_QUEUES=1 V5_CASCADE_FORCE=1 V5_BAND_VOXEL_DISPATCH=1
echo "fast submit default on: $(grep -c 'V5_FAST_SUBMIT\", \"1\"' experiment/v5/utils/simulator_v5.py)"
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
for T in $TRIALS; do
  echo; echo "=== $FAMILY trial $T: K=1 reference on all 8 GPUs simultaneously ($(date +%T)) ==="
  PIDS=""
  for G in 0 1 2 3 4 5 6 7; do bench ${FAMILY}_ref_t${T}_g$G $(casedir 1) 1 $G "${CPUL[$G]}" & PIDS="$PIDS $!"; done
  wait $PIDS
  for K in $KS; do
    W=$(python -c "print(','.join(['1']*$K))"); D=$(python -c "print(','.join(str(i) for i in range($K)))")
    echo "=== $FAMILY trial $T: K=$K on GPUs 0..$((K-1)) ($(date +%T)) ==="
    bench ${FAMILY}_k${K}_t$T $(casedir $K) $W $D
  done
  cp "$SHM"/*.log "$LOG"/ 2>/dev/null; cp "$SHM/telemetry.csv" "$LOG"/ 2>/dev/null
done
kill $TELEPID 2>/dev/null
cp -r "$SHM"/. "$LOG"/ && rm -rf "$SHM"; ls "$LOG" | wc -l
echo "=== job done, total $(( $(date +%s) - T0 )) s, $(date) ==="
