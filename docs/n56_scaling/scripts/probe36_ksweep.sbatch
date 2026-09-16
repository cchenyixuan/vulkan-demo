#!/bin/bash
#SBATCH -p hp_5090
#SBATCH --gpus=8
#SBATCH -w wqd10nbj04g2
#SBATCH --time=03:15:00
#SBATCH -J probe36_ksweep
#SBATCH -o /data/run01/scxm138/logs/probe36_ksweep_%j.out
# Task A — fixed-N K sweep with error bars (standardized protocol, docs/n56_scaling 2026-09-15).
#   For N in {64M, 32M}, K in {2, 4, 8}: 3 alternating trials of (reference, K-run).  Reference = K=1 on GPUs 0..K-1
#   SIMULTANEOUSLY (other GPUs idle), same case, same window (chain bench 3000 steps, steady = last 2000), --anatomy
#   on every run (per-GPU A/B/C/gaps at f2000, f3000).  Topology add-on: 64M K=2 on the cross-NUMA pair (3,4) x2.
#   RESULT label format: <size>_k<K>_ref_t<T>_g<G> (reference GPU G) / <size>_k<K>_run_t<T> / <size>_k2x_ref_t<T>_g<G>,
#   <size>_k2x_run_t<T> for the cross-NUMA pair.
set -u
COMMIT=${COMMIT:-367706a}
source ~/run/tools/env.sh
export MPLCONFIGDIR=/dev/shm/scxm138/mplconfig; mkdir -p "$MPLCONFIGDIR"
cd ~/run/vulkan-demo
SHM=/dev/shm/scxm138/probe36_$SLURM_JOB_ID; mkdir -p "$SHM"; LOG=~/run/logs/probe36_$SLURM_JOB_ID; mkdir -p "$LOG"
T0=$(date +%s)
echo "=== node $(hostname) job $SLURM_JOB_ID probe36_ksweep commit=$(timeout 30 git rev-parse --short HEAD 2>/dev/null || echo unknown) expected=$COMMIT CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-} $(date) ==="
nvidia-smi --query-gpu=index,name,pci.bus_id,power.limit --format=csv,noheader | head -8
echo "--- nvidia-smi topo -m ---"; nvidia-smi topo -m 2>/dev/null | head -14
CPUL=(); SYS=""; for bus in $(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader | sed 's/^0000//' | tr 'A-Z' 'a-z'); do p=/sys/bus/pci/devices/$bus; n=$(cat $p/numa_node 2>/dev/null || echo 0); [ "$n" -lt 0 ] && n=0; CPUL+=("$(cat /sys/devices/system/node/node$n/cpulist)"); [ -d "$p" ] && SYS="$SYS $p"; done
AFF=""; for p in $SYS; do n=$(cat $p/numa_node); AFF="$AFF$(cat /sys/devices/system/node/node$n/cpulist);"; done; export V5_WORKER_AFFINITY="${AFF%;}"
# explicit switches (do not rely on code defaults)
export V5_FAST_SUBMIT=1 V5_CASCADE_FORCE=1 V5_BAND_VOXEL_DISPATCH=1 V5_PHASE_A_NO_WAIT=0 V5_PER_SIM_PIPELINE=0
export V5_WORKER_COUNT_AWARE=1 V5_GHOST_POOL_FACTOR=0.25 V5_SWITCH_INTERVAL_MS=0.2 V5_SPLIT_TRANSFER_QUEUES=1 V5_RECEIVER_CACHED=0 V5_LOOP_TRACE=0 V5_BENCH_PARITY=1
env | grep -E '^V5_' | sort | tr '\n' ' '; echo
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,power.draw,temperature.gpu,clocks.current.sm --format=csv -l 5 > "$SHM/telemetry.csv" 2>/dev/null &
TELEPID=$!

bench() {  # label case weights devmap [cpulist]  -> prints RESULT (anatomy in the per-run log)
  local L=$1 C=$2 W=$3 D=$4 PREFIX=""
  [ -n "${5:-}" ] && PREFIX="taskset -c $5"
  local TA=$(date +%s)
  $PREFIX python -c "import sys; sys.setswitchinterval(0.0002); import runpy; sys.argv=['x','--case','$C','--weights','$W','--device-map','$D','--depth','2','--pool-safety','1.2','--max-steps','3000','--warmup','1000','--anatomy','--no-seam-check']; runpy.run_path('experiment/v5/_run_v5_chain_bench.py', run_name='__main__')" > "$SHM/$L.log" 2>&1
  local RC=$?
  local FPS=$(grep -a "STEADY" "$SHM/$L.log" | sed -E 's/.*= ([0-9.]+) fps.*/\1/' | tail -1)
  local DRIFT=$(grep -a "\[chain_v5\] final" "$SHM/$L.log" | sed -E 's/.*drift=([-0-9]+).*/\1/' | tail -1)
  local STAMPS=$(grep -a "\[chain_v5\] final" "$SHM/$L.log" | sed -E 's/.*stamp_errors gpu=([0-9]+) host=([0-9]+).*/\1+\2/' | tail -1)
  echo "RESULT label=$L rc=$RC steady_fps=${FPS:-nan} drift=${DRIFT:-nan} stamps=${STAMPS:-nan} wall=$(( $(date +%s) - TA ))s $(date +%T)"
  grep -aE "Traceback|Error" "$SHM/$L.log" | head -2
}
run_matrix() {  # size case K gpulist(space-separated) tag
  local SIZE=$1 C=$2 K=$3 GPUS=$4 TAG=$5
  local W=$(python -c "print(','.join(['1']*$K))") D=$(echo $GPUS | tr ' ' ',')
  for T in 1 2 3; do
    [ "$TAG" = "k2x" ] && [ "$T" = 3 ] && break     # topology add-on: 2 trials
    echo; echo "=== $SIZE $TAG trial $T: K=1 reference on GPUs [$GPUS] simultaneously, then K=$K run ($(date +%T)) ==="
    PIDS=""
    for G in $GPUS; do bench ${SIZE}_${TAG}_ref_t${T}_g$G $C 1 $G "${CPUL[$G]}" & PIDS="$PIDS $!"; done
    wait $PIDS
    bench ${SIZE}_${TAG}_run_t$T $C $W $D
    cp "$SHM"/*.log "$LOG"/ 2>/dev/null; cp "$SHM/telemetry.csv" "$LOG"/ 2>/dev/null
  done
}
for SIZE in 64m 32m; do
  CASE=cases/lid_driven_cavity_2d_$SIZE/case.yaml
  run_matrix $SIZE $CASE 2 "0 1" k2
  run_matrix $SIZE $CASE 4 "0 1 2 3" k4
  run_matrix $SIZE $CASE 8 "0 1 2 3 4 5 6 7" k8
done
# topology add-on: 64M K=2 on the cross-NUMA pair (GPU 3 = NUMA 0, GPU 4 = NUMA 1)
run_matrix 64m cases/lid_driven_cavity_2d_64m/case.yaml 2 "3 4" k2x
kill $TELEPID 2>/dev/null
cp -r "$SHM"/. "$LOG"/ && rm -rf "$SHM"; ls "$LOG" | wc -l
echo "=== job done, total $(( $(date +%s) - T0 )) s, $(date) ==="
