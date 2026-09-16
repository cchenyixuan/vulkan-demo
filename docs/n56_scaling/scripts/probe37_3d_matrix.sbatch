#!/bin/bash
#SBATCH -p hp_5090
#SBATCH --gpus=8
#SBATCH --time=03:15:00
#SBATCH -J probe37_3d
#SBATCH -o /data/run01/scxm138/logs/probe37_3d_%j.out
# 3-D matrix (docs/n56_scaling/3d_matrix_budget.md, 2026-09-17) — NOT to be submitted without confirmation.
#   Protocol: chain bench 1500 steps, steady = last 1000 (warmup 500), reference and K-run share the window;
#   per N ONE 8-GPU simultaneous K=1 reference per job, K-runs x3; --anatomy everywhere; V5_GHOST_POOL_FACTOR=1.0;
#   border-4 cases (gen_3d_families.sh default); other switches as the 2-D curve jobs. Bootstrap defrag is in
#   the frozen code (eda4b8f), so no slow first-1000 phase.
#   Order: strong 64M stretched (K=2/4/8) -> weak 8M (K=2/4; K=8 shared with the strong block) -> strong 32M
#   (K=2/4/8) -> weak 4M (K=2/4) -> cube 64M K=8 control (no reference: fps + anatomy only).
#   RESULT labels: <block>_ref_g<G> (K=1 on GPU G, 8 at once) / <block>_k<K>_run_t<T>.
set -u
COMMIT=${COMMIT:-eda4b8f}
STEPS=${STEPS:-1500}; WARMUP=${WARMUP:-500}
source ~/run/tools/env.sh
export MPLCONFIGDIR=/dev/shm/scxm138/mplconfig; mkdir -p "$MPLCONFIGDIR"
cd ~/run/vulkan-demo
SHM=/dev/shm/scxm138/probe37_$SLURM_JOB_ID; mkdir -p "$SHM"; LOG=~/run/logs/probe37_$SLURM_JOB_ID; mkdir -p "$LOG"
T0=$(date +%s)
HEAD=$(timeout 30 git rev-parse --short HEAD 2>/dev/null || echo unknown)
echo "=== node $(hostname) job $SLURM_JOB_ID probe37_3d commit=$HEAD expected=$COMMIT steps=$STEPS warmup=$WARMUP CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-} $(date) ==="
[ "$HEAD" = "$COMMIT" ] || { echo "ABORT: HEAD $HEAD != expected $COMMIT"; exit 2; }
for M in .cavity3d_weak4_ready .cavity3d_weak8_ready .cavity3d_cube64_ready; do [ -f cases/$M ] || { echo "ABORT: missing cases marker $M"; exit 3; }; done
nvidia-smi --query-gpu=index,name,pci.bus_id,power.limit --format=csv,noheader | head -8
echo "--- nvidia-smi topo -m ---"; nvidia-smi topo -m 2>/dev/null | head -14
CPUL=(); SYS=""; for bus in $(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader | sed 's/^0000//' | tr 'A-Z' 'a-z'); do p=/sys/bus/pci/devices/$bus; n=$(cat $p/numa_node 2>/dev/null || echo 0); [ "$n" -lt 0 ] && n=0; CPUL+=("$(cat /sys/devices/system/node/node$n/cpulist)"); [ -d "$p" ] && SYS="$SYS $p"; done
AFF=""; for p in $SYS; do n=$(cat $p/numa_node); AFF="$AFF$(cat /sys/devices/system/node/node$n/cpulist);"; done; export V5_WORKER_AFFINITY="${AFF%;}"
# explicit switches (do not rely on code defaults)
export V5_FAST_SUBMIT=1 V5_CASCADE_FORCE=1 V5_BAND_VOXEL_DISPATCH=1 V5_BAND_SLOT_LANES=0 V5_FAKE_BAND_TEST=0 V5_PHASE_A_NO_WAIT=0 V5_PER_SIM_PIPELINE=0
export V5_WORKER_COUNT_AWARE=1 V5_GHOST_POOL_FACTOR=1.0 V5_SWITCH_INTERVAL_MS=0.2 V5_SPLIT_TRANSFER_QUEUES=1 V5_RECEIVER_CACHED=0 V5_LOOP_TRACE=0 V5_BENCH_PARITY=1
env | grep -E '^V5_' | sort | tr '\n' ' '; echo
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,power.draw,temperature.gpu,clocks.current.sm --format=csv -l 5 > "$SHM/telemetry.csv" 2>/dev/null &
TELEPID=$!

bench() {  # label case weights devmap [cpulist]  -> prints RESULT (anatomy in the per-run log)
  local L=$1 C=$2 W=$3 D=$4 PREFIX=""
  [ -n "${5:-}" ] && PREFIX="taskset -c $5"
  local TA=$(date +%s)
  $PREFIX python -c "import sys; sys.setswitchinterval(0.0002); import runpy; sys.argv=['x','--case','$C','--weights','$W','--device-map','$D','--depth','2','--pool-safety','1.2','--max-steps','$STEPS','--warmup','$WARMUP','--anatomy','--no-seam-check']; runpy.run_path('experiment/v5/_run_v5_chain_bench.py', run_name='__main__')" > "$SHM/$L.log" 2>&1
  local RC=$?
  local FPS=$(grep -a "STEADY" "$SHM/$L.log" | sed -E 's/.*= ([0-9.]+) fps.*/\1/' | tail -1)
  local DRIFT=$(grep -a "\[chain_v5\] final" "$SHM/$L.log" | sed -E 's/.*drift=([-0-9]+).*/\1/' | tail -1)
  local STAMPS=$(grep -a "\[chain_v5\] final" "$SHM/$L.log" | sed -E 's/.*stamp_errors gpu=([0-9]+) host=([0-9]+).*/\1+\2/' | tail -1)
  local LOAD=$(grep -a "total loaded" "$SHM/$L.log" | head -1 | sed -E 's/.*: ([0-9,]+) particles.*/\1/')
  echo "RESULT label=$L rc=$RC steady_fps=${FPS:-nan} drift=${DRIFT:-nan} stamps=${STAMPS:-nan} particles=${LOAD:-nan} wall=$(( $(date +%s) - TA ))s $(date +%T)"
  grep -aE "Traceback|Error" "$SHM/$L.log" | head -2
  cp "$SHM/$L.log" "$LOG"/ 2>/dev/null; cp "$SHM/telemetry.csv" "$LOG"/ 2>/dev/null
}
reference8() {  # block case : K=1 on all 8 GPUs simultaneously (one set per job)
  local BLOCK=$1 C=$2
  echo; echo "=== $BLOCK: K=1 reference on GPUs 0-7 simultaneously ($C) ($(date +%T)) ==="
  PIDS=""
  for G in 0 1 2 3 4 5 6 7; do bench ${BLOCK}_ref_g$G $C 1 $G "${CPUL[$G]}" & PIDS="$PIDS $!"; done
  wait $PIDS
}
kruns() {  # block case K trials : K-run x trials on GPUs 0..K-1
  local BLOCK=$1 C=$2 K=$3 N=$4
  local W=$(python -c "print(','.join(['1']*$K))") D=$(python -c "print(','.join(str(i) for i in range($K)))")
  for T in $(seq 1 $N); do
    echo; echo "=== $BLOCK K=$K run trial $T ($C) ($(date +%T)) ==="
    bench ${BLOCK}_k${K}_run_t$T $C $W $D
  done
}
# 1. strong 64M stretched (= weak8 K=8 geometry): reference + K=2/4/8 x3
C64=cases/cavity3d_weak8_k8_64m/case.yaml
reference8 strong64m $C64
kruns strong64m $C64 2 3; kruns strong64m $C64 4 3; kruns strong64m $C64 8 3
# 2. weak 8M/GPU: reference on the K=1 case, K=2/4 x3 (K=8 = strong64m_k8)
reference8 weak8 cases/cavity3d_weak8_k1_8m/case.yaml
kruns weak8 cases/cavity3d_weak8_k2_16m/case.yaml 2 3; kruns weak8 cases/cavity3d_weak8_k4_32m/case.yaml 4 3
# 3. strong 32M stretched (= weak4 K=8 geometry): reference + K=2/4/8 x3
C32=cases/cavity3d_weak4_k8_32m/case.yaml
reference8 strong32m $C32
kruns strong32m $C32 2 3; kruns strong32m $C32 4 3; kruns strong32m $C32 8 3
# 4. weak 4M/GPU: reference on the K=1 case, K=2/4 x3 (K=8 = strong32m_k8)
reference8 weak4 cases/cavity3d_weak4_k1_4m/case.yaml
kruns weak4 cases/cavity3d_weak4_k2_8m/case.yaml 2 3; kruns weak4 cases/cavity3d_weak4_k4_16m/case.yaml 4 3
# 5. cube 64M K=8 control (1-D slab in a cube; fps + anatomy vs strong64m_k8; no reference)
kruns cube64m cases/cavity3d_cube_64m/case.yaml 8 3
kill $TELEPID 2>/dev/null
cp -r "$SHM"/. "$LOG"/ && rm -rf "$SHM"; ls "$LOG" | wc -l
echo "=== job done, total $(( $(date +%s) - T0 )) s, $(date) ==="
