#!/bin/bash
#SBATCH -p hp_5090
#SBATCH --gpus=8
#SBATCH --time=00:40:00
#SBATCH -J probe38_anat
#SBATCH -o /data/run01/scxm138/logs/probe38_anat_%j.out
# 2-D K=8 anatomy for the strong-vs-size points that probe34 measured without --anatomy (4M, 16M, 128M),
# so the unified hiding-margin figure has a measured transport slack for every strong point. Same
# switches as the curve jobs; 1500 steps, warmup 500; one run per size. No solver change (freeze eda4b8f).
set -u
source ~/run/tools/env.sh
export MPLCONFIGDIR=/dev/shm/scxm138/mplconfig; mkdir -p "$MPLCONFIGDIR"
cd ~/run/vulkan-demo
SHM=/dev/shm/scxm138/probe38_$SLURM_JOB_ID; mkdir -p "$SHM"; LOG=~/run/logs/probe38_$SLURM_JOB_ID; mkdir -p "$LOG"
echo "=== node $(hostname) job $SLURM_JOB_ID probe38_anat commit=$(timeout 30 git rev-parse --short HEAD 2>/dev/null) $(date) ==="
SYS=""; for bus in $(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader | sed 's/^0000//' | tr 'A-Z' 'a-z'); do p=/sys/bus/pci/devices/$bus; [ -d "$p" ] && SYS="$SYS $p"; done
AFF=""; for p in $SYS; do n=$(cat $p/numa_node); [ "$n" -lt 0 ] && n=0; AFF="$AFF$(cat /sys/devices/system/node/node$n/cpulist);"; done; export V5_WORKER_AFFINITY="${AFF%;}"
export V5_FAST_SUBMIT=1 V5_CASCADE_FORCE=1 V5_BAND_VOXEL_DISPATCH=1 V5_BAND_SLOT_LANES=0 V5_FAKE_BAND_TEST=0 V5_PHASE_A_NO_WAIT=0 V5_PER_SIM_PIPELINE=0
export V5_WORKER_COUNT_AWARE=1 V5_GHOST_POOL_FACTOR=0.25 V5_SWITCH_INTERVAL_MS=0.2 V5_SPLIT_TRANSFER_QUEUES=1 V5_RECEIVER_CACHED=0 V5_LOOP_TRACE=0 V5_BENCH_PARITY=1
env | grep -E '^V5_' | sort | tr '\n' ' '; echo
for SIZE in 4m 16m 128m; do
  L=2d${SIZE}_k8_anat; TA=$(date +%s)
  python -c "import sys; sys.setswitchinterval(0.0002); import runpy; sys.argv=['x','--case','cases/lid_driven_cavity_2d_$SIZE/case.yaml','--weights','1,1,1,1,1,1,1,1','--device-map','0,1,2,3,4,5,6,7','--depth','2','--pool-safety','1.2','--max-steps','1500','--warmup','500','--anatomy','--no-seam-check']; runpy.run_path('experiment/v5/_run_v5_chain_bench.py', run_name='__main__')" > "$SHM/$L.log" 2>&1
  FPS=$(grep -a STEADY "$SHM/$L.log" | sed -E 's/.*= ([0-9.]+) fps.*/\1/' | tail -1); DR=$(grep -a "final" "$SHM/$L.log" | sed -E 's/.*drift=([-0-9]+).*/\1/' | tail -1)
  echo "RESULT label=$L steady_fps=${FPS:-nan} drift=${DR:-nan} wall=$(( $(date +%s) - TA ))s $(date +%T)"
  cp "$SHM/$L.log" "$LOG"/
done
cp -r "$SHM"/. "$LOG"/ && rm -rf "$SHM"; echo "=== job done $(date) ==="
