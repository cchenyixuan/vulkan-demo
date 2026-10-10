#!/bin/bash
#SBATCH -p hp_5090
#SBATCH -A hp5090
#SBATCH -N 1
#SBATCH --gpus=8
#SBATCH --time=00:40:00
#SBATCH -J wake_repro
#SBATCH -o /data/run01/scxm138/logs/e7_wake_repro_%j.out
# E7 late-wake reproduction, standalone (docs/cluster_v6/scripts/e7_wake_repro.py, no solver code): a whole node, one
# card (GPU 0) with the process pinned to that card's NUMA cpulist as the solver's K = 1 processes are, every case
# (4000 iterations each); then a short second pass on GPU 4 (the other socket). Records the kernel's CONFIG_HZ, the
# driver and the clock source. Files: $E7_REPRO_DIR (default ~/run/e7_wake_repro) holds the script; results go to
# ~/run/logs/e7_wake_repro_<job>.
set -u
DIR=${E7_REPRO_DIR:-$HOME/run/e7_wake_repro}
OUT=$HOME/run/logs/e7_wake_repro_${SLURM_JOB_ID}
SHM=/dev/shm/scxm138/e7_wake_repro_${SLURM_JOB_ID}
mkdir -p "$OUT" "$SHM"
echo "=== host $(hostname) job ${SLURM_JOB_ID} $(date -Is)"
sha256sum "$DIR/e7_wake_repro.py" "$0" 2>/dev/null
echo "=== kernel"
uname -a
KCONFIG=/boot/config-$(uname -r)
if [ -r "$KCONFIG" ]; then
    grep -E '^CONFIG_HZ|^# CONFIG_HZ_|^CONFIG_NO_HZ|^CONFIG_HIGH_RES_TIMERS|^CONFIG_PREEMPT|^CONFIG_FUTEX' "$KCONFIG"
elif [ -r /proc/config.gz ]; then
    zcat /proc/config.gz | grep -E '^CONFIG_HZ|^CONFIG_NO_HZ|^CONFIG_HIGH_RES_TIMERS|^CONFIG_PREEMPT|^CONFIG_FUTEX'
else
    echo "kernel config not readable ($KCONFIG, /proc/config.gz)"
fi
echo "CLK_TCK $(getconf CLK_TCK)"
echo "clocksource $(cat /sys/devices/system/clocksource/clocksource0/current_clocksource 2>/dev/null)"
echo "cmdline $(cat /proc/cmdline)"
echo "timer_migration $(cat /proc/sys/kernel/timer_migration 2>/dev/null)"
echo "=== driver"
cat /proc/driver/nvidia/version
nvidia-smi --query-gpu=index,name,uuid,driver_version,pci.bus_id,persistence_mode,compute_mode --format=csv
nvidia-smi topo -m
lscpu | grep -E 'Model name|^Socket|^NUMA|Thread|^Core'
source ~/run/tools/env.sh >/dev/null 2>&1 || { echo "ABORT: env.sh"; exit 2; }
python -c "import sys, importlib.metadata as m; print(sys.version); print('vulkan package', m.version('vulkan'))"
cpus_of_gpu() {
    local bus
    bus=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader -i "$1" | tr 'A-F' 'a-f' | tr -d ' ')
    local node
    node=$(cat "/sys/bus/pci/devices/0000:${bus#*:}/numa_node")
    cat "/sys/devices/system/node/node${node}/cpulist"
}
CPUS0=$(cpus_of_gpu 0)
CPUS4=$(cpus_of_gpu 4)
echo "GPU 0 cpulist $CPUS0; GPU 4 cpulist $CPUS4"
echo "=== GPU 0, every case $(date -Is)"
taskset -c "$CPUS0" python -u "$DIR/e7_wake_repro.py" --device 0 --out "$SHM/gpu0" --iterations 4000 --cases all
echo "rc=$? $(date -Is)"
echo "=== GPU 4, short pass $(date -Is)"
taskset -c "$CPUS4" python -u "$DIR/e7_wake_repro.py" --device 4 --out "$SHM/gpu4" --iterations 2000 \
    --cases baseline,host_w4_infinite_load,gpu_w2_infinite_load,gpu_w4_infinite_load,gpu_w2_relay_condition_load,gpu_w2_timeout0.2_load
echo "rc=$? $(date -Is)"
for part in gpu0 gpu4; do [ -f "$SHM/$part/waits.csv" ] && gzip -f "$SHM/$part/waits.csv"; done
cp -r "$SHM"/* "$OUT"/
ls -la "$OUT"/*
rm -rf "$SHM"
echo "=== E7 wake repro done $(date -Is)"
