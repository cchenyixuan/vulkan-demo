#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus=4
#SBATCH --time=00:30:00
#SBATCH -J e30_a100
#SBATCH -o /home/bingxing2/home/scx6633/e30/logs/e30_a100_%j.out
# E30 A100 portability smoke on N32-H (4 x A100-PCIE-40GB, Kunpeng-920 aarch64, driver 535): the v6 solver at the
#   E32 HEAD (phase-A no-wait default, nearest-column cuts, --weights auto / --weights-file) plus the E30 fix
#   a6bce2b (--step-trace falls back to VK_EXT_calibrated_timestamps; driver 535 has no KHR), release defaults.
#   Scaling at equal weights, 2-D 16M (2000 steps, warmup 500) and 3-D 8M (1000 steps, warmup 500): K=1 on all
#   4 GPUs at once, K=2 on 0,1, K=4 on 0-3. Then 2-D 16M K=4 with pilot-calibrated weights (--weights auto
#   --calibrate-only --weights-file, then --weights-file), one --step-trace run, last an OPAQUE_FD probe.
#   Every run in a timeout (2-D 6 min, 3-D 10 min); a timeout is recorded and skipped. Only V6_WORKER_AFFINITY
#   is exported; GIL 0.2 ms in run_chain_v6.py; parsed .obj arrays cached on /dev/shm (--obj-cache).
export E30_REPO=${E30_REPO:-$HOME/e30/vulkan-demo-v6}
export E30_ENV_SCRIPT=${E30_ENV_SCRIPT:-$HOME/e30/tools/env.sh}
export E30_SHM_BASE=${E30_SHM_BASE:-/dev/shm/$USER}
export E30_LOG_BASE=${E30_LOG_BASE:-$HOME/e30/logs}
E30_TIMEOUT_SKIP=1
E30_CONFIG_CASES="--case cases/lid_driven_cavity_2d_16m/case.yaml --case cases/cavity3d_weak4_k2_8m_b4/case.yaml"
source "$E30_REPO/docs/cluster_v6/scripts/e30_lib.sh" || { echo "ABORT: no e30_lib.sh"; exit 2; }
e30_prelude e30_a100

# Kunpeng-920 + driver 535: inside vkCmdBindDescriptorSets the NVIDIA driver copies into a Device-type
# /dev/nvidiaN (BAR1) mapping with glibc's memcpy, whose source-aligned 16-byte stores fault there: SIGBUS,
# si_code BUS_ADRALN (jobs 1550014, 1550133, 1550139). devmem_safe_copy.so sends only the copies that touch
# /dev/nvidiaN mappings through naturally aligned accesses (job 1550151: 1M K=1 and K=2 pass, 4-8 such calls
# per run). Environment only; experiment/v6 is unchanged.
if [ "$E30_LOCAL" != 1 ]; then
    E30_SHIM=${E30_SHIM:-$HOME/e30/tools/devmem_safe_copy.so}
    [ -s "$E30_SHIM" ] || e30_fail "shim $E30_SHIM missing"
    export LD_PRELOAD=$E30_SHIM
    echo "LD_PRELOAD=$LD_PRELOAD sha256=$(sha256sum "$E30_SHIM" | cut -d' ' -f1)"
fi

E30_WRAPPER_ARGS="--obj-cache $SHM/obj_cache"
C16=cases/lid_driven_cavity_2d_16m/case.yaml
C3D=cases/cavity3d_weak4_k2_8m_b4/case.yaml
COMMON="--sync-scheme per-direction --depth 2 --pool-safety 1.2"
WINDOW_2D="--max-steps 2000 --warmup 500"
WINDOW_3D="--max-steps 1000 --warmup 500"
if [ "$E30_LOCAL" = 1 ]; then
    EXPECTED_DEVICES=2; MAP2=0,1; MAP4=0,1,0,1
else
    EXPECTED_DEVICES=$(nvidia-smi -L | wc -l); MAP2=0,1; MAP4=0,1,2,3
fi

e30_step s0_env 600 -- $PY -u remote/bringup_check_v6.py --stage env --log-dir "$SHM/bringup" \
    --expected-devices "$EXPECTED_DEVICES"
e30_step s1_caps 300 --optional -- $PY -u docs/cluster_v6/scripts/caps_v6.py --out "$SHM/caps.json" --live-sample

# scaling, equal weights
e30_run_k1_all k1_2d16m 360 -- --case $C16 $COMMON $WINDOW_2D
e30_run k2_2d16m 360 -- --case $C16 --weights 1,1 --device-map $MAP2 $COMMON $WINDOW_2D
e30_run k4_2d16m 360 -- --case $C16 --weights 1,1,1,1 --device-map $MAP4 $COMMON $WINDOW_2D
e30_run_k1_all k1_3d8m 600 -- --case $C3D $COMMON $WINDOW_3D
e30_run k2_3d8m 600 -- --case $C3D --weights 1,1 --device-map $MAP2 $COMMON $WINDOW_3D
e30_run k4_3d8m 600 -- --case $C3D --weights 1,1,1,1 --device-map $MAP4 $COMMON $WINDOW_3D

# 2-D 16M K=4: pilot weight calibration (E32), then the timed run on the calibrated weights
e30_step k4_2d16m_calibrate 360 --optional -- $PY -u docs/cluster_v6/scripts/run_chain_v6.py $E30_WRAPPER_ARGS -- \
    --case $C16 --weights auto --calibrate-only --weights-file "$SHM/weights_2d16m_k4.json" --device-map $MAP4 $COMMON
if [ -s "$SHM/weights_2d16m_k4.json" ]; then
    e30_run k4_2d16m_weighted 360 --optional -- --case $C16 --weights-file "$SHM/weights_2d16m_k4.json" \
        --device-map $MAP4 $COMMON $WINDOW_2D
fi
# 2-D 16M K=4 step trace (E29 instrument, default detail): first Linux / A100 run of the clock path
e30_run k4_2d16m_steptrace 360 --optional -- --case $C16 --weights 1,1,1,1 --device-map $MAP4 $COMMON $WINDOW_2D \
    --step-trace "$SHM/step_trace_2d16m_k4"
# optional: can Vulkan share device memory between two of these GPUs on Linux (OPAQUE_FD)?
e30_step p2p_probe 120 --optional -- $PY -u docs/cluster_v6/scripts/p2p_probe_linux.py
e30_finish
