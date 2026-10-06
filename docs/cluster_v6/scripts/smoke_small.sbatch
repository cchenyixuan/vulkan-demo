#!/bin/bash
#SBATCH -p gpu_5090
#SBATCH -N 1
#SBATCH --gpus=2
#SBATCH --time=00:25:00
#SBATCH -J e30_smoke
#SBATCH -o /data/run01/scxm138/logs/e30_smoke_%j.out
# E30 small smoke: the v6 solver on N56 on 2 or 4 GPUs before the whole-node E30 jobs.
#   S0 bring-up and S1 capability query on the allocated GPUs, then short chain runs at the
#   release defaults (depth 2, equal weights, 1500 steps / warmup 500, seam check on).
#   E30_PLAN=k2 (--gpus=2): 1M K=1 + K=2 bring-up, 2-D 16M K=2, 3-D 8M K=2, optional E29 step trace.
#   E30_PLAN=kN, N=3..8 (--gpus=N): 2-D 16M K=N, 3-D 8M K=N (first v6 3-D interior slabs), 2-D 16M K=N --anatomy;
#   options E30_BRINGUP=all (1M K=1/K=2 bring-up as in k2), E30_SINGLE=1 (K=1 references of both cases on GPU 0).
#   Other clusters (N32-H A100 portability smoke): sbatch -p gpu -o ... with E30_REPO, E30_ENV_SCRIPT,
#   E30_SHM_BASE, E30_LOG_BASE exported (see e30_lib.sh).
#   Only V6_WORKER_AFFINITY is exported; GIL 0.2 ms in run_chain_v6.py; stop at the first hard failure.
E30_PLAN=${E30_PLAN:-k2}
E30_CONFIG_CASES="--case cases/lid_driven_cavity_2d_n1000/case.yaml --case cases/lid_driven_cavity_2d_16m/case.yaml --case cases/cavity3d_weak4_k2_8m_b4/case.yaml"
source "${E30_REPO:-$HOME/run/vulkan-demo-v6}/docs/cluster_v6/scripts/e30_lib.sh" || { echo "ABORT: no e30_lib.sh"; exit 2; }
e30_prelude "e30_smoke_$E30_PLAN"

C1M=cases/lid_driven_cavity_2d_n1000/case.yaml
C16=cases/lid_driven_cavity_2d_16m/case.yaml
C3D=cases/cavity3d_weak4_k2_8m_b4/case.yaml
COMMON="--sync-scheme per-direction --depth 2 --pool-safety 1.2"
if [ "$E30_LOCAL" = 1 ]; then
    EXPECTED_DEVICES=2                        # the local rig has two 5090s: K>2 as a virtual map 0,1,0,...
else
    EXPECTED_DEVICES=$(nvidia-smi -L | wc -l)
fi

for case_path in $C1M $C16 $C3D; do
    e30_step "s0_case_$(basename "$(dirname "$case_path")")" 120 -- \
        $PY -u remote/bringup_check_v6.py --stage case --case "$case_path" --log-dir "$SHM/bringup"
done

case "$E30_PLAN" in
k2)
    [ "$EXPECTED_DEVICES" -ge 2 ] || e30_fail "plan k2 needs 2 GPUs, $EXPECTED_DEVICES visible"
    e30_step s0_bringup 900 -- $PY -u remote/bringup_check_v6.py --stage all --case $C1M \
        --log-dir "$SHM/bringup" --expected-devices "$EXPECTED_DEVICES" --timeout 300
    e30_step s1_caps 300 --optional -- $PY -u docs/cluster_v6/scripts/caps_v6.py --out "$SHM/caps.json" --live-sample
    e30_run k2_2d16m 480 -- --case $C16 --weights 1,1 --device-map 0,1 $COMMON --max-steps 1500 --warmup 500
    e30_run k2_3d8m 600 -- --case $C3D --weights 1,1 --device-map 0,1 $COMMON --max-steps 1500 --warmup 500
    e30_run k2_1m_steptrace 300 --optional -- --case $C1M --weights 1,1 --device-map 0,1 $COMMON \
        --max-steps 600 --warmup 200 --step-trace "$SHM/step_trace_k2_1m"
    ;;
k[3-8])
    K=${E30_PLAN#k}
    if [ "$E30_LOCAL" != 1 ] && [ "$EXPECTED_DEVICES" -lt "$K" ]; then
        e30_fail "plan $E30_PLAN needs $K GPUs, $EXPECTED_DEVICES visible"
    fi
    WEIGHTS=$(printf '1,%.0s' $(seq 1 "$K")); WEIGHTS=${WEIGHTS%,}
    if [ "$E30_LOCAL" = 1 ]; then
        MAP=$(for slab in $(seq 0 $((K - 1))); do printf '%d,' $((slab % 2)); done); MAP=${MAP%,}
    else
        MAP=$(seq -s, 0 $((K - 1)))
    fi
    # E30_BRINGUP=all: full S0 (contexts + 1M K=1 / K=2 300 steps) instead of the env stage only
    e30_step s0_bringup_${E30_BRINGUP:-env} 900 -- $PY -u remote/bringup_check_v6.py --stage "${E30_BRINGUP:-env}" \
        --case $C1M --log-dir "$SHM/bringup" --expected-devices "$EXPECTED_DEVICES" --timeout 300
    e30_step s1_caps 300 --optional -- $PY -u docs/cluster_v6/scripts/caps_v6.py --out "$SHM/caps.json" --live-sample
    if [ "${E30_SINGLE:-0}" = 1 ]; then       # single-GPU magnitude references on GPU 0 (taskset to its NUMA cpulist)
        [ -n "${CPUL[0]:-}" ] && E30_PREFIX="taskset -c ${CPUL[0]}"
        e30_run k1_2d16m 600 -- --case $C16 --weights 1 --device-map 0 $COMMON --max-steps 1500 --warmup 500
        e30_run k1_3d8m 900 -- --case $C3D --weights 1 --device-map 0 $COMMON --max-steps 1000 --warmup 300
        E30_PREFIX=""
    fi
    e30_run k${K}_2d16m 480 -- --case $C16 --weights $WEIGHTS --device-map $MAP $COMMON --max-steps 1500 --warmup 500
    e30_run k${K}_3d8m 600 -- --case $C3D --weights $WEIGHTS --device-map $MAP $COMMON --max-steps 1500 --warmup 500
    e30_run k${K}_2d16m_anatomy 480 -- --case $C16 --weights $WEIGHTS --device-map $MAP $COMMON \
        --max-steps 1500 --warmup 500 --anatomy
    ;;
*)
    e30_fail "unknown E30_PLAN=$E30_PLAN"
    ;;
esac
e30_finish
