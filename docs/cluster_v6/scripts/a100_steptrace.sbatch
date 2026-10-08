#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus=4
#SBATCH --time=00:12:00
#SBATCH -J e30_a100_st
#SBATCH -o /home/bingxing2/home/scx6633/e30/logs/e30_a100_st_%j.out
# E30 A100, plan item 3 (step trace) after the fix a6bce2b: driver 535 offers only VK_EXT_calibrated_timestamps,
#   so the k4_2d16m_steptrace run of a100_smoke.sbatch (deployed c411b11) could not create its devices. Same run
#   as there: 2-D 16M K=4, equal weights, 2000 steps, warmup 500, depth 2, E29 default detail; then the E29 model
#   summary of the trace (time domain, map residuals, causal checks, coverage, t_chain hops); then the OPAQUE_FD
#   probe again (its read-back line was broken in 1550154).
export E30_REPO=${E30_REPO:-$HOME/e30/vulkan-demo-v6}
export E30_ENV_SCRIPT=${E30_ENV_SCRIPT:-$HOME/e30/tools/env.sh}
export E30_SHM_BASE=${E30_SHM_BASE:-/dev/shm/$USER}
export E30_LOG_BASE=${E30_LOG_BASE:-$HOME/e30/logs}
E30_TIMEOUT_SKIP=1
E30_CONFIG_CASES="--case cases/lid_driven_cavity_2d_16m/case.yaml"
source "$E30_REPO/docs/cluster_v6/scripts/e30_lib.sh" || { echo "ABORT: no e30_lib.sh"; exit 2; }
e30_prelude e30_a100_st

# Kunpeng-920 + driver 535 SIGBUS workaround, as in a100_smoke.sbatch (environment only).
if [ "$E30_LOCAL" != 1 ]; then
    E30_SHIM=${E30_SHIM:-$HOME/e30/tools/devmem_safe_copy.so}
    [ -s "$E30_SHIM" ] || e30_fail "shim $E30_SHIM missing"
    export LD_PRELOAD=$E30_SHIM
    echo "LD_PRELOAD=$LD_PRELOAD sha256=$(sha256sum "$E30_SHIM" | cut -d' ' -f1)"
fi

E30_WRAPPER_ARGS="--obj-cache $SHM/obj_cache"
C16=cases/lid_driven_cavity_2d_16m/case.yaml
COMMON="--sync-scheme per-direction --depth 2 --pool-safety 1.2"
if [ "$E30_LOCAL" = 1 ]; then MAP4=0,1,0,1; else MAP4=0,1,2,3; fi
TRACE="$SHM/step_trace_2d16m_k4"

e30_run k4_2d16m_steptrace 360 -- --case $C16 --weights 1,1,1,1 --device-map $MAP4 $COMMON \
    --max-steps 2000 --warmup 500 --step-trace "$TRACE"
if [ -s "$TRACE/run_meta.json" ]; then
    e30_step k4_2d16m_steptrace_model 300 --optional -- bash -c \
        "$PY -m experiment.$E30_SOLVER.analysis.step_trace_model --run '$TRACE' > '$TRACE/model.json'"
fi
# plan item 4 again: in job 1550154 the OPAQUE_FD import on GPU 1 went through, then the probe's own read-back
# line failed (python-vulkan's vkMapMemory already returns a buffer); fixed probe, same check.
e30_step p2p_probe 120 --optional -- $PY -u docs/cluster_v6/scripts/p2p_probe_linux.py
e30_finish
