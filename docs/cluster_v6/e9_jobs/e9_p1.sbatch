#!/bin/bash
#SBATCH -p hp_5090
#SBATCH -A hp5090
#SBATCH -N 1
#SBATCH --gpus=2
#SBATCH --time=00:40:00
#SBATCH -J e9_p1
#SBATCH --exclude=wqd10nbm13g5,wqd10nah09g1,wqd10nbj04g3,wqd10nbj04g5
#SBATCH -o /data/run01/scxm138/logs/e9_p1_%j.out
# E9 short job (user, 2026-10-27, item 8): the prefix defrag order (tag v7-rc4: experiment/v7's defrag.comp prefix branch
# fed by the host scan of utils/defrag_prefix.py before every defrag; kernels unchanged) on two cards of one node, 300
# steps per item, before the formal E9 batch. Solver = tag v7-rc4 (E7_EXPECTED_TREE), harness = e7_lib.sh of this
# deployment (~/run/vulkan-demo-e9; cases/ -> the shared case store of ~/run/vulkan-demo-v7rc1, caches in ~/run/e7_objcache).
#   1  8M cube (cavity3d_n192, 200^3) K = 1: every built-in check, particle count conserved, V7_DEFRAG_AUDIT=1 (the pid
#      order after the bootstrap defrag: 0 block descents expected) - the K = 1 ns per particle-step against E8 phase 1's
#      6.28 ns (atomic order) and the local 5.79 ns (prefix order, 575 W).
#   2  8M cube K = 2 (cards 0, 1): step-traced with the seam check and the pool peaks, V7_DEFRAG_AUDIT=1 on both slabs;
#      then the field comparison at step 300 (dump_state K = 1 / K = 2 keyed by the global particle id, e8_field_compare).
#   3  the 64M cube cavity3d_n392 (400^3) at K = 2 (seam check, pool peaks): the largest per-card slab of the matrix.
# A hard failure of a chain run stops the job (e30_fail, no retries); the dumps and comparison are optional steps.
# Outputs under /dev/shm during the job, synced to ~/run/logs/e7_p1_<job> (e30_lib.sh naming: E30_NAME = e7_$E7_JOB).
export E30_EXPECT_COMMIT=${E30_EXPECT_COMMIT:?"the deployment commit prefix: sbatch --export=ALL,E30_EXPECT_COMMIT=..."}
export E7_SOLVER_TAG=${E7_SOLVER_TAG:-e9p1}
export E7_EXPECTED_TREE=${E7_EXPECTED_TREE:?"experiment/v7 tree hash of the deployment (git rev-parse <commit>:experiment/v7)"}
export E7_EXPECTED_DEVICES=2
export E7_ARMS=C
export E7_RAW=${E7_RAW:-1}
export E7_TRACE_DETAIL=${E7_TRACE_DETAIL:-full}
export E7_UNIFORM_POWER_CAP=${E7_UNIFORM_POWER_CAP:-1}
export E7_POWER_CAP_W=${E7_POWER_CAP_W:-575}
source "${E30_REPO:-$HOME/run/vulkan-demo-e9}/docs/cluster_v6/scripts/e7_lib.sh" || { echo "ABORT: no e7_lib.sh"; exit 2; }
e7_job_begin p1 S cavity2d_n1000 cavity3d_n192 cavity3d_n392
C8M=cases/aligned/cavity3d_n192/case.yaml
C64=cases/aligned/cavity3d_n392/case.yaml
PIN0=""
[ -n "${CPUL[0]:-}" ] && PIN0="taskset -c ${CPUL[0]}"
mkdir -p "$SHM/dumps" "$SHM/compare"
dump() {            # LABEL CASE K MAP: the seam-audit dump at step 300 (optional step; obj cache installed)
    e30_step "$1" 900 --optional -- $PY -u docs/cluster_v6/scripts/e8_dump_state.py --obj-cache "$NODE_CACHE" -- \
        --version v7 --case "$2" --slabs "$3" --device-map "$4" --horizons 300 --depth 2 --pool-safety 1.2 \
        --sync-scheme per-direction --out-dir "$SHM/dumps" --run-name "$1" --stall-timeout 300
}
compare() {         # LABEL REF_LABEL TEST_LABEL
    e30_step "$1" 600 --optional -- $PY -u docs/cluster_v6/scripts/e8_field_compare.py \
        --reference "$SHM/dumps/${2}_N300.npz" --test "$SHM/dumps/${3}_N300.npz" \
        --md "$SHM/compare/$1.md" --json "$SHM/compare/$1.json"
}

# ---- 1. 8M cube K = 1 (timed, pinned, pid-order audit)
e7_index 8m_K1 run e9p1 cavity3d_n192 1 0 - - 8m
E30_PREFIX="$PIN0 env V7_DEFRAG_AUDIT=1" e30_run 8m_K1 900 -- --case $C8M --weights 1 --device-map 0 $COMMON --max-steps 300 --warmup 100 --no-seam-check

# ---- 2. 8M cube K = 2: step-traced with the seam check, pool peaks, audit; dumps and the field comparison
e7_index 8m_K2_trace run e9p1 cavity3d_n192 2 0 - - 8m
E30_PREFIX="env V7_POOL_PEAKS=1 V7_DEFRAG_AUDIT=1" e30_run 8m_K2_trace 900 -- --case $C8M --weights 1,1 --device-map 0,1 $COMMON \
    --max-steps 300 --warmup 100 --step-trace "$SHM/traces/8m_K2"
dump dump_8m_K1 $C8M 1 0
dump dump_8m_K2 $C8M 2 0,1
compare compare_8m dump_8m_K1 dump_8m_K2

# ---- 3. the 64M cube at K = 2 (seam check, pool peaks)
e7_index 64m_K2 run e9p1 cavity3d_n392 2 0 - - 64m
E30_PREFIX="env V7_POOL_PEAKS=1" e30_run 64m_K2 1200 -- --case $C64 --weights 1,1 --device-map 0,1 $COMMON --max-steps 300 --warmup 100

[ "$E30_DRY" = 1 ] || { echo "--- defrag order / audit lines of every run"; grep -h "defrag_order=\|defrag_prefix\] audit" "$SHM"/*.log 2>/dev/null | cut -c1-200; \
    echo "--- RESULT lines"; grep -h "RESULT label=" "$SHM"/*.log 2>/dev/null | cut -c1-400; }
e7_job_end
