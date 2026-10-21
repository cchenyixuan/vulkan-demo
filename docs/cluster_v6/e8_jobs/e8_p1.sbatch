#!/bin/bash
#SBATCH -p hp_5090
#SBATCH -A hp5090
#SBATCH -N 1
#SBATCH --gpus=2
#SBATCH --time=00:30:00
#SBATCH -J e8_p1
#SBATCH -o /data/run01/scxm138/logs/e8_p1_%j.out
# E8 phase 1 (user, 2026-10-22): 3-D cubic-cavity smoke on two cards of one node, 300 steps per item, in one job
# of at most 30 minutes. Solver = tag e8-3d-p1 (v7-rc3 + the partitioner floor 3 own columns; experiment/v7 tree
# E7_EXPECTED_TREE), harness = e7_lib.sh of this deployment (~/run/vulkan-demo-e8; cases/ -> the shared case store
# of ~/run/vulkan-demo-v7rc1, the E8 cases generated there by gen_aligned_cases.sh, caches in ~/run/e7_objcache).
#   1  8M cube (cavity3d_n192, 200^3) K = 1 and K = 2: every built-in check, particle count conserved; the K = 2 run
#      is step-traced (phase timing: T_B, the three hops, B_link) with the seam check; then the field comparison at
#      step 300 (dump_state K = 1 / K = 2 keyed by the global particle id, e8_field_compare: max |dv|/U and
#      |drho|/rho0 by distance to the seam).
#   2  the narrow-slab stand-in cavity3d_n48 (56^3 = 175,616 particles, 6 fluid columns per card at K = 2 = the
#      8M K = 8 condition): K = 1 and K = 2 (seam check), the same field comparison.
#   3  the 2 x 1 x 1 weak box cavity3d_n192_k2 (15.68M) at K = 2 (seam check).
#   4  the 64M cube cavity3d_n392 (400^3) at K = 2 (seam check): the generator's output at the largest size and the
#      per-card VRAM against the phase-0 table (telemetry peak + the simulator's buffer prints).
#   5  (from 1) the K = 1 ns per particle-step and the K = 2 trace.
# Items run in the user's order; a hard failure of a chain run stops the job (e30_fail, no retries); the dumps and
# comparisons are optional steps (recorded, the job goes on). Outputs under /dev/shm during the job, synced to
# ~/run/logs/e7_p1_<job> (e30_lib.sh naming: E30_NAME = e7_$E7_JOB).
export E30_EXPECT_COMMIT=${E30_EXPECT_COMMIT:?"the deployment commit prefix: sbatch --export=ALL,E30_EXPECT_COMMIT=..."}
export E7_SOLVER_TAG=${E7_SOLVER_TAG:-e8p1}
export E7_EXPECTED_TREE=${E7_EXPECTED_TREE:?"experiment/v7 tree hash of the deployment (git rev-parse <commit>:experiment/v7)"}
export E7_EXPECTED_DEVICES=2
export E7_ARMS=C
source "${E30_REPO:-$HOME/run/vulkan-demo-e8}/docs/cluster_v6/scripts/e7_lib.sh" || { echo "ABORT: no e7_lib.sh"; exit 2; }
e7_job_begin p1 S cavity2d_n1000 cavity3d_n192 cavity3d_n48 cavity3d_n192_k2 cavity3d_n392
C8M=cases/aligned/cavity3d_n192/case.yaml
C48=cases/aligned/cavity3d_n48/case.yaml
CW2=cases/aligned/cavity3d_n192_k2/case.yaml
C64=cases/aligned/cavity3d_n392/case.yaml
PIN0=""; PIN01=""
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

# ---- 1. 8M cube: K = 1 (timed, pinned), K = 2 step-traced with the seam check, the dumps and the comparison
e7_index 8m_K1 run phase1 cavity3d_n192 1 0 - - 8m
E30_PREFIX="$PIN0" e30_run 8m_K1 900 -- --case $C8M --weights 1 --device-map 0 $COMMON --max-steps 300 --warmup 100 --no-seam-check
e7_index 8m_K2_trace run phase1 cavity3d_n192 2 0 - - 8m
e30_run 8m_K2_trace 900 -- --case $C8M --weights 1,1 --device-map 0,1 $COMMON --max-steps 300 --warmup 100 \
    --step-trace "$SHM/traces/8m_K2"
dump dump_8m_K1 $C8M 1 0
dump dump_8m_K2 $C8M 2 0,1
compare compare_8m dump_8m_K1 dump_8m_K2

# ---- 2. the 56^3 narrow-slab stand-in: K = 1, K = 2 (6 fluid columns per card), dumps, comparison
e7_index s48_K1 run phase1 cavity3d_n48 1 0 - - s48
E30_PREFIX="$PIN0" e30_run s48_K1 600 -- --case $C48 --weights 1 --device-map 0 $COMMON --max-steps 300 --warmup 100 --no-seam-check
e7_index s48_K2 run phase1 cavity3d_n48 2 0 - - s48
e30_run s48_K2 600 -- --case $C48 --weights 1,1 --device-map 0,1 $COMMON --max-steps 300 --warmup 100
dump dump_s48_K1 $C48 1 0
dump dump_s48_K2 $C48 2 0,1
compare compare_s48 dump_s48_K1 dump_s48_K2

# ---- 3. the 2 x 1 x 1 weak box at K = 2
e7_index weak2_K2 run phase1 cavity3d_n192_k2 2 0 - - weak2
e30_run weak2_K2 900 -- --case $CW2 --weights 1,1 --device-map 0,1 $COMMON --max-steps 300 --warmup 100

# ---- 4. the 64M cube at K = 2 (generator output at the largest size, per-card VRAM)
e7_index 64m_K2 run phase1 cavity3d_n392 2 0 - - 64m
e30_run 64m_K2 1200 -- --case $C64 --weights 1,1 --device-map 0,1 $COMMON --max-steps 300 --warmup 100

[ "$E30_DRY" = 1 ] || { echo "--- clocks / temperatures (telemetry.csv) per run are in results.jsonl (telemetry_loop)"; \
    grep -h "RESULT label=" "$SHM"/*.log 2>/dev/null | cut -c1-400; }
e7_job_end
