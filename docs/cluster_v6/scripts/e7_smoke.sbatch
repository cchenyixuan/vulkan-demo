#!/bin/bash
#SBATCH -p hp_5090
#SBATCH -A hp5090
#SBATCH -N 1
#SBATCH --gpus=8
#SBATCH --time=01:30:00
#SBATCH -J e7_smoke
#SBATCH -o /data/run01/scxm138/logs/e7_smoke_%j.out
# E7 step 1: smoke of v7 (tag v7-rc1) on one whole hp_5090 node, 8 x RTX 5090. The data is not for the paper.
#   Deployment ~/run/vulkan-demo-v7rc1 (deploy_v6.sh E30_DEPLOY_ALL=1, the v7-rc1 subset); particle files and
#   --obj-cache files built beforehand on the login node (gen_aligned_cases.sh), cache in ~/run/e7_objcache.
#   Host settings (e30_lib.sh, E30_SOLVER=v7): every inherited V5_ / V6_ / V7_ variable unset, only
#   V7_WORKER_AFFINITY exported (one NUMA cpulist per GPU in nvidia-smi order), GIL switch interval 0.2 ms
#   (run_chain_v6.py), two transfer queues and every other switch at the code default. The one other V7_*
#   variable set anywhere is V7_POOL_PEAKS=1 on the two E15 runs (S5), on their command lines only.
#   Every step and run has its own timeout; a hard failure of a run stops the job (no retries).
#   S0  configuration print (prelude), Vulkan layer inventory, bring-up (env, devices, the 22 SPIR-V files of v7,
#       1M K=1 and K=2), device order check (caps_v6.py: Vulkan discrete-first = nvidia-smi), obj cache check
#   S1  fused_single_step.py (B1): 2-D 1M (aligned n1000) developed 2000 steps at K=1, compared at K=2, K=8, K=1;
#       3-D n104 border 9 developed 2000 steps, compared at K=2 and K=1 (L / kernel sum / grad rho 0 ULP?, rho)
#   S2  K = 2 / 4 / 8 on 2-D 16M (cavity2d_n4000) and the 3-D cube (cavity3d_n416), 1500 steps each (warmup 500,
#       the in-loop defrag at step 1000), every invariant 0
#   S3  Khronos validation layer: K = 8 2-D 16M 500 steps if the node has the layer, otherwise recorded as absent
#   S4  the largest weak case cavity2d_n5640_k8 (255.6M) at K = 8 with --obj-cache: chain build time and host
#       memory peak; then 8 simultaneous K = 1 references (cavity2d_n5640, 32M): host memory peak
#   S5  E15 pool peaks (V7_POOL_PEAKS=1): K = 8 2-D 16M and 3-D cube, 3000 steps, not used for timing
export E30_SOLVER=v7
export E30_REPO=${E30_REPO:-$HOME/run/vulkan-demo-v7rc1}
OBJ_CACHE=${E7_OBJ_CACHE:-$HOME/run/e7_objcache}
# the cases (E7_* overrides only for the local pre-run of this script with small stand-ins)
C1M=${E7_C1M:-cases/aligned/cavity2d_n1000/case.yaml}
C3S=${E7_C3S:-cases/aligned/cavity3d_n104_b9/case.yaml}
C16=${E7_C16:-cases/aligned/cavity2d_n4000/case.yaml}
CUBE=${E7_CUBE:-cases/aligned/cavity3d_n416/case.yaml}
W255=${E7_W255:-cases/aligned/cavity2d_n5640_k8/case.yaml}
W32=${E7_W32:-cases/aligned/cavity2d_n5640/case.yaml}
E30_CONFIG_CASES="--case $C1M --case $C3S --case $C16 --case $CUBE --case $W255 --case $W32"
source "$E30_REPO/docs/cluster_v6/scripts/e30_lib.sh" || { echo "ABORT: no e30_lib.sh"; exit 2; }
e30_prelude e7_smoke
echo "SCRIPTS_COMMIT: $(cat SCRIPTS_COMMIT 2>/dev/null || echo none)"     # the harness commit the scripts came from
COMMON="--sync-scheme per-direction --depth 2 --pool-safety 1.2"
if [ "$E30_LOCAL" = 1 ]; then
    DEVICES=${E7_LOCAL_DEVICES:-2}      # local pre-run (two 5090s): K > 2 as a virtual map 0,1,0,1,...
else
    DEVICES=$(nvidia-smi -L | wc -l)
    [ "$DEVICES" -eq 8 ] || e30_fail "expected 8 GPUs, $DEVICES visible"
fi
ones() { local text; text=$(printf '1,%.0s' $(seq 1 "$1")); echo "${text%,}"; }
device_map() {      # K -> 0,1,...,K-1 (locally: modulo the visible devices)
    local slab text=""
    for slab in $(seq 0 $(($1 - 1))); do text="$text$((slab % DEVICES)),"; done
    echo "${text%,}"
}
E30_WRAPPER_ARGS="--obj-cache $OBJ_CACHE"

# host memory of the node / job / run processes, 1 Hz, for the whole job; windows are cut per S4 run
$PY -u docs/cluster_v6/scripts/host_memory_sampler.py --out "$SHM/host_memory.csv" --interval 1 &
MEMPID=$!
memory_window() {   # LABEL SINCE UNTIL
    [ "$E30_DRY" = 1 ] && return 0
    $PY docs/cluster_v6/scripts/host_memory_sampler.py --summary "$SHM/host_memory.csv" --since "$2" --until "$3" \
        --label "$1" --json "$SHM/memory_windows.jsonl"
}

# ---- S0
e30_step s0_layers 120 --optional -- bash -c '
    echo "VK_LAYER_PATH=${VK_LAYER_PATH:-} VK_ADD_LAYER_PATH=${VK_ADD_LAYER_PATH:-} VK_LOADER_LAYERS_DISABLE=${VK_LOADER_LAYERS_DISABLE:-}"
    for d in /etc/vulkan/explicit_layer.d /usr/local/share/vulkan/explicit_layer.d /usr/share/vulkan/explicit_layer.d \
             /etc/vulkan/implicit_layer.d /usr/share/vulkan/implicit_layer.d "$HOME/.local/share/vulkan/explicit_layer.d"; do
        [ -d "$d" ] && echo "$d: $(ls "$d" | tr "\n" " ")"
    done
    ldconfig -p 2>/dev/null | grep -i vklayer
    find /usr /opt -maxdepth 6 -iname "*VkLayer_khronos_validation*" 2>/dev/null | head
    echo "inventory done"'
KHRONOS_LAYER=$(grep -l -i '"VK_LAYER_KHRONOS_validation"' /etc/vulkan/explicit_layer.d/*.json \
    /usr/local/share/vulkan/explicit_layer.d/*.json /usr/share/vulkan/explicit_layer.d/*.json \
    "$HOME/.local/share/vulkan/explicit_layer.d"/*.json 2>/dev/null | head -1)
echo "Khronos validation layer manifest: ${KHRONOS_LAYER:-none}"
e30_step s0_bringup 1200 -- $PY -u remote/bringup_check_v6.py $E30_SOLVER_ARGS --stage all --case $C1M \
    --log-dir "$SHM/bringup" --expected-devices "$DEVICES" --timeout 300
e30_step s0_caps 300 --optional -- $PY -u docs/cluster_v6/scripts/caps_v6.py --out "$SHM/caps.json"
e30_step s0_obj_cache 300 --optional -- $PY -u docs/cluster_v6/scripts/obj_cache_build.py $E30_SOLVER_ARGS \
    --obj-cache "$OBJ_CACHE" --check --case $C16 --case $CUBE --case $W255 --case $W32

# ---- S1: B1 single step (one process per configuration; the tool clears inherited V7_* itself, the affinity
#      is passed back explicitly; it does not change the numbers)
FSS="$PY -u experiment/seam_audit/fused_single_step.py"
AFFINITY_ENV=()
[ -n "${V7_WORKER_AFFINITY:-}" ] && AFFINITY_ENV=(--env "V7_WORKER_AFFINITY=$V7_WORKER_AFFINITY")
# developed states (150-250 MB each) live next to $SHM, not in it (e30_sync copies $SHM after every step)
STATES=${SHM}_fss_states
mkdir -p "$SHM/fss" "$STATES"
e30_step s1_fss_2d_k2 900 --optional -- $FSS --case $C1M --develop 2000 --save-state "$STATES/state_2d_N2000.npz" \
    --device-map "$(device_map 2)" "${AFFINITY_ENV[@]}" --out "$SHM/fss/k2_2d.json"
e30_step s1_fss_2d_k8 900 --optional -- $FSS --case $C1M --state "$STATES/state_2d_N2000.npz" \
    --device-map "$(device_map 8)" "${AFFINITY_ENV[@]}" --out "$SHM/fss/k8_2d.json"
e30_step s1_fss_2d_k1 900 --optional -- $FSS --case $C1M --state "$STATES/state_2d_N2000.npz" \
    --device-map 0 "${AFFINITY_ENV[@]}" --out "$SHM/fss/k1_2d.json"
e30_step s1_fss_3d_k2 900 --optional -- $FSS --case $C3S --develop 2000 --save-state "$STATES/state_3d_N2000.npz" \
    --device-map "$(device_map 2)" "${AFFINITY_ENV[@]}" --out "$SHM/fss/k2_3d.json"
e30_step s1_fss_3d_k1 900 --optional -- $FSS --case $C3S --state "$STATES/state_3d_N2000.npz" \
    --device-map 0 "${AFFINITY_ENV[@]}" --out "$SHM/fss/k1_3d.json"
[ "$E30_DRY" = 1 ] || rm -f "$STATES"/state_*.npz

# ---- S2
for K in 2 4 8; do
    e30_run k${K}_2d16m 900 -- --case $C16 --weights "$(ones $K)" --device-map "$(device_map $K)" $COMMON \
        --max-steps 1500 --warmup 500
done
for K in 2 4 8; do
    e30_run k${K}_cube 2400 -- --case $CUBE --weights "$(ones $K)" --device-map "$(device_map $K)" $COMMON \
        --max-steps 1500 --warmup 500
done

# ---- S3
if [ -n "$KHRONOS_LAYER" ]; then
    E30_PREFIX="env -u VK_LOADER_LAYERS_DISABLE" e30_run k8_2d16m_validation 1200 -- --case $C16 --weights "$(ones 8)" \
        --device-map "$(device_map 8)" $COMMON --max-steps 500 --warmup 100 --validation
else
    echo "S3: no Khronos validation layer manifest on $(hostname) (searched /etc, /usr/local/share, /usr/share, ~/.local);"
    echo "S3: the K = 8 validation run is not possible here -> recorded as absent"
fi

# ---- S4
since=$(date +%s.%N)
e30_run k8_w255 2400 -- --case $W255 --weights "$(ones 8)" --device-map "$(device_map 8)" $COMMON \
    --max-steps 300 --warmup 100
memory_window k8_w255 "$since" "$(date +%s.%N)"
since=$(date +%s.%N)
e30_run_k1_all k1_w32 1500 -- --case $W32 $COMMON --max-steps 300 --warmup 100
memory_window k1_w32_x8 "$since" "$(date +%s.%N)"

# ---- S5 (E15)
E30_PREFIX="env V7_POOL_PEAKS=1" e30_run k8_2d16m_poolpeaks 1200 -- --case $C16 --weights "$(ones 8)" \
    --device-map "$(device_map 8)" $COMMON --max-steps 3000 --warmup 500
E30_PREFIX="env V7_POOL_PEAKS=1" e30_run k8_cube_poolpeaks 2400 -- --case $CUBE --weights "$(ones 8)" \
    --device-map "$(device_map 8)" $COMMON --max-steps 3000 --warmup 500

kill "$MEMPID" 2>/dev/null; wait "$MEMPID" 2>/dev/null
memory_window job 0 "$(date +%s.%N)"
e30_finish
