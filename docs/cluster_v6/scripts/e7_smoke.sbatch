#!/bin/bash
#SBATCH -p hp_5090
#SBATCH -A hp5090
#SBATCH -N 1
#SBATCH --gpus=8
#SBATCH --time=02:00:00
#SBATCH -J e7_smoke
#SBATCH -o /data/run01/scxm138/logs/e7_smoke_%j.out
# E7 step 1: smoke of v7 (tag v7-rc1) on one whole hp_5090 node, 8 x RTX 5090. The data is not for the paper.
#   Deployment ~/run/vulkan-demo-v7rc1 (deploy_v6.sh E30_DEPLOY_ALL=1, the v7-rc1 subset); particle files and
#   --obj-cache files built beforehand on the login node (gen_aligned_cases.sh), shared cache ~/run/e7_objcache,
#   staged once per job to node-local /dev/shm (s0_obj_cache, outside every timed run).
#   Host settings (e30_lib.sh, E30_SOLVER=v7): every inherited V5_ / V6_ / V7_ variable unset, only
#   V7_WORKER_AFFINITY exported (one NUMA cpulist per GPU in nvidia-smi order), GIL switch interval 0.2 ms
#   (run_chain_v6.py), two transfer queues and every other switch at the code default. The one other V7_*
#   variable set anywhere is V7_POOL_PEAKS=1 on the two E15 runs (S5), on their command lines only.
#   Every step and run has its own timeout; a hard failure stops the job (no retries): a chain run that the
#   parser classifies hard_failure, any of the 8 simultaneous K = 1 references, a fused_single_step check
#   that ends INVALID or without its JSON (a DIFFERENT verdict is the measured result and is recorded).
#   S0  configuration print (prelude), Vulkan layer inventory, bring-up (env, devices, the 22 SPIR-V files of v7,
#       1M K=1 and K=2), device order check (caps_v6.py: Vulkan discrete-first = nvidia-smi), obj cache check
#       and stage-in
#   S1  fused_single_step.py (B1): 2-D 1M (aligned n1000) developed 2000 steps at K=1, compared at K=2, K=8, K=1;
#       3-D n104 border 9 developed 2000 steps, compared at K=2 and K=1 (L / kernel sum / grad rho 0 ULP?, rho)
#   S2  K = 2 / 4 / 8 on 2-D 16M (cavity2d_n4000) and the 3-D cube (cavity3d_n416), 1500 steps each (warmup 500,
#       the in-loop defrag at step 1000), every invariant 0
#   S4  the largest weak case cavity2d_n5640_k8 (255.6M) at K = 8 with --obj-cache: chain build time and host
#       memory peak; then 8 simultaneous K = 1 references (cavity2d_n5640, 32M): host memory peak
#   S5  E15 pool peaks (V7_POOL_PEAKS=1): K = 8 2-D 16M and 3-D cube, 3000 steps, not used for timing
#   S3  (last, so that a failure coming from the layer cannot cost S4 / S5) Khronos validation layer: K = 8 2-D 16M
#       500 steps if the loader offers the layer in the runs' environment, otherwise recorded as absent
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
    NODE_CACHE=$OBJ_CACHE
else
    DEVICES=$(timeout 60 nvidia-smi -L | wc -l)
    [ "$DEVICES" -eq 8 ] || e30_fail "expected 8 GPUs, $DEVICES visible"
    NODE_CACHE=${SHM}_obj_cache         # next to $SHM, not in it (e30_sync copies $SHM after every step)
fi
ones() { local text; text=$(printf '1,%.0s' $(seq 1 "$1")); echo "${text%,}"; }
device_map() {      # K -> 0,1,...,K-1 (locally: modulo the visible devices)
    local slab text=""
    for slab in $(seq 0 $(($1 - 1))); do text="$text$((slab % DEVICES)),"; done
    echo "${text%,}"
}
E30_WRAPPER_ARGS="--obj-cache $NODE_CACHE"

# host memory of the node / job / run processes, 1 Hz, for the whole job; windows are cut per S4 run
$PY -u docs/cluster_v6/scripts/host_memory_sampler.py --out "$SHM/host_memory.csv" --interval 1 &
MEMPID=$!
trap 'kill "$MEMPID" 2>/dev/null' EXIT            # also on e30_fail: the sampler then writes its last peak file
memory_window() {   # LABEL SINCE UNTIL
    [ "$E30_DRY" = 1 ] && return 0
    timeout 120 $PY docs/cluster_v6/scripts/host_memory_sampler.py --summary "$SHM/host_memory.csv" \
        --since "$2" --until "$3" --label "$1" --json "$SHM/memory_windows.jsonl"
}
check_rows() {      # LABEL_PREFIX: stop the job if a results.jsonl row with this prefix is a hard failure
    [ "$E30_DRY" = 1 ] && return 0
    local failed
    failed=$($PY -c 'import json, sys
rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
print(" ".join(row["label"] for row in rows if row["label"].startswith(sys.argv[2]) and row["status"] == "hard_failure"))' \
        "$E30_RESULTS" "$1")
    [ -z "$failed" ] || e30_fail "hard failure in $failed"
}
FSS="$PY -u experiment/seam_audit/fused_single_step.py"
fss_step() {        # LABEL JSON -- ARGUMENTS: one fused_single_step check (the JSON is written only by a finished run)
    local label=$1 json=$2 verdict
    shift 2
    [ "${1:-}" = "--" ] && shift
    e30_step "$label" 600 --optional -- $FSS "$@" --out "$json"
    [ "$E30_DRY" = 1 ] && return 0
    verdict=$($PY -c 'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("verdict"))' \
        "$json" 2>/dev/null)
    echo "FSS $label verdict=${verdict:-none}"
    case "$verdict" in
        PASS|DIFFERENT) ;;          # DIFFERENT = a field is not bit-identical: the measured result, recorded
        *) e30_fail "$label (fused_single_step verdict ${verdict:-none: no JSON - crash, stall or timeout})" ;;
    esac
}

# ---- S0
e30_step s0_layers 120 --optional -- bash -c '
    echo "VK_LAYER_PATH=${VK_LAYER_PATH:-} VK_ADD_LAYER_PATH=${VK_ADD_LAYER_PATH:-} VK_LOADER_LAYERS_DISABLE=${VK_LOADER_LAYERS_DISABLE:-}"
    for d in /etc/vulkan/explicit_layer.d /usr/local/share/vulkan/explicit_layer.d /usr/share/vulkan/explicit_layer.d \
             /etc/vulkan/implicit_layer.d /usr/share/vulkan/implicit_layer.d "$HOME/.local/share/vulkan/explicit_layer.d" \
             /etc/xdg/vulkan/explicit_layer.d "$HOME/.config/vulkan/explicit_layer.d"; do
        [ -d "$d" ] && echo "$d: $(ls "$d" | tr "\n" " ")"
    done
    ldconfig -p 2>/dev/null | grep -i vklayer
    find /usr /opt -maxdepth 6 -iname "*VkLayer_khronos_validation*" 2>/dev/null | head
    echo "inventory done"'
e30_step s0_bringup 1200 -- $PY -u remote/bringup_check_v6.py $E30_SOLVER_ARGS --stage all --case $C1M \
    --log-dir "$SHM/bringup" --expected-devices "$DEVICES" --timeout 300
e30_step s0_caps 300 --optional -- $PY -u docs/cluster_v6/scripts/caps_v6.py --out "$SHM/caps.json"
STAGE_ARGUMENTS=()
[ "$NODE_CACHE" != "$OBJ_CACHE" ] && STAGE_ARGUMENTS=(--stage-to "$NODE_CACHE")
e30_step s0_obj_cache 900 --optional -- $PY -u docs/cluster_v6/scripts/obj_cache_build.py $E30_SOLVER_ARGS \
    --obj-cache "$OBJ_CACHE" --check --case $C16 --case $CUBE --case $W255 --case $W32 "${STAGE_ARGUMENTS[@]}"

# ---- S1: B1 single step (one process per configuration; the tool clears inherited V7_* itself, the affinity
#      is passed back explicitly; it does not change the numbers)
AFFINITY_ENV=()
[ -n "${V7_WORKER_AFFINITY:-}" ] && AFFINITY_ENV=(--env "V7_WORKER_AFFINITY=$V7_WORKER_AFFINITY")
STATES=${SHM}_fss_states            # developed states (150-250 MB each), next to $SHM, not in it
mkdir -p "$SHM/fss" "$STATES"
fss_step s1_fss_2d_k2 "$SHM/fss/k2_2d.json" -- --case $C1M --develop 2000 --save-state "$STATES/state_2d_N2000.npz" \
    --device-map "$(device_map 2)" "${AFFINITY_ENV[@]}"
fss_step s1_fss_2d_k8 "$SHM/fss/k8_2d.json" -- --case $C1M --state "$STATES/state_2d_N2000.npz" \
    --device-map "$(device_map 8)" "${AFFINITY_ENV[@]}"
fss_step s1_fss_2d_k1 "$SHM/fss/k1_2d.json" -- --case $C1M --state "$STATES/state_2d_N2000.npz" \
    --device-map 0 "${AFFINITY_ENV[@]}"
fss_step s1_fss_3d_k2 "$SHM/fss/k2_3d.json" -- --case $C3S --develop 2000 --save-state "$STATES/state_3d_N2000.npz" \
    --device-map "$(device_map 2)" "${AFFINITY_ENV[@]}"
fss_step s1_fss_3d_k1 "$SHM/fss/k1_3d.json" -- --case $C3S --state "$STATES/state_3d_N2000.npz" \
    --device-map 0 "${AFFINITY_ENV[@]}"
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

# ---- S4
since=$(date +%s.%N)
e30_run k8_w255 2400 -- --case $W255 --weights "$(ones 8)" --device-map "$(device_map 8)" $COMMON \
    --max-steps 300 --warmup 100
memory_window k8_w255 "$since" "$(date +%s.%N)"
since=$(date +%s.%N)
e30_run_k1_all k1_w32 1500 -- --case $W32 $COMMON --max-steps 300 --warmup 100
memory_window k1_w32_x8 "$since" "$(date +%s.%N)"
check_rows k1_w32

# ---- S5 (E15)
E30_PREFIX="env V7_POOL_PEAKS=1" e30_run k8_2d16m_poolpeaks 1200 -- --case $C16 --weights "$(ones 8)" \
    --device-map "$(device_map 8)" $COMMON --max-steps 3000 --warmup 500
E30_PREFIX="env V7_POOL_PEAKS=1" e30_run k8_cube_poolpeaks 2400 -- --case $CUBE --weights "$(ones 8)" \
    --device-map "$(device_map 8)" $COMMON --max-steps 3000 --warmup 500

# ---- S3: ask the loader, in the runs' environment without env.sh's layer switch-off, which layers it offers
LAYERS_TEXT=$(timeout 120 env -u VK_LOADER_LAYERS_DISABLE $PY -c 'import vulkan
def version(value):
    return f"{(value >> 22) & 0x7F}.{(value >> 12) & 0x3FF}.{value & 0xFFF}"
for layer in vulkan.vkEnumerateInstanceLayerProperties():
    print(f"layer {layer.layerName} spec={version(layer.specVersion)} implementation={layer.implementationVersion}")' 2>&1)
echo "$LAYERS_TEXT" | sed 's/^/S3 loader: /'
if echo "$LAYERS_TEXT" | grep -q '^layer VK_LAYER_KHRONOS_validation '; then
    E30_PREFIX="env -u VK_LOADER_LAYERS_DISABLE" e30_run k8_2d16m_validation 1200 -- --case $C16 --weights "$(ones 8)" \
        --device-map "$(device_map 8)" $COMMON --max-steps 500 --warmup 100 --validation
else
    echo "S3: the loader on $(hostname) offers no VK_LAYER_KHRONOS_validation -> the K = 8 validation run is not possible (recorded as absent)"
fi

kill "$MEMPID" 2>/dev/null; wait "$MEMPID" 2>/dev/null
trap - EXIT
memory_window job 0 "$(date +%s.%N)"
e30_finish
