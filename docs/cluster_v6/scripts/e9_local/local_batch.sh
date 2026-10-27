#!/bin/bash
# E9 local validation (user 2026-10-27, items 4-6), GPU 1 at 575 W, the E9 tree of the e8 worktree:
#   4  K = 1 8M + 64M, 300 steps (window 100-299), V7_DEFRAG_AUDIT=1 (pid order after the bootstrap defrag)
#   5  K = 2 8M on one card (--device-map 1,1): seam check + trace + pool peaks; dumps K1 / K2 at step 300 + field
#      comparison (phase-1 format); K = 2 with defrag cadence 100 (periodic defrags at 100 and 200 inside the run)
#   6  2-D cavity2d_n1000 (1M) and cavity2d_n2840 (8M) K = 1: prefix (tree default) vs atomic (variant yaml)
set -u
SPE=/c/Users/cchen/AppData/Local/Temp/claude/C--Users-cchen-PycharmProjects-vulkan-demo/fd68689d-e4b4-4b3f-8636-fbe3d1d866de/scratchpad/e9
REPO=/c/Users/cchen/PycharmProjects/vulkan-demo-e8
PY=/c/Users/cchen/PycharmProjects/vulkan-demo/.venv/Scripts/python.exe
until grep -q "K1 DONE" "$SPE/k1_runs.log" 2>/dev/null; do sleep 15; done
cd "$REPO" || exit 2
echo "v7 tree before: $(git rev-parse HEAD:experiment/v7) dirty: $(git status --short experiment/v7 | wc -l)"
C8=cases/aligned/cavity3d_n192/case.yaml; C64=cases/aligned/cavity3d_n392/case.yaml
C2D1=cases/aligned/cavity2d_n1000/case.yaml; C2D8=cases/aligned/cavity2d_n2840/case.yaml
run() { echo "##### $1 ($(date +%T))"; "$SPE/run_k.sh" "$@" 2>&1 | grep -E "STEADY|final:|seam|rc=|Traceback|Error|audit:|defrag_prefix\] own|defrag_event call" | cut -c1-200; }
# ---- item 4
run e9_8m_k1_audit  $C8  1 V7_DEFRAG_AUDIT=1
run e9_64m_k1_audit $C64 1 V7_DEFRAG_AUDIT=1
# ---- item 5
run e9_8m_k2_trace  $C8  2 V7_DEFRAG_AUDIT=1 V7_POOL_PEAKS=1
mkdir -p "$SPE/dumps" "$SPE/compare"
dump() {  # NAME K MAP
  echo "##### dump $1 ($(date +%T))"
  env VK_LOADER_LAYERS_DISABLE=VK_LAYER_KHRONOS_validation $PY -u docs/cluster_v6/scripts/e8_dump_state.py --obj-cache logs/e8_local/obj_cache -- \
    --version v7 --case $C8 --slabs $2 --device-map $3 --horizons 300 --depth 2 --pool-safety 1.2 --sync-scheme per-direction \
    --out-dir "$SPE/dumps" --run-name $1 --stall-timeout 300 > "$SPE/dumps/$1.log" 2>&1; echo "rc=$?"; grep -E "dumped|wrote|Traceback|Error|alive" "$SPE/dumps/$1.log" | tail -4 | cut -c1-200
}
dump dump_8m_K1 1 1
dump dump_8m_K2 2 1,1
echo "##### compare_8m ($(date +%T))"
PYTHONIOENCODING=utf-8 $PY -u docs/cluster_v6/scripts/e8_field_compare.py --reference "$SPE/dumps/dump_8m_K1_N300.npz" --test "$SPE/dumps/dump_8m_K2_N300.npz" \
  --md "$SPE/compare/compare_8m.md" --json "$SPE/compare/compare_8m.json" > "$SPE/compare/compare_8m.log" 2>&1; echo "rc=$?"; tail -5 "$SPE/compare/compare_8m.log" | cut -c1-200
EXTRA="--defrag-cadence 100 --defrag-log" run e9_8m_k2_cad100 $C8 2 V7_DEFRAG_AUDIT=1
# ---- item 6
$PY "$SPE/make_atomic_case.py" $C2D1 "$SPE/cases_atomic/cavity2d_n1000/case.yaml"
$PY "$SPE/make_atomic_case.py" $C2D8 "$SPE/cases_atomic/cavity2d_n2840/case.yaml"
STEPS=1300 WARMUP=300 run e9_2d1m_prefix $C2D1 1 V7_DEFRAG_AUDIT=1
STEPS=1300 WARMUP=300 run e9_2d1m_atomic "$SPE/cases_atomic/cavity2d_n1000/case.yaml" 1 V7_DEFRAG_AUDIT=1
STEPS=1300 WARMUP=300 run e9_2d8m_prefix $C2D8 1 V7_DEFRAG_AUDIT=1
STEPS=1300 WARMUP=300 run e9_2d8m_atomic "$SPE/cases_atomic/cavity2d_n2840/case.yaml" 1 V7_DEFRAG_AUDIT=1
echo "v7 tree after: $(git rev-parse HEAD:experiment/v7) dirty: $(git status --short experiment/v7 | wc -l)"
echo "##### LOCAL DONE $(date +%T)"
