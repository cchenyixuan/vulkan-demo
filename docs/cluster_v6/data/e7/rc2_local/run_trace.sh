#!/bin/bash
# E7 B2 after the review fixes: one step trace per solver (2-D 2M, K = 2) and interleaved untraced timing, 3 runs per
# solver in the order rc1 rc2 rc2 rc1 rc1 rc2, at 2-D 2M (transport hidden) and 2-D 1M (transport exposed, ~1150 fps).
# rc1 = pristine v7-rc1 export, rc2 = worktree vulkan-demo-v7-wake (defaults: relay / counter).
W=C:/Users/cchen/AppData/Local/Temp/claude/C--Users-cchen-PycharmProjects-vulkan-demo/4f9257e4-cae2-4304-9c80-7006d6fe4db6/scratchpad/e7/rc2_work
WAKE=C:/Users/cchen/PycharmProjects/vulkan-demo-v7-wake
RC1=$W/rc1_tree
O=${1:-$W/ffix/trace}; mkdir -p $O
PY=C:/Users/cchen/PycharmProjects/vulkan-demo/.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8 PYTHONPYCACHEPREFIX=$W/pycache
export VK_LOADER_LAYERS_DISABLE=VK_LAYER_KHRONOS_validation
unset $(env | grep -oE '^V[0-9]_[A-Z0-9_]+' )
FLAGS="--weights 1,1 --device-map 0,1 --sync-scheme per-direction --depth 2 --pool-safety 1.2"
CASE2M="--case $WAKE/cases/lid_driven_cavity_2d_2m/case.yaml --max-steps 3000 --warmup 1000"
CASE1M="--case $WAKE/cases/lid_driven_cavity_2d_gen/case.yaml --max-steps 6000 --warmup 1000"
run() {  # tag tree extra-args...
  local tag=$1 tree=$2; shift 2
  nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | grep -i python && { echo "python on a GPU: abort"; exit 3; }
  echo "=== $tag ($(date +%T)) GPU $(nvidia-smi --query-gpu=index,clocks.sm,temperature.gpu,power.draw --format=csv,noheader | tr '\n' ';')"
  local t0=$SECONDS
  (cd $tree && timeout 1800 $PY -u experiment/v7/_run_v7_chain_bench.py $FLAGS "$@") > $O/$tag.log 2>&1
  echo "rc=$? $((SECONDS - t0)) s"
  grep -E "STEADY|final:|VALIDATION FAILED|dest guard:|DIED|\*\*\*|step_trace\] wrote" $O/$tag.log | cut -c1-240
}
echo "### start $(date)"
run trace_rc1 $RC1 $CASE2M --step-trace $O/trace_rc1
run trace_rc2 $WAKE $CASE2M --step-trace $O/trace_rc2
for pair in a:rc1:rc2 b:rc2:rc1 c:rc1:rc2; do
  IFS=: read -r trial first second <<< "$pair"
  for solver in $first $second; do
    tree=$RC1; [ $solver = rc2 ] && tree=$WAKE
    run time2m_${solver}_$trial $tree $CASE2M
  done
done
for pair in a:rc1:rc2 b:rc2:rc1 c:rc1:rc2; do
  IFS=: read -r trial first second <<< "$pair"
  for solver in $first $second; do
    tree=$RC1; [ $solver = rc2 ] && tree=$WAKE
    run time1m_${solver}_$trial $tree $CASE1M
  done
done
echo "### end $(date)"
