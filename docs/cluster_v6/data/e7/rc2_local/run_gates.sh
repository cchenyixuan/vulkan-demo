#!/bin/bash
# E7 B2 after the review fixes (drain-first stop, counter pre-check default): bitwise gate rc1 vs rc2 at K = 2 / 3 / 4
# and for every V7_DEST_GUARD / V7_DEST_GUARD_PRECHECK setting; validation layer at K = 2 and K = 4.
# rc1 = pristine v7-rc1 export (rc2_work/rc1_tree), rc2 = worktree vulkan-demo-v7-wake.
W=C:/Users/cchen/AppData/Local/Temp/claude/C--Users-cchen-PycharmProjects-vulkan-demo/4f9257e4-cae2-4304-9c80-7006d6fe4db6/scratchpad/e7/rc2_work
WAKE=C:/Users/cchen/PycharmProjects/vulkan-demo-v7-wake
RC1=$W/rc1_tree
O=${1:-$W/ffix/gates}; mkdir -p $O/bitwise $O/validation
PY=C:/Users/cchen/PycharmProjects/vulkan-demo/.venv/Scripts/python.exe
HARNESS=$WAKE/experiment/seam_audit/canonical_dump.py
export PYTHONIOENCODING=utf-8 PYTHONPYCACHEPREFIX=$W/pycache
unset $(env | grep -oE '^V[0-9]_[A-Z0-9_]+' )
S=$WAKE/cases/lid_driven_cavity_2d_n250_xi0p001_eps0p0025/case.yaml
D3=$WAKE/cases/cavity3d_1m/case.yaml
M1=$WAKE/cases/lid_driven_cavity_2d_gen/case.yaml
POOL3D="--env V7_GHOST_POOL_FACTOR=0.5 --env V7_MIGRANT_POOL_FACTOR=0.02 --env V7_DEPARTED_FACE_FRACTION=0.64"
gpu_free() { nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | grep -i python && { echo "python on a GPU: abort"; exit 3; }; }
dump() {  # tag repo args...
  local tag=$1 repo=$2; shift 2
  gpu_free
  echo "=== $tag ($(date +%T))"
  VK_LOADER_LAYERS_DISABLE=VK_LAYER_KHRONOS_validation timeout 1200 $PY $HARNESS --solver v7 --repo $repo "$@" \
      --out $O/bitwise/$tag.npz > $O/bitwise/$tag.log 2>&1
  echo "rc=$? $(grep -E '^\[canonical_dump\] v7' $O/bitwise/$tag.log | tail -1 | cut -c1-160)"
  grep -E "dest guard:|DIED|\*\*\*" $O/bitwise/$tag.log | cut -c1-200
}
compare() {  # first second name
  echo "=== compare $3: $( (cd $WAKE && $PY -m experiment.seam_audit.canonical_dump --compare $O/bitwise/$1.npz $O/bitwise/$2.npz) > $O/bitwise/compare_$3.txt 2>&1; tail -1 $O/bitwise/compare_$3.txt)"
}
validation() {  # tag device-map weights case steps
  gpu_free
  local t0=$SECONDS
  (cd $WAKE && unset VK_LOADER_LAYERS_DISABLE && timeout 3600 $PY -u experiment/v7/_run_v7_chain_bench.py --validation \
      --case $4 --weights $3 --device-map $2 --sync-scheme per-direction --depth 2 --pool-safety 1.2 --max-steps $5 \
      --warmup 200) > $O/validation/$1.log 2>&1
  echo "=== validation $1: rc=$? $((SECONDS - t0)) s; validation messages $(grep -c '\[Vulkan ' $O/validation/$1.log)"
  grep -E "v7 switches|dest guard|final:|STEADY|VALIDATION FAILED|WARNING: validation|DIED|\*\*\*" $O/validation/$1.log | cut -c1-240
}
echo "### start $(date)"
dump rc1__k2_n250_200 $RC1 --case $S --device-map 0,1 --steps 200 --canonical-lists
dump rc2__k2_n250_200 $WAKE --case $S --device-map 0,1 --steps 200 --canonical-lists
dump rc2b__k2_n250_200 $WAKE --case $S --device-map 0,1 --steps 200 --canonical-lists
dump rc2zero__k2_n250_200 $WAKE --case $S --device-map 0,1 --steps 200 --canonical-lists --env V7_DEST_GUARD_PRECHECK=zero_wait
dump rc2none__k2_n250_200 $WAKE --case $S --device-map 0,1 --steps 200 --canonical-lists --env V7_DEST_GUARD_PRECHECK=none
dump rc2wait__k2_n250_200 $WAKE --case $S --device-map 0,1 --steps 200 --canonical-lists --env V7_DEST_GUARD=wait
dump rc1__k2_cross999 $RC1 --case $S --device-map 0,1 --steps 999 --canonical-lists --transport-extension
dump rc2__k2_cross999 $WAKE --case $S --device-map 0,1 --steps 999 --canonical-lists --transport-extension
dump rc1__k2_3d1m_200 $RC1 --case $D3 --device-map 0,1 --steps 200 --canonical-lists $POOL3D
dump rc2__k2_3d1m_200 $WAKE --case $D3 --device-map 0,1 --steps 200 --canonical-lists $POOL3D
dump rc1__k3_n250_300 $RC1 --case $S --device-map 0,1,0 --steps 300 --canonical-lists
dump rc2__k3_n250_300 $WAKE --case $S --device-map 0,1,0 --steps 300 --canonical-lists
dump rc1__k4_n250_300 $RC1 --case $S --device-map 0,1,0,1 --steps 300 --canonical-lists
dump rc2__k4_n250_300 $WAKE --case $S --device-map 0,1,0,1 --steps 300 --canonical-lists
dump rc1__k4_2d1m_300 $RC1 --case $M1 --device-map 0,1,0,1 --steps 300 --canonical-lists
dump rc2__k4_2d1m_300 $WAKE --case $M1 --device-map 0,1,0,1 --steps 300 --canonical-lists
echo "### dumps done $(date)"
compare rc1__k2_n250_200 rc2__k2_n250_200 rc1_rc2__k2_n250_200
compare rc2__k2_n250_200 rc2b__k2_n250_200 rc2_rc2b__k2_n250_200
compare rc1__k2_n250_200 rc2zero__k2_n250_200 rc1_rc2zero__k2_n250_200
compare rc1__k2_n250_200 rc2none__k2_n250_200 rc1_rc2none__k2_n250_200
compare rc1__k2_n250_200 rc2wait__k2_n250_200 rc1_rc2wait__k2_n250_200
compare rc1__k2_cross999 rc2__k2_cross999 rc1_rc2__k2_cross999
compare rc1__k2_3d1m_200 rc2__k2_3d1m_200 rc1_rc2__k2_3d1m_200
compare rc1__k3_n250_300 rc2__k3_n250_300 rc1_rc2__k3_n250_300
compare rc1__k4_n250_300 rc2__k4_n250_300 rc1_rc2__k4_n250_300
compare rc1__k4_2d1m_300 rc2__k4_2d1m_300 rc1_rc2__k4_2d1m_300
echo "### validation $(date)"
validation k2_2d1m 0,1 1,1 $M1 2000
validation k4_2d1m 0,1,0,1 1,1,1,1 $M1 2000
echo "### end $(date)"
