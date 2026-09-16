#!/bin/bash
# 3-D case families for the cluster (Task B.2). Generator: utils/geometry/_demo_cavity_case_3d.py (--half-x added
# 2026-09-16; chunked, bit-identical to the old cubic output when half_x = half).
#   weak4  : 4M/GPU, half=79  -> 159^3 per slab; stretched half_x = 79*K + K//2   (K=1,2,4,8; K=8 = 32M strong geometry)
#   weak8  : 8M/GPU, half=100 -> 201^3 per slab; stretched half_x = 100*K + K//2  (K=1,2,4,8; K=8 = 64M strong geometry)
#   cubes  : cavity3d_cube_32m (half 159 -> 319^3) and cavity3d_cube_64m (half 200 -> 401^3): 1-D-slab-in-cube controls
# BORDER (wall shell layers): 4 = one support radius (h = 4 dx) is the default since 2026-09-17 (verified
# equivalent to the old 9 = 2*hdx+1 on 8M K=1 and the stretched K=2 case; -13% particles, +7-9% fps).
# Pass BORDER=9 to build the thick-wall variants (suffix _b9).
source ~/run/tools/env.sh 2>/dev/null; cd "$(dirname "$0")/.." 2>/dev/null || cd ~/run/vulkan-demo
BORDER=${BORDER:-4}; SUF=""; [ "$BORDER" != 4 ] && SUF="_b$BORDER"
gen() {  # name half half_x
  local NAME=$1$SUF; local T=$(date +%s)
  if [ -f cases/$NAME/domain.obj ]; then echo "[$(date +%T)] $NAME exists, skip"; return; fi
  echo "[$(date +%T)] $NAME half=$2 half_x=$3 border=$BORDER"
  python utils/geometry/_demo_cavity_case_3d.py --half $2 --half-x $3 --border $BORDER --out cases/$NAME --no-preview 2>&1 | grep -E "lattice|domain|pool_size"
  echo "[$(date +%T)] done $NAME in $(( $(date +%s) - T )) s: $(du -sh cases/$NAME | cut -f1)"
}
for K in 1 2 4 8; do gen cavity3d_weak4_k${K}_$(( 4 * K ))m 79 $(( 79 * K + K / 2 )); done
touch cases/.cavity3d_weak4${SUF}_ready
for K in 1 2 4 8; do gen cavity3d_weak8_k${K}_$(( 8 * K ))m 100 $(( 100 * K + K / 2 )); done
touch cases/.cavity3d_weak8${SUF}_ready
gen cavity3d_cube_32m 159 159
gen cavity3d_cube_64m 200 200
touch cases/.cavity3d_cubes${SUF}_ready
echo GEN3D_DONE
