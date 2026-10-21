#!/bin/bash
# e8_gen_remote.sh — generate the E8 cases' particle files and .obj caches on the N56 login node (CPU only, not
# billed) with gen_aligned_cases.sh of the E8 deployment, into the shared case store (~/run/vulkan-demo-e8/cases ->
# ~/run/vulkan-demo-v7rc1/cases). The python environment lives in /dev/shm and dies with the last login session:
# run this inside tmux (e8_gen_launch.sh does) and leave the session until the summary line appears.
# Usage: bash e8_gen_remote.sh [CASE ...]   (default: the four cubes, the three weak boxes and the 56^3 stand-in,
#                                            the largest first so that the 64M case does not end the batch alone)
set -u
export E30_REPO=$HOME/run/vulkan-demo-e8
export E30_SOLVER=v7
export E7_OBJ_CACHE=$HOME/run/e7_objcache
export E7_GEN_PARALLEL=${E7_GEN_PARALLEL:-3}
export E7_GEN_LOG=$HOME/run/logs/e8_setup/$(date +%Y%m%d)
export PYTHONDONTWRITEBYTECODE=1
CASES=${*:-cavity3d_n392 cavity3d_n192_k8 cavity3d_n312 cavity3d_n192_k4 cavity3d_n244 cavity3d_n192_k2 cavity3d_n192 cavity3d_n48}
source "$HOME/run/tools/env.sh" || { echo "ABORT: env script failed"; exit 2; }
echo "=== e8 gen $(date -Is) on $(hostname): $CASES"
df -h "$HOME/run" | tail -1
bash "$E30_REPO/docs/cluster_v6/scripts/gen_aligned_cases.sh" $CASES
STATUS=$?
echo "=== e8 gen done $(date -Is) status $STATUS; cache $(du -sh "$E7_OBJ_CACHE" | cut -f1); store:"
for name in $CASES; do
    directory=$E30_REPO/cases/aligned/$name
    printf '%-20s objs %s lines domain %s\n' "$name" "$(ls "$directory"/*.obj 2>/dev/null | wc -l)" "$(wc -l < "$directory/domain.obj" 2>/dev/null)"
done
cd "$E30_REPO" && python docs/cluster_v6/scripts/obj_cache_build.py --solver v7 --obj-cache "$E7_OBJ_CACHE" --check \
    $(for name in $CASES cavity2d_n1000; do echo "--case cases/aligned/$name/case.yaml"; done) 2>&1 | tail -12
echo "=== E8 GEN STATUS $STATUS"
exit $STATUS
