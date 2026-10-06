#!/usr/bin/env bash
# E36: one Re = 1000 cavity validation run with the release switches of experiment/validation/cavity_runner.py
# (RELEASE_ENVIRONMENT, --expect release) under the solver's prefix, from the v7-wall-bc worktree.
#   run_cavity.sh <v6|v7> <wall_bc or -> <case.yaml> <run dir> <device> <device uuid> [runner options...]
set -euo pipefail
solver=$1; wall_bc=$2; case_path=$3; run_dir=$4; device=$5; uuid=$6; shift 6
prefix=$(echo "$solver" | tr a-z A-Z)_
for key in $(env | grep -oE "^V[0-9]+_[A-Z0-9_]+" || true); do unset "$key"; done
export VK_LOADER_LAYERS_DISABLE=VK_LAYER_KHRONOS_validation
for pair in KEEP_DEPARTED=1 GHOST_LAYERS=2 LEAN_TRANSPORT=1 COMPACT_GHOST_LISTS=1 PACKED_REPLICAS=1 \
            BAND_SLOT_LANES=64 GHOST_POOL_FACTOR=0.29 MIGRANT_POOL_FACTOR=0.05 DEPARTED_FACE_FRACTION=0.8 \
            INIT_SEAM_CLAMP=1 WORKER_COUNT_AWARE=1 SPLIT_TRANSFER_QUEUES=1; do
    export "${prefix}${pair}"
done
solver_options=(--solver "$solver")
if [ "$solver" = v7 ]; then
    export V7_WALL_BC="$wall_bc"
    solver_options+=(--wall-bc "$wall_bc")
fi
cd "$(dirname "$0")/../../.."
exec ../vulkan-demo/.venv/Scripts/python.exe -m experiment.validation.cavity_runner "${solver_options[@]}" \
    --case "$case_path" --run-dir "$run_dir" --slabs 1 --device-map "$device" --expect release \
    --require-uuid "$uuid" "$@"
