#!/bin/bash
# weak-scaling families at 4M / 8M / 32M per slab (same rule as weak16: half_x = half*K + floor(K/2)).
#   weak4 : half=1000 -> 2000^2 = 4.0M per slab    weak8 : half=1414 -> 2828^2 = 8.0M    weak32: half=2828 -> 5656^2 = 32.0M
# Runs on the login node (CPU only). Touches cases/.<family>_ready when a family is complete.
source ~/run/tools/env.sh; cd ~/run/vulkan-demo
gen_family() {  # family half
  local F=$1 H=$2 M
  case $F in weak4) M=4;; weak8) M=8;; weak32) M=32;; esac
  for K in 1 2 4 8; do
    local HX=$(( H * K + K / 2 )) NAME=cavity_${F}_k${K}_$(( M * K ))m T=$(date +%s)
    if [ -f cases/$NAME/domain.obj ]; then echo "[$(date +%T)] $NAME exists, skip"; continue; fi
    echo "[$(date +%T)] $F K=$K half=$H half_x=$HX -> $NAME"
    python utils/geometry/_demo_cavity_case.py --half $H --half-x $HX --out cases/$NAME --no-preview 2>&1 | grep -E "domain|wrote|pool"
    echo "[$(date +%T)] done $NAME in $(( $(date +%s) - T )) s: $(du -sh cases/$NAME | cut -f1), fluid=$(grep -c "^v " cases/$NAME/domain.obj)"
  done
  touch cases/.${F}_ready; echo "[$(date +%T)] FAMILY_READY $F"
}
gen_family weak4 1000
gen_family weak8 1414
gen_family weak32 2828
echo GEN_DONE
