#!/usr/bin/env bash
# instrument_single.sh <label> <device> <steps> — sustained SINGLE-card load
# with cooldown gate + 2s clock/temp telemetry (thermal isolation test).
set -u
source ~/swq/env.sh
cd ~/swq/vulkan-demo
LABEL="$1"; DEV="$2"; STEPS="$3"
echo "[cooldown] waiting for all cards <= 48C ..."
for i in $(seq 60); do
    MAXT=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | sort -n | tail -1)
    if [ "$MAXT" -le 48 ]; then break; fi
    sleep 10
done
echo "[cooldown] start temp max=${MAXT}C after $((i*10))s"
rm -f "logs/campaign_a/${LABEL}_clocks.csv"
( while true; do
    echo "$(date +%s),$(nvidia-smi --query-gpu=index,temperature.gpu,clocks.sm,clocks_event_reasons.active --format=csv,noheader | tr '
' ';')"         >> "logs/campaign_a/${LABEL}_clocks.csv"
    sleep 2
  done ) &
SAMPLER=$!
python experiment/v5/_run_v5_single_bench.py     --case cases/cavity_weak_k1_2m/case.yaml --device "$DEV"     --max-steps "$STEPS" --warmup 2000 --bench-window $((STEPS-2000))     > "logs/campaign_a/${LABEL}.log" 2>&1
kill $SAMPLER 2>/dev/null
grep -aE "steps in|final" "logs/campaign_a/${LABEL}.log" | tail -2
awk -F, -v dev="$DEV" 'BEGIN{n=0} {split($0,a,";"); row=a[dev+1]; split(row,b,","); t=b[2]+0; c=b[3]+0; if(c>100){n++; if(t>maxt)maxt=t; if(minc==0||c<minc)minc=c; sumc+=c}} END{if(n)printf "[thermal] dev%s load-samples=%d temp_max=%dC sm_min=%dMHz sm_mean=%dMHz
", dev, n, maxt, minc, sumc/n}'     "logs/campaign_a/${LABEL}_clocks.csv"
