#!/bin/bash
#SBATCH -p hp_5090
#SBATCH -A hp5090
#SBATCH -N 1
#SBATCH --gpus=8
#SBATCH --time=02:30:00
#SBATCH -J cav_k8
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --exclude=wqd10nbm13g5,wqd10nah09g1
#SBATCH -o /data/run01/scxm138/logs/cavity_k8_%j.out
# Paper 6.2 supplement (user 2026-10-20 night, plan a): the 1000^2 adami Re = 1000 cavity continued at K = 8 (equal
# weights, the chain cut by the particle distribution of the starting state) from the K = 1 run's checkpoint at
# t = 89.3025 (c0011907000.npz) to t = 100, sampled / snapshotted on the same global steps as the K = 1 and K = 2 runs
# (experiment/validation/cavity_runner.py --start-from; the source's steady.json gives the same stop step 13,334,000).
# Deployment ~/run/vulkan-demo-v7rc3 (solver tag v7-rc3 = 64eecf0, experiment/v7 tree f33a46ba..., runner patched
# 2026-10-20). Any whole 8 x 575 W node but wqd10nbm13g5 / wqd10nah09g1; the job stops before any GPU work unless the
# deployment, the checkpoint and the power caps check out. Steps: (1) a 27,000-step smoke at K = 8 from the checkpoint
# (one sample), (2) the run. Outputs under /dev/shm during the job, archived to ~/run/logs/cavity_k8_<job id>.
# Submit: sbatch --export=ALL,EXPECT_COMMIT=<rc3 deployment commit prefix> docs/validation/cluster/cavity_k8.sbatch
set -u
REPO=$HOME/run/vulkan-demo-v7rc3
INPUT=$HOME/run/cavity_k8_input/n1000_k1_float32_xi0p001_eps0p0025_adami
CHECKPOINT=$INPUT/checkpoints/c0011907000.npz
CHECKPOINT_SHA=db77ef290e1011b268c5175beb5e70f7c69ee4e057f642fc1531b6d7da483cf9
EXPECT_COMMIT=${EXPECT_COMMIT:?"the deployment commit prefix is missing: sbatch --export=ALL,EXPECT_COMMIT=..."}
EXPECT_TREE=f33a46ba021064e92a64d3aba49897c607234850
CASE=cases/lid_driven_cavity_2d_n1000_xi0p001_eps0p0025_adami/case.yaml
RUN_NAME=n1000_k8_float32_xi0p001_eps0p0025_adami
SHM=/dev/shm/scxm138/cavity_k8_$SLURM_JOB_ID
OUT=$HOME/run/logs/cavity_k8_$SLURM_JOB_ID
T0=$(date +%s)
echo "=== cavity K = 8 continuation, job $SLURM_JOB_ID on $(hostname), $(date -Is), CPUs $(nproc), $(free -g | awk '/Mem/ {print $2}') GiB"
source "$HOME/run/tools/env.sh" || { echo "ABORT: env script failed"; exit 2; }
mkdir -p "$SHM" "$OUT" || { echo "ABORT: cannot create $SHM / $OUT"; exit 2; }
cd "$REPO" || { echo "ABORT: no $REPO"; exit 2; }
archive() { cp -r "$SHM/." "$OUT/" 2>/dev/null; echo "archived to $OUT ($(du -sh "$OUT" | cut -f1))"; }

echo "--- deployment"; cat COMMIT
grep -q "^$EXPECT_COMMIT" COMMIT || { echo "ABORT: COMMIT is not $EXPECT_COMMIT"; exit 2; }
sha256sum --quiet -c MANIFEST.sha256 && echo "MANIFEST OK ($(wc -l < MANIFEST.sha256) files)" || { echo "ABORT: MANIFEST.sha256 mismatch"; exit 2; }
TREE=$(python docs/cluster_v6/scripts/tree_hash.py experiment/v7 | tail -1)
[ "$TREE" = "$EXPECT_TREE" ] || { echo "ABORT: experiment/v7 tree $TREE is not v7-rc3's $EXPECT_TREE"; exit 2; }
echo "experiment/v7 tree $TREE = v7-rc3"
echo "--- GPUs"
nvidia-smi -L || { echo "ABORT: nvidia-smi failed"; exit 2; }
nvidia-smi --query-gpu=index,pci.bus_id,name,driver_version,power.limit,memory.total --format=csv
COUNT=$(nvidia-smi -L | grep -c '^GPU ')
[ "$COUNT" -eq 8 ] || { echo "ABORT: $COUNT GPUs listed, 8 expected"; exit 2; }
CAPS=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | tr -d ' ')
AT_575=$(echo "$CAPS" | awk '$1 + 0 == 575 {n++} END {print n + 0}')
[ "$AT_575" -eq 8 ] || { echo "ABORT: power caps are not 8 x 575 W: $(echo "$CAPS" | tr '\n' ' ')"; exit 2; }
echo "power caps: all 8 GPUs at 575 W on $(hostname)"
echo "--- input"
[ -f "$CHECKPOINT" ] || { echo "ABORT: no checkpoint $CHECKPOINT"; exit 2; }
echo "$CHECKPOINT_SHA  $CHECKPOINT" | sha256sum --quiet -c - && echo "checkpoint sha256 OK" || { echo "ABORT: checkpoint sha256 differs"; exit 2; }
[ -f "$INPUT/meta.json" ] && [ -f "$INPUT/steady.json" ] || { echo "ABORT: source meta.json / steady.json missing"; exit 2; }
cat "$INPUT/steady.json" | tr -d '\n' ; echo
# the runner's release switches (--expect release: the V7_* set must be exactly this), nothing else V7_ / V6_ / V5_
for name in $(env | grep -o '^V[567]_[A-Za-z0-9_]*'); do unset "$name"; done
export V7_KEEP_DEPARTED=1 V7_GHOST_LAYERS=2 V7_LEAN_TRANSPORT=1 V7_COMPACT_GHOST_LISTS=1 V7_PACKED_REPLICAS=1 \
       V7_BAND_SLOT_LANES=64 V7_GHOST_POOL_FACTOR=0.29 V7_MIGRANT_POOL_FACTOR=0.05 V7_DEPARTED_FACE_FRACTION=0.8 \
       V7_INIT_SEAM_CLAMP=1 V7_WORKER_COUNT_AWARE=1 V7_SPLIT_TRANSFER_QUEUES=1
export VK_LOADER_LAYERS_DISABLE=VK_LAYER_KHRONOS_validation PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
RUNNER="python -m experiment.validation.cavity_runner --solver v7 --case $CASE --expect release --slabs 8 --device-map 0,1,2,3,4,5,6,7 --start-from $CHECKPOINT --t-end 100 --t-max 200 --average-span 20 --pool-safety 1.2"
echo "--- partition the runner will build (CPU)"
python docs/validation/cluster/cavity_k8_partition_check.py --case "$CASE" --checkpoint "$CHECKPOINT" --slabs 8 --json "$SHM/partition_check.json" 2>&1 | grep -v '^\[' | tr -d '\n' | tr -s ' '; echo

echo "=== 1. smoke: 27,000 steps at K = 8 from the checkpoint ($(date +%T))"
$RUNNER --run-dir "$SHM/smoke_$RUN_NAME" --max-steps 27000 > "$SHM/smoke.log" 2>&1; RC=$?
grep -E "^\[cavity|restart state loaded|restarted from|partition_v7\] chain" "$SHM/smoke.log" | cut -c1-220
cat "$SHM/smoke_$RUN_NAME/segments.jsonl" 2>/dev/null
if [ $RC -ne 0 ] || ! grep -q '"status": "complete"' "$SHM/smoke_$RUN_NAME/segments.jsonl" 2>/dev/null; then
    echo "ABORT: smoke rc=$RC"; tail -30 "$SHM/smoke.log"; archive; exit 2
fi
echo "smoke OK: $(tail -1 "$SHM/smoke_$RUN_NAME/samples.jsonl" | cut -c1-160)"

echo "=== 2. the run: t = 89.3025 -> 100 ($(date +%T))"
( while sleep 60; do cp "$SHM/$RUN_NAME/heartbeat.json" "$OUT/heartbeat.json" 2>/dev/null; cp "$SHM/$RUN_NAME/run.log" "$OUT/run.log" 2>/dev/null; done ) &
WATCH=$!
$RUNNER --run-dir "$SHM/$RUN_NAME" > "$SHM/run_stdout.log" 2>&1; RC=$?
kill $WATCH 2>/dev/null
echo "=== run rc=$RC ($(date +%T)), $(( $(date +%s) - T0 )) s since the job started"
grep -E "^\[cavity" "$SHM/run_stdout.log" | tail -8 | cut -c1-220
cat "$SHM/$RUN_NAME/segments.jsonl" 2>/dev/null
cat "$SHM/$RUN_NAME/result.json" 2>/dev/null
echo "samples $(wc -l < "$SHM/$RUN_NAME/samples.jsonl" 2>/dev/null), snapshots $(ls "$SHM/$RUN_NAME/snapshots" 2>/dev/null | wc -l), checkpoints $(ls "$SHM/$RUN_NAME/checkpoints" 2>/dev/null | grep -c npz)"
grep -E "dest guard:" "$SHM/run_stdout.log" | cut -c1-200
archive
sacct -j "$SLURM_JOB_ID" -o JobID,JobName,State,Start,End,Elapsed,NodeList -P
exit $RC
