# e30_lib.sh — shared prelude and run helpers for the E30 v6 cluster jobs (sourced by bash).
#
# Cluster (default): checkout ~/run/vulkan-demo-v6 (archive deploy, see deploy_v6.sh), env
# ~/run/tools/env.sh, run logs in node-local /dev/shm during the job (JuiceFS appends stalled
# runs in September), copied to ~/run/logs/<name>_<job> after every step.
# Local pre-run: E30_LOCAL=1 E30_REPO=<checkout> E30_PY=<python> — no env.sh, no sysfs
# affinity (the Windows rig has none), logs under <checkout>/logs/e30_local.
# E30_DRY=1 prints every command instead of running it.
#
# Host settings (E30 "主机侧设置"): every inherited V5_* / V6_* variable is unset and only
# V6_WORKER_AFFINITY is exported (one NUMA-node cpulist per GPU, in nvidia-smi order = the
# v6 discrete-first Vulkan order; caps_v6.py checks that equality); the GIL switch interval
# (0.2 ms) is set inside run_chain_v6.py; two transfer queues are the code default.
#
# Solver (E39): E30_SOLVER=v6 (default) | v7 is the one switch. v7 runs experiment/v7 (the
# checkout must carry it: E30_SOLVER=v7 deploy_v6.sh SHA): every inherited V5_* / V6_* / V7_*
# variable is unset, V7_WORKER_AFFINITY is exported instead of V6_WORKER_AFFINITY (same cpulists,
# same discrete-first order), the configuration print, every run and its parse get --solver v7
# (E30_SOLVER_ARGS, also for the job scripts' direct calls: bring-up, calibration), and the
# environment listing is env | grep ^V7_. Unset or v6: everything as before.

e30_prelude() {   # NAME
    E30_NAME=$1
    E30_LOCAL=${E30_LOCAL:-0}
    E30_DRY=${E30_DRY:-0}
    E30_SOLVER=${E30_SOLVER:-v6}
    case "$E30_SOLVER" in
        v6) E30_SOLVER_ARGS=""; E30_SOLVER_PREFIX=V6_; E30_INHERITED_PATTERN='^V[56]_' ;;
        v7) E30_SOLVER_ARGS="--solver v7"; E30_SOLVER_PREFIX=V7_; E30_INHERITED_PATTERN='^V[567]_' ;;
        *) echo "ABORT: E30_SOLVER=$E30_SOLVER (expected v6 or v7)"; exit 2 ;;
    esac
    E30_JOB=${SLURM_JOB_ID:-local$(date +%Y%m%d_%H%M%S)}
    if [ "$E30_LOCAL" = 1 ]; then
        E30_REPO=${E30_REPO:?E30_REPO must name the checkout in local mode}
        PY=${E30_PY:-python}
        SHM=${E30_SHM_ROOT:-$E30_REPO/logs/e30_local}/${E30_NAME}_$E30_JOB
        LOG=$SHM
    else
        # N56 defaults; another cluster sets E30_REPO / E30_ENV_SCRIPT / E30_SHM_BASE / E30_LOG_BASE
        # (N32-H: ~/e30/vulkan-demo-v6, ~/e30/tools/env.sh, /dev/shm/scx6633, ~/e30/logs)
        E30_REPO=${E30_REPO:-$HOME/run/vulkan-demo-v6}
        E30_SHM_BASE=${E30_SHM_BASE:-/dev/shm/scxm138}
        source "${E30_ENV_SCRIPT:-$HOME/run/tools/env.sh}" || { echo "ABORT: env script failed"; exit 2; }
        export MPLCONFIGDIR=$E30_SHM_BASE/mplconfig; mkdir -p "$MPLCONFIGDIR"
        PY=python
        SHM=$E30_SHM_BASE/${E30_NAME}_$E30_JOB
        LOG=${E30_LOG_BASE:-$HOME/run/logs}/${E30_NAME}_$E30_JOB
    fi
    mkdir -p "$SHM" "$LOG" || { echo "ABORT: cannot create $SHM / $LOG"; exit 2; }
    cd "$E30_REPO" || { echo "ABORT: no checkout at $E30_REPO"; exit 2; }
    E30_T0=$(date +%s)
    E30_RESULTS=$SHM/results.jsonl
    echo "=== E30 $E30_NAME job $E30_JOB node $(hostname) $(date) local=$E30_LOCAL dry=$E30_DRY$([ "$E30_SOLVER" != v6 ] && echo " solver=$E30_SOLVER") ==="

    # provenance: deployed commit and manifests (archive deploy: there is no .git)
    if [ -f COMMIT ]; then echo "COMMIT: $(cat COMMIT)"; else echo "COMMIT: none (git $(git rev-parse HEAD 2>/dev/null || echo -))"; fi
    if [ -n "${E30_EXPECT_COMMIT:-}" ]; then
        grep -q "^$E30_EXPECT_COMMIT" COMMIT 2>/dev/null || { echo "ABORT: COMMIT does not start with $E30_EXPECT_COMMIT"; exit 2; }
    fi
    local manifest
    for manifest in MANIFEST.sha256 SCRIPTS.sha256; do
        if [ -f "$manifest" ]; then
            if sha256sum --quiet -c "$manifest"; then
                echo "manifest $manifest: OK ($(wc -l < "$manifest") files)"
            else
                echo "ABORT: manifest $manifest mismatch"; exit 2
            fi
        else
            echo "manifest $manifest: absent"
        fi
    done

    # environment: nothing V5_/V6_ (v7: also V7_) inherited from the submit shell; only the worker affinity
    local variable
    for variable in $(compgen -e | grep -E "$E30_INHERITED_PATTERN"); do unset "$variable"; done
    CPUL=()
    if [ "$E30_LOCAL" != 1 ]; then
        local bus device_path node affinity="" count=0
        for bus in $(timeout 60 nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader | sed 's/^0000//' | tr 'A-Z' 'a-z'); do
            device_path=/sys/bus/pci/devices/$bus
            [ -d "$device_path" ] || { echo "ABORT: no sysfs entry for GPU bus $bus"; exit 2; }
            node=$(cat "$device_path/numa_node" 2>/dev/null || echo 0); [ "$node" -lt 0 ] && node=0
            CPUL+=("$(cat /sys/devices/system/node/node$node/cpulist)")
            affinity="$affinity${CPUL[$count]};"
            count=$((count + 1))
        done
        [ "$count" -ge 1 ] || { echo "ABORT: no GPU visible"; exit 2; }
        export "${E30_SOLVER_PREFIX}WORKER_AFFINITY=${affinity%;}"
    fi
    echo "--- env | grep ^$E30_SOLVER_PREFIX"; env | grep "^$E30_SOLVER_PREFIX" | sort || echo "(none)"
    echo "--- host"
    echo "job cpus: $(taskset -cp $$ 2>/dev/null || echo n/a)"
    timeout 60 nvidia-smi --query-gpu=index,name,pci.bus_id,uuid,driver_version,memory.total,power.limit --format=csv,noheader
    echo "--- nvidia-smi topo -m"; timeout 60 nvidia-smi topo -m 2>&1 | head -16
    local index
    for index in "${!CPUL[@]}"; do echo "gpu$index numa cpulist ${CPUL[$index]}"; done
    echo "--- effective configuration ($E30_SOLVER resolvers, this environment)"
    timeout 300 $PY -u docs/cluster_v6/scripts/run_chain_v6.py $E30_SOLVER_ARGS --config-only ${E30_CONFIG_CASES:-}         || { echo "ABORT: config print failed"; exit 2; }
    # telemetry every second, with memory.used for the per-run VRAM peak
    nvidia-smi --query-gpu=timestamp,index,pci.bus_id,memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu,clocks.current.sm \
        --format=csv -l 1 > "$SHM/telemetry.csv" 2>/dev/null &
    TELEPID=$!
}

e30_sync() {      # copy the node-local logs to ~/run/logs (cluster only): new or changed files, bounded in time
    [ "$E30_LOCAL" = 1 ] || timeout 300 cp -ru "$SHM"/. "$LOG"/ 2>/dev/null
    return 0
}

e30_finish() {
    if [ -n "${TELEPID:-}" ]; then kill "$TELEPID" 2>/dev/null; wait "$TELEPID" 2>/dev/null; TELEPID=""; fi
    e30_sync
    local runs=0
    [ -f "$E30_RESULTS" ] && runs=$(grep -c . "$E30_RESULTS")
    echo
    echo "=== E30 $E30_NAME done: $runs runs recorded, wall $(( $(date +%s) - E30_T0 ))s, logs $LOG ==="
}

e30_fail() {      # LABEL
    echo
    echo "*** E30 HARD FAILURE at $1 — stopping the job (stop at the failing step; no retries on the cluster) ***"
    e30_finish
    exit 3
}

e30_step() {      # LABEL TIMEOUT_S [--optional] -- COMMAND ...   (a non-bench step: bring-up, capability query)
    local label=$1 limit=$2 optional=0
    shift 2
    if [ "${1:-}" = "--optional" ]; then optional=1; shift; fi
    [ "${1:-}" = "--" ] && shift
    echo; echo "=== STEP $label ($(date +%T)) timeout ${limit}s$([ "$optional" = 1 ] && echo ' (optional)') ==="
    if [ "$E30_DRY" = 1 ]; then echo "DRY: $*"; return 0; fi
    timeout -k 30 "$limit" "$@" > "$SHM/$label.log" 2>&1
    local return_code=$?
    tail -n 40 "$SHM/$label.log"
    echo "STEP label=$label rc=$return_code"
    e30_sync
    if [ "$return_code" -ne 0 ]; then
        if [ "$optional" = 1 ]; then echo "(optional step $label failed — recorded, job continues)"; return 0; fi
        e30_fail "$label"
    fi
}

e30_run_k1_all() {   # LABEL TIMEOUT_S -- CHAIN BENCH ARGUMENTS (no --weights / --device-map)
    # K=1 on every GPU of the job at the same time (the simultaneous single-GPU references of the
    # efficiency rule), each process pinned to its GPU's NUMA cpulist; failures are recorded per GPU.
    local label=$1 limit=$2 gpu pids=""
    shift 2
    [ "${1:-}" = "--" ] && shift
    local count=${#CPUL[@]}
    [ "$E30_LOCAL" = 1 ] && count=2
    for gpu in $(seq 0 $((count - 1))); do
        (
            E30_PREFIX=""
            [ -n "${CPUL[$gpu]:-}" ] && E30_PREFIX="taskset -c ${CPUL[$gpu]}"
            e30_run "${label}_g$gpu" "$limit" --optional -- "$@" --weights 1 --device-map "$gpu"
        ) &
        pids="$pids $!"
    done
    wait $pids
}

e30_run() {       # LABEL TIMEOUT_S [--optional] -- CHAIN BENCH ARGUMENTS ...   (E30_PREFIX, e.g. "taskset -c LIST")
    local label=$1 limit=$2 optional=0
    shift 2
    if [ "${1:-}" = "--optional" ]; then optional=1; shift; fi
    [ "${1:-}" = "--" ] && shift
    local log="$SHM/$label.log"
    echo; echo "=== RUN $label ($(date +%T)) timeout ${limit}s ${E30_PREFIX:-}$([ "$optional" = 1 ] && echo ' (optional)') ==="
    echo "args: $*"
    if [ "$E30_DRY" = 1 ]; then echo "DRY: ${E30_PREFIX:-} $PY -u docs/cluster_v6/scripts/run_chain_v6.py ${E30_SOLVER_ARGS:+$E30_SOLVER_ARGS }-- $*"; return 0; fi
    local start end return_code verdict
    start=$(date +%s.%N)
    timeout -k 30 "$limit" ${E30_PREFIX:-} $PY -u docs/cluster_v6/scripts/run_chain_v6.py ${E30_SOLVER_ARGS:-} ${E30_WRAPPER_ARGS:-} \
        -- "$@" > "$log" 2>&1
    return_code=$?
    end=$(date +%s.%N)
    $PY docs/cluster_v6/scripts/parse_run_v6.py ${E30_SOLVER_ARGS:-} --log "$log" --label "$label" --rc "$return_code" \
        --start "$start" --end "$end" --telemetry "$SHM/telemetry.csv" --results "$E30_RESULTS" --node "$(hostname)"
    verdict=$?
    grep -aE "Traceback|Error|VALIDATION FAILED|STALL|DIED|STALE" "$log" | head -5
    e30_sync
    if [ "${E30_TIMEOUT_SKIP:-0}" = 1 ] && { [ "$return_code" = 124 ] || [ "$return_code" = 137 ]; }; then
        echo "(TIMEOUT: $label exceeded ${limit}s — recorded, skipped, job continues)"
        return 0
    fi
    if [ "$verdict" -ne 0 ]; then
        if [ "$optional" = 1 ]; then echo "(optional run $label failed — recorded, job continues)"; return 0; fi
        e30_fail "$label"
    fi
}
