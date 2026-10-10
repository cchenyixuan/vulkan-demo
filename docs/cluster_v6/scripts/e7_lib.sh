# e7_lib.sh — E7 full campaign on N56 (hp_5090, whole nodes, v7-rc1; the validation job: rc2 against v7-rc1):
# the protocol the generated job scripts (docs/cluster_v6/e7_jobs/e7_<job>.sbatch, written by e7_plan.py
# --emit-jobs / --emit-validation) run, on top of e30_lib.sh. Sourced by bash; every function is called from a
# job script in this order:
#
#   e7_job_begin JOB LINE CASE...
#       node-local NVIDIA shader disk cache, one directory per job (__GL_SHADER_DISK_CACHE_PATH, a larger
#       __GL_SHADER_DISK_CACHE_SIZE, __GL_SHADER_DISK_CACHE_SKIP_CLEANUP=1); e30_prelude (manifests, every
#       inherited V5_ / V6_ / V7_ variable unset, V7_WORKER_AFFINITY = one NUMA cpulist per GPU in nvidia-smi
#       order, the configuration print of every case of the job); provenance (hostname, harness commit,
#       experiment/v7 tree hash = E7_EXPECTED_TREE (default v7-rc1's) or the job stops, driver; with E7_RC1_REPO
#       the read-only checks of v7-rc1's deployment, below); host memory sampler (1 Hz, whole job);
#       bring-up (env, devices, SPIR-V manifest, 1M K = 1 and K = 2); device order check; the shader cache
#       check (files in the job's directory, none new in ~/.cache/nvidia); the job's .obj caches staged to
#       node-local /dev/shm.
#   e7_precheck CASE KIND PARTIES STEPS TIMEOUT
#       user item 9: PARTIES simultaneous K = 1 processes of CASE on GPUs 0.. (KIND k1) or K = 2 pairs on
#       (0,1),(2,3),.. (KIND pairs), behind the start barrier, STEPS steps: VRAM peak per card (telemetry),
#       host memory peak (sampler window). Fits -> the reference stays K = 1; a memory failure (out of device
#       or host memory, or killed before its timeout) -> refmode pairs in the state directory (a later point of
#       this line uses K = 2 pairs, eta x K/2, flagged; at K = 2 no reference); any other failure stops the job.
#   e7_point FAMILY CASE K TRIALS STEPS WARMUP T_CAL T_RUN T_TRACE TRACES REF:TIMEOUT...
#       one point: calibration (--weights auto --calibrate-only --weights-file, explicit --device-map), TRIALS
#       trials in rotated order (R E C / E C R / C R E, continued), R = every reference set of the point
#       (REF = same | pairs | <per-card case>), E = equal weights, C = --weights-file + --device-map, both
#       timed without the seam check; then TRACES step-traced runs (E29, with the seam check), one per arm.
#   e7_cross ORIGINAL_LINE <e7_point arguments>
#       user item 13: the same point on this node, labels x_*; skipped (recorded) when this node is the one
#       the original point ran on (its points.tsv record: a job of another solver than rc1 looks up
#       <point>@<tag> first, then the bare <point>; an rc1 job the bare one only; the CROSS line names it).
#   e7_extra KIND LABEL CASE K STEPS WARMUP TIMEOUT      (KIND anatomy | fulltrace | soak | selftest)
#       untimed runs, a failure is recorded and the job goes on: anatomy = --anatomy with V7_LOOP_TRACE=1 and
#       --defrag-log (items 5, 6), the loop rows written to traces/<label>/loop_trace.csv (run_chain_v6.py
#       --loop-trace, item 11); fulltrace = --step-trace --step-trace-detail full; soak = V7_POOL_PEAKS=1
#       with --defrag-log and the seam check (item 7); selftest = every E7 output path at once on a small case.
#   e7_pair_point FAMILY CASE K TRIALS STEPS WARMUP T_CAL T_RUN ARM REF:TIMEOUT...      (validation job, item 7)
#       an A/B point of two solvers, the job's (E7_SOLVER_TAG, its deployment E30_REPO) and v7-rc1 (E7_RC1_REPO,
#       read only): point ab_<short>_K<K>; ARM C = one calibration by the job's solver (cal_<point>, its weights
#       file read by both solvers: same device map and particles, so the same cuts) or E = equal weights; TRIALS
#       trials, each the point's reference sets (the job's solver, as in e7_point) and then the two K runs back
#       to back, the solver order alternating (R rc1 <job> / R <job> rc1 / ...), labels <point>_t<t>_<ARM>_rc1
#       and <point>_t<t>_<ARM>_<job tag> (role calibrated / equal); no traces. An rc1 run executes rc1's own
#       run_chain_v6.py (E30_RUN_REPO, e30_lib.sh) with the wrapper options that copy knows (--solver,
#       --obj-cache), both K runs with PYTHONDONTWRITEBYTECODE=1 (rc1 stays read only; the two alike).
#   e7_job_end
#       the provenance checks again (provenance_end.json): a tree, manifest or commit that changed during the
#       job is reported and the job ends with status 3. A job that stops before it (e30_fail, SIGTERM) runs the
#       same check, bounded in time, in its EXIT / TERM cleanup (e7_cleanup) and syncs it to the log directory.
#
# Solvers (validation job, line V, job e7_V): the job runs from its deployment E30_REPO (default: the deployment
# this file belongs to) with tag E7_SOLVER_TAG and tree E7_EXPECTED_TREE (both exported by the job script; the
# defaults are v7-rc1's); E7_RC1_REPO (set only there) names v7-rc1's deployment: its COMMIT (prefix
# E7_RC1_COMMIT), MANIFEST.sha256, SCRIPTS.sha256 and experiment/v7 tree (E7_RC1_TREE) are checked at the start
# (the job stops on a mismatch) and at the end; nothing is written there (batch-3 jobs run from it).
# Device maps: K slabs on GPUs 0..K-1 (line A K <= 4 on GPUs 0-3 = NUMA node 0; line B 0-7). A reference
# process (K = 1, or a K = 2 pair) is pinned with taskset to its GPU's NUMA cpulist; a K-slab run is not pinned
# (its transport workers are, by V7_WORKER_AFFINITY), as in the smoke.
# Stop rule: a hard failure of a timed run, a calibration, a reference member or a traced run stops the job
# (e30_fail); the next job of the line still starts (singleton dependency). Extras never stop the job.
# State shared by the jobs of the campaign: $E7_STATE (default ~/run/logs/e7_state): refmode_<case>_<line>
# from the pre-checks, points.tsv (point, job, slurm job, host, time, seconds) for the cross-node repeats: the
# point's bare name for an rc1 job, <point>@<tag> for a job of any other solver (e7_point_record; batch-1 records
# and rc1's own e7_cross keep reading bare names only); a dry run prints the line it would append.
# Every run gets one line in $SHM/index.tsv: label, role, family, case, K, trial, reference kind, reference
# case, point, line, host, solver (the tag of the code that ran it; e7_full_summary.py groups by it). The results
# row of each run (parse_run_v6.py) carries the same tag, the tree and the root the process printed.
# E30_DRY=1 prints every command instead of running it (e30_lib.sh).

# v7-rc1 (d0c8dcb): git rev-parse v7-rc1:experiment/v7, and the COMMIT prefix of its deployment. Constants: a
# cluster job (E30_LOCAL != 1) whose environment sets E7_RC1_TREE / E7_RC1_COMMIT to anything else, the empty
# value included, stops in e7_job_begin; a local pre-run may override them (stand-in deployments).
E7_RC1_TREE_CONSTANT=82dd6a740fa2a97a44e6a31d75392f75e486163b
E7_RC1_COMMIT_CONSTANT=d0c8dcb7c848
E7_RC1_TREE=${E7_RC1_TREE-$E7_RC1_TREE_CONSTANT}
E7_RC1_COMMIT=${E7_RC1_COMMIT-$E7_RC1_COMMIT_CONSTANT}
# the job's solver: git rev-parse <its commit>:experiment/v7 and its tag (v7-rc1's by default; a job of another
# deployment exports both, e7_plan.py; a local pre-run from a CRLF worktree overrides the tree only)
E7_EXPECTED_TREE=${E7_EXPECTED_TREE:-$E7_RC1_TREE}
E7_SOLVER_TAG=${E7_SOLVER_TAG:-$([ "$E7_EXPECTED_TREE" = "$E7_RC1_TREE" ] && echo rc1)}
E7_EXPECTED_DEVICES=${E7_EXPECTED_DEVICES:-8}                   # the local pre-run rig has 2
E7_SHADER_CACHE_SIZE=17179869184                               # 16 GiB (the driver's default cap is 1 GiB)
E7_LIBRARY_REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)    # the deployment this file belongs to

e7_short() {        # CASE -> 2d_n4000 / 3d_n416_k8 (labels)
    echo "$1" | sed 's/^cavity2d_/2d_/; s/^cavity3d_/3d_/'
}

e7_device_map() {   # K -> 0,1,...,K-1
    local slab text=""
    for slab in $(seq 0 $(($1 - 1))); do text="$text$slab,"; done
    echo "${text%,}"
}

e7_ones() {         # K -> 1,1,...,1
    local text
    text=$(printf '1,%.0s' $(seq 1 "$1"))
    echo "${text%,}"
}

e7_rotation() {     # TRIAL (1-based) -> R E C / E C R / C R E, continued past trial 3
    case $(( ($1 - 1) % 3 )) in
        0) echo "R E C" ;;
        1) echo "E C R" ;;
        2) echo "C R E" ;;
    esac
}

e7_pair_order() {   # TRIAL (1-based) -> R rc1 <job tag> / R <job tag> rc1, alternating
    if [ $(( $1 % 2 )) -eq 1 ]; then echo "R rc1 $E7_SOLVER_TAG"; else echo "R $E7_SOLVER_TAG rc1"; fi
}

e7_run_tag() {      # the solver tag of the code an e30_run call made here executes (E30_RUN_REPO: E30_RUN_TAG)
    if [ -n "${E30_RUN_REPO:-}" ]; then echo "${E30_RUN_TAG:-}"; else echo "$E7_SOLVER_TAG"; fi
}

e7_index() {        # LABEL ROLE FAMILY CASE K TRIAL REFERENCE_KIND REFERENCE_CASE POINT   (+ line, host, solver tag)
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9" \
        "$E7_LINE" "$(hostname)" "$(e7_run_tag)" >> "$SHM/index.tsv"
}

e7_state_get() {    # NAME -> content of $E7_STATE/NAME (empty when absent)
    [ -f "$E7_STATE/$1" ] && cat "$E7_STATE/$1"
    return 0
}

e7_state_set() {    # NAME VALUE
    [ "$E30_DRY" = 1 ] && { echo "DRY: state $1 = $2"; return 0; }
    echo "$2" > "$E7_STATE/$1.tmp.$$" && mv -f "$E7_STATE/$1.tmp.$$" "$E7_STATE/$1"
}

e7_state_append() { # NAME LINE
    [ "$E30_DRY" = 1 ] && { echo "DRY: state $1 += $2"; return 0; }
    echo "$2" >> "$E7_STATE/$1"
}

e7_point_record() { # POINT -> its name in points.tsv: POINT for an rc1 job (and an untagged one), POINT@<tag> for a job of
    # any other solver, so that a lookup of the bare name (rc1's jobs; rc1's own e7_cross too) never finds another
    # solver's run of the same point
    if [ -n "${E7_SOLVER_TAG:-}" ] && [ "$E7_SOLVER_TAG" != rc1 ]; then echo "$1@$E7_SOLVER_TAG"; else echo "$1"; fi
}

e7_group_rows() {   # LABEL -> "<member rows> <hard-failure labels...>" (members = LABEL_g<i> / LABEL_p<i> exactly)
    $PY -c 'import json, re, sys
rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
member = re.compile(re.escape(sys.argv[2]) + r"_[gp]\d+")
members = [row for row in rows if member.fullmatch(row["label"])]
print(len(members), " ".join(row["label"] for row in members if row["status"] == "hard_failure"))' "$E30_RESULTS" "$1"
}

e7_check_group() {  # LABEL MEMBERS: stop the job when a member of the group failed or has no row
    [ "$E30_DRY" = 1 ] && return 0
    local result found failed
    result=$(e7_group_rows "$1")
    found=${result%% *}
    failed=${result#* }
    [ -z "${failed// /}" ] || e30_fail "hard failure in $failed"
    [ "$found" = "$2" ] || e30_fail "$1: $found rows of $2 members"
}

e7_glcache_check() {   # STAGE: does the driver write its shader cache into this job's directory, and only there?
    local files size newer verdict
    files=$(find "$E7_GLCACHE" -type f 2>/dev/null | wc -l)
    size=$(du -sm "$E7_GLCACHE" 2>/dev/null | cut -f1)
    newer=$(timeout 120 find -L "$HOME/.cache/nvidia" -type f -newer "$E7_GLCACHE_MARKER" 2>/dev/null | wc -l)
    verdict="CONFIRMED"
    { [ "$files" -gt 0 ] && [ "$newer" -eq 0 ]; } || verdict="NOT CONFIRMED"
    echo "SHADER CACHE ($1): $verdict - $E7_GLCACHE holds $files files ($size MiB); ~/.cache/nvidia: $newer files newer than the job start"
    find "$E7_GLCACHE" -maxdepth 3 2>/dev/null | head -8 | sed 's/^/  /'
    printf '{"stage": "%s", "verdict": "%s", "directory": "%s", "files": %s, "mib": %s, "default_location_new_files": %s, "time": %s}\n' \
        "$1" "$verdict" "$E7_GLCACHE" "$files" "${size:-0}" "$newer" "$(date +%s)" >> "$SHM/shader_cache.jsonl"
}

e7_job_begin() {    # JOB LINE CASE...
    E7_JOB=$1
    E7_LINE=$2
    shift 2
    E7_CASES=("$@")
    export E30_SOLVER=v7
    export E30_REPO=${E30_REPO:-$E7_LIBRARY_REPO}
    # this file must belong to the deployment the job runs from (else the job would run another deployment's
    # code under this file's checks and labels)
    [ "$(cd "$E30_REPO" 2>/dev/null && pwd -P)" = "$(cd "$E7_LIBRARY_REPO" && pwd -P)" ] \
        || { echo "ABORT: e7_lib.sh of $E7_LIBRARY_REPO sourced for the deployment E30_REPO=$E30_REPO"; exit 2; }
    # v7-rc1's identity is fixed in a cluster job: an override (the empty value too) would let the tag rc1 and the
    # rc1 deployment checks accept another tree or commit
    if [ "${E30_LOCAL:-0}" != 1 ] && { [ "$E7_RC1_TREE" != "$E7_RC1_TREE_CONSTANT" ] \
            || [ "$E7_RC1_COMMIT" != "$E7_RC1_COMMIT_CONSTANT" ]; }; then
        echo "ABORT: E7_RC1_TREE='$E7_RC1_TREE' E7_RC1_COMMIT='$E7_RC1_COMMIT' are not v7-rc1's constants" \
             "($E7_RC1_TREE_CONSTANT, $E7_RC1_COMMIT_CONSTANT): a cluster job does not take them from the environment"
        exit 2
    fi
    # the solver tag labels every index and results row: it must name the expected tree's solver
    if [ -z "$E7_SOLVER_TAG" ]; then
        [ "${E30_LOCAL:-0}" = 1 ] \
            || { echo "ABORT: E7_EXPECTED_TREE=$E7_EXPECTED_TREE is not v7-rc1's: export E7_SOLVER_TAG with it"; exit 2; }
        E7_SOLVER_TAG=local
    fi
    local tag_is_rc1=0 tree_is_rc1=0          # the tag rc1 goes with v7-rc1's tree and no other (cluster runs)
    [ "$E7_SOLVER_TAG" = rc1 ] && tag_is_rc1=1
    [ "$E7_EXPECTED_TREE" = "$E7_RC1_TREE" ] && tree_is_rc1=1
    if [ "${E30_LOCAL:-0}" != 1 ] && [ "$tag_is_rc1" != "$tree_is_rc1" ]; then
        echo "ABORT: E7_SOLVER_TAG=$E7_SOLVER_TAG with E7_EXPECTED_TREE=$E7_EXPECTED_TREE (v7-rc1's tree is $E7_RC1_TREE)"
        exit 2
    fi
    if [ -n "${E7_RC1_REPO:-}" ] && [ "$E7_SOLVER_TAG" = rc1 ]; then
        echo "ABORT: E7_RC1_REPO names the second solver of a job of another solver; this job's solver is rc1"; exit 2
    fi
    E7_EXPECTED_LABEL=v7-$E7_SOLVER_TAG
    E7_OBJ_CACHE=${E7_OBJ_CACHE:-$HOME/run/e7_objcache}
    E7_STATE=${E7_STATE:-$HOME/run/logs/e7_state}
    E30_DRY=${E30_DRY:-0}
    # user item 10: the driver's shader disk cache in node-local memory, one directory per job; set before the
    # first Vulkan process of the job (the configuration print creates no Vulkan instance)
    E7_GLCACHE=${E7_GLCACHE_BASE:-/dev/shm/scxm138}/glcache_${SLURM_JOB_ID:-$$}
    mkdir -p "$E7_GLCACHE" "$E7_STATE" || { echo "ABORT: cannot create $E7_GLCACHE / $E7_STATE"; exit 2; }
    export __GL_SHADER_DISK_CACHE=1
    export __GL_SHADER_DISK_CACHE_PATH=$E7_GLCACHE
    export __GL_SHADER_DISK_CACHE_SIZE=$E7_SHADER_CACHE_SIZE
    export __GL_SHADER_DISK_CACHE_SKIP_CLEANUP=1
    E30_CONFIG_CASES=""
    local case_name
    for case_name in "${E7_CASES[@]}"; do E30_CONFIG_CASES="$E30_CONFIG_CASES --case cases/aligned/$case_name/case.yaml"; done
    source "$E30_REPO/docs/cluster_v6/scripts/e30_lib.sh" || { echo "ABORT: no e30_lib.sh"; exit 2; }
    e30_prelude "e7_$E7_JOB"
    E7_GLCACHE_MARKER=$SHM/.job_start_marker
    touch "$E7_GLCACHE_MARKER"
    echo "--- shader cache: $(env | grep '^__GL_SHADER_DISK_CACHE' | sort | tr '\n' ' ')"
    COMMON="--sync-scheme per-direction --depth 2 --pool-safety 1.2"
    NODE_CACHE=${SHM}_obj_cache            # next to $SHM, not in it (e30_sync copies $SHM after every step)
    E7_BARRIERS=${SHM}_barriers
    mkdir -p "$E7_BARRIERS" "$SHM/weights" "$SHM/traces"
    e7_provenance
    if [ "$E30_DRY" = 1 ]; then         # dry run (login node, E30_LOCAL=1): stand-ins for the GPUs' NUMA cpulists
        DEVICES=8
        [ "${#CPUL[@]}" -gt 0 ] || CPUL=(cpus_gpu0 cpus_gpu1 cpus_gpu2 cpus_gpu3 cpus_gpu4 cpus_gpu5 cpus_gpu6 cpus_gpu7)
    else
        DEVICES=$(timeout 60 nvidia-smi -L | wc -l)
        [ "$DEVICES" -eq "$E7_EXPECTED_DEVICES" ] || e30_fail "expected $E7_EXPECTED_DEVICES GPUs, $DEVICES visible"
        $PY -u docs/cluster_v6/scripts/host_memory_sampler.py --out "$SHM/host_memory.csv" --interval 1 &
        E7_MEMPID=$!
    fi
    # a job that stops before e7_job_end (e30_fail, a SIGTERM at the time limit or from scancel) still gets the
    # end-of-job provenance check, in its cleanup (e7_cleanup)
    trap 'e7_cleanup exit $?' EXIT
    trap 'echo "*** E7 job $E7_JOB: SIGTERM ***"; e7_cleanup term 143; exit 143' TERM
    e30_step s0_bringup 1200 -- $PY -u remote/bringup_check_v6.py $E30_SOLVER_ARGS --stage all \
        --case cases/aligned/$E7_BRINGUP_CASE/case.yaml --log-dir "$SHM/bringup" --expected-devices "$DEVICES" --timeout 300
    e30_step s0_caps 300 --optional -- $PY -u docs/cluster_v6/scripts/caps_v6.py --out "$SHM/caps.json"
    e7_glcache_check after_bringup
    local stage_cases=()
    for case_name in "${E7_CASES[@]}"; do stage_cases+=(--case "cases/aligned/$case_name/case.yaml"); done
    e30_step s0_obj_cache 1800 -- $PY -u docs/cluster_v6/scripts/obj_cache_build.py $E30_SOLVER_ARGS \
        --obj-cache "$E7_OBJ_CACHE" --check "${stage_cases[@]}" --stage-to "$NODE_CACHE"
    E30_WRAPPER_ARGS="--obj-cache $NODE_CACHE"
    echo "=== E7 job $E7_JOB (line $E7_LINE) ready at $(date +%T): ${#E7_CASES[@]} cases staged"
}
E7_BRINGUP_CASE=cavity2d_n1000

e7_tree_hash() {    # DIRECTORY -> its git tree hash (the job's tree_hash.py; reads only)
    timeout 300 $PY docs/cluster_v6/scripts/tree_hash.py "$1" 2>&1 | tail -1
}

e7_manifest_state() {   # DIRECTORY MANIFEST -> OK | MISMATCH | absent   (sha256sum -c inside DIRECTORY; reads only)
    if [ ! -f "$1/$2" ]; then echo absent; return 0; fi
    if (cd "$1" && timeout 900 sha256sum --quiet -c "$2" > /dev/null 2>&1); then echo OK; else echo MISMATCH; fi
}

e7_deployment_state() {     # DIRECTORY NAME: NAME_ROOT, _COMMIT, _TREE (experiment/v7), _MANIFEST, _SCRIPTS, _HARNESS
    local directory=$1 name=$2 root
    root=$(cd "$directory" 2>/dev/null && pwd -P)
    printf -v "${name}_ROOT" '%s' "${root:-missing}"
    printf -v "${name}_COMMIT" '%s' "$(head -c 40 "$directory/COMMIT" 2>/dev/null)"
    printf -v "${name}_TREE" '%s' "$(e7_tree_hash "$directory/experiment/v7")"
    printf -v "${name}_MANIFEST" '%s' "$(e7_manifest_state "$directory" MANIFEST.sha256)"
    printf -v "${name}_SCRIPTS" '%s' "$(e7_manifest_state "$directory" SCRIPTS.sha256)"
    printf -v "${name}_HARNESS" '%s' "$(head -c 40 "$directory/SCRIPTS_COMMIT" 2>/dev/null)"
}

e7_write_json() {   # FILE KEY VALUE ...: one JSON object (VALUE true / false -> a boolean, anything else a string)
    $PY -c 'import json, sys
items = sys.argv[2:]
data = {items[index]: {"true": True, "false": False}.get(items[index + 1], items[index + 1])
        for index in range(0, len(items) - 1, 2)}
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    handle.write(json.dumps(data) + "\n")' "$@"
}

e7_rc1_mismatches() {   # -> the ways $E7_RC1_REPO (state E7_RC1_FOUND_*) is not v7-rc1's intact deployment ("" = none)
    local text=""
    [ "$E7_RC1_FOUND_ROOT" != missing ] || text="$text no deployment at E7_RC1_REPO=$E7_RC1_REPO;"
    [ "$E7_RC1_FOUND_ROOT" != "$E7_JOB_ROOT" ] || text="$text E7_RC1_REPO is the job's own deployment;"
    case "$E7_RC1_FOUND_COMMIT" in
        "$E7_RC1_COMMIT"*) ;;
        *) text="$text rc1 COMMIT '$E7_RC1_FOUND_COMMIT' does not start with $E7_RC1_COMMIT;" ;;
    esac
    [ "$E7_RC1_FOUND_TREE" = "$E7_RC1_TREE" ] || text="$text rc1 experiment/v7 tree $E7_RC1_FOUND_TREE is not $E7_RC1_TREE;"
    [ "$E7_RC1_FOUND_MANIFEST" = OK ] || text="$text rc1 MANIFEST.sha256 $E7_RC1_FOUND_MANIFEST;"
    [ "$E7_RC1_FOUND_SCRIPTS" = OK ] || text="$text rc1 SCRIPTS.sha256 $E7_RC1_FOUND_SCRIPTS;"
    echo "$text"
}

e7_provenance() {   # user item 14: harness commit, experiment/v7 tree hash, driver, hostname; with E7_RC1_REPO the
    # read-only checks of v7-rc1's deployment (the second solver of a validation job). The job stops on a mismatch.
    local driver failures="" fields
    e7_deployment_state . E7_JOB
    driver=$(timeout 60 nvidia-smi --query-gpu=driver_version --format=csv,noheader | tr -d '\r' | sort -u | tr '\n' ' ' | sed 's/ $//')
    # every run of the job's deployment is recorded with this tag and tree (e30_run -> parse_run_v6.py)
    E30_SOLVER_TAG=$E7_SOLVER_TAG
    E30_SOLVER_TREE=$E7_JOB_TREE
    echo "PROVENANCE host=$(hostname) slurm_job=$E30_JOB job=$E7_JOB line=$E7_LINE harness=${E7_JOB_HARNESS:-none} experiment_v7_tree=$E7_JOB_TREE ($E7_EXPECTED_LABEL $E7_EXPECTED_TREE) driver=$driver solver=$E7_SOLVER_TAG root=$E7_JOB_ROOT"
    [ "$E7_JOB_TREE" = "$E7_EXPECTED_TREE" ] || failures="experiment/v7 tree $E7_JOB_TREE is not $E7_EXPECTED_LABEL's $E7_EXPECTED_TREE;"
    fields=(host "$(hostname)" slurm_job "$E30_JOB" job "$E7_JOB" line "$E7_LINE" harness_commit "${E7_JOB_HARNESS:-none}"
            experiment_v7_tree "$E7_JOB_TREE" expected_tree "$E7_EXPECTED_TREE"
            tree_match "$([ "$E7_JOB_TREE" = "$E7_EXPECTED_TREE" ] && echo true || echo false)"
            driver "$driver" commit "$E7_JOB_COMMIT" time "$(date -Is)"
            solver_tag "$E7_SOLVER_TAG" expected_label "$E7_EXPECTED_LABEL" root "$E7_JOB_ROOT")
    if [ -n "${E7_RC1_REPO:-}" ]; then
        e7_deployment_state "$E7_RC1_REPO" E7_RC1_FOUND
        local rc1_failures
        rc1_failures=$(e7_rc1_mismatches)
        echo "PROVENANCE rc1 root=$E7_RC1_FOUND_ROOT commit=$E7_RC1_FOUND_COMMIT (expected $E7_RC1_COMMIT) experiment_v7_tree=$E7_RC1_FOUND_TREE (v7-rc1 $E7_RC1_TREE) manifest=$E7_RC1_FOUND_MANIFEST scripts=$E7_RC1_FOUND_SCRIPTS harness=${E7_RC1_FOUND_HARNESS:-none}"
        failures="$failures$rc1_failures"
        fields+=(rc1_repository "$E7_RC1_REPO" rc1_root "$E7_RC1_FOUND_ROOT" rc1_commit "$E7_RC1_FOUND_COMMIT"
                 rc1_expected_commit "$E7_RC1_COMMIT" rc1_tree "$E7_RC1_FOUND_TREE" rc1_expected_tree "$E7_RC1_TREE"
                 rc1_tree_match "$([ "$E7_RC1_FOUND_TREE" = "$E7_RC1_TREE" ] && echo true || echo false)"
                 rc1_manifest "$E7_RC1_FOUND_MANIFEST" rc1_scripts "$E7_RC1_FOUND_SCRIPTS"
                 rc1_harness_commit "${E7_RC1_FOUND_HARNESS:-none}"
                 rc1_match "$([ -z "$rc1_failures" ] && echo true || echo false)")
    fi
    e7_write_json "$SHM/provenance.json" "${fields[@]}"
    if [ -n "$failures" ]; then
        [ "$E30_DRY" = 1 ] && { echo "DRY: provenance mismatch not enforced:$failures"; return 0; }
        e30_fail "provenance:$failures"
    fi
}

e7_provenance_end() {   # [WHERE] the provenance checks again at the end (every run of the job lies between the two):
    # provenance_end.json (checked_in WHERE: job_end = e7_job_end, the default; exit / term = the cleanup of a job that
    # stopped before it, e7_cleanup); returns 1 when a tree, manifest, harness or commit of either deployment changed
    local where=${1:-job_end} changes="" fields name
    e7_deployment_state . E7_END
    for name in ROOT COMMIT TREE MANIFEST SCRIPTS HARNESS; do
        local before="E7_JOB_$name" after="E7_END_$name"
        [ "${!before}" = "${!after}" ] || changes="$changes job ${name,,} ${!before} -> ${!after};"
    done
    fields=(job "$E7_JOB" slurm_job "$E30_JOB" time "$(date -Is)" checked_in "$where" root "$E7_END_ROOT"
            commit "$E7_END_COMMIT" experiment_v7_tree "$E7_END_TREE" manifest "$E7_END_MANIFEST"
            scripts "$E7_END_SCRIPTS" harness_commit "${E7_END_HARNESS:-none}")
    if [ -n "${E7_RC1_REPO:-}" ]; then
        e7_deployment_state "$E7_RC1_REPO" E7_RC1_END
        for name in ROOT COMMIT TREE MANIFEST SCRIPTS HARNESS; do
            local before="E7_RC1_FOUND_$name" after="E7_RC1_END_$name"
            [ "${!before}" = "${!after}" ] || changes="$changes rc1 ${name,,} ${!before} -> ${!after};"
        done
        fields+=(rc1_root "$E7_RC1_END_ROOT" rc1_commit "$E7_RC1_END_COMMIT" rc1_tree "$E7_RC1_END_TREE"
                 rc1_manifest "$E7_RC1_END_MANIFEST" rc1_scripts "$E7_RC1_END_SCRIPTS"
                 rc1_harness_commit "${E7_RC1_END_HARNESS:-none}")
    fi
    fields+=(unchanged "$([ -z "$changes" ] && echo true || echo false)" changes "$changes")
    e7_write_json "$SHM/provenance_end.json" "${fields[@]}"
    if [ -n "$changes" ]; then
        echo "*** PROVENANCE CHANGED DURING THE JOB:$changes ***"
        [ "$E30_DRY" = 1 ] && return 0
        return 1
    fi
    echo "PROVENANCE_END unchanged$([ "$where" != job_end ] && echo " (checked in the $where cleanup)"): experiment_v7_tree=$E7_END_TREE manifest=$E7_END_MANIFEST scripts=$E7_END_SCRIPTS$([ -n "${E7_RC1_REPO:-}" ] && echo " rc1_tree=$E7_RC1_END_TREE rc1_manifest=$E7_RC1_END_MANIFEST rc1_scripts=$E7_RC1_END_SCRIPTS")"
}

e7_provenance_end_bounded() {   # WHERE: e7_provenance_end WHERE in a child bash that gets this shell's functions and
    # state, killed with its children (timeout's process group) after E7_PROVENANCE_END_TIMEOUT s (default 300) or,
    # in the TERM cleanup, E7_PROVENANCE_END_TERM_TIMEOUT s (default 20: inside the scheduler's grace period before
    # its SIGKILL); returns the check's status (0 unchanged, 1 changed, 124 / 137 cut off). A check that wrote no
    # provenance_end.json leaves one that says so ("incomplete").
    local where=$1 limit definitions status=0
    if [ "$where" = term ]; then limit=${E7_PROVENANCE_END_TERM_TIMEOUT:-20}; else limit=${E7_PROVENANCE_END_TIMEOUT:-300}; fi
    echo "--- end-of-job provenance check in the $where cleanup (the job stopped before e7_job_end; at most $limit s)"
    definitions="$(declare -f e7_provenance_end e7_deployment_state e7_tree_hash e7_manifest_state e7_write_json)
$(declare -p PY SHM E30_JOB E30_DRY E7_JOB ${!E7_JOB_@} ${!E7_RC1_FOUND_@})
$([ -n "${E7_RC1_REPO:-}" ] && declare -p E7_RC1_REPO)"
    timeout -k 5 "$limit" "$BASH" -c "$definitions
e7_provenance_end $where" || status=$?
    if [ ! -f "$SHM/provenance_end.json" ]; then
        echo "*** END-OF-JOB PROVENANCE CHECK INCOMPLETE ($where cleanup, status $status, limit $limit s) ***"
        e7_write_json "$SHM/provenance_end.json" job "$E7_JOB" slurm_job "$E30_JOB" time "$(date -Is)" \
            checked_in "$where" incomplete "the end check did not finish within $limit s (status $status)"
    fi
    return $status
}

e7_cleanup() {      # [WHERE STATUS]: stop the sampler, free the node-local caches (logs stay in $SHM, synced). From the
    # EXIT / TERM traps (WHERE exit / term, STATUS the job's exit status) of a job that stopped before e7_job_end
    # (no provenance_end.json yet, provenance checked at the start): the end-of-job provenance check
    # (e7_provenance_end_bounded), then e30_sync as at the normal end; a change it finds turns an exit status 0 into 3.
    # e7_job_end calls it without arguments (its own check ran).
    local where=${1:-} status=${2:-0} end_status=0
    trap - EXIT TERM
    if [ -n "${E7_MEMPID:-}" ]; then kill "$E7_MEMPID" 2>/dev/null; wait "$E7_MEMPID" 2>/dev/null; E7_MEMPID=""; fi
    [ -n "${E7_GLCACHE:-}" ] && [ -d "$E7_GLCACHE" ] && rm -rf "$E7_GLCACHE"
    [ -n "${NODE_CACHE:-}" ] && [ -d "$NODE_CACHE" ] && [ "$NODE_CACHE" != "$E7_OBJ_CACHE" ] && rm -rf "$NODE_CACHE"
    [ -n "${E7_BARRIERS:-}" ] && [ -d "$E7_BARRIERS" ] && rm -rf "$E7_BARRIERS"
    if [ -n "$where" ] && [ -n "${E7_JOB_ROOT:-}" ] && [ ! -f "$SHM/provenance_end.json" ]; then
        e7_provenance_end_bounded "$where" || end_status=$?
        e30_sync
        if [ "$end_status" = 1 ] && [ "$status" = 0 ]; then exit 3; fi
    fi
    return 0
}

e7_memory_window() {    # LABEL SINCE UNTIL
    [ "$E30_DRY" = 1 ] && return 0
    timeout 120 $PY docs/cluster_v6/scripts/host_memory_sampler.py --summary "$SHM/host_memory.csv" \
        --since "$2" --until "$3" --label "$1" --json "$SHM/memory_windows.jsonl"
}

e7_group() {        # LABEL KIND CASE PARTIES STEPS WARMUP TIMEOUT ROLE FAMILY TRIAL REFERENCE_KIND POINT
    # PARTIES processes started together, each behind the start barrier: KIND k1 = K = 1 on GPU 0..PARTIES-1,
    # KIND pairs = K = 2 on GPUs (2i, 2i+1); every member pinned to its (first) GPU's NUMA cpulist; failures
    # recorded per member (the caller decides).
    local label=$1 kind=$2 case_name=$3 parties=$4 steps=$5 warmup=$6 limit=$7 role=$8 family=$9
    local trial=${10} reference_kind=${11} point=${12}
    local yaml=cases/aligned/$case_name/case.yaml member gpu map weights suffix pids="" slabs
    local barrier=$E7_BARRIERS/$label
    mkdir -p "$barrier"
    for member in $(seq 0 $((parties - 1))); do
        if [ "$kind" = pairs ]; then
            gpu=$((2 * member)); map="$gpu,$((gpu + 1))"; weights=1,1; suffix=p$member; slabs=2
        else
            gpu=$member; map=$gpu; weights=1; suffix=g$member; slabs=1
        fi
        e7_index "${label}_$suffix" "$role" "$family" "$case_name" "$slabs" "$trial" "$reference_kind" "$case_name" "$point"
        (
            E30_PREFIX=""
            [ -n "${CPUL[$gpu]:-}" ] && E30_PREFIX="taskset -c ${CPUL[$gpu]}"      # local pre-run: no cpulists
            # a member still missing after half the run timeout has failed: the others start anyway (status timeout)
            E30_WRAPPER_ARGS="--obj-cache $NODE_CACHE --barrier $barrier --barrier-parties $parties --barrier-timeout $((limit / 2))"
            e30_run "${label}_$suffix" "$limit" --optional -- --case "$yaml" --weights "$weights" --device-map "$map" \
                $COMMON --max-steps "$steps" --warmup "$warmup" --no-seam-check
        ) &
        pids="$pids $!"
    done
    wait $pids
}

e7_precheck() {     # CASE KIND PARTIES STEPS TIMEOUT
    local case_name=$1 kind=$2 parties=$3 steps=$4 limit=$5
    local label since until verdict
    label=pre_$(e7_short "$case_name")_${kind}_x$parties
    echo; echo "##### PRECHECK $case_name $kind x$parties ($(date +%T)) host=$(hostname)"
    since=$(date +%s.%N)
    e7_group "$label" "$kind" "$case_name" "$parties" "$steps" 10 "$limit" precheck precheck 0 "$kind" "$label"
    until=$(date +%s.%N)
    e7_memory_window "$label" "$since" "$until"
    [ "$E30_DRY" = 1 ] && return 0
    verdict=$($PY - "$E30_RESULTS" "${label}_" "$limit" "$SHM" <<'PYEOF'
import json, pathlib, re, sys
rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
rows = [row for row in rows if re.fullmatch(re.escape(sys.argv[2]) + r"[gp]\d+", row["label"])]
limit, directory = float(sys.argv[3]), pathlib.Path(sys.argv[4])
memory = re.compile(r"OutOfDeviceMemory|OutOfHostMemory|OUT_OF_DEVICE_MEMORY|OUT_OF_HOST_MEMORY|MemoryError|"
                    r"Cannot allocate memory|bad_alloc")
def log_text(row):
    path = directory / (row["label"] + ".log")
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
if rows and all(row["status"] == "pass" for row in rows):
    print("fits")
elif any(memory.search(log_text(row)) or (row.get("rc") == 137 and (row.get("wall_seconds") or limit) < limit)
         for row in rows if row["status"] != "pass"):
    print("too_large")
else:
    print("failed")
PYEOF
)
    echo "PRECHECK $case_name $kind x$parties: $verdict"
    printf '{"case": "%s", "kind": "%s", "parties": %s, "line": "%s", "verdict": "%s", "host": "%s", "job": "%s"}\n' \
        "$case_name" "$kind" "$parties" "$E7_LINE" "$verdict" "$(hostname)" "$E30_JOB" >> "$SHM/prechecks.jsonl"
    case "$verdict" in
        fits) [ "$kind" = k1 ] && e7_state_set "refmode_${case_name}_$E7_LINE" k1 ;;
        too_large)
            if [ "$kind" = k1 ]; then
                e7_state_set "refmode_${case_name}_$E7_LINE" pairs
                echo "PRECHECK: K = 1 of $case_name does not fit at x$parties -> its references on line $E7_LINE become K = 2 pairs (eta x K/2, flagged)"
            else
                e7_state_set "refmode_${case_name}_$E7_LINE" none
                echo "PRECHECK: the K = 2 pairs of $case_name do not fit -> no reference for it on line $E7_LINE"
            fi ;;
        *) e30_fail "precheck $label ($verdict)" ;;
    esac
}

e7_reference() {    # POINT_BASE TRIAL FAMILY POINT_CASE K SPEC STEPS WARMUP   (SPEC = same|pairs|<case> : TIMEOUT)
    local base=$1 trial=$2 family=$3 point_case=$4 slabs=$5 spec=$6 steps=$7 warmup=$8
    local kind=${spec%%:*} limit=${spec##*:} reference_case group_kind mode parties label
    case "$kind" in
        same) reference_case=$point_case; group_kind=k1 ;;
        pairs) reference_case=$point_case; group_kind=pairs ;;
        *) reference_case=$kind; group_kind=k1 ;;
    esac
    mode=$(e7_state_get "refmode_${reference_case}_${E7_REFERENCE_LINE:-$E7_LINE}")
    if [ "$mode" = none ] || { [ "$mode" = pairs ] && [ "$slabs" -le 2 ]; }; then
        echo "REFERENCE ${base} t$trial: no reference ($reference_case refmode=$mode at K=$slabs) - recorded"
        e7_index "${base}_t${trial}_R_none" reference_skipped "$family" "$point_case" "$slabs" "$trial" "$kind" "$reference_case" "$base"
        return 0
    fi
    [ "$mode" = pairs ] && group_kind=pairs
    if [ "$group_kind" = pairs ]; then parties=$((slabs / 2)); else parties=$slabs; fi
    label=${base}_t${trial}_R$([ "$group_kind" = pairs ] && echo P || echo 1)_$(e7_short "$reference_case")
    e7_group "$label" "$group_kind" "$reference_case" "$parties" "$steps" "$warmup" "$limit" reference "$family" \
        "$trial" "$kind" "$base"
    e7_check_group "$label" "$parties"
}

e7_calibrate() {    # POINT FAMILY CASE K YAML MAP WEIGHTS_FILE TIMEOUT: the point's weight calibration (a hard step)
    local base=$1 family=$2 case_name=$3 slabs=$4 yaml=$5 map=$6 weights_file=$7 limit=$8
    e7_index "cal_$base" calibration "$family" "$case_name" "$slabs" 0 - - "$base"
    e30_step "cal_$base" "$limit" -- $PY -u ${E30_RUN_REPO:+$E30_RUN_REPO/}docs/cluster_v6/scripts/run_chain_v6.py \
        $E30_SOLVER_ARGS --obj-cache "$NODE_CACHE" -- \
        --case "$yaml" --weights auto --calibrate-only --weights-file "$weights_file" --device-map "$map" $COMMON
    if [ "$E30_DRY" != 1 ]; then
        [ -s "$weights_file" ] || e30_fail "cal_$base wrote no weights file"
        grep -h "\[calibrate\]" "$SHM/cal_$base.log" | tail -4
    fi
}

e7_point() {        # FAMILY CASE K TRIALS STEPS WARMUP T_CAL T_RUN T_TRACE TRACES REF:TIMEOUT...
    local family=$1 case_name=$2 slabs=$3 trials=$4 steps=$5 warmup=$6 t_cal=$7 t_run=$8 t_trace=$9 traces=${10}
    shift 10
    local references=("$@")
    local base=${E7_PREFIX:-}$(e7_short "$case_name")_K$slabs
    local yaml=cases/aligned/$case_name/case.yaml map weights_file trial item reference started
    map=$(e7_device_map "$slabs")
    weights_file=$SHM/weights/$base.json
    started=$(date +%s)
    echo; echo "##### POINT $base family=$family case=$case_name K=$slabs trials=$trials traces=$traces references=${references[*]} host=$(hostname) ($(date +%T))"
    e7_calibrate "$base" "$family" "$case_name" "$slabs" "$yaml" "$map" "$weights_file" "$t_cal"
    for trial in $(seq 1 "$trials"); do
        for item in $(e7_rotation "$trial"); do
            case "$item" in
                R)
                    for reference in "${references[@]}"; do
                        e7_reference "$base" "$trial" "$family" "$case_name" "$slabs" "$reference" "$steps" "$warmup"
                    done ;;
                E)
                    e7_index "${base}_t${trial}_E" equal "$family" "$case_name" "$slabs" "$trial" - - "$base"
                    e30_run "${base}_t${trial}_E" "$t_run" -- --case "$yaml" --weights "$(e7_ones "$slabs")" --device-map "$map" \
                        $COMMON --max-steps "$steps" --warmup "$warmup" --no-seam-check ;;
                C)
                    e7_index "${base}_t${trial}_C" calibrated "$family" "$case_name" "$slabs" "$trial" - - "$base"
                    e30_run "${base}_t${trial}_C" "$t_run" -- --case "$yaml" --weights-file "$weights_file" --device-map "$map" \
                        $COMMON --max-steps "$steps" --warmup "$warmup" --no-seam-check ;;
            esac
        done
    done
    if [ "$traces" -ge 1 ]; then
        e7_index "${base}_trace_E" trace_equal "$family" "$case_name" "$slabs" 0 - - "$base"
        e30_run "${base}_trace_E" "$t_trace" -- --case "$yaml" --weights "$(e7_ones "$slabs")" --device-map "$map" \
            $COMMON --max-steps "$steps" --warmup "$warmup" --step-trace "$SHM/traces/${base}_E"
    fi
    if [ "$traces" -ge 2 ]; then
        e7_index "${base}_trace_C" trace_calibrated "$family" "$case_name" "$slabs" 0 - - "$base"
        e30_run "${base}_trace_C" "$t_trace" -- --case "$yaml" --weights-file "$weights_file" --device-map "$map" \
            $COMMON --max-steps "$steps" --warmup "$warmup" --step-trace "$SHM/traces/${base}_C"
    fi
    # the shared record for e7_cross: <base> for an rc1 job, <base>@<tag> for any other solver (e7_point_record)
    e7_state_append points.tsv "$(printf '%s\t%s\t%s\t%s\t%s\t%s' "$(e7_point_record "$base")" "$E7_JOB" "$E30_JOB" "$(hostname)" "$(date -Is)" "$(( $(date +%s) - started ))")"
    echo "##### POINT $base done in $(( $(date +%s) - started )) s"
}

e7_pair_run() {     # SOLVER LABEL ROLE FAMILY CASE K TRIAL POINT TIMEOUT -- CHAIN BENCH ARGUMENTS
    # one K run of a pair point: SOLVER rc1 = v7-rc1's deployment (E7_RC1_REPO; its own run_chain_v6.py, so only
    # the wrapper options that copy knows), any other = the job's. The index row and the run carry the same tag.
    local solver=$1 label=$2 role=$3 family=$4 case_name=$5 slabs=$6 trial=$7 point=$8 limit=$9
    shift 9
    [ "${1:-}" = "--" ] && shift
    if [ "$solver" = rc1 ]; then
        E30_RUN_REPO=$E7_RC1_REPO E30_RUN_TAG=rc1 E30_RUN_TREE=$E7_RC1_FOUND_TREE \
            e7_index "$label" "$role" "$family" "$case_name" "$slabs" "$trial" - - "$point"
        E30_RUN_REPO=$E7_RC1_REPO E30_RUN_TAG=rc1 E30_RUN_TREE=$E7_RC1_FOUND_TREE \
            E30_PREFIX="env PYTHONDONTWRITEBYTECODE=1" E30_WRAPPER_ARGS="--obj-cache $NODE_CACHE" \
            e30_run "$label" "$limit" -- "$@"
    elif [ "$solver" = "$E7_SOLVER_TAG" ]; then
        e7_index "$label" "$role" "$family" "$case_name" "$slabs" "$trial" - - "$point"
        E30_PREFIX="env PYTHONDONTWRITEBYTECODE=1" E30_WRAPPER_ARGS="--obj-cache $NODE_CACHE" \
            e30_run "$label" "$limit" -- "$@"
    else
        e30_fail "e7_pair_run $label: solver $solver is neither rc1 nor the job's $E7_SOLVER_TAG"
    fi
}

e7_pair_point() {   # FAMILY CASE K TRIALS STEPS WARMUP T_CAL T_RUN ARM REF:TIMEOUT...   (validation job, item 7)
    local family=$1 case_name=$2 slabs=$3 trials=$4 steps=$5 warmup=$6 t_cal=$7 t_run=$8 arm=$9
    shift 9
    local references=("$@")
    local base=ab_$(e7_short "$case_name")_K$slabs
    local yaml=cases/aligned/$case_name/case.yaml map weights_file trial item reference started role arm_arguments
    [ -n "${E7_RC1_REPO:-}" ] || e30_fail "e7_pair_point $base: E7_RC1_REPO (the second solver's deployment) is not set"
    case "$arm" in
        C) role=calibrated ;;
        E) role=equal ;;
        *) e30_fail "e7_pair_point $base: arm $arm (expected C or E)" ;;
    esac
    map=$(e7_device_map "$slabs")
    weights_file=$SHM/weights/$base.json
    started=$(date +%s)
    echo; echo "##### POINT $base family=$family case=$case_name K=$slabs trials=$trials traces=0 references=${references[*]} solvers=rc1,$E7_SOLVER_TAG arm=$arm host=$(hostname) ($(date +%T))"
    if [ "$arm" = C ]; then
        # by the job's solver; both solvers read the file (same device map and particles: the same cuts)
        e7_calibrate "$base" "$family" "$case_name" "$slabs" "$yaml" "$map" "$weights_file" "$t_cal"
        arm_arguments=(--weights-file "$weights_file")
    else
        arm_arguments=(--weights "$(e7_ones "$slabs")")
    fi
    for trial in $(seq 1 "$trials"); do
        for item in $(e7_pair_order "$trial"); do
            if [ "$item" = R ]; then
                for reference in "${references[@]}"; do
                    e7_reference "$base" "$trial" "$family" "$case_name" "$slabs" "$reference" "$steps" "$warmup"
                done
                continue
            fi
            e7_pair_run "$item" "${base}_t${trial}_${arm}_$item" "$role" "$family" "$case_name" "$slabs" "$trial" "$base" \
                "$t_run" -- --case "$yaml" "${arm_arguments[@]}" --device-map "$map" $COMMON --max-steps "$steps" \
                --warmup "$warmup" --no-seam-check
        done
    done
    # recorded under the ab_ name (both solvers ran it; no e7_cross looks an ab_ name up)
    e7_state_append points.tsv "$(printf '%s\t%s\t%s\t%s\t%s\t%s' "$base" "$E7_JOB" "$E30_JOB" "$(hostname)" "$(date -Is)" "$(( $(date +%s) - started ))")"
    echo "##### POINT $base done in $(( $(date +%s) - started )) s"
}

e7_cross() {        # ORIGINAL_LINE <e7_point arguments>   (user item 13)
    local original_line=$1
    shift
    local case_name=$2 slabs=$3 original original_host="" record="" record_text candidate
    original=$(e7_short "$case_name")_K$slabs
    # the original's record in points.tsv (the last line of the name): a job of another solver than rc1 looks up its
    # own <point>@<tag> first, then the bare <point> (rc1's run of it); an rc1 job only the bare name
    local candidates=("$original")
    [ "$(e7_point_record "$original")" != "$original" ] && candidates=("$(e7_point_record "$original")" "$original")
    for candidate in "${candidates[@]}"; do
        original_host=$(awk -F'\t' -v point="$candidate" '$1 == point {host = $4} END {print host}' "$E7_STATE/points.tsv" 2>/dev/null)
        if [ -n "$original_host" ]; then record=$candidate; break; fi
    done
    if [ -z "$record" ]; then
        record_text="no record: ${candidates[*]}"
    elif [ "$record" = "$original" ]; then
        record_text="bare record $record, host $original_host"
    else
        record_text="qualified record $record, host $original_host"
    fi
    if [ -n "$original_host" ] && [ "$original_host" = "$(hostname)" ]; then
        echo "CROSS $original (line $original_line): SKIPPED - this node $(hostname) ran the original ($record_text); resubmit with --exclude=$(hostname)"
        e7_index "x_${original}_skipped" cross_skipped "$1" "$case_name" "$slabs" 0 - - "x_$original"
        e7_state_append cross_skipped.tsv "$(printf '%s\t%s\t%s\t%s' "$(e7_point_record "$original")" "$E7_JOB" "$E30_JOB" "$(hostname)")"
        return 0
    fi
    echo "CROSS $original (line $original_line): original on ${original_host:-<not run yet>} ($record_text), repeat on $(hostname)"
    E7_PREFIX=x_ E7_REFERENCE_LINE=$original_line e7_point "$@"
}

e7_extra() {        # KIND LABEL CASE K STEPS WARMUP TIMEOUT
    local kind=$1 label=$2 case_name=$3 slabs=$4 steps=$5 warmup=$6 limit=$7
    local yaml=cases/aligned/$case_name/case.yaml map pin=""
    map=$(e7_device_map "$slabs")
    [ "$slabs" -eq 1 ] && [ -n "${CPUL[0]:-}" ] && pin="taskset -c ${CPUL[0]} "
    echo; echo "##### EXTRA $kind $label case=$case_name K=$slabs steps=$steps host=$(hostname) ($(date +%T))"
    e7_index "$label" "$kind" extra "$case_name" "$slabs" 0 - - "$label"
    case "$kind" in
        anatomy)        # items 5 + 6: per-kernel GPU time, install kernels, density copy, defrag, host loop submit / wait;
                        # item 11: every loop row in traces/<label>/loop_trace.csv (the [loop] lines reset them)
            E30_PREFIX="${pin}env V7_LOOP_TRACE=1" \
                E30_WRAPPER_ARGS="--obj-cache $NODE_CACHE --defrag-log --loop-trace $SHM/traces/$label" \
                e30_run "$label" "$limit" --optional -- --case "$yaml" --weights "$(e7_ones "$slabs")" --device-map "$map" \
                $COMMON --max-steps "$steps" --warmup "$warmup" --anatomy --no-seam-check ;;
        fulltrace)      # every kernel's ticks for every step (the anatomy samples one frame per defrag)
            E30_PREFIX="${pin}" e30_run "$label" "$limit" --optional -- --case "$yaml" --weights "$(e7_ones "$slabs")" \
                --device-map "$map" $COMMON --max-steps "$steps" --warmup "$warmup" --step-trace "$SHM/traces/$label" \
                --step-trace-detail full --no-seam-check ;;
        soak)           # item 7: invariants after a long run, pool demand per 1000 frames, per-defrag pool watermarks
            E30_PREFIX="${pin}env V7_POOL_PEAKS=1" E30_WRAPPER_ARGS="--obj-cache $NODE_CACHE --defrag-log" \
                e30_run "$label" "$limit" --optional -- --case "$yaml" --weights "$(e7_ones "$slabs")" --device-map "$map" \
                $COMMON --max-steps "$steps" --warmup "$warmup" ;;
        selftest)       # every E7 output path once (anatomy_all, defrag log, [loop], loop rows, pool series) on a small case
            E30_PREFIX="${pin}env V7_LOOP_TRACE=1 V7_POOL_PEAKS=1" \
                E30_WRAPPER_ARGS="--obj-cache $NODE_CACHE --defrag-log --loop-trace $SHM/traces/$label" \
                e30_run "$label" "$limit" --optional -- --case "$yaml" --weights "$(e7_ones "$slabs")" --device-map "$map" \
                $COMMON --max-steps "$steps" --warmup "$warmup" --anatomy --no-seam-check
            [ "$E30_DRY" = 1 ] || $PY - "$E30_RESULTS" "$label" <<'PYEOF'
import json, sys
rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
row = next((row for row in rows if row["label"] == sys.argv[2]), None)
if row is None:
    print("SELFTEST: no row"); sys.exit(0)
checks = {"status pass": row["status"] == "pass", "anatomy_frames": bool(row.get("anatomy_frames")),
          "anatomy paired": row.get("anatomy_unpaired") == 0, "install_sum": any("install_sum" in frame for frame in row.get("anatomy_frames", [])),
          "defrag_reports": bool(row.get("defrag_reports")), "defrag_times gpu": any(entry.get("gpu_us") is not None for entry in row.get("defrag_times", [])),
          "loop_intervals": bool(row.get("loop_intervals")), "pool_series": bool(row.get("pool_series")),
          "loop_trace": bool((row.get("loop_trace") or {}).get("rows")),
          "loop_trace complete": (row.get("loop_trace") or {}).get("rows") == row.get("total_steps"),
          "stage epochs": all((stage or {}).get("epoch") for stage in (row.get("stages") or {}).values()),
          "telemetry_loop": bool(row.get("telemetry_loop"))}
print("SELFTEST " + ("OK" if all(checks.values()) else "INCOMPLETE") + ": " + " ".join(f"{name}={'ok' if value else 'MISSING'}" for name, value in checks.items()))
PYEOF
            ;;
        *) echo "unknown extra kind $kind" ;;
    esac
}

e7_job_end() {
    local status=0
    e7_glcache_check job_end
    if [ -n "${E7_MEMPID:-}" ]; then kill "$E7_MEMPID" 2>/dev/null; wait "$E7_MEMPID" 2>/dev/null; E7_MEMPID=""; fi
    e7_memory_window job 0 "$(date +%s.%N)"
    e7_provenance_end || status=3
    e30_finish
    e7_cleanup
    return $status
}
