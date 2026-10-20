#!/bin/bash
# gen_aligned_cases.sh CASE_NAME ... — E7: write the particle files (.obj) of cases/aligned/<CASE_NAME> in the
# deployed checkout and build their run_chain_v6.py --obj-cache files, on the login node (CPU only).
#
#   - The generator line is the one in the case's generate.txt (utils/geometry/_demo_cavity_case_aligned.py
#     ... --objs-only; case.yaml is never written), run with this environment's python instead of the dev
#     box's .venv/Scripts/python.exe. The generator streams the lattice in x chunks and checks the column
#     alignment itself; a case whose domain.obj already exists is skipped unless E7_GEN_FORCE=1.
#   - Then obj_cache_build.py fills E7_OBJ_CACHE (default ~/run/e7_objcache) with the same .npy files the
#     jobs' --obj-cache would write (same key: resolved path, size, mtime), so no billed run parses text.
#     Regenerating a case changes its files' mtime, i.e. the keys: the old cache files are then dead weight.
#   - E7_GEN_PARALLEL cases at a time (default 3), one log per case under E7_GEN_LOG
#     (default ~/run/logs/e7_setup/<date>), a summary line per case on stdout.
# The python environment lives in the login node's /dev/shm, which is wiped when the last session of the user
# ends: run this from a session that stays open until it returns (it prints a line per finished case).
set -u
REPO=${E30_REPO:-$HOME/run/vulkan-demo-v7rc1}
CACHE=${E7_OBJ_CACHE:-$HOME/run/e7_objcache}
LOGDIR=${E7_GEN_LOG:-$HOME/run/logs/e7_setup/$(date +%Y%m%d)}
PARALLEL=${E7_GEN_PARALLEL:-3}
SOLVER=${E30_SOLVER:-v7}
[ "$#" -ge 1 ] || { echo "usage: gen_aligned_cases.sh CASE_NAME ..."; exit 2; }
cd "$REPO" || { echo "ABORT: no checkout at $REPO"; exit 2; }
mkdir -p "$LOGDIR" "$CACHE" || { echo "ABORT: cannot create $LOGDIR / $CACHE"; exit 2; }
command -v python > /dev/null || { echo "ABORT: no python (source ~/run/tools/env.sh first)"; exit 2; }

generate_one() {
    local name=$1 directory=cases/aligned/$1 log=$LOGDIR/$1.log line started return_code
    started=$(date +%s)
    (
        echo "=== $name $(date -Is) host $(hostname)"
        [ -f "$directory/case.yaml" ] && [ -f "$directory/generate.txt" ] \
            || { echo "ABORT: $directory has no case.yaml / generate.txt"; exit 2; }
        line=$(grep -E 'utils/geometry/_demo_cavity_case_aligned\.py .*--objs-only' "$directory/generate.txt" | head -1)
        [ -n "$line" ] || { echo "ABORT: no generator line with --objs-only in $directory/generate.txt"; exit 2; }
        line=${line#*python.exe }                     # drop the dev box's interpreter path
        case "$line" in utils/geometry/_demo_cavity_case_aligned.py*) ;; *) echo "ABORT: unexpected line: $line"; exit 2;; esac
        case "$line" in *"--out cases/aligned/$name "*|*"--out cases/aligned/$name") ;; *) echo "ABORT: --out is not $directory: $line"; exit 2;; esac
        if [ -f "$directory/domain.obj" ] && [ "${E7_GEN_FORCE:-0}" != 1 ]; then
            echo "domain.obj exists: generation skipped"
        else
            yaml_before=$(sha256sum "$directory/case.yaml" | cut -d' ' -f1)
            echo "+ python -u $line"
            # shellcheck disable=SC2086
            timeout 7200 python -u $line || { echo "ABORT: generator rc=$?"; exit 3; }
            [ "$(sha256sum "$directory/case.yaml" | cut -d' ' -f1)" = "$yaml_before" ] \
                || { echo "ABORT: case.yaml changed during generation"; exit 3; }
        fi
        ls -l "$directory"/*.obj
        # the generator writes in place and never re-reads its files: count the lines against generate.txt's
        # first line ("... = N, wall W, lid L = T particles") so that a truncated file cannot pass
        counts=$(python -c 'import re, sys
match = re.search(r"= ([\d,]+), wall ([\d,]+), lid ([\d,]+) = ([\d,]+) particles", open(sys.argv[1], encoding="utf-8").readline())
print(" ".join(value.replace(",", "") for value in match.groups()) if match else "")' "$directory/generate.txt")
        [ -n "$counts" ] || { echo "ABORT: no particle counts in $directory/generate.txt"; exit 5; }
        read -r fluid wall lid total <<< "$counts"
        for pair in "domain.obj:$fluid" "wall.obj:$wall" "wall_top.obj:$lid" "frame.obj:21"; do
            file=${pair%%:*}; expected=${pair##*:}
            lines=$(wc -l < "$directory/$file")
            [ "$lines" = "$expected" ] || { echo "ABORT: $file has $lines lines, expected $expected (E7_GEN_FORCE=1 regenerates)"; exit 5; }
        done
        echo "line counts OK: fluid $fluid wall $wall lid $lid (total $total) + frame 21"
        echo "+ obj_cache_build.py --solver $SOLVER --obj-cache $CACHE --case $directory/case.yaml"
        timeout 7200 python -u docs/cluster_v6/scripts/obj_cache_build.py --solver "$SOLVER" --obj-cache "$CACHE" \
            --case "$directory/case.yaml" || { echo "ABORT: cache build rc=$?"; exit 4; }
        echo "=== $name done $(date -Is)"
    ) > "$log" 2>&1
    return_code=$?
    echo "CASE $name rc=$return_code $(( $(date +%s) - started ))s obj=$(du -cb "$directory"/*.obj 2>/dev/null | tail -1 | cut -f1)B log=$log"
    return $return_code
}
export -f generate_one
export REPO CACHE LOGDIR SOLVER

echo "=== gen_aligned_cases $(date -Is): $# cases, $PARALLEL at a time, cache $CACHE, logs $LOGDIR"
printf '%s\n' "$@" | xargs -P "$PARALLEL" -I{} bash -c 'generate_one "$@"' _ {}
status=$?
echo "=== gen_aligned_cases done $(date -Is) xargs status $status; cache: $(du -sh "$CACHE" | cut -f1)"
exit $status
