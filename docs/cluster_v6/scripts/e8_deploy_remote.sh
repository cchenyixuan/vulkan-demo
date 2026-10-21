#!/bin/bash
# e8_deploy_remote.sh — deploy the E8 tag into ~/run/vulkan-demo-e8 on the N56 login node (never on a compute node).
# Nothing is submitted; the E8 cases' particle files are generated separately (e8_gen_remote.sh).
#
# Uploaded to ~/run/tools beforehand (built on the dev box with core.autocrlf=false, every file LF):
#   vd_e8_<SHA>.tar.gz   git archive --prefix=vulkan-demo-e8/ <tag> experiment/__init__.py experiment/v7 experiment/seam_audit
#                        materials utils/__init__.py utils/geometry/{__init__.py,_demo_cavity_case_aligned.py}
#                        remote/bringup_check_v6.py docs/cluster_v6/scripts docs/cluster_v6/e8_jobs
#                        cases/aligned/cavity2d_n1000/{case.yaml,generate.txt} cases/aligned/cavity3d_n{192,244,312,392}
#                        cases/aligned/cavity3d_n192_k{2,4,8} cases/aligned/cavity3d_n48
#   e8_sha.txt           the 40-hex SHA of the tag's commit
# Layout: the deployment's cases/ becomes a symbolic link to ~/run/vulkan-demo-v7rc1/cases, the shared case store of
# the E7 deployments (rc2 links it the same way: one copy of every particle file and one .obj cache key per case);
# the E8 case directories of the archive (case.yaml + generate.txt) are moved into that store, new directories only
# (an existing directory there is left alone and compared).
# Usage: [SHA=<40 hex>] [EXPECT_TREE=<sha>] [UPDATE=1] bash e8_deploy_remote.sh   (UPDATE=1 moves an existing deployment aside)
set -u
SHA=${SHA:-$(cat "$HOME/run/tools/e8_sha.txt" 2>/dev/null | tr -d '[:space:]')}
EXPECT_TREE=${EXPECT_TREE:?"the experiment/v7 tree hash of the tag (git rev-parse <tag>:experiment/v7)"}
DEST=$HOME/run/vulkan-demo-e8
STORE=$HOME/run/vulkan-demo-v7rc1
TOOLS=$HOME/run/tools
TARBALL=$TOOLS/vd_e8_$SHA.tar.gz
E8_CASES="cavity3d_n192 cavity3d_n244 cavity3d_n312 cavity3d_n392 cavity3d_n192_k2 cavity3d_n192_k4 cavity3d_n192_k8 cavity3d_n48"
FAILS=0
fail() { echo "FAIL: $*"; FAILS=$((FAILS + 1)); }
T0=$(date +%s)
echo "=== e8 deploy $(date) host $(hostname): $SHA -> $DEST (tree $EXPECT_TREE)"

# ---------------------------------------------------------------- guards
[[ "$SHA" =~ ^[0-9a-f]{40}$ ]] || { echo "ABORT: '$SHA' is not a 40-character sha"; exit 1; }
[ -s "$TARBALL" ] || { echo "ABORT: no $TARBALL"; exit 1; }
[ -d "$STORE/cases/aligned" ] || { echo "ABORT: no case store $STORE/cases/aligned"; exit 1; }
QUEUE=$(timeout 60 squeue -u scxm138 -h -o '%i|%j|%T|%Z|%o') || { echo "ABORT: squeue failed"; exit 1; }
if echo "$QUEUE" | grep -F "vulkan-demo-e8"; then echo "ABORT: a job of $DEST is queued or running"; exit 1; fi
echo "queue: $(echo "$QUEUE" | grep -c .) jobs of scxm138, none of $DEST"
if [ -e "$DEST" ]; then
    if [ "${UPDATE:-0}" = 1 ]; then
        ASIDE=$DEST.old.$(date +%Y%m%d_%H%M%S); mv "$DEST" "$ASIDE" || { echo "ABORT: cannot move $DEST aside"; exit 1; }
        echo "existing deployment moved to $ASIDE"
    else
        echo "ABORT: $DEST exists (UPDATE=1 moves it aside)"; exit 1
    fi
fi

# ---------------------------------------------------------------- extract
ARCHIVE_COMMIT=$(gzip -dc "$TARBALL" | git get-tar-commit-id)
[ "$ARCHIVE_COMMIT" = "$SHA" ] || { echo "ABORT: archive commit $ARCHIVE_COMMIT != $SHA"; exit 1; }
mkdir -p "$DEST" || { echo "ABORT: cannot create $DEST"; exit 1; }
tar xzf "$TARBALL" --strip-components=1 -C "$DEST" || { echo "ABORT: cannot extract $TARBALL"; exit 1; }
echo "extracted: $(find "$DEST" -type f | wc -l) files, $(du -sh "$DEST" | cut -f1)"
CR=$(find "$DEST" -type f \( -name '*.py' -o -name '*.sh' -o -name '*.sbatch' -o -name '*.yaml' -o -name '*.txt' \) -exec grep -l $'\r' {} + 2>/dev/null | head -3)
[ -z "$CR" ] && echo "CR check OK" || fail "CR in: $CR"

# ---------------------------------------------------------------- the case store
for name in $E8_CASES cavity2d_n1000; do
    if [ -d "$STORE/cases/aligned/$name" ]; then
        if cmp -s "$DEST/cases/aligned/$name/case.yaml" "$STORE/cases/aligned/$name/case.yaml"; then
            echo "case $name: already in the store, case.yaml identical"
        else
            fail "case $name: the store's case.yaml differs from the archive's"
        fi
    else
        mv "$DEST/cases/aligned/$name" "$STORE/cases/aligned/$name" || fail "cannot move $name into the store"
        echo "case $name: moved into the store (new)"
    fi
done
rm -rf "$DEST/cases"
ln -s ../vulkan-demo-v7rc1/cases "$DEST/cases" || fail "cannot link cases"
[ -f "$DEST/cases/aligned/cavity3d_n192/case.yaml" ] && echo "cases link OK" || fail "cases link"
[ -f "$DEST/cases/aligned/cavity2d_n1000/domain.obj" ] && echo "bring-up case has its particle files" || fail "cavity2d_n1000 particle files missing in the store"

# ---------------------------------------------------------------- provenance files
echo "$SHA tarball_sha256=$(sha256sum "$TARBALL" | cut -c1-64) deployed=$(date -Is) solver_tag=e8-3d-p1 base=v7-rc3" > "$DEST/COMMIT"
echo "$SHA" > "$DEST/SCRIPTS_COMMIT"
(cd "$DEST" && find . -type f ! -name MANIFEST.sha256 ! -name SCRIPTS.sha256 ! -name COMMIT ! -path '*/__pycache__/*' | sort | xargs sha256sum) > "$DEST/MANIFEST.sha256"
(cd "$DEST" && find docs/cluster_v6/scripts docs/cluster_v6/e8_jobs remote -type f ! -path '*/__pycache__/*' | sort | xargs sha256sum) > "$DEST/SCRIPTS.sha256"
echo "COMMIT: $(cat "$DEST/COMMIT")"; echo "MANIFEST.sha256: $(wc -l < "$DEST/MANIFEST.sha256") files; SCRIPTS.sha256: $(wc -l < "$DEST/SCRIPTS.sha256") files"

# ---------------------------------------------------------------- checks (CPU)
source "$TOOLS/env.sh" > /dev/null 2>&1 || { echo "ABORT: env script failed"; exit 1; }
cd "$DEST" || exit 1
sha256sum --quiet -c MANIFEST.sha256 && echo "MANIFEST OK" || fail "MANIFEST.sha256"
sha256sum --quiet -c SCRIPTS.sha256 && echo "SCRIPTS OK" || fail "SCRIPTS.sha256"
python docs/cluster_v6/scripts/tree_hash.py experiment/v7 --expect "$EXPECT_TREE" && echo "experiment/v7 tree OK ($EXPECT_TREE)" || fail "experiment/v7 tree"
python -m py_compile docs/cluster_v6/scripts/run_chain_v6.py docs/cluster_v6/scripts/parse_run_v6.py docs/cluster_v6/scripts/e8_dump_state.py \
    docs/cluster_v6/scripts/e8_field_compare.py docs/cluster_v6/scripts/e8_vram.py docs/cluster_v6/scripts/e8_plan.py \
    experiment/seam_audit/dump_state.py experiment/v7/_run_v7_chain_bench.py && echo "py_compile OK" || fail "py_compile"
python experiment/v7/_test_chain_cuts.py | tail -1 | grep -q "ALL PASS" && echo "_test_chain_cuts ALL PASS (floor 3)" || fail "_test_chain_cuts"
python docs/cluster_v6/scripts/e8_vram.py --validate | grep "largest relative error" || fail "e8_vram"
for name in $E8_CASES; do
    python utils/geometry/_demo_cavity_case_aligned.py $(grep -o -- '--dimension 3 --n [0-9]* --n-x [0-9]* --border 4' "cases/aligned/$name/generate.txt") --out /dev/null --check-only > /dev/null \
        && echo "generator check-only OK: $name" || fail "generator check-only $name"
done
bash -n docs/cluster_v6/e8_jobs/e8_p1.sbatch && echo "sbatch syntax OK" || fail "bash -n e8_p1.sbatch"
# dry run of the job on the login node (no GPU work, every command printed)
DRY_LOG=$HOME/run/logs/e8_deploy_dry_$(date +%Y%m%d_%H%M%S).log
E30_LOCAL=1 E30_DRY=1 E30_REPO=$DEST E30_PY=python E30_EXPECT_COMMIT=${SHA:0:12} E7_EXPECTED_TREE=$EXPECT_TREE E7_SOLVER_TAG=e8p1 \
    E7_GLCACHE_BASE=/dev/shm/scxm138 E7_STATE=/dev/shm/scxm138/e8_state_dry E7_OBJ_CACHE=$HOME/run/e7_objcache E30_SHM_ROOT=/dev/shm/scxm138/e8_dry \
    timeout 900 bash docs/cluster_v6/e8_jobs/e8_p1.sbatch > "$DRY_LOG" 2>&1
DRY_RUNS=$(grep -c "^DRY: .*run_chain_v6.py" "$DRY_LOG"); DRY_STEPS=$(grep -c "^=== STEP" "$DRY_LOG")
grep -q "ABORT\|HARD FAILURE" "$DRY_LOG" && fail "dry run aborted (see $DRY_LOG)" || echo "dry run OK: $DRY_RUNS chain runs, $DRY_STEPS steps ($DRY_LOG)"
rm -rf /dev/shm/scxm138/e8_dry /dev/shm/scxm138/e8_state_dry 2>/dev/null
# test-only submission
TEST_ONLY=$(timeout 60 sbatch --test-only --export=ALL,E30_EXPECT_COMMIT=${SHA:0:12},E7_EXPECTED_TREE=$EXPECT_TREE docs/cluster_v6/e8_jobs/e8_p1.sbatch 2>&1 | tail -1)
echo "$TEST_ONLY"
echo "$TEST_ONLY" | grep -q "to start at" || fail "test-only did not accept the job: $TEST_ONLY"
for other in "$HOME/run/vulkan-demo-v7rc2" "$HOME/run/vulkan-demo-v7rc3"; do
    [ -z "$(find "$other" -newer "$DEST/COMMIT" ! -path '*/__pycache__*' 2>/dev/null | head -1)" ] && echo "$(basename "$other") untouched" || fail "$other has entries newer than this deploy"
done
echo
if [ "$FAILS" -eq 0 ]; then
    echo "=== DEPLOY OK in $(( $(date +%s) - T0 ))s. Next: e8_gen_remote.sh (particle files + caches), then"
    echo "    cd $DEST && sbatch --export=ALL,E30_EXPECT_COMMIT=${SHA:0:12},E7_EXPECTED_TREE=$EXPECT_TREE docs/cluster_v6/e8_jobs/e8_p1.sbatch"
else
    echo "=== DEPLOY HAS $FAILS FAILURE(S): do not submit"; exit 2
fi
