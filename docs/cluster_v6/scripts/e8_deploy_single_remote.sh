#!/bin/bash
# e8_deploy_single_remote.sh — deploy the v7_single solver (branch v7-single, experiment/v7_single on v7-rc1's
# experiment/v7 and SPIR-V) into ~/run/vulkan-demo-v7single on the N56 login node, for the E8 single-GPU-solver
# comparison (e7_lib.sh e8_single_ratio: run by the E8 job through e8_single_bench.py from this directory).
# Nothing is submitted. Login node only.
#
# Uploaded to ~/run/tools beforehand (built on the dev box with core.autocrlf=false, every file LF):
#   vd_v7single_<SHA>.tar.gz   git archive --prefix=vulkan-demo-v7single/ <v7-single head> experiment/__init__.py
#                              experiment/v7 experiment/v7_single materials
#   v7single_sha.txt           the 40-hex SHA of that commit
# The deployment's cases/ links the shared store (~/run/vulkan-demo-v7rc1/cases) like the E8 deployment: the same
# particle files and .obj cache keys. JuiceFS creates files slowly: /dev/shm stage + 16 parallel copies.
# Usage: [SHA=<40 hex>] [UPDATE=1] bash e8_deploy_single_remote.sh
set -u
SHA=${SHA:-$(cat "$HOME/run/tools/v7single_sha.txt" 2>/dev/null | tr -d '[:space:]')}
EXPECT_TREE=82dd6a740fa2a97a44e6a31d75392f75e486163b       # v7-rc1's experiment/v7 (the branch v7-single never edits it)
DEST=$HOME/run/vulkan-demo-v7single
TOOLS=$HOME/run/tools
TARBALL=$TOOLS/vd_v7single_$SHA.tar.gz
STAGE=/dev/shm/scxm138/v7single_stage_$SHA
FAILS=0
fail() { echo "FAIL: $*"; FAILS=$((FAILS + 1)); }
T0=$(date +%s)
echo "=== v7single deploy $(date) host $(hostname): $SHA -> $DEST"
[[ "$SHA" =~ ^[0-9a-f]{40}$ ]] || { echo "ABORT: '$SHA' is not a 40-character sha"; exit 1; }
[ -s "$TARBALL" ] || { echo "ABORT: no $TARBALL"; exit 1; }
QUEUE=$(timeout 60 squeue -u scxm138 -h -o '%i|%j|%T|%Z|%o') || { echo "ABORT: squeue failed"; exit 1; }
if echo "$QUEUE" | grep -F "vulkan-demo-e8"; then echo "ABORT: an E8 job is queued or running (it reads this deployment)"; exit 1; fi
if [ -e "$DEST" ]; then
    if [ "${UPDATE:-0}" = 1 ]; then
        ASIDE=$DEST.old.$(date +%Y%m%d_%H%M%S); mv "$DEST" "$ASIDE" || { echo "ABORT: cannot move $DEST aside"; exit 1; }
        echo "existing deployment moved to $ASIDE"
    else
        echo "ABORT: $DEST exists (UPDATE=1 moves it aside)"; exit 1
    fi
fi
ARCHIVE_COMMIT=$(gzip -dc "$TARBALL" | git get-tar-commit-id)
[ "$ARCHIVE_COMMIT" = "$SHA" ] || { echo "ABORT: archive commit $ARCHIVE_COMMIT != $SHA"; exit 1; }
rm -rf "$STAGE"; mkdir -p "$STAGE" || { echo "ABORT: cannot create $STAGE"; exit 1; }
tar xzf "$TARBALL" --strip-components=1 -C "$STAGE" || { echo "ABORT: cannot extract $TARBALL"; exit 1; }
mkdir -p "$DEST" || { echo "ABORT: cannot create $DEST"; exit 1; }
(cd "$STAGE" && find . -type d | sed 's|^\./||' | grep -v '^\.$' | sort) | while read -r directory; do mkdir -p "$DEST/$directory"; done
(cd "$STAGE" && find . -type f | sed 's|^\./||' | sort) > "$STAGE.files"
TC=$(date +%s)
(cd "$STAGE" && xargs -a "$STAGE.files" -P 16 -I{} cp -p "{}" "$DEST/{}")
echo "copied $(wc -l < "$STAGE.files") files in $(( $(date +%s) - TC )) s"
MISSING=$(cd "$STAGE" && while read -r file; do [ -f "$DEST/$file" ] || echo "$file"; done < "$STAGE.files" | head -3)
[ -z "$MISSING" ] && echo "copy complete" || fail "files missing after the copy: $MISSING"
rm -rf "$STAGE" "$STAGE.files"
CR=$(find "$DEST" -type f \( -name '*.py' -o -name '*.sh' -o -name '*.yaml' -o -name '*.txt' \) -exec grep -l $'\r' {} + 2>/dev/null | head -3)
[ -z "$CR" ] && echo "CR check OK" || fail "CR in: $CR"
ln -s ../vulkan-demo-v7rc1/cases "$DEST/cases" || fail "cannot link cases"
[ -f "$DEST/cases/aligned/cavity3d_n192/domain.obj" ] && echo "cases link OK (8M cube present)" || fail "cases link / 8M particle files"
echo "$SHA tarball_sha256=$(sha256sum "$TARBALL" | cut -c1-64) deployed=$(date -Is) solver_tag=v7-single base=v7-rc1" > "$DEST/COMMIT"
(cd "$DEST" && find . -type f ! -name MANIFEST.sha256 ! -name COMMIT ! -path '*/__pycache__/*' | sort | xargs sha256sum) > "$DEST/MANIFEST.sha256"
echo "COMMIT: $(cat "$DEST/COMMIT")"; echo "MANIFEST.sha256: $(wc -l < "$DEST/MANIFEST.sha256") files"
source "$TOOLS/env.sh" > /dev/null 2>&1 || { echo "ABORT: env script failed"; exit 1; }
cd "$DEST" || exit 1
sha256sum --quiet -c MANIFEST.sha256 && echo "MANIFEST OK" || fail "MANIFEST.sha256"
python "$HOME/run/vulkan-demo-e8/docs/cluster_v6/scripts/tree_hash.py" experiment/v7 --expect "$EXPECT_TREE" && echo "experiment/v7 tree OK (v7-rc1)" || fail "experiment/v7 tree"
python -m py_compile experiment/v7_single/_run_single_bench.py experiment/v7_single/utils/simulator_single.py && echo "py_compile OK" || fail "py_compile"
python -c "import experiment.v7_single.utils.simulator_single, experiment.v7_single.utils.spirv_manifest; print('imports OK')" || fail "imports"
python experiment/v7_single/_run_single_bench.py --help | head -3 | cut -c1-120
for spv in correction_density.comp.spv force.comp.spv predict.comp.spv update_voxel.comp.spv defrag.comp.spv; do
    cmp -s "experiment/v7/shaders/spv/$spv" "$HOME/run/vulkan-demo-e8/experiment/v7/shaders/spv/$spv" && echo "SPIR-V $spv = e8's" || fail "SPIR-V $spv differs from e8's"
done
echo
if [ "$FAILS" -eq 0 ]; then echo "=== DEPLOY OK in $(( $(date +%s) - T0 ))s"; else echo "=== DEPLOY HAS $FAILS FAILURE(S)"; exit 2; fi
