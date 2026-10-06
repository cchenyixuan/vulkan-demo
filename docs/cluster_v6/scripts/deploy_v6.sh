#!/bin/bash
# deploy_v6.sh SHA — E30: deploy commit SHA of github.com/cchenyixuan/vulkan-demo into
# ~/run/vulkan-demo-v6 from the GitHub archive (via gh-proxy; a full checkout on JuiceFS costs
# ~0.65 s per file and git fetch through the proxy failed in September). Only the paths the v6
# runtime needs are extracted: experiment/__init__.py, experiment/v6, materials and the E30 case
# yamls. Writes COMMIT (sha, tarball sha256, time) and MANIFEST.sha256 (every deployed file).
# The v5 checkout ~/run/vulkan-demo is never touched. Run on the login node.
#   UPDATE=1 deploy_v6.sh SHA   overwrites an existing deployment.
#   TARBALL=<path> deploy_v6.sh SHA   uses an uploaded archive instead of the download (gh-proxy ran at
#   ~85 KB/s on 2026-10-05): git -c core.autocrlf=false archive --format=tar.gz --prefix=vulkan-demo-SHA/
#   SHA <paths> on the dev box; git get-tar-commit-id still identifies the commit.
set -u
SHA=${1:?full 40-character commit sha}
DEST=${DEST:-$HOME/run/vulkan-demo-v6}
STAGE=/dev/shm/scxm138/deploy_$SHA
CASES="lid_driven_cavity_2d_n1000 lid_driven_cavity_2d_16m lid_driven_cavity_2d_64m lid_driven_cavity_2d_128m cavity3d_weak4_k2_8m_b4 cavity3d_weak8_k8_64m_b4"
mkdir -p "$STAGE" && cd "$STAGE" || { echo "ABORT: no stage dir"; exit 2; }
if [ -n "${TARBALL:-}" ]; then
    cp "$TARBALL" vd.tar.gz || { echo "ABORT: cannot copy $TARBALL"; exit 2; }
fi
if [ ! -s vd.tar.gz ]; then
    timeout 600 curl -fsSL -o vd.tar.gz "https://gh-proxy.com/https://github.com/cchenyixuan/vulkan-demo/archive/$SHA.tar.gz" \
        || { echo "ABORT: download failed"; rm -f vd.tar.gz; exit 2; }
fi
ARCHIVE_COMMIT=$(gzip -dc vd.tar.gz | git get-tar-commit-id)
[ "$ARCHIVE_COMMIT" = "$SHA" ] || { echo "ABORT: archive commit $ARCHIVE_COMMIT != $SHA"; exit 2; }
TOP=$(tar tzf vd.tar.gz | head -1 | cut -d/ -f1)
MEMBERS="$TOP/experiment/__init__.py $TOP/experiment/v6 $TOP/materials"
for case_name in $CASES; do MEMBERS="$MEMBERS $TOP/cases/$case_name/case.yaml"; done
if [ -e "$DEST/COMMIT" ] && [ "${UPDATE:-0}" != 1 ]; then
    echo "ABORT: $DEST already deployed ($(cat "$DEST/COMMIT")); UPDATE=1 to overwrite"; exit 2
fi
mkdir -p "$DEST"
tar xzf vd.tar.gz -C "$DEST" --strip-components=1 $MEMBERS || { echo "ABORT: extract failed"; exit 2; }
echo "$SHA tarball_sha256=$(sha256sum vd.tar.gz | cut -d' ' -f1) deployed=$(date -Is)" > "$DEST/COMMIT"
cd "$DEST" || exit 2
find . -type f ! -name MANIFEST.sha256 ! -name COMMIT ! -name SCRIPTS.sha256 ! -path './docs/*' ! -path './remote/*' \
    | sort | xargs sha256sum > MANIFEST.sha256
cp "$STAGE/vd.tar.gz" "$HOME/run/tools/vulkan-demo-$SHA.tar.gz" 2>/dev/null
echo "deployed $(wc -l < MANIFEST.sha256) files to $DEST"
cat COMMIT
