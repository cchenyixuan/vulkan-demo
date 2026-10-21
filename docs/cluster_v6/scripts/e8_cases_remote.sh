#!/bin/bash
# e8_cases_remote.sh — put the E8 case definitions (case.yaml + generate.txt of the archive) into the shared case
# store ~/run/vulkan-demo-v7rc1/cases/aligned (new directories only; an existing one is compared) and start their
# particle-file generation (e8_gen_remote.sh) in a tmux session, so that the long generation does not wait for the
# deployment. Login node only.
# Usage: [SHA=<40 hex>] bash e8_cases_remote.sh
set -u
SHA=${SHA:-$(cat "$HOME/run/tools/e8_sha.txt" 2>/dev/null | tr -d '[:space:]')}
STORE=$HOME/run/vulkan-demo-v7rc1
TARBALL=$HOME/run/tools/vd_e8_$SHA.tar.gz
STAGE=/dev/shm/scxm138/e8_cases_$SHA
E8_CASES="cavity3d_n152 cavity3d_n192 cavity3d_n244 cavity3d_n312 cavity3d_n392 cavity3d_n416 cavity3d_n496 cavity3d_n192_k2 cavity3d_n192_k4 cavity3d_n192_k8 cavity3d_n48"
FAILS=0
fail() { echo "FAIL: $*"; FAILS=$((FAILS + 1)); }
[ -s "$TARBALL" ] || { echo "ABORT: no $TARBALL"; exit 1; }
rm -rf "$STAGE"; mkdir -p "$STAGE"
tar xzf "$TARBALL" --strip-components=1 -C "$STAGE" --wildcards 'vulkan-demo-e8/cases/aligned/*' || { echo "ABORT: cannot extract the cases"; exit 1; }
for name in $E8_CASES; do
    if [ -d "$STORE/cases/aligned/$name" ]; then
        cmp -s "$STAGE/cases/aligned/$name/case.yaml" "$STORE/cases/aligned/$name/case.yaml" \
            && echo "case $name: already in the store, case.yaml identical" || fail "case $name: the store's case.yaml differs"
    else
        mkdir -p "$STORE/cases/aligned/$name" \
            && cp -p "$STAGE/cases/aligned/$name/case.yaml" "$STAGE/cases/aligned/$name/generate.txt" "$STORE/cases/aligned/$name/" \
            && echo "case $name: added to the store" || fail "cannot add $name"
    fi
done
rm -rf "$STAGE"
[ "$FAILS" -eq 0 ] || { echo "=== CASES HAVE $FAILS FAILURE(S)"; exit 2; }
GEN_LOG=$HOME/run/logs/e8_gen_$(date +%Y%m%d_%H%M%S).log
mkdir -p "$HOME/run/logs"
GEN_CASES=${E8_GEN_CASES:-}        # the cases to generate now (default: the list of e8_gen_remote.sh; phase 2: cavity3d_n152)
tmux new-session -d -s e8gen "bash -lc 'bash $HOME/run/tools/e8_gen_remote.sh $GEN_CASES > $GEN_LOG 2>&1'" \
    && echo "=== CASES OK; generation started in tmux session e8gen, log $GEN_LOG" || fail "tmux start"
sleep 5; tmux ls 2>&1 | head -3; head -5 "$GEN_LOG" 2>/dev/null
