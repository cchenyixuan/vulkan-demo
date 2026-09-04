#!/usr/bin/env bash
# install_kit.sh — set up the V5 solver on the AIR-GAPPED GPU server.
# Run from inside the extracted kit directory:
#     tar xzf v5_offline_kit_*.tar.gz && cd v5_offline_kit && bash install_kit.sh
#
# Creates ~/vulkan-demo with:  repo checkout + precompiled .spv + .venv from
# the bundled wheels (no network access needed at any point).
set -euo pipefail

KIT_DIR="$(cd "$(dirname "$0")" && pwd)"
TARGET="${TARGET:-$HOME/vulkan-demo}"
PYTHON="${PYTHON:-python3}"

echo "[install] python: $($PYTHON --version 2>&1)  target: $TARGET"
TAG=$($PYTHON -c 'import sys; print(f"cp{sys.version_info[0]}{sys.version_info[1]}")')
WHEELS="$KIT_DIR/wheels/$TAG"
if [ ! -d "$WHEELS" ]; then
    echo "[install] ERROR: no bundled wheels for $TAG (have: $(ls "$KIT_DIR/wheels"))"
    echo "[install] set PYTHON=<other python3.x> and retry"
    exit 1
fi

echo "[install] 1/4 extract repo"
mkdir -p "$TARGET"
tar -xf "$KIT_DIR/repo.tar" -C "$TARGET"

echo "[install] 2/4 install precompiled SPIR-V"
mkdir -p "$TARGET/experiment/v5/shaders/spv"
cp "$KIT_DIR"/spv/*.spv "$TARGET/experiment/v5/shaders/spv/"
echo "[install]   $(ls "$TARGET/experiment/v5/shaders/spv" | wc -l) .spv installed"

echo "[install] 3/4 create venv from bundled wheels"
if ! $PYTHON -m venv "$TARGET/.venv" 2>/dev/null; then
    echo "[install] WARNING: 'python3 -m venv' unavailable (missing python3-venv?)"
    echo "[install] falling back to --user install"
    $PYTHON -m pip install --user --no-index --find-links "$WHEELS" \
        numpy PyYAML matplotlib scipy vulkan
    VENV_PY="$PYTHON"
else
    "$TARGET/.venv/bin/pip" install --no-index --find-links "$WHEELS" \
        --upgrade pip setuptools wheel >/dev/null
    "$TARGET/.venv/bin/pip" install --no-index --find-links "$WHEELS" \
        numpy PyYAML matplotlib scipy vulkan
    VENV_PY="$TARGET/.venv/bin/python"
fi

echo "[install] 4/4 environment self-check"
cp "$KIT_DIR/bringup_check.py" "$TARGET/remote/bringup_check.py" 2>/dev/null || true
cd "$TARGET"
"$VENV_PY" remote/bringup_check.py --stage env || {
    echo "[install] env check FAILED — see bringup_report.txt"; exit 1; }

echo
echo "[install] DONE. Next:"
echo "    cd $TARGET"
echo "    .venv/bin/python remote/bringup_check.py --stage all   # case gen + GPU smokes"
