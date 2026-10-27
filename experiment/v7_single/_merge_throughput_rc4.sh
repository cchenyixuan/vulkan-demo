#!/bin/bash
# E9 item 7, fig:baseline: merge the re-measured points (throughput_rc4b: v7_single d1/d2 + v7_chain d2 at 2M, 4M gen, 8M,
# 16M, 32M; prefix order, 600 W) into r3 (everything else unchanged: 1M, the old 4M, 3-D 8M, the V / Vx rows), summarize,
# redraw Figure 1 (single panel, S rows) as single_gpu_throughput.{pdf,png}; the previous figure is kept as
# single_gpu_throughput_e8_atomic.* in both figure directories.
set -eu
cd /c/Users/cchen/PycharmProjects/vulkan-demo-v7-single
PY=/c/Users/cchen/PycharmProjects/vulkan-demo/.venv/Scripts/python.exe
M=logs/v7_single/throughput_rc4_merged
mkdir -p "$M"
$PY - <<'EOF'
import json, pathlib
r3 = pathlib.Path("logs/v7_single/throughput_r3/results.jsonl")
new = pathlib.Path("logs/v7_single/throughput_rc4b/results.jsonl")
out = pathlib.Path("logs/v7_single/throughput_rc4_merged/results.jsonl")
REPLACED = {("2m", "v7_single"), ("4m_gen", "v7_single"), ("8m", "v7_single"), ("16m", "v7_single"), ("32m", "v7_single"),
            ("2m", "v7_chain"), ("4m_gen", "v7_chain"), ("8m", "v7_chain"), ("16m", "v7_chain"), ("32m", "v7_chain")}
kept, dropped = [], 0
for line in r3.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    if (row.get("tag"), row.get("solver")) in REPLACED:
        dropped += 1
        continue
    kept.append(line)
added = [l for l in new.read_text(encoding="utf-8").splitlines() if l.strip()]
out.write_text("\n".join(kept + added) + "\n", encoding="utf-8")
print(f"merged: kept {len(kept)} r3 rows, dropped {dropped} (re-measured), added {len(added)} rc4b rows -> {out}")
EOF
cat logs/v7_single/throughput_r3/telemetry.csv logs/v7_single/throughput_rc4b/telemetry.csv > "$M/telemetry.csv"
PYTHONIOENCODING=utf-8 $PY _summarize_v7_single_campaign.py --dir "$M" 2>&1 | tail -3
for ext in pdf png; do
  [ -f manuscripts/fig/single_gpu_throughput_e8_atomic.$ext ] || cp -p manuscripts/fig/single_gpu_throughput.$ext manuscripts/fig/single_gpu_throughput_e8_atomic.$ext
done
PYTHONIOENCODING=utf-8 $PY _plot_single_baseline_paper.py --dir "$M" --solver v7_single --figure-a-name single_gpu_throughput --no-figure-b --single-panel --font-size 8 --exclude-sizes 4m 2>&1 | tail -3
ls -la manuscripts/fig/single_gpu_throughput*.png
MAIN=/c/Users/cchen/PycharmProjects/vulkan-demo/manuscripts/fig
for ext in pdf png; do
  [ -f $MAIN/single_gpu_throughput_e8_atomic.$ext ] || cp -p $MAIN/single_gpu_throughput.$ext $MAIN/single_gpu_throughput_e8_atomic.$ext 2>/dev/null || true
  cp -p manuscripts/fig/single_gpu_throughput.$ext $MAIN/single_gpu_throughput.$ext
done
echo "figure copied to $MAIN"
