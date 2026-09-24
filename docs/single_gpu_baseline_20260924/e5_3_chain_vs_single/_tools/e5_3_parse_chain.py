"""Parse one _run_v5_chain_bench.py log into a JSON record (stdout)."""
import json
import re
import sys
import time

log_path, trial, tag = sys.argv[1], int(sys.argv[2]), sys.argv[3]
text = open(log_path, encoding="utf-8", errors="replace").read()
record = {"kind": "chain_k1", "tag": tag, "trial": trial, "log": log_path,
          "status": "failed", "epoch_end": time.time()}
m = re.search(r"STEADY \(post-warmup (\d+)\): (\d+) steps in ([\d.]+)s = ([\d.]+) fps", text)
if m:
    record.update({"warmup": int(m.group(1)), "steady_frames": int(m.group(2)),
                   "steady_s": float(m.group(3)), "steady_fps": float(m.group(4))})
m = re.search(r"TOTAL: (\d+) steps in ([\d.]+)s = ([\d.]+) fps", text)
if m:
    record.update({"total_frames": int(m.group(1)), "total_s": float(m.group(2)),
                   "total_fps": float(m.group(3))})
m = re.search(r"final: total=([\d,]+)", text)
if m:
    record["alive_total"] = int(m.group(1).replace(",", ""))
m = re.search(r"\(expected ([\d,]+)\)", text)
if m:
    record["expected"] = int(m.group(1).replace(",", ""))
    if "alive_total" in record:
        record["drift"] = record["alive_total"] - record["expected"]
m = re.search(r"sim0 \(dev(\d+)\): alive=([\d,]+)", text)
if m:
    record["device"] = int(m.group(1))
    record.setdefault("alive_total", int(m.group(2).replace(",", "")))
if "steady_fps" in record and record.get("drift", 0) == 0 and "Traceback" not in text:
    record["status"] = "ok"
print(json.dumps(record))
