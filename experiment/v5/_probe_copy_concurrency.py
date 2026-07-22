"""UNDER REVIEW — memcpy concurrency probe (2026-07-22).

Measures how many ghost-worker memcpys are in flight concurrently during
a K=8 VGPU run, from worker.timestamps host stamps (wait_ns = copy start,
copy_ns = copy end, absolute perf_counter_ns).

KNOWN SUSPECT (flagged by user, under audit): the stamps are taken by
PYTHON THREADS — after the numpy copy (GIL released during the copy) the
thread must REACQUIRE the GIL to record copy_ns; under 14-thread + main
submit-loop contention the recorded interval may substantially exceed
the true copy duration. Conclusions drawn from per-copy durations are
provisional until the audit closes.
"""
import sys, pathlib
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiment.v5.utils.case_loader_v5 import load_case_v5
from experiment.v5.utils.orchestrator_v5 import ChainOrchestratorV5
from experiment.v5.utils.partition_v5 import compute_chain_partition
from experiment.v5.utils.simulator_v5 import SphSimulatorV5
from experiment.v5.utils.vulkan_context_v5 import VulkanContextV5


def main() -> int:
    K = 8
    case = load_case_v5("cases/lid_driven_cavity_2d/case.yaml")
    chain = compute_chain_partition(case, [1.0] * K, pool_safety=1.2)
    device_map = [i % 2 for i in range(K)]
    contexts, sims = [], []
    for i in range(K):
        contexts.append(VulkanContextV5.create(
            device_index=device_map[i], application_name=f"conc_probe_{i}"))
        sims.append(SphSimulatorV5(contexts[-1], chain.slabs[i],
                                   sync_scheme="per-direction"))
    try:
        with ChainOrchestratorV5(sims, defrag_cadence=10**9) as orch:
            orch.bootstrap_all()
            orch.run_pipelined(800, depth=2, warmup=0)
            intervals = []
            for w in orch.workers:
                for frame, st in w.timestamps.items():
                    if 300 <= frame < 700 and "wait_ns" in st and "copy_ns" in st:
                        intervals.append((st["wait_ns"], st["copy_ns"]))
    finally:
        for sim in sims:
            sim.destroy()
        for ctx in contexts:
            ctx.destroy()

    intervals.sort()
    events = []
    for start, end in intervals:
        events.append((start, +1))
        events.append((end, -1))
    events.sort()
    active = max_active = 0
    time_at_level = {}
    prev_t = events[0][0]
    for t, delta in events:
        if t > prev_t and active > 0:
            time_at_level[active] = time_at_level.get(active, 0) + (t - prev_t)
        active += delta
        max_active = max(max_active, active)
        prev_t = t
    total_busy = sum(time_at_level.values())
    mean_conc = sum(l * ns for l, ns in time_at_level.items()) / total_busy
    copy_total = sum(e - s for s, e in intervals)
    wall = max(e for _, e in intervals) - min(s for s, _ in intervals)
    print(f"copies: {len(intervals)}  max_concurrent: {max_active}  "
          f"mean_concurrency: {mean_conc:.2f}")
    print(f"levels: " + ", ".join(f"{l}:{100*ns/total_busy:.0f}%"
                                  for l, ns in sorted(time_at_level.items())))
    print(f"copy_sum {copy_total/1e6:.1f} ms vs wall {wall/1e6:.1f} ms "
          f"(ratio {copy_total/wall:.2f}); mean copy "
          f"{copy_total/len(intervals)/1e3:.0f} us")
    return 0


if __name__ == "__main__":
    sys.exit(main())
