"""
stress.py - claude/wall_store_report.md section 12 (round 6): the overnight stress test of the final defaults (git tag
crossing-wall-store-tuned) on the production runners, each run under a watchdog. No solver code is changed: the
watchdog wraps three methods of ChainOrchestratorV7 from outside (the frame submission, the frame wait, the
pipelined loop's defrag callback) for the run's process only.

    # one run under the watchdog (box_run / hydrostatic / cavity_runner_entry, their own arguments after --)
    python -m experiment.wall_crossing.stress wrap --run-dir <dir> --module experiment.wall_crossing.box_run \
        [--deadline-seconds S] -- --case ... --name ... --device 0,1 --time 1000
    # a plan of runs one after another, each in its own process, with an outer supervisor
    python -m experiment.wall_crossing.stress queue --plan <plan.json> --out <dir> [--deadline-unix T]
    # nvidia-smi of every card every 30 s
    python -m experiment.wall_crossing.stress smi --out <file.csv> --hours 9
    # the volume-constraint-off variants of the cavity cases (cases/stress6_<case>_volume_off/case.yaml)
    python -m experiment.wall_crossing.stress make-cases

Watchdog (wrap). Every frame's frame_done is awaited in 1 s slices (as ChainOrchestratorV7._wait_frame does). A frame
not done after --stall-seconds (120) ends the run: <run-dir>/watchdog.json gets the semaphore autopsy
(vkGetSemaphoreCounterValue, never blocks) and every slab's sync state, the last submitted frame and the frames in
flight, the last 64 frames' submit / completion times, three nvidia-smi samples of every card of the run, a
responsiveness probe on every slab's transfer and compute queue (soak.py) and every Python thread's stack; then the
process exits with code 2 (os._exit: a wedged device can hang the teardown). A monitor thread does the same when
nothing progresses for --stall-seconds + 60 s anywhere else (a host wait outside the frame loop; before the first
frame the budget is --boot-seconds). At every defrag boundary (frames drained) one row goes to <run-dir>/defrag.csv:
the interval's wall time and steps per second (the sampling of the runner excluded), every slab's global status
summed (alive, overflow counters, far migrations, GPU stamp errors, the fallback counters, the largest speed), the
host stamp errors, the process's resident memory and its device-local memory (VK_EXT_memory_budget heap usage, this
process only). --deadline-seconds: at the first defrag boundary after it the runner's own stop is raised
(box_run.StopCondition: watch_stop.json, exit 3; cavity_runner.StopRun: a normal end).

Queue. Plan: {"jobs": [{"label", "module", "args" (with {run_dir} / {name} / {out}), "run_dir" (default
<out>/box_run/<name>), "env" {...}, "env_preset" ("cavity_release": the cavity runner's release environment),
"device_uuids" [...], "deadline_seconds", "budget_seconds", "not_after_unix", "keep_npz"}]}. Each job runs in its own
process (<run-dir>/stdout.log); the supervisor kills the process tree when the run's heartbeat file (the monitor
thread, every 5 s) is older than 400 s or the job exceeds budget_seconds, after writing external_kill.json (three
nvidia-smi samples). One line per job goes to <out>/status.jsonl. A file STOP in <out> ends the queue before the
next job; no job starts after --deadline-unix or after its not_after_unix. final.npz / stop_state.npz of a run are
deleted afterwards (frames are not kept) unless the job says keep_npz or the run stalled.
"""
from __future__ import annotations

import argparse
import collections
import ctypes
import importlib
import json
import math
import os
import pathlib
import subprocess
import sys
import threading
import time
import traceback

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

STALL_EXIT_CODE = 2
HEARTBEAT_NAME = "stress_heartbeat.json"
DEFRAG_COLUMNS = ("pid", "frame", "unix", "interval_s", "frames", "steps_per_second", "alive", "maximum_velocity",
                  "overflow", "far_migration", "stamp_error_gpu", "stamp_error_host", "gradient_fallbacks",
                  "correction_fallbacks", "report_overflow", "report_far_migration", "rss_mb", "vram_process_mb")
SMI_FIELDS = ("timestamp,index,uuid,temperature.gpu,power.draw,clocks.sm,clocks.mem,utilization.gpu,memory.used,"
              "pstate,clocks_throttle_reasons.active")


# ---- process and card state ---------------------------------------------------------------------------------------

def resident_megabytes() -> float:
    """The process's working set (Windows) or VmRSS (Linux), in MiB; nan when unavailable."""
    try:
        if os.name == "nt":
            class Counters(ctypes.Structure):
                _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                            ("PeakPagefileUsage", ctypes.c_size_t)]
            counters = Counters()
            counters.cb = ctypes.sizeof(Counters)
            kernel32 = ctypes.WinDLL("kernel32")
            kernel32.GetCurrentProcess.restype = ctypes.c_void_p
            kernel32.K32GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
            kernel32.K32GetProcessMemoryInfo.restype = ctypes.c_int
            if kernel32.K32GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
                return counters.WorkingSetSize / 2 ** 20
            return math.nan
        for line in pathlib.Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    except Exception:                       # noqa: BLE001 - a diagnostic must not raise
        pass
    return math.nan


def device_local_megabytes(context) -> float:
    """This process's usage of the device-local heaps (VK_EXT_memory_budget heapUsage), MiB; nan when unavailable."""
    try:
        from vulkan import (VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, VkPhysicalDeviceMemoryBudgetPropertiesEXT,
                            VkPhysicalDeviceMemoryProperties2, vkGetPhysicalDeviceMemoryProperties2)
        budget = VkPhysicalDeviceMemoryBudgetPropertiesEXT()
        properties = VkPhysicalDeviceMemoryProperties2(pNext=budget)
        vkGetPhysicalDeviceMemoryProperties2(context.physical_device, properties)
        heaps = properties.memoryProperties
        total = 0
        for index in range(heaps.memoryHeapCount):
            if heaps.memoryHeaps[index].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT:
                total += int(budget.heapUsage[index])
        return total / 2 ** 20
    except Exception:                       # noqa: BLE001
        return math.nan


def dashed_uuid(uuid: str) -> str:
    uuid = uuid.replace("-", "").lower()
    return f"GPU-{uuid[:8]}-{uuid[8:12]}-{uuid[12:16]}-{uuid[16:20]}-{uuid[20:]}"


def smi_samples(uuids: list, count: int = 3) -> dict:
    from experiment.wall_crossing.soak import nvidia_smi_samples
    return {uuid: nvidia_smi_samples(uuid, count) for uuid in uuids}


# ---- the watchdog -------------------------------------------------------------------------------------------------

class Watchdog:
    def __init__(self, run_dir: pathlib.Path, stall_seconds: float, boot_seconds: float, deadline_unix: float | None):
        self.run_dir = run_dir
        self.stall_seconds = stall_seconds
        self.boot_seconds = boot_seconds
        self.deadline_unix = deadline_unix
        self.orchestrator = None
        self.submitted = {}
        self.last_submitted = -1
        self.last_completed = -1
        self.frames_completed = 0
        self.recent = collections.deque(maxlen=64)
        self.started = time.perf_counter()
        self.last_progress = self.started
        self.interval_start = None
        self.interval_frame = 0
        self.vram = {"at": 0.0, "value": math.nan}
        self.fired = False
        self.finished = False
        self.lock = threading.Lock()
        self.stop_factory = None
        new_file = not (run_dir / "defrag.csv").exists()
        self.defrag_file = open(run_dir / "defrag.csv", "a", encoding="utf-8", newline="\n")
        if new_file:
            self.defrag_file.write(",".join(DEFRAG_COLUMNS) + "\n")
            self.defrag_file.flush()

    # -- frame events
    def on_submit(self, orchestrator, frame: int) -> None:
        self.orchestrator = orchestrator
        self.submitted[frame] = time.perf_counter()
        self.last_submitted = frame

    def on_complete(self, frame: int) -> None:
        now = time.perf_counter()
        submitted = self.submitted.pop(frame, math.nan)
        self.recent.append((frame, round(submitted - self.started, 4), round(now - self.started, 4)))
        self.last_completed = frame
        self.frames_completed += 1
        self.last_progress = now

    # -- the end of a run that stopped progressing
    def diagnostics(self, reason: str, frame: int | None, sim_index: int | None) -> dict:
        orchestrator = self.orchestrator
        record = {"reason": reason, "unix": time.time(), "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                  "frame_waited": frame, "sim_waited": sim_index, "last_submitted": self.last_submitted,
                  "last_completed": self.last_completed, "frames_completed": self.frames_completed,
                  "in_flight": sorted(self.submitted),
                  "seconds_since_progress": round(time.perf_counter() - self.last_progress, 1),
                  "recent": list(self.recent), "rss_mb": resident_megabytes()}
        uuids = []
        if orchestrator is not None:
            try:
                record["autopsy"] = orchestrator.dump_stall_autopsy(frame if frame is not None
                                                                    else self.last_completed + 1)
            except Exception as error:      # noqa: BLE001
                record["autopsy"] = f"autopsy failed: {error!r}"
            record["sync_states"] = []
            for sim in orchestrator.sims:
                try:
                    record["sync_states"].append(str(sim.sync_state()))
                except Exception as error:  # noqa: BLE001
                    record["sync_states"].append(f"failed: {error!r}")
                try:
                    from experiment.wall_crossing.run_log import vulkan_device_uuid
                    uuids.append(vulkan_device_uuid(sim.ctx))
                except Exception:           # noqa: BLE001
                    pass
        record["device_uuids"] = uuids
        record["nvidia_smi"] = smi_samples(uuids) if uuids else {}
        if orchestrator is not None:
            from experiment.wall_crossing.soak import queue_probe
            record["probes"] = []
            for sim in orchestrator.sims:
                context = sim.ctx
                record["probes"].append({
                    "transfer_queue": queue_probe(context, context.transfer_queue, context.transfer_command_pool, 10.0),
                    "compute_queue": queue_probe(context, context.compute_queue, context.command_pool, 3.0)})
            record["sync_states_after_probes"] = [str(sim.sync_state()) for sim in orchestrator.sims]
        stacks = {}
        for thread_id, stack_frame in sys._current_frames().items():
            stacks[str(thread_id)] = "".join(traceback.format_stack(stack_frame))
        record["threads"] = stacks
        return record

    def fire(self, reason: str, frame: int | None, sim_index: int | None) -> None:
        with self.lock:
            if self.fired or self.finished:
                return
            self.fired = True
        print(f"[stress] WATCHDOG: {reason} (frame {frame}, sim {sim_index}); collecting diagnostics", flush=True)
        try:
            record = self.diagnostics(reason, frame, sim_index)
        except Exception as error:          # noqa: BLE001
            record = {"reason": reason, "frame_waited": frame, "diagnostics_error": repr(error)}
        (self.run_dir / "watchdog.json").write_text(json.dumps(record, indent=1, default=str), encoding="utf-8")
        print(f"[stress] WATCHDOG record written to {self.run_dir / 'watchdog.json'}; ending the run", flush=True)
        self.write_wrapper_record({"exit_code": STALL_EXIT_CODE, "status": f"stall: {reason}"})
        sys.stdout.flush()
        os._exit(STALL_EXIT_CODE)

    def monitor(self) -> None:
        while not self.finished:
            time.sleep(5.0)
            now = time.perf_counter()
            try:
                (self.run_dir / HEARTBEAT_NAME).write_text(json.dumps({
                    "unix": time.time(), "last_submitted": self.last_submitted, "last_completed": self.last_completed,
                    "frames_completed": self.frames_completed, "seconds_since_progress": round(now - self.last_progress, 1),
                    "rss_mb": resident_megabytes(), "pid": os.getpid()}), encoding="utf-8")
            except OSError:
                pass
            budget = self.boot_seconds if self.frames_completed == 0 else self.stall_seconds + 60.0
            if not self.finished and now - self.last_progress > budget:
                self.fire(f"no progress for {now - self.last_progress:.0f} s (monitor thread)",
                          self.last_completed + 1 if self.frames_completed else None, None)

    # -- defrag boundary
    def on_defrag(self, orchestrator, frame: int, report: list) -> None:
        now = time.perf_counter()
        if self.interval_start is None:
            self.interval_start = self.started
        frames = frame - self.interval_frame
        interval = now - self.interval_start
        statuses = [sim.readback_global_status() for sim in orchestrator.sims]
        if time.time() - self.vram["at"] > 10.0:
            self.vram = {"at": time.time(),
                         "value": sum(device_local_megabytes(sim.ctx) for sim in orchestrator.sims)}

        def total(key: str) -> int:
            return int(sum(status.get(key, 0) for status in statuses))
        row = {"pid": os.getpid(), "frame": frame, "unix": round(time.time(), 3), "interval_s": round(interval, 4),
               "frames": frames, "steps_per_second": round(frames / interval, 2) if interval > 0 else "",
               "alive": total("alive_particle_count"),
               "maximum_velocity": max(float(status.get("maximum_velocity", 0.0)) for status in statuses),
               "overflow": int(sum(value for status in statuses for key, value in status.items()
                                   if key.startswith("overflow_"))),
               "far_migration": total("far_migration_count"), "stamp_error_gpu": total("stamp_error_count"),
               "stamp_error_host": int(sum(getattr(worker, "stamp_error_count", 0) for worker in orchestrator.workers)),
               "gradient_fallbacks": total("wall_crossing_gradient_fallback_count"),
               "correction_fallbacks": total("correction_fallback_count"),
               "report_overflow": int(sum(value for entry in report for key, value in entry.items()
                                          if key.startswith("overflow_"))),
               "report_far_migration": int(sum(entry.get("far_migration", 0) for entry in report)),
               "rss_mb": round(resident_megabytes(), 1), "vram_process_mb": round(self.vram["value"], 1)}
        self.defrag_file.write(",".join(str(row[name]) for name in DEFRAG_COLUMNS) + "\n")
        self.defrag_file.flush()
        self.interval_frame = frame

    def write_wrapper_record(self, extra: dict) -> None:
        record = {"pid": os.getpid(), "unix": time.time(), "seconds": round(time.perf_counter() - self.started, 1),
                  "frames_completed": self.frames_completed, "last_completed": self.last_completed,
                  "rss_mb": resident_megabytes(), **extra}
        with open(self.run_dir / "wrapper.jsonl", "a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, default=str) + "\n")


def install(watchdog: Watchdog) -> None:
    from experiment.v7.utils.orchestrator_v7 import ChainOrchestratorV7
    original_submit = ChainOrchestratorV7._submit_frame
    original_run = ChainOrchestratorV7.run_pipelined
    test_fire_frame = int(os.environ.get("STRESS_TEST_FIRE_AT_FRAME", "-1"))

    def submit_frame(self, frame_n: int) -> None:
        watchdog.on_submit(self, frame_n)
        return original_submit(self, frame_n)

    def wait_frame(self, frame_n: int, stall_timeout_s: float = 120.0) -> None:
        # ChainOrchestratorV7._wait_frame with the watchdog's end instead of its RuntimeError
        started = time.perf_counter()
        for sim_index, sim in enumerate(self.sims):
            frame_semaphore, target_value = sim.sync.frame_done_op(frame_n)
            while not sim.wait_semaphore(frame_semaphore, target_value, timeout_ns=int(1e9)):
                self._raise_if_worker_died(frame_n)
                if time.perf_counter() - started > watchdog.stall_seconds:
                    watchdog.fire(f"frame_done not reached after {watchdog.stall_seconds:.0f} s", frame_n, sim_index)
        self._raise_if_worker_died(frame_n)
        on_frame_done = getattr(self, "on_frame_done", None)
        if on_frame_done is not None:
            on_frame_done(frame_n, None)
        watchdog.on_complete(frame_n)
        if frame_n == test_fire_frame:          # STRESS_TEST_FIRE_AT_FRAME: try the diagnostics on a live run
            watchdog.fire("test of the diagnostics (STRESS_TEST_FIRE_AT_FRAME)", frame_n + 1, 0)

    def run_pipelined(self, max_steps: int, *, depth: int = 2, warmup: int = 0, on_defrag=None,
                      stall_timeout_s: float = 120.0) -> dict:
        watchdog.orchestrator = self
        watchdog.interval_start = time.perf_counter()
        watchdog.interval_frame = 0

        def wrapped(frame: int, report: list) -> None:
            watchdog.on_defrag(self, frame, report)
            if on_defrag is not None:
                on_defrag(frame, report)
            watchdog.last_progress = time.perf_counter()
            watchdog.interval_start = time.perf_counter()
            if watchdog.deadline_unix is not None and time.time() >= watchdog.deadline_unix \
                    and watchdog.stop_factory is not None:
                raise watchdog.stop_factory()
        return original_run(self, max_steps, depth=depth, warmup=warmup, on_defrag=wrapped,
                            stall_timeout_s=stall_timeout_s)

    ChainOrchestratorV7._submit_frame = submit_frame
    ChainOrchestratorV7._wait_frame = wait_frame
    ChainOrchestratorV7.run_pipelined = run_pipelined


def wrap(arguments, rest: list) -> int:
    run_dir = pathlib.Path(arguments.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(REPOSITORY_ROOT)
    deadline = time.time() + arguments.deadline_seconds if arguments.deadline_seconds else None
    watchdog = Watchdog(run_dir, arguments.stall_seconds, arguments.boot_seconds, deadline)
    install(watchdog)
    threading.Thread(target=watchdog.monitor, name="stress-watchdog", daemon=True).start()
    watchdog.write_wrapper_record({"event": "start", "module": arguments.module, "argv": rest,
                                   "deadline_unix": deadline})
    code: object = 0
    status = "ok"
    try:
        if arguments.module.endswith("cavity_runner_entry"):
            from experiment.validation.cavity_runner import StopRun
            watchdog.stop_factory = StopRun
            import runpy
            sys.argv = [arguments.module] + rest
            runpy.run_module(arguments.module, run_name="__main__", alter_sys=True)
        else:
            module = importlib.import_module(arguments.module)
            stop = getattr(module, "StopCondition", None)
            if stop is not None:
                watchdog.stop_factory = lambda: stop("wall-clock deadline of the stress segment")
            sys.argv = [arguments.module] + rest
            code = module.main()
    except SystemExit as exit_request:
        code = exit_request.code
    except BaseException as error:          # noqa: BLE001 - recorded
        code = 1
        status = f"error: {error!r}"
        traceback.print_exc()
    watchdog.finished = True
    code = code if isinstance(code, int) else (0 if code is None else 1)
    watchdog.write_wrapper_record({"event": "end", "exit_code": code, "status": status})
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


# ---- the nvidia-smi log -------------------------------------------------------------------------------------------

def smi(arguments) -> int:
    out = pathlib.Path(arguments.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    new_file = not out.exists()
    end = time.time() + arguments.hours * 3600.0
    with open(out, "a", encoding="utf-8", newline="\n") as handle:
        if new_file:
            handle.write("unix," + SMI_FIELDS + "\n")
        while time.time() < end:
            if (out.parent / "STOP_SMI").exists():
                break
            started = time.time()
            try:
                result = subprocess.run(["nvidia-smi", f"--query-gpu={SMI_FIELDS}", "--format=csv,noheader,nounits"],
                                        capture_output=True, text=True, timeout=20)
                for line in result.stdout.strip().splitlines():
                    handle.write(f"{started:.1f},{line.strip()}\n")
                if result.returncode != 0:
                    handle.write(f"{started:.1f},ERROR {result.stderr.strip()!r}\n")
            except Exception as error:      # noqa: BLE001
                handle.write(f"{started:.1f},ERROR {error!r}\n")
            handle.flush()
            time.sleep(max(1.0, arguments.interval - (time.time() - started)))
    return 0


# ---- the queue ----------------------------------------------------------------------------------------------------

def kill_tree(process_id: int) -> None:
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process_id), "/T", "/F"], capture_output=True)
    else:
        subprocess.run(["kill", "-9", str(process_id)], capture_output=True)


def substitute(value: str, mapping: dict) -> str:
    for key, replacement in mapping.items():
        value = value.replace("{" + key + "}", str(replacement))
    return value


def queue(arguments) -> int:
    plan = json.loads(pathlib.Path(arguments.plan).read_text(encoding="utf-8"))
    out = pathlib.Path(arguments.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "queue.pid").write_text(str(os.getpid()), encoding="utf-8")
    status_path = out / "status.jsonl"
    sequence = sum(1 for line in status_path.read_text(encoding="utf-8").splitlines() if line.strip()) \
        if status_path.exists() else 0

    def status_line(record: dict) -> None:
        with open(status_path, "a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record) + "\n")

    for job in plan["jobs"]:
        now = time.time()
        if (out / "STOP").exists():
            status_line({"event": "stopped (STOP file)", "time": time.strftime("%Y-%m-%d %H:%M:%S")})
            break
        if arguments.deadline_unix and now >= arguments.deadline_unix:
            status_line({"event": "deadline", "time": time.strftime("%Y-%m-%d %H:%M:%S")})
            break
        if job.get("not_after_unix") and now >= job["not_after_unix"]:
            continue
        sequence += 1
        name = f"{sequence:04d}_{job['label']}"
        mapping = {"name": name, "out": out}
        run_dir = pathlib.Path(substitute(job.get("run_dir", "{out}/box_run/{name}"), mapping)).resolve()
        mapping["run_dir"] = run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        environment = dict(os.environ)
        if job.get("env_preset") == "cavity_release":
            from experiment.validation import cavity_campaign
            environment.update(cavity_campaign.environment("release", "v7"))
        environment.update({"OPENBLAS_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8",
                            "VK_LOADER_LAYERS_DISABLE": "VK_LAYER_KHRONOS_validation",
                            "WALL_CROSSING_LOG_ROOT": str(out)})
        environment.update({key: substitute(str(value), mapping) for key, value in job.get("env", {}).items()})
        command = [sys.executable, "-m", "experiment.wall_crossing.stress", "wrap", "--run-dir", str(run_dir),
                   "--module", job["module"], "--stall-seconds", str(arguments.stall_seconds),
                   "--boot-seconds", str(job.get("boot_seconds", arguments.boot_seconds))]
        if job.get("deadline_seconds"):
            command += ["--deadline-seconds", str(job["deadline_seconds"])]
        command += ["--"] + [substitute(str(item), mapping) for item in job["args"]]
        started = time.time()
        record = {"seq": sequence, "name": name, "label": job["label"], "combo": job.get("combo"),
                  "start": time.strftime("%Y-%m-%d %H:%M:%S"), "start_unix": started, "run_dir": str(run_dir)}
        external = None
        with open(run_dir / "stdout.log", "a", encoding="utf-8", newline="\n") as log:
            process = subprocess.Popen(command, cwd=REPOSITORY_ROOT, env=environment, stdout=log,
                                       stderr=subprocess.STDOUT)
            heartbeat = run_dir / HEARTBEAT_NAME
            while process.poll() is None:
                time.sleep(5.0)
                beat = heartbeat.stat().st_mtime if heartbeat.exists() else started
                age = time.time() - max(beat, started)
                over_budget = job.get("budget_seconds") and time.time() - started > job["budget_seconds"]
                if age > arguments.external_seconds or over_budget:
                    external = (f"heartbeat {age:.0f} s old" if age > arguments.external_seconds
                                else f"over the budget of {job['budget_seconds']} s")
                    (run_dir / "external_kill.json").write_text(json.dumps({
                        "reason": external, "unix": time.time(),
                        "nvidia_smi": smi_samples(job.get("device_uuids", [])) if job.get("device_uuids") else {}},
                        indent=1), encoding="utf-8")
                    kill_tree(process.pid)
                    try:
                        process.wait(timeout=900)
                    except subprocess.TimeoutExpired:
                        external += "; the process did not exit within 900 s of the kill"
                    break
        code = process.poll()
        stalled = (run_dir / "watchdog.json").exists()
        if not (job.get("keep_npz") or stalled or external):
            for leftover in ("final.npz", "stop_state.npz"):
                (run_dir / leftover).unlink(missing_ok=True)
        record.update({"exit_code": code, "seconds": round(time.time() - started, 1),
                       "end": time.strftime("%Y-%m-%d %H:%M:%S"), "watchdog": stalled, "external_kill": external})
        status_line(record)
        print(f"[stress queue] {record['end']} {name}: exit {code}"
              + (" WATCHDOG" if stalled else "") + (f" KILLED ({external})" if external else ""), flush=True)
        if code not in (0, 3) or stalled or external:
            time.sleep(20)
    status_line({"event": "queue_done", "time": time.strftime("%Y-%m-%d %H:%M:%S")})
    return 0


# ---- the volume-constraint-off cavity cases -----------------------------------------------------------------------

VOLUME_OFF_BASES = ("lid_driven_cavity_2d_n250_xi0p001_eps0p0025_crossing",
                    "lid_driven_cavity_2d_n500_xi0p001_eps0p0025_crossing",
                    "cavity3d_n50_crossing", "cavity3d_n70_crossing")


def make_cases(arguments) -> int:
    for base in VOLUME_OFF_BASES:
        text = (REPOSITORY_ROOT / "cases" / base / "case.yaml").read_text(encoding="utf-8")
        lines = text.splitlines()
        index = next(position for position, line in enumerate(lines) if line.strip().startswith("volume_constraint:"))
        if lines[index + 1].strip() != "enabled: true":
            sys.exit(f"{base}: unexpected volume_constraint block {lines[index:index + 2]}")
        lines[index + 1] = lines[index + 1].replace("enabled: true", "enabled: false")
        header = (f"# STRESS VARIANT (report section 12, round 6) of cases/{base}/case.yaml: "
                  "numerics.volume_constraint enabled false; everything else identical.")
        target = REPOSITORY_ROOT / "cases" / f"stress6_{base}_volume_off" / "case.yaml"
        target.parent.mkdir(exist_ok=True)
        target.write_text(header + "\n" + "\n".join(lines) + "\n", encoding="utf-8", newline="\n")
        print(target.relative_to(REPOSITORY_ROOT).as_posix())
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action", required=True)
    wrap_parser = sub.add_parser("wrap")
    wrap_parser.add_argument("--run-dir", required=True)
    wrap_parser.add_argument("--module", required=True)
    wrap_parser.add_argument("--stall-seconds", type=float, default=120.0)
    wrap_parser.add_argument("--boot-seconds", type=float, default=900.0)
    wrap_parser.add_argument("--deadline-seconds", type=float, default=0.0)
    queue_parser = sub.add_parser("queue")
    queue_parser.add_argument("--plan", required=True)
    queue_parser.add_argument("--out", required=True)
    queue_parser.add_argument("--deadline-unix", type=float, default=0.0)
    queue_parser.add_argument("--stall-seconds", type=float, default=120.0)
    queue_parser.add_argument("--boot-seconds", type=float, default=900.0)
    queue_parser.add_argument("--external-seconds", type=float, default=400.0)
    smi_parser = sub.add_parser("smi")
    smi_parser.add_argument("--out", required=True)
    smi_parser.add_argument("--hours", type=float, required=True)
    smi_parser.add_argument("--interval", type=float, default=30.0)
    sub.add_parser("make-cases")
    if len(sys.argv) > 1 and sys.argv[1] == "wrap" and "--" in sys.argv:
        split = sys.argv.index("--")
        arguments = parser.parse_args(sys.argv[1:split])
        return wrap(arguments, sys.argv[split + 1:])
    arguments = parser.parse_args()
    if arguments.action == "queue":
        return queue(arguments)
    if arguments.action == "smi":
        return smi(arguments)
    if arguments.action == "make-cases":
        return make_cases(arguments)
    parser.error("wrap needs -- before the runner's arguments")
    return 2


if __name__ == "__main__":
    sys.exit(main())
