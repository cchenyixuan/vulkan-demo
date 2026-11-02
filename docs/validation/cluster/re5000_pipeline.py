"""re5000_pipeline.py (user 2026-11-02): the Re = 5000 cavity on the workstation's two RTX 5090, unattended, by the procedure
of the Re = 3200 runs (solver experiment/v7 = tree of tag v7-rc3, release switches, adami walls).

    1. warm_start_velocity.py: the finished Re 3200 1000^2 state -> initial velocities of the Re 5000 1000^2 case
    2. wait until the cards are free
    3. coarse: 1000^2 from them, to the runner's steady rule (1e-3 per window, then 20 t of averaging). User 2026-11-02, after
       an early stop at 3.3e-3 ran for 40 minutes: that stop came from the cold-started Re 3200 run and was never checked on
       this start. Snapshots are kept from t = 40, so the state the fine run starts from can still be chosen afterwards
    4. warm_start_velocity.py: the coarse final snapshot -> initial velocities of the 2500^2 case
    5. fine: 2500^2 K = 2 from them, every checkpoint kept, one every 5 t, for the 5090 + 7900 XTX continuation. At t = 80
       pipeline.log says the card may be swapped; the run goes on towards the full stop rule until it is
    6. stop. The mixed continuation needs the card swapped by hand: docs/validation/cluster/re5000_plan.md

Idempotent: a stage with its result.json / marker is skipped, so a rerun after a reboot continues (the runner resumes from
its newest checkpoint). A crash is retried with --resume, after the cards are idle again. An invariant violation stops the
pipeline: it is never retried. Two quick failures in a row (a set-up error) stop it too. One instance only (pipeline.pid).

    python docs/validation/cluster/re5000_pipeline.py          # detached: see re5000_plan.md
    python docs/validation/cluster/re5000_pipeline.py --check  # the commands and the current card / disk state, starts nothing
Log logs/validation/cavity_re5000_v7/pipeline.log, state pipeline_state/, runner output <run>.out.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT))

from experiment.validation.cavity_runner import INVARIANT_EXIT_CODE, expected_environments  # noqa: E402

RUNS = _ROOT / "logs/validation/cavity_re5000_v7"
STATE = RUNS / "pipeline_state"
LOG = RUNS / "pipeline.log"
PID_FILE = STATE / "pipeline.pid"
COARSE_CASE = "cases/lid_driven_cavity_2d_n1000_re5000_adami/case.yaml"
FINE_CASE = "cases/lid_driven_cavity_2d_n2500_re5000_adami/case.yaml"
COARSE_RUN = RUNS / "n1000_k2_float32_xi0p001_eps0p0025_adami_re5000"
FINE_RUN = RUNS / "n2500_k2_float32_xi0p001_eps0p0025_adami_re5000_warm"
WARM_FILE = RUNS / "warm_n2500_from_n1000_re5000.npz"
COARSE_SOURCE = _ROOT / "logs/validation/cavity_re3200_v7/n1000_k1_float32_xi0p001_eps0p0025_adami_re3200"
COARSE_WARM_FILE = RUNS / "warm_n1000_re5000_from_re3200.npz"
SWAP_READY = STATE / "ready_for_swap"
SWAP_READY_TIME = 80.0  # 1.6 x the Re 3200 t = 50, where its error against Marchi had flattened (tail tau 78 against 48.7)
# device map 0,1 = the display 5090, the headless 5090: anything else in the box and the runner refuses to start
DEVICE_UUIDS = "fb83f0dafd9e1d38dc05e7ae643e22a8,ae137c668a40f5acaab90a83f7cda175"
COMMON = ["--slabs", "2", "--device-map", "0,1", "--reynolds", "5000", "--keep-checkpoints", "0", "--expect", "release",
          "--solver", "v7", "--require-uuid", DEVICE_UUIDS]
# t_max is a cap, far above the estimates (fine steady near t = 125). The runner keeps snapshots from t_end - span = 40 on
COARSE_ARGUMENTS = ["--t-end", "60", "--t-max", "300", "--steady-tol", "1e-3", "--average-span", "20", "--initial-velocity", str(COARSE_WARM_FILE)]
FINE_ARGUMENTS = ["--t-end", "40", "--t-max", "300", "--average-span", "20", "--checkpoint-time", "5"]
SINGLE_CARD = False
# the ZeroTier machine's run (--single-card there), taken over by --take-over here; REMOTE_PW comes from the environment only
REMOTE_HOST = "10.147.20.1"
REMOTE_USER = "Administrator"
REMOTE_RUNS = "D:/vulkan-demo-re5000/logs/validation/cavity_re5000_v7"
REMOTE_FINE = "n2500_k1_float32_xi0p001_eps0p0025_adami_re5000_warm"
REMOTE_WARM = "warm_n2500_from_n1000_k1_re5000.npz"
REMOTE_POLL_SECONDS = 120
REMOTE_GIVE_UP_SECONDS = 1800
IDLE_MINUTES = 20
RETRY_IDLE_MINUTES = 3
HEADLESS_UTILIZATION_LIMIT = 5        # per cent; the display card also draws the desktop, hence its own limit
DISPLAY_UTILIZATION_LIMIT = 30
HEADLESS_MEMORY_LIMIT_MIB = 500       # 18 MiB idle; a paused job still holds its memory
MINIMUM_FREE_DISK_GB = 40
MAXIMUM_ATTEMPTS = 6
QUICK_FAILURE_SECONDS = 180
POLL_SECONDS = 30
HEARTBEAT_LOG_SECONDS = 3600


def configure_single_card(uuid: str) -> None:
    """The ZeroTier machine (user 2026-11-02): its one 5090 starts the work while the two local cards are busy. K = 1, no
    wait for idle cards, a checkpoint every 2 t so that handing over to the local K = 2 loses little, k1 run directories."""
    global SINGLE_CARD, COMMON, COARSE_RUN, FINE_RUN, WARM_FILE, FINE_ARGUMENTS
    SINGLE_CARD = True
    COMMON = ["--slabs", "1", "--device-map", "0", "--reynolds", "5000", "--keep-checkpoints", "0", "--expect", "release",
              "--solver", "v7", "--require-uuid", uuid]
    COARSE_RUN = RUNS / "n1000_k1_float32_xi0p001_eps0p0025_adami_re5000"
    FINE_RUN = RUNS / "n2500_k1_float32_xi0p001_eps0p0025_adami_re5000_warm"
    WARM_FILE = RUNS / "warm_n2500_from_n1000_k1_re5000.npz"
    FINE_ARGUMENTS = ["--t-end", "40", "--t-max", "300", "--average-span", "20", "--checkpoint-time", "2"]


def log(message: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def stop(reason: str) -> int:
    (STATE / "STOPPED").write_text(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {reason}\n", encoding="utf-8")
    log(f"STOPPED: {reason}")
    return 1


def process_alive(pid: int) -> bool:
    listing = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW)
    return str(pid) in listing.stdout


def card_states() -> list[tuple[int, int]]:
    """(utilization per cent, memory MiB) of nvidia-smi index 0 (display 5090) and 1 (headless 5090)."""
    text = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used", "--format=csv,noheader,nounits"], capture_output=True,
                          text=True, timeout=60, creationflags=subprocess.CREATE_NO_WINDOW).stdout
    rows = [[int(item) for item in line.split(",")] for line in text.strip().splitlines()]
    return [(rows[index][1], rows[index][2]) for index in range(2)]


def cards_idle() -> bool:
    (display_utilization, _), (headless_utilization, headless_memory) = card_states()
    return (display_utilization <= DISPLAY_UTILIZATION_LIMIT and headless_utilization <= HEADLESS_UTILIZATION_LIMIT
            and headless_memory <= HEADLESS_MEMORY_LIMIT_MIB)


def wait_for_idle(minutes: int) -> None:
    """Idle at every poll for the given minutes on end: a gap between two phases of another job must not start a run."""
    if SINGLE_CARD:
        return
    needed = minutes * 60 // POLL_SECONDS
    streak = 0
    last_report = 0.0
    log(f"waiting for {minutes} idle minutes on end (limits: display {DISPLAY_UTILIZATION_LIMIT} %, headless {HEADLESS_UTILIZATION_LIMIT} % and {HEADLESS_MEMORY_LIMIT_MIB} MiB)")
    while streak < needed:
        try:
            streak = streak + 1 if cards_idle() else 0
            if time.time() - last_report >= HEARTBEAT_LOG_SECONDS:
                last_report = time.time()
                log(f"cards {card_states()}, idle streak {streak * POLL_SECONDS // 60} min")
        except (subprocess.SubprocessError, OSError, ValueError, IndexError) as error:
            log(f"nvidia-smi failed: {error!r}")
            streak = 0
        if streak < needed:
            time.sleep(POLL_SECONDS)
    log("cards idle")


def runner_environment() -> dict:
    environment = {key: value for key, value in os.environ.items() if not key.startswith("V7_") and key != "REMOTE_PW"}
    environment.update(expected_environments("v7")["release"])
    environment.update({"VK_LOADER_LAYERS_DISABLE": "VK_LAYER_KHRONOS_validation", "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"})
    return environment


def announce_swap(run_dir: pathlib.Path) -> None:
    """Once, when the fine run passes SWAP_READY_TIME. The heartbeat may be caught mid-write: the next poll reads it again."""
    if SWAP_READY.exists():
        return
    try:
        heartbeat = json.loads((run_dir / "heartbeat.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if heartbeat.get("time", 0.0) >= SWAP_READY_TIME:
        SWAP_READY.write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"), encoding="utf-8")
        log(f"fine reached t = {heartbeat['time']:.2f}: the 7900 XTX may be swapped in now (re5000_plan.md); the run goes on until then")


def run_stage(name: str, case: str, run_dir: pathlib.Path, extra: list[str], announce: bool = False) -> int:
    """The runner until the run has its result.json. 0 = done; anything else stops the pipeline."""
    if (run_dir / "result.json").exists():
        log(f"{name}: result.json exists, skipped")
        return 0
    quick_failures = 0
    for attempt in range(MAXIMUM_ATTEMPTS):
        free = shutil.disk_usage(_ROOT).free / 1e9
        if free < MINIMUM_FREE_DISK_GB:
            return stop(f"{name}: {free:.0f} GB free on the disk")
        checkpoints = list((run_dir / "checkpoints").glob("c*.npz")) if run_dir.exists() else []
        if run_dir.exists() and not checkpoints:
            aside = run_dir.with_name(f"{run_dir.name}.failed{int(time.time())}")
            run_dir.rename(aside)
            log(f"{name}: no checkpoint in the failed run, moved aside to {aside.name}")
        command = [sys.executable, "-m", "experiment.validation.cavity_runner", "--case", case, "--run-dir", str(run_dir), *COMMON, *extra]
        if checkpoints:
            command.append("--resume")
        log(f"{name}: attempt {attempt}: {' '.join(command[1:])}")
        started = time.time()
        with open(RUNS / f"{run_dir.name}.out", "a", encoding="utf-8") as output:
            process = subprocess.Popen(command, cwd=_ROOT, env=runner_environment(), stdout=output, stderr=subprocess.STDOUT,
                                       creationflags=subprocess.CREATE_NO_WINDOW)
            last_report = time.time()
            while process.poll() is None:
                time.sleep(POLL_SECONDS)
                if announce:
                    announce_swap(run_dir)
                if time.time() - last_report >= HEARTBEAT_LOG_SECONDS:
                    last_report = time.time()
                    heartbeat = run_dir / "heartbeat.json"
                    log(f"{name}: {heartbeat.read_text(encoding='utf-8')[:200] if heartbeat.exists() else 'no heartbeat yet'}")
        elapsed = time.time() - started
        log(f"{name}: attempt {attempt} exited {process.returncode} after {elapsed / 3600:.2f} h")
        if process.returncode == 0 and (run_dir / "result.json").exists():
            result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
            log(f"{name}: done, status {result.get('status')}, steady {result.get('steady')}, stop step {result.get('stop_step')}")
            return 0 if result.get("status") == "complete" else stop(f"{name}: result status {result.get('status')}")
        if process.returncode == INVARIANT_EXIT_CODE:
            return stop(f"{name}: invariant violation, see {run_dir / 'segments.jsonl'}; not retried")
        quick_failures = quick_failures + 1 if elapsed < QUICK_FAILURE_SECONDS else 0
        if quick_failures >= 2:
            return stop(f"{name}: two failures within {QUICK_FAILURE_SECONDS} s each, see {run_dir.name}.out")
        wait_for_idle(RETRY_IDLE_MINUTES)
    return stop(f"{name}: {MAXIMUM_ATTEMPTS} attempts failed")


def run_tool(marker: pathlib.Path, command: list[str], required: bool) -> int:
    """One python tool, once. A failed required tool stops the pipeline, a failed report only logs."""
    if marker.exists():
        return 0
    with open(marker.with_suffix(".out"), "w", encoding="utf-8") as output:
        completed = subprocess.run(command, cwd=_ROOT, env=runner_environment(), stdout=output, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
    log(f"{marker.name}: exit {completed.returncode}")
    if completed.returncode == 0:
        marker.write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"), encoding="utf-8")
        return 0
    return stop(f"{marker.name} failed, see {marker.with_suffix('.out').name}") if required else 0


def remote_session():
    import paramiko
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(REMOTE_HOST, username=REMOTE_USER, password=os.environ["REMOTE_PW"], timeout=30, allow_agent=False, look_for_keys=False)
    return client, client.open_sftp()


def remote_text(sftp, path: str) -> str | None:
    """The file's text, None when it is not there."""
    try:
        with sftp.open(path) as handle:
            return handle.read().decode("utf-8")
    except FileNotFoundError:
        return None


def remote_fetch(sftp, source: str, target: pathlib.Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    sftp.get(source, str(partial))
    if partial.stat().st_size != sftp.stat(source).st_size:
        raise OSError(f"{source}: short copy")
    partial.replace(target)


def finish_take_over(client, sftp, marker: pathlib.Path) -> list[str]:
    """Stop the remote pipeline, then its runner (in this order: the pipeline would retry), and fetch what the fine run
    starts from: the newest checkpoint of the manifest (checkpoints are atomic, the manifest line comes after the file), or
    the remote warm file when it has no checkpoint yet."""
    fine = f"{REMOTE_RUNS}/{REMOTE_FINE}"
    for path in (f"{REMOTE_RUNS}/pipeline_state/pipeline.pid", f"{fine}/runner.pid"):
        pid = (remote_text(sftp, path) or "").strip()
        if pid.isdigit():
            _, stdout, _ = client.exec_command(f"taskkill /F /T /PID {pid}")
            log(f"take-over: stopping remote pid {pid}: exit {stdout.channel.recv_exit_status()}")
    time.sleep(5)
    manifest = (remote_text(sftp, f"{fine}/checkpoints/manifest.jsonl") or "").strip()
    if manifest:
        entry = json.loads(manifest.splitlines()[-1])
        local = RUNS / REMOTE_FINE
        checkpoint = local / "checkpoints" / entry["file"]
        remote_fetch(sftp, f"{fine}/meta.json", local / "meta.json")
        steady = remote_text(sftp, f"{fine}/steady.json")
        if steady is not None:
            (local / "steady.json").write_text(steady, encoding="utf-8")
        remote_fetch(sftp, f"{fine}/checkpoints/{entry['file']}", checkpoint)
        start = ["--start-from", str(checkpoint)]
        log(f"take-over: fine continues from the remote t = {entry['time']:.2f}, {entry['file']}")
    else:
        remote_fetch(sftp, f"{REMOTE_RUNS}/{REMOTE_WARM}", RUNS / REMOTE_WARM)
        start = ["--initial-velocity", str(RUNS / REMOTE_WARM)]
        log("take-over: no remote fine checkpoint yet, fine starts from the remote warm file")
    marker.write_text(json.dumps(start), encoding="utf-8")
    return start


def take_over_from_remote() -> list[str] | None:
    """The cards are free here: the fine run takes over from the ZeroTier machine, once that has finished the coarse run and
    made the warm file. Returns the fine run's starting arguments, or None = no usable remote run, do the coarse run here."""
    import paramiko
    marker = STATE / "take_over.json"
    if marker.exists():
        return json.loads(marker.read_text(encoding="utf-8"))
    if "REMOTE_PW" not in os.environ:
        log("take-over: REMOTE_PW is not set, coarse run here")
        return None
    log("take-over: waiting for the remote warm file")
    unreachable_since = None
    while True:
        try:
            client, sftp = remote_session()
            try:
                if remote_text(sftp, f"{REMOTE_RUNS}/pipeline_state/STOPPED") is not None:
                    log("take-over: the remote pipeline has stopped, coarse run here")
                    return None
                if remote_text(sftp, f"{REMOTE_RUNS}/pipeline_state/warm_start_done") is not None:
                    return finish_take_over(client, sftp, marker)
            finally:
                client.close()
            unreachable_since = None
        except (OSError, EOFError, paramiko.SSHException) as error:
            unreachable_since = unreachable_since or time.time()
            log(f"take-over: remote failed: {error!r}")
            if time.time() - unreachable_since > REMOTE_GIVE_UP_SECONDS:
                log("take-over: remote unreachable, coarse run here")
                return None
        time.sleep(REMOTE_POLL_SECONDS)


def report(run_dir: pathlib.Path, name: str) -> None:
    """The window against Marchi 2021 Re 5000 for the log; the pipeline goes on whatever it says (the user chose the stages)."""
    run_tool(STATE / f"{name}_decide_done", [sys.executable, "docs/validation/cluster/re3200_decide.py", "--run-dir", str(run_dir), "--reynolds", "5000",
                                             "--no-stream-function", "--json", str(STATE / f"{name}_decision.json")], required=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="print the commands and the card / disk state, start nothing")
    parser.add_argument("--single-card", default=None, metavar="UUID", help="one 5090, K = 1, no wait (the ZeroTier machine)")
    parser.add_argument("--take-over", action="store_true", help="when the cards are free, continue the ZeroTier machine's run")
    arguments = parser.parse_args()
    if arguments.single_card:
        configure_single_card(arguments.single_card)
    if arguments.check:
        print("coarse warm:", "--source-run", COARSE_SOURCE, "--case", COARSE_CASE, "--out", COARSE_WARM_FILE)
        print("coarse:", "--case", COARSE_CASE, "--run-dir", COARSE_RUN, *COMMON, *COARSE_ARGUMENTS)
        print("warm:", "--source-run", COARSE_RUN, "--case", FINE_CASE, "--out", WARM_FILE)
        print("fine:", "--case", FINE_CASE, "--run-dir", FINE_RUN, *COMMON, *FINE_ARGUMENTS, "--initial-velocity", WARM_FILE)
        print("cards", card_states(), "idle now:", cards_idle(), "free disk GB", round(shutil.disk_usage(_ROOT).free / 1e9), "release switches", len(expected_environments("v7")["release"]))
        return 0
    STATE.mkdir(parents=True, exist_ok=True)
    if PID_FILE.exists():
        other = int(PID_FILE.read_text(encoding="utf-8").strip() or 0)
        if other and other != os.getpid() and process_alive(other):
            print(f"pipeline already running as pid {other}")
            return 1
    PID_FILE.write_text(str(os.getpid()), encoding="utf-8")
    (STATE / "STOPPED").unlink(missing_ok=True)
    log(f"pipeline started, pid {os.getpid()}")

    # before the wait: a set-up error shows now, not when the cards come free
    if run_tool(STATE / "coarse_warm_done", [sys.executable, "docs/validation/cluster/warm_start_velocity.py", "--source-run", str(COARSE_SOURCE),
                                             "--case", COARSE_CASE, "--out", str(COARSE_WARM_FILE)], required=True):
        return 1
    if not (COARSE_RUN / "result.json").exists() or not (FINE_RUN / "result.json").exists():
        if not any(path.exists() for path in (COARSE_RUN / "meta.json", FINE_RUN / "meta.json")) or not (STATE / "started").exists():
            wait_for_idle(IDLE_MINUTES)
            (STATE / "started").write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"), encoding="utf-8")
        else:
            wait_for_idle(RETRY_IDLE_MINUTES)

    start = take_over_from_remote() if arguments.take_over else None
    if start is None:
        if run_stage("coarse", COARSE_CASE, COARSE_RUN, COARSE_ARGUMENTS):
            return 1
        report(COARSE_RUN, "coarse")
        if run_tool(STATE / "warm_start_done", [sys.executable, "docs/validation/cluster/warm_start_velocity.py", "--source-run", str(COARSE_RUN),
                                                "--case", FINE_CASE, "--out", str(WARM_FILE)], required=True):
            return 1
        start = ["--initial-velocity", str(WARM_FILE)]
    if run_stage("fine", FINE_CASE, FINE_RUN, [*FINE_ARGUMENTS, *start], announce=not SINGLE_CARD):
        return 1
    report(FINE_RUN, "fine")
    (STATE / "fine_done").write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"), encoding="utf-8")
    log("fine stage done: ready for the 5090 + 7900 XTX continuation (re5000_plan.md)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
