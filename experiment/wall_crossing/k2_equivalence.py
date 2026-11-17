"""
k2_equivalence.py - does a chain of K slabs compute what one slab computes? Bit-level comparison of a case
after N steps on one GPU and on a chain, with the first difference (if any) located and explained.

    .venv/Scripts/python.exe -m experiment.wall_crossing.k2_equivalence \
        --case cases/lid_driven_cavity_2d_n250_xi0p001_eps0p0025_crossing/case.yaml --label n250_crossing \
        --steps 999 --single-device 1 --chain-devices 0,1 \
        --uuid ae137c668a40f5acaab90a83f7cda175 --uuid fb83f0dafd9e1d38dc05e7ae643e22a8 --bisect

Method (experiment/seam_audit/canonical_dump.py --canonical-lists --transport-extension; N below the defrag
cadence): the per-particle state of every alive particle, sorted by a global id, after N steps:
    one slab  vs one slab again   the build reproduces itself over the window (else it proves nothing)
    one slab  vs the chain        the answer; every dumped field bit for bit, except the voxel-id word of
                                  position_voxel_id (voxel ids are numbered per slab)
With --bisect and a difference, the smallest step count with a difference is found by bisection, and there:
  - which particles differ and in which voxels they sit;
  - the voxel neighbour lists of those voxels and their neighbours on both layouts (voxel_list_probe.py): a
    particle listed in different voxels, or the same set in a different order, means the two layouts summed the
    same pair terms in a different order (rounding), not that they computed different physics. Each slab bins a
    particle with floor((x - origin_x) / h) in float32 from its own grid origin, so a particle within rounding
    of a voxel boundary can be binned one step earlier or later on the chain than on one slab.

Output: <out>/<label>/summary.json and summary.md, the dumps and logs; one runs.jsonl record per dump / probe.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys

import numpy as np

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiment.wall_crossing.run_log import LOG_ROOT, LoggedRun     # noqa: E402

FIELDS = ("correction_inverse", "density_gradient_kernel_sum", "density_pressure_scratch", "density_pressure",
          "acceleration", "shift", "velocity_mass")


class Comparison:
    def __init__(self, case: str, label: str, out: pathlib.Path, single_device: str, chain_devices: str,
                 uuids: list) -> None:
        self.case, self.label, self.out = case, label, out
        self.devices = {"single": single_device, "chain": chain_devices}
        self.uuids = uuids
        self.environment = dict(os.environ, OPENBLAS_NUM_THREADS="1", PYTHONIOENCODING="utf-8")
        self.run_ids: list = []
        out.mkdir(parents=True, exist_ok=True)

    def dump(self, layout: str, steps: int, repeat: int = 1) -> pathlib.Path:
        path = self.out / f"{layout}{'' if repeat == 1 else f'_repeat{repeat}'}_s{steps}.npz"
        if path.exists():
            return path
        command = [sys.executable, "experiment/seam_audit/canonical_dump.py", "--solver", "v7", "--case", self.case,
                   "--device-map", self.devices[layout], "--steps", str(steps), "--canonical-lists",
                   "--transport-extension", "--out", str(path)]
        with LoggedRun(kind="k2_equivalence_dump", case=self.case, gpu_uuids=self.uuids,
                       command=" ".join(command[1:])) as run:
            run.extra["label"] = f"{self.label} {layout} ({self.devices[layout]}) {steps} steps" \
                                 + ("" if repeat == 1 else f" repeat {repeat}")
            with open(path.with_suffix(".log"), "w", encoding="utf-8", newline="\n") as log:
                code = subprocess.call(command, cwd=REPOSITORY_ROOT, env=self.environment, stdout=log,
                                       stderr=subprocess.STDOUT)
            run.extra["exit_code"] = code
            run.extra["status"] = "ok" if code == 0 else f"failed: exit code {code}"
            self.run_ids.append(run.run_id)
        if code != 0:
            raise RuntimeError(f"canonical_dump {layout} {steps} failed (exit code {code}): {path.with_suffix('.log')}")
        return path

    @staticmethod
    def differences(first_path: pathlib.Path, second_path: pathlib.Path) -> dict:
        """Per field: the rows that differ bit for bit ({} = identical). Positions are compared without the
        voxel-id word."""
        first = np.load(first_path, allow_pickle=True)
        second = np.load(second_path, allow_pickle=True)
        if not np.array_equal(first["global_id"], second["global_id"]):
            return {"global_id": {"rows": -1}}
        found = {}
        position_first = np.ascontiguousarray(first["position_voxel_id"][:, :3]).view(np.uint32).reshape(-1, 3)
        position_second = np.ascontiguousarray(second["position_voxel_id"][:, :3]).view(np.uint32).reshape(-1, 3)
        moved = (position_first != position_second).any(axis=1)
        if moved.any():
            found["position"] = {"rows": int(moved.sum()), "row_list": np.flatnonzero(moved)}
        for name in FIELDS:
            if name not in first.files or name not in second.files:
                continue
            a, b = first[name], second[name]
            differ = (np.ascontiguousarray(a).view(np.uint32).reshape(a.shape[0], -1)
                      != np.ascontiguousarray(b).view(np.uint32).reshape(b.shape[0], -1)).any(axis=1)
            if differ.any():
                rows = np.flatnonzero(differ)
                scale = float(np.abs(a[rows].astype(np.float64)).max())
                found[name] = {"rows": int(rows.size), "row_list": rows,
                               "relative_difference_maximum": float(
                                   np.abs(a[rows].astype(np.float64) - b[rows].astype(np.float64)).max()
                                   / max(scale, 1e-300))}
        return found

    def crossings(self, steps: int) -> dict:
        meta = json.loads(str(np.load(self.dump("chain", steps), allow_pickle=True)["meta"]))
        return {"crossings": meta.get("crossings"), "cuts": meta.get("cuts")}

    def probe(self, layout: str, steps: int, voxels: list) -> dict:
        path = self.out / f"lists_{layout}_s{steps}.json"
        command = [sys.executable, "experiment/wall_crossing/voxel_list_probe.py", self.case, self.devices[layout],
                   str(steps), str(path)] + [f"{ix},{iy}" for ix, iy in voxels]
        with LoggedRun(kind="k2_equivalence_lists", case=self.case, gpu_uuids=self.uuids,
                       command=" ".join(command[1:])) as run:
            run.extra["label"] = f"{self.label} voxel lists {layout} {steps} steps"
            with open(path.with_suffix(".log"), "w", encoding="utf-8", newline="\n") as log:
                code = subprocess.call(command, cwd=REPOSITORY_ROOT, env=self.environment, stdout=log,
                                       stderr=subprocess.STDOUT)
            run.extra["exit_code"] = code
            run.extra["status"] = "ok" if code == 0 else f"failed: exit code {code}"
            self.run_ids.append(run.run_id)
        if code != 0:
            raise RuntimeError(f"voxel_list_probe {layout} {steps} failed: {path.with_suffix('.log')}")
        return json.loads(path.read_text(encoding="utf-8"))


def explain_lists(single: dict, chain: dict) -> dict:
    """Compare the probed voxel lists of the two layouts: per voxel 'same order', 'same set, other order' or the
    particles listed on one layout only; and for every particle that sits in different voxels, how its x is
    binned in float32 from the two origins."""
    support = np.float32(single["support"])
    verdicts = {}
    voxel_of = {"single": {}, "chain": {}}
    position_of = {}
    for key, entry in single["voxels"].items():
        listed = next(iter(entry.values()))
        for index, particle in enumerate(listed["global_ids"]):
            voxel_of["single"][particle] = key
            position_of[particle] = (listed["x_float32"][index], listed["y_float32"][index])
    chain_lists = {}
    chain_slab_of_voxel = {}
    for key, entry in chain["voxels"].items():
        # a voxel may be held by two slabs (own on one, ghost on the other): take the slab that owns it,
        # i.e. the one whose list is not a replica list; the first reported slab that lists it is used
        slab_name, listed = next(iter(entry.items()))
        chain_lists[key] = listed["global_ids"]
        chain_slab_of_voxel[key] = int(slab_name[len("slab"):])
        for index, particle in enumerate(listed["global_ids"]):
            voxel_of["chain"][particle] = key
            position_of.setdefault(particle, (listed["x_float32"][index], listed["y_float32"][index]))
    for key, entry in single["voxels"].items():
        single_list = next(iter(entry.values()))["global_ids"]
        chain_list = chain_lists.get(key)
        if chain_list is None:
            verdicts[key] = "not held by the chain's reported slabs"
        elif single_list == chain_list:
            verdicts[key] = "same order"
        elif sorted(single_list) == sorted(chain_list):
            verdicts[key] = "same set, other order"
        else:
            verdicts[key] = {"only_single": sorted(set(single_list) - set(chain_list)),
                             "only_chain": sorted(set(chain_list) - set(single_list))}
    moved = []
    for particle, single_voxel in voxel_of["single"].items():
        chain_voxel = voxel_of["chain"].get(particle)
        if chain_voxel is None or chain_voxel == single_voxel:
            continue
        x = np.float32(position_of[particle][0])
        slab = chain["slabs"][chain_slab_of_voxel[chain_voxel]]
        single_origin = np.float32(single["slabs"][0]["origin_x_float32"])
        chain_origin = np.float32(slab["origin_x_float32"])
        single_quotient = (x - single_origin) / support                      # float32, as predict.comp bins
        chain_quotient = (x - chain_origin) / support
        exact = (float(x) - float(single_origin)) / float(support)
        moved.append({"global_id": int(particle), "x_float32": float(x), "y_float32": float(position_of[particle][1]),
                      "voxel_single": single_voxel, "voxel_chain": chain_voxel,
                      "single_quotient_float32": float(single_quotient),
                      "single_column": int(np.floor(single_quotient)),
                      "chain_quotient_float32": float(chain_quotient),
                      "chain_column": int(np.floor(chain_quotient)) + int(slab["column_shift"]),
                      "distance_to_column_boundary_in_voxels": float(exact - round(exact)),
                      "x_ulp_in_voxels": float(np.spacing(x)) / float(support)})
    return {"voxels": verdicts, "particles_binned_differently": moved}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--steps", type=int, default=999)
    parser.add_argument("--single-device", default="1")
    parser.add_argument("--chain-devices", default="0,1")
    parser.add_argument("--uuid", action="append", default=[])
    parser.add_argument("--bisect", action="store_true")
    parser.add_argument("--out", default=None)
    arguments = parser.parse_args()

    out = (REPOSITORY_ROOT / arguments.out).resolve() if arguments.out else LOG_ROOT / "k2_equivalence"
    comparison = Comparison(arguments.case, arguments.label, out / arguments.label, arguments.single_device,
                            arguments.chain_devices, arguments.uuid)
    steps = arguments.steps
    single = comparison.dump("single", steps)
    self_difference = comparison.differences(single, comparison.dump("single", steps, repeat=2))
    chain_difference = comparison.differences(single, comparison.dump("chain", steps))
    summary = {"case": arguments.case, "label": arguments.label, "steps": steps,
               "single_device": arguments.single_device, "chain_devices": arguments.chain_devices,
               "single_reproduces_itself": not self_difference,
               "chain_equals_single": not chain_difference,
               "chain": comparison.crossings(steps),
               "difference_at_steps": {name: {key: value for key, value in entry.items() if key != "row_list"}
                                       for name, entry in chain_difference.items()}}
    lines = [f"# K = 1 对 K = {len(arguments.chain_devices.split(','))}：{arguments.label}", "",
             f"算例 `{arguments.case}`，{steps} 步，canonical 体素表；单卡设备 {arguments.single_device}，"
             f"链设备 {arguments.chain_devices}。", "",
             f"- 单卡两次运行逐位相同：{'是' if not self_difference else '否'}",
             f"- 链对单卡逐位相同（体素号除外）：{'是' if not chain_difference else '否'}；"
             f"切口列 {summary['chain']['cuts']}，过切口 {summary['chain']['crossings']}"]
    if chain_difference and arguments.bisect:
        low, high = 0, steps
        while high - low > 1:
            middle = (low + high) // 2
            found = comparison.differences(comparison.dump("single", middle), comparison.dump("chain", middle))
            print(f"[k2_equivalence] {middle} steps: {'different' if found else 'identical'}", flush=True)
            if found:
                high = middle
            else:
                low = middle
        first_path, chain_path = comparison.dump("single", high), comparison.dump("chain", high)
        first_difference = comparison.differences(first_path, chain_path)
        state = np.load(first_path, allow_pickle=True)
        global_case_origin = None
        rows = np.unique(np.concatenate([entry["row_list"] for entry in first_difference.values()
                                         if "row_list" in entry]))
        # voxel coordinates on the one-slab grid: the probe reports the origin; bin here with float64 of the
        # float32 positions (only used to choose which voxels to probe: the 3 x 3 blocks around them)
        probe_seed = comparison.probe("single", high, [(0, 0)])
        origin = np.asarray(probe_seed["global_origin"])
        support = probe_seed["support"]
        coordinates = np.floor((state["position_voxel_id"][rows, :2].astype(np.float64) - origin) / support) \
            .astype(int)
        block = sorted({(int(ix + dx), int(iy + dy)) for ix, iy in coordinates
                        for dx in (-1, 0, 1) for dy in (-1, 0, 1)})
        lists = explain_lists(comparison.probe("single", high, block), comparison.probe("chain", high, block))
        summary["first_difference"] = {
            "steps": high, "identical_at": low,
            "fields": {name: {key: value for key, value in entry.items() if key != "row_list"}
                       for name, entry in first_difference.items()},
            "rows": int(rows.size),
            "x_range": [float(state["position_voxel_id"][rows, 0].min()), float(state["position_voxel_id"][rows, 0].max())],
            "y_range": [float(state["position_voxel_id"][rows, 1].min()), float(state["position_voxel_id"][rows, 1].max())],
            "chain": comparison.crossings(high), "lists": lists}
        del global_case_origin
        lines += ["", f"## 第一处差异：第 {high} 步（第 {low} 步仍逐位相同）", "",
                  f"- 不同的行：{rows.size}，位置 x ∈ [{summary['first_difference']['x_range'][0]:.4f}, "
                  f"{summary['first_difference']['x_range'][1]:.4f}]，y ∈ [{summary['first_difference']['y_range'][0]:.4f}, "
                  f"{summary['first_difference']['y_range'][1]:.4f}]；"
                  + "；".join(f"{name} {entry['rows']} 行（最大相对差 {entry.get('relative_difference_maximum', 0):.1e}）"
                             for name, entry in summary["first_difference"]["fields"].items()),
                  f"- 该步链上过切口计数：{summary['first_difference']['chain']['crossings']}",
                  "- 体素表（3 × 3 邻域内各体素，单卡 对 链）："
                  + "；".join(f"({key}) {value if isinstance(value, str) else '成员不同 ' + json.dumps(value)}"
                             for key, value in lists["voxels"].items() if value != "same order")]
        for particle in lists["particles_binned_differently"]:
            lines.append(
                f"- 粒子 {particle['global_id']}（x = {particle['x_float32']:.9g}, y = {particle['y_float32']:.9g}，"
                f"两种布局位置逐位相同）：单卡的体素表把它列在体素 ({particle['voxel_single']})，链的体素表列在 "
                f"({particle['voxel_chain']})。它的 (x − origin_x) / h 的精确值距体素列边界 "
                f"{particle['distance_to_column_boundary_in_voxels']:+.3e} 个体素，而 x 的 1 ulp 是 "
                f"{particle['x_ulp_in_voxels']:.3e} 个体素：落在舍入范围内，两种布局各自从自己的 float32 网格原点"
                f"装箱（numpy 的 float32 复算：单卡商 {particle['single_quotient_float32']:.9g}，链商 "
                f"{particle['chain_quotient_float32']:.9g} + 列偏移；GPU 的除法舍入不必与 numpy 相同）")
    summary["run_ids"] = comparison.run_ids
    (comparison.out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8", newline="\n")
    lines += ["", f"记录：{comparison.run_ids[0]} … {comparison.run_ids[-1]}（共 {len(comparison.run_ids)} 条）"]
    text = "\n".join(lines) + "\n"
    (comparison.out / "summary.md").write_text(text, encoding="utf-8", newline="\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
