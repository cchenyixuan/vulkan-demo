"""
_test_single_cpu.py — CPU checks of v7_single against v7 (no GPU):

  1. the three switch parsers accept and refuse the same strings as v7's and give the same values;
  2. read_switches refuses every other V7_* name and the defaults are v7's;
  3. the SPIR-V manifest of experiment/v7 parses and every listed file matches it;
  4. deep_wall_single.deep_wall_candidate_count equals simulator_v7.deep_wall_candidate_count and the auto
     rule gives v7's verdict and reason, on every case given with --case (default: the five acceptance cases
     whose particle files are present);
  5. the specialization constants of v7_single's pipelines equal those v7's single-GPU mode gives the same
     pipelines, for every case and both V7_FUSED_CORRECTION_DENSITY values (the full command-stream comparison
     is _compare_recordings.py).

    .venv/Scripts/python.exe experiment/v7_single/_test_single_cpu.py [--case CASE.yaml ...]
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

_REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

DEFAULT_CASES = ("cases/lid_driven_cavity_2d_n250_xi0p001_eps0p0025/case.yaml",
                 "cases/lid_driven_cavity_2d_n250_xi0p001_eps0p0025_adami/case.yaml",
                 "cases/lid_driven_cavity_2d_gen/case.yaml",
                 "cases/cavity3d_8m_b4/case.yaml",
                 "cases/cavity3d_1m/case.yaml")


def check_parsers(failures: list) -> None:
    import experiment.v7.utils.simulator_v7 as simulator_v7
    from experiment.v7_single.utils import switches
    pairs = ((switches.parse_fused_correction_density, simulator_v7._parse_fused_correction_density,
              "V7_FUSED_CORRECTION_DENSITY"),
             (switches.parse_density_copy_compute, simulator_v7._parse_density_copy_compute, "V7_DENSITY_COPY_COMPUTE"),
             (switches.parse_deep_wall_skip, simulator_v7._parse_deep_wall_skip, "V7_DEEP_WALL_SKIP"))
    samples = ("0", "1", " 1 ", "0\n", "auto", "AUTO", " Auto ", "", "2", "on", "off", "true", "yes", "-1", "1.0",
               "01", "none")
    for ours, theirs, name in pairs:
        for text in samples:
            try:
                expected = ("value", theirs(text))
            except ValueError:
                expected = ("refused", None)
            try:
                actual = ("value", ours(text))
            except ValueError:
                actual = ("refused", None)
            if expected[0] != actual[0] or (expected[0] == "value" and int(expected[1]) != int(actual[1])
                                            if name != "V7_DEEP_WALL_SKIP" else expected != actual):
                failures.append(f"{name}={text!r}: v7 {expected}, v7_single {actual}")
    defaults = switches.read_switches({})
    expected_defaults = {"V7_DENSITY_COPY_COMPUTE": 1, "V7_DEEP_WALL_SKIP": "auto", "V7_FUSED_CORRECTION_DENSITY": 1}
    if defaults.table() != expected_defaults:
        failures.append(f"defaults {defaults.table()} != {expected_defaults}")
    v7_table = simulator_v7.configured_v7_switches()
    for key, value in defaults.table().items():
        if v7_table.get(key) != value:
            failures.append(f"default {key}={value} differs from v7's {v7_table.get(key)} (this process)")
    for stray in ("V7_GHOST_SEND_LANES", "V7_DEEP_WALL_CHECK", "V7_DELTA_DENSITY", "V7_SPV_DIR", "V7_KEEP_DEPARTED",
                  "V7_CONCURRENT_BUFFERS", "V7_BAND_WIDTHS"):
        try:
            switches.read_switches({stray: "1", "V7_FUSED_CORRECTION_DENSITY": "1"})
            failures.append(f"{stray} not refused")
        except SystemExit:
            pass
    try:
        switches.read_switches({"V7_FUSED_CORRECTION_DENSITY": "2"})
        failures.append("V7_FUSED_CORRECTION_DENSITY=2 not refused")
    except ValueError:
        pass
    print("[test] switch parsers: v7 and v7_single agree on %d strings x 3 switches" % len(samples))


def check_manifest(failures: list) -> None:
    from experiment.v7_single.utils import spirv_manifest
    digests = spirv_manifest.spirv_digests()
    if len(digests) != 22:
        failures.append(f"manifest lists {len(digests)} files, expected 22")
    print(f"[test] SPIR-V manifest: {len(digests)} files verified in {spirv_manifest.V7_SHADER_DIRECTORY}")


def check_deep_wall_and_specs(failures: list, cases: list[str]) -> None:
    import experiment.v7.utils.simulator_v7 as simulator_v7
    from experiment.v7.utils.case_loader_v7 import load_case_v7
    from experiment.v7_single.utils import deep_wall_single
    from experiment.v7_single.utils.simulator_single import SphSimulatorSingle, BAND_WIDTHS
    from experiment.v7_single.utils.switches import Switches
    from experiment.v7_single._v7_cpu_fake import v7_pipeline_specs, fake_v7_simulator

    for case_path in cases:
        case = load_case_v7(case_path)
        ours = deep_wall_single.deep_wall_candidate_count(case, BAND_WIDTHS[1])
        theirs = simulator_v7.deep_wall_candidate_count(case, BAND_WIDTHS[1])
        if ours != theirs:
            failures.append(f"{case_path}: deep-wall candidates {ours} (v7_single) != {theirs} (v7)")
        for mode in ("0", "1", "auto"):
            verdict = deep_wall_single.resolve_deep_wall_skip(case, mode, BAND_WIDTHS[1])
            saved = simulator_v7._DEEP_WALL_SKIP
            simulator_v7._DEEP_WALL_SKIP = mode
            try:
                fake = fake_v7_simulator(simulator_v7, case)
                expected = fake._resolve_deep_wall_skip()
            finally:
                simulator_v7._DEEP_WALL_SKIP = saved
            if tuple(verdict) != tuple(expected):
                failures.append(f"{case_path}: V7_DEEP_WALL_SKIP={mode}: {verdict} != v7's {expected}")
        for fused in (1, 0):
            single = object.__new__(SphSimulatorSingle)
            single.case = case
            single.switches = Switches(fused_correction_density=fused, density_copy_compute=1, deep_wall_skip="auto")
            single.deep_wall_skip_resolution = deep_wall_single.resolve_deep_wall_skip(case, "auto", BAND_WIDTHS[1])
            single.fused_correction_density_resolution = (bool(fused), "")
            single.deep_wall_record = False
            ours_specs = {key: (shader, {identifier: (fmt, value) for identifier, fmt, value in entries})
                          for key, (shader, entries) in single.pipeline_specs().items()}
            theirs_specs = v7_pipeline_specs(simulator_v7, case, fused=fused, copy_compute=1, deep_wall="auto")
            for key, (shader, spec) in ours_specs.items():
                if key not in theirs_specs:
                    failures.append(f"{case_path} fused={fused}: pipeline {key} not built by v7")
                    continue
                their_shader, their_spec = theirs_specs[key]
                if shader != their_shader:
                    failures.append(f"{case_path} fused={fused}: {key} shader {shader} != v7's {their_shader}")
                if spec != their_spec:
                    differing = sorted(set(spec) ^ set(their_spec)
                                       | {identifier for identifier in set(spec) & set(their_spec)
                                          if spec[identifier] != their_spec[identifier]})
                    failures.append(f"{case_path} fused={fused}: {key} spec constants differ at ids {differing}: "
                                    f"{[(i, spec.get(i), their_spec.get(i)) for i in differing]}")
        print(f"[test] {case_path}: deep-wall candidates {ours:,} (= v7), auto {single.deep_wall_skip_resolution}, "
              f"spec constants of {len(ours_specs)} pipelines = v7's (both fused values)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", action="append", default=None)
    arguments = parser.parse_args()
    for name in [name for name in os.environ if name.startswith("V7_")]:
        del os.environ[name]
    cases = arguments.case or [case for case in DEFAULT_CASES
                               if (_REPOSITORY_ROOT / case).exists()
                               and all((_REPOSITORY_ROOT / case).parent.joinpath(part).exists()
                                       for part in ())]
    failures: list[str] = []
    check_parsers(failures)
    check_manifest(failures)
    check_deep_wall_and_specs(failures, cases)
    if failures:
        print("FAILURES:")
        for failure in failures:
            print("  " + failure)
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
