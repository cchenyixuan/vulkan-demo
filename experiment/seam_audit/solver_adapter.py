"""
solver_adapter.py — resolve the solver modules of one version directory
(experiment/v5 or experiment/v6) behind a single namespace, so the seam audit
drives either solver with the same code.

experiment/v6 is a renamed copy of experiment/v5: modules ``*_v6.py``, classes
``SphSimulatorV6`` / ``ChainOrchestratorV6`` / ``VulkanContextV6``, loader
``load_case_v6`` and environment switches ``V6_*`` instead of ``V5_*``.

IMPORTANT: the solvers read most of their environment switches (``V5_CASCADE_FORCE``,
``V5_BAND_VOXEL_DISPATCH``, ...) at MODULE IMPORT time. The caller must set
``os.environ`` BEFORE calling :func:`load_solver`; switching environment after
the first import in a process has no effect on those module-level constants.

Importing the GPU modules loads the Vulkan loader library but creates no
instance or device; only ``Context.create(...)`` touches a GPU.
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace

SUPPORTED_VERSIONS = ("v5", "v6", "v7")


def _check_version(version: str) -> None:
    if version not in SUPPORTED_VERSIONS:
        raise ValueError(
            f"unknown solver version {version!r}; expected one of {SUPPORTED_VERSIONS}")


def environment_prefix(version: str) -> str:
    """Prefix of the solver's environment switches: 'V5_' or 'V6_'."""
    _check_version(version)
    return version.upper() + "_"


def load_solver(version: str, *, include_gpu_modules: bool = True) -> SimpleNamespace:
    """Import one solver version and return its entry points.

    Attributes of the returned namespace:
      version, env_prefix (also environment_prefix),
      load_case(case_yaml_path) -> degenerate global case,
      compute_chain_partition(global_case, weights, pool_safety) -> ChainPartition,
      Simulator, Orchestrator, Context (None when include_gpu_modules=False),
      plus the raw modules for introspection (case_loader_module, partition_module,
      simulator_module, orchestrator_module, context_module).

    ``include_gpu_modules=False`` imports only the CPU-side loader and
    partitioner (no python-vulkan import at all) — used by dump_state --dry-run.
    """
    _check_version(version)
    upper_version = version.upper()
    package = f"experiment.{version}.utils"

    case_loader_module = importlib.import_module(f"{package}.case_loader_{version}")
    partition_module = importlib.import_module(f"{package}.partition_{version}")

    simulator_module = orchestrator_module = context_module = None
    simulator_class = orchestrator_class = context_class = None
    if include_gpu_modules:
        simulator_module = importlib.import_module(f"{package}.simulator_{version}")
        orchestrator_module = importlib.import_module(f"{package}.orchestrator_{version}")
        context_module = importlib.import_module(f"{package}.vulkan_context_{version}")
        simulator_class = getattr(simulator_module, f"SphSimulator{upper_version}")
        orchestrator_class = getattr(orchestrator_module, f"ChainOrchestrator{upper_version}")
        context_class = getattr(context_module, f"VulkanContext{upper_version}")

    prefix = environment_prefix(version)
    return SimpleNamespace(
        version=version,
        env_prefix=prefix,
        environment_prefix=prefix,
        load_case=getattr(case_loader_module, f"load_case_{version}"),
        compute_chain_partition=partition_module.compute_chain_partition,
        Simulator=simulator_class,
        Orchestrator=orchestrator_class,
        Context=context_class,
        case_loader_module=case_loader_module,
        partition_module=partition_module,
        simulator_module=simulator_module,
        orchestrator_module=orchestrator_module,
        context_module=context_module,
    )
