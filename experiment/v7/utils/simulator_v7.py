"""
simulator_v7.py — V5 per-GPU SPH simulator.

V5 differences vs V1 (see docs/sph_v5_design.md):

  - Timeline semaphore (uint64 counter, 3 values per frame: 3N+1 / 3N+2 / 3N+3)
    replaces V1's fence-wait round trips.

  - 3 pre-recorded cmd buffers per frame (§14.1 / §5):
        phase_a: predict → update_voxel → ghost_send → readback   (signal 3N+1)
        phase_b: correction(INTERIOR)                              (queue-ordered)
        phase_c: upload → install_migration → correction(BOUNDARY) → density →
                 force                                              (wait 3N+2 signal 3N+3)

  - correction.comp split into INTERIOR + BOUNDARY pipelines (CORRECTION_MODE
    spec const id=47). Phase B runs INTERIOR; Phase C runs BOUNDARY after
    install_migration.

  - 3-hop CPU-staged transport (§14.5). Sender staging HOST_CACHED, receiver
    HOST_COHERENT; sim owns both stagings per direction. Readback / upload
    vkCmdCopyBuffer folded into phase_a / phase_c.

  - apiVersion=1.3 (sync2 core), shader target-env=vulkan1.2 (SPIR-V 1.5).

V5 is fully self-contained: no imports from experiment/v1/ or utils/sph/.
"""

from __future__ import annotations

import os
import pathlib
import struct
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from vulkan import *  # noqa: F401, F403
from vulkan._vulkancache import ffi

from experiment.v7.utils.case_v7 import (
    CaseV7,
    KIND_BOUNDARY,
    KIND_FLUID,
)
import threading

from experiment.v7.utils.partition_v7 import (
    configured_compact_ghost_lists,
    configured_delta_density,
    configured_lean_transport,
    configured_transport_extension,
    transported_particle_fields,
    configured_init_seam_clamp,
    configured_packed_replicas,
    configured_ghost_self_kernels,
    configured_diagnostic_poison_inner_replica,
    configured_band_widths,
    COMPACT_DISPATCH_BAND_WIDTHS,
)
from experiment.v7.utils.sync_scheme_v7 import make_sync_scheme
from experiment.v7.utils.vulkan_context_v7 import VulkanContextV7

# Serialization of driver-entry calls that MUTATE queue/semaphore state
# (vkQueueSubmit2, vkSignalSemaphore).
#
# History: added 2026-07-18 as a PROCESS-WIDE lock, a diagnostic mitigation
# for the K=8 soak wedges (sim6/sim2 — a submitted transfer batch with its
# semaphore wait objectively satisfied never executed on an idle, non-lost
# device). The wedges recurred WITH the lock in place (theory A refuted;
# scoped as environmental — see docs M4 closure), and the 2026-07-22
# instruments audit flagged the GLOBAL scope itself as a confound for the
# K-sweep: every vkQueueSubmit2 (all from the single orchestrator step
# thread) contends with every worker thread's vkSignalSemaphore across BOTH
# physical GPUs — coupling the two GPUs' submit paths through one mutex,
# which the spec never requires.
#
# Facts that bound what locking is actually needed here:
#   - All vkQueueSubmit2 calls come from ONE thread (the orchestrator step
#     loop), so per-queue external synchronization is satisfied lock-free.
#   - vkSignalSemaphore on timeline semaphores has no externally-
#     synchronized parameters (host-side timeline ops are concurrency-safe
#     by spec).
# So no lock is REQUIRED by the spec; it exists only as insurance against
# driver/binding thread-safety bugs. Scope is configurable for A/B:
#   V7_SUBMIT_LOCK_SCOPE=device  (default) one lock per PHYSICAL GPU —
#                                cross-GPU driver entries never serialize
#   V7_SUBMIT_LOCK_SCOPE=global  the historical process-wide lock
#   V7_SUBMIT_LOCK_SCOPE=none    no locking (spec-legal here, see above)
# vkWaitSemaphores is deliberately never locked (it blocks for seconds).
_SUBMIT_LOCK_SCOPE = os.environ.get("V7_SUBMIT_LOCK_SCOPE", "device")
# V3.3 cascading force (2026-09-15, N56 K=8 hiding-window work): move
# force_deep_interior (boundary band = 3 voxel columns by default, V7_BAND_WIDTHS;
# density source = scratch) into Phase B so the transfer chain hides behind correction +
# density + force instead of correction + density only; Phase C then runs
# force_boundary instead of force_all. Off by default until validated.
_CASCADE_FORCE = os.environ.get("V7_CASCADE_FORCE", "1") == "1"   # default ON since the 2026-09-17 freeze (every N56 curve job ran with it; verifier + seam evidence in docs/n56_scaling)
# V3.4 band-voxel dispatch (2026-09-15): Phase C boundary pipelines launch
# one thread per (band voxel, slot) instead of one per own pool slot with
# early return (spec const 57; see common.glsl / helpers.glsl). Off by
# default until validated.
_BAND_VOXEL_DISPATCH = os.environ.get("V7_BAND_VOXEL_DISPATCH", "1") == "1"   # default ON since the 2026-09-17 freeze
# V3.8: lanes per band voxel for the band-dispatch pipelines (spec const 58); 0 = one thread per slot.
# Default 64 since E6b (the release set; v6_opt.md "phase C").
_BAND_SLOT_LANES = int(os.environ.get("V7_BAND_SLOT_LANES", "64"))
# V7_BAND_COMPACT_DISPATCH (port of exp/band-compact): phase C band kernels over
# a compacted pid list, one thread per particle, indirect dispatch (needs
# V7_BAND_VOXEL_DISPATCH=1 for the band definition). Bit-identical results.
_BAND_COMPACT = os.environ.get("V7_BAND_COMPACT_DISPATCH", "0") == "1"
# Diagnostic (2026-09-16): V7_FAKE_BAND_TEST=<column> places the boundary band
# at own columns [column, column + range) of a SINGLE-GPU run (spec const 59)
# and runs the boundary pipelines on it (no ghosts: all neighbours local) to
# separate ghost-pool locality from the band code path. 0 = off. Ignored on
# sims that have a peer.
_FAKE_BAND_COLUMN = int(os.environ.get("V7_FAKE_BAND_TEST", "0"))
# Diagnostic (V7_GHOST_LAYERS = 2 cost split): which band kernels recompute the
# inner ghost column as self. Default both (the seam fix); see _ghost_self_layer.
_DIAG_GHOST_SELF_KERNELS = configured_ghost_self_kernels()
# V3.5 fast submit (2026-09-15): pre-built cffi submit batches + raw cffi
# entry points instead of python-vulkan's per-call struct building. Same
# semaphore ops, one vkQueueSubmit2 per queue per frame. Off by default.
_FAST_SUBMIT = os.environ.get("V7_FAST_SUBMIT", "1") == "1"   # default ON since 2026-09-15 (validated: N56 probe29/30/31/32, local verifier)
# Phase A does not wait on its OWN frame_done(n-1) (V3.6, 2026-09-15; the
# code default since E32, 2026-10-06; V7_PHASE_A_NO_WAIT=0 restores the wait).
# The wait is redundant: A's command buffer opens with a same-queue
# compute->compute barrier and is submitted after C(n-1) on the same queue,
# so submission order already puts it after C(n-1) — C's force writes, and
# through C(n-1)'s own upload_done(n-1) wait the readback / worker / upload
# of n-1 as well. The semaphore was only satisfied at the very end of
# C(n-1), and the GPU scheduler re-evaluated it late: E31 (2 x 5090) C->A
# gap 37 -> 5.6 us per step at K = 2, +2.0 % (2-D 1M) / +4.2 % (2-D 10k)
# fps. Validated at depth 1 (E31: single-step and repeated A/B gates) and at
# depth 2 (E32: 300 / 2000-step restarts of 2-D 1M K = 2 / 4 and 3-D 1M
# K = 2, six =0 and six =1 runs each: no difference beyond the run-to-run
# spread). The frame_done SIGNAL stays (host, workers and the next frame's
# readback fence rely on it). Read once at import.
_PHASE_A_NO_WAIT = os.environ.get("V7_PHASE_A_NO_WAIT", "1") == "1"
# E39 B9 (audit H05): the density scratch -> primary copy as a compute pass
# (density_scratch_copy.comp: raw 32-bit words, one slot per invocation) over
# the same slot regions as the v6 vkCmdCopyBuffer, between compute -> compute
# barriers. The v6 copy started at byte own_first_pid * 8, 8 (mod 16) for
# K = 1 and slab 0, where NVIDIA's copy runs at about half bandwidth; E39
# step trace at 2-D 1M K = 1: 38.5 -> 7.9 us per step (+1.5 % fps).
# Bit-identical (a bit copy). Also closes the v6 recording's K = 1
# synchronization-validation report (WRITE_AFTER_WRITE between consecutive
# frames' copies: the compute -> transfer barrier grants TRANSFER_READ only).
# 0 = the v6 recording (vkCmdCopyBuffer between compute -> transfer /
# transfer -> compute barriers). Read once at import.
_DENSITY_COPY_COMPUTE = os.environ.get("V7_DENSITY_COPY_COMPUTE", "1") == "1"
# Regions density_scratch_copy.comp takes (spec constants 101-108): the own
# range, two inner replica regions, the departed pool.
_DENSITY_COPY_REGION_LIMIT = 4
# density_scratch_copy.comp's fixed local_size_x (the shader comment has the
# measurement behind 256).
_DENSITY_COPY_LOCAL_SIZE = 256
# E39 B6 (audit H04): ghost_send with one lane group per (face voxel, layer)
# (ghost_send_lanes.comp). ghost_send.comp gives every (y, z) face voxel ONE
# thread that copies both layers' replicas one after another, a dependent load
# chain per replica (206 threads = 2 workgroups at 2-D 1M). Here the group
# leader does the pair's one atomicAdd and the overflow handling, broadcasts
# the base slot through shared memory, the group's lanes copy slot k to
# base + k with the same record expressions, and the leader then sends the
# voxel's migrants serially as before; the bootstrap ghost round and the
# single-layer (V7_GHOST_LAYERS = 1) pool take the same kernel. Bit-identical:
# in-voxel order and record bits are unchanged, the voxels' block placement is
# atomic arrival order as before (nothing sums in it). E39 step trace, K = 2,
# a_voxel_end -> a_ghost_end (counter fills + barriers + dispatch), s0 / s1:
# 2-D 62k 28.7 / 34.8 -> 4.1 / 4.1 us, 2-D 1M 43.0 / 51.2 -> 6.2 / 6.2 us,
# 3-D 8M 145.4 / 176.1 -> 12.3 / 12.3 us at the default 32 lanes (the shader
# comment has the lanes / local size table). V7_GHOST_SEND_LANES = lanes per
# group, one of _GHOST_SEND_LANES_ACCEPTED (the validated values); 0 =
# ghost_send.comp with its dispatch, command for command. Read once at import.
_GHOST_SEND_LANES_ACCEPTED = (0, 4, 8, 16, 32)
# ghost_send_lanes.comp's workgroup size (spec constant 110, a multiple of the
# lanes; the shader comment has the measurement behind it) and its shared
# arrays' entries (one per group of a workgroup).
_GHOST_SEND_LOCAL_SIZE = 64
_GHOST_SEND_MAXIMUM_GROUPS_PER_WORKGROUP = 256


def _parse_ghost_send_lanes(text: str) -> int:
    """V7_GHOST_SEND_LANES: lanes per ghost_send lane group, one of
    _GHOST_SEND_LANES_ACCEPTED (0 = ghost_send.comp)."""
    try:
        lanes = int(text.strip())
    except ValueError:
        lanes = None
    if lanes not in _GHOST_SEND_LANES_ACCEPTED:
        raise ValueError(f"V7_GHOST_SEND_LANES={text!r}: accepted values are "
                         f"{', '.join(map(str, _GHOST_SEND_LANES_ACCEPTED))} "
                         "(lanes per (face voxel, layer) group; 0 = the one-thread-per-face-voxel ghost_send.comp)")
    return lanes


_GHOST_SEND_LANES = _parse_ghost_send_lanes(os.environ.get("V7_GHOST_SEND_LANES", "32"))
# E39 B4 (audit H06): deep walls skip correction and the density neighbour loop
# (shaders/deep_wall_skip.glsl has the consumers and the criterion). A wall
# particle whose 3^d voxels (the voxels its loops visit) list no particle of a
# kind density counts for a wall (every kind but BOUNDARY) has a dead L /
# kernel sum and a no-op density loop; deep_wall_marker.comp marks such voxels
# every step from that step's lists (phase B start / single-cmd step / K = 1
# bootstrap, two dispatches), correction_interior / density_deep_interior (and
# correction_all / density_all on a slab without peers) built from the
# DEEP_WALL_SKIP variants of correction.comp / density.comp return / skip the
# loop for them; density's epilogue still stores the same scratch value. Every
# output another kernel reads is bit-identical; only a skipped wall's own L and
# density gradient / kernel sum keep their previous values (canonical_dump
# masks exactly those rows of those two fields).
# V7_DEEP_WALL_SKIP: 0 = the previous build command for command (no marker, no
# extra buffer, the old pipelines); 1 = forced on (every slab); auto (default) =
# _resolve_deep_wall_skip's per-slab rule. Read once at import.
_DEEP_WALL_SKIP_ACCEPTED = ("0", "1", "auto")


def _parse_deep_wall_skip(text: str) -> str:
    """V7_DEEP_WALL_SKIP: one of _DEEP_WALL_SKIP_ACCEPTED (case and surrounding blanks ignored)."""
    value = text.strip().lower()
    if value not in _DEEP_WALL_SKIP_ACCEPTED:
        raise ValueError(f"V7_DEEP_WALL_SKIP={text!r}: accepted values are "
                         f"{', '.join(_DEEP_WALL_SKIP_ACCEPTED)} (0 = off, the previous build; 1 = forced on; "
                         "auto = per-slab rule from the dimension and the initial deep-wall candidates)")
    return value


def _parse_deep_wall_check(text: str) -> int:
    """V7_DEEP_WALL_CHECK (debug): 0 or 1."""
    value = text.strip()
    if value not in ("0", "1"):
        raise ValueError(f"V7_DEEP_WALL_CHECK={text!r}: accepted values are 0, 1 (1 = every skipped wall also "
                         "runs density's neighbour test into overflow_deep_wall_skip_count)")
    return int(value)


_DEEP_WALL_SKIP = _parse_deep_wall_skip(os.environ.get("V7_DEEP_WALL_SKIP", "auto"))
# V7_DEEP_WALL_CHECK=1 (debug, needs the skip on a slab to do anything): every
# skipped wall also walks density's neighbour test and adds the neighbours its
# sums would count to global_status.overflow_deep_wall_skip_count (an
# overflow_* invariant: the chain bench, canonical_dump and cavity_runner fail
# on it), and the decisions are recorded as with _DEEP_WALL_RECORD_DECISIONS.
# Outputs are those of 0. Read once at import.
_DEEP_WALL_CHECK = _parse_deep_wall_check(os.environ.get("V7_DEEP_WALL_CHECK", "0"))
# Test hook, not a switch (experiment/seam_audit/canonical_dump.py sets it before
# building the simulators): record every skip decision (frame_stamp + 1 per
# skipped particle and kernel, deep_wall_skip_record) so a dump can name the
# rows whose L / kernel sum were kept. Production runs never record.
_DEEP_WALL_RECORD_DECISIONS = False
# The AUTO rule (_resolve_deep_wall_skip): on for a 3-D slab whose initial state
# has deep-wall candidates (walls the marker skips at step 0) of at least this
# fraction of its particles. E39 B4 chain bench, interleaved 0 / 1 / 1 / 0,
# two RTX 5090: 3-D 9-layer walls gain - cavity3d_1m K = 1 (16.6 % candidates)
# 77.7 / 77.5 -> 81.8 / 81.9 fps (+5.5 %), K = 2 121.5 -> 126.1 fps (+3.8 %,
# step trace: marker 23.8 us, correction_interior -7 / -11 %,
# density_deep_interior -4 / -6 %, phase A and the readback start unchanged),
# cavity3d_8m K = 1 (9.3 %) 12.3 / 12.3 -> 12.7 / 12.7 fps (+3.2 %); 3-D
# 4-layer walls have no candidates (cavity3d_weak4_k2_8m_b4 K = 2, forced on:
# 25.7 / 25.6 -> 25.9 / 25.9 fps, the marker's cost is below the noise); 2-D
# loses although it has candidates - n250 62k (5.8 %) K = 1 3016 -> 2914 fps
# (-3.3 %), K = 2 -2.0 %, 2-D 1M (1.6 %) K = 1 -0.4 %, K = 2 -0.7 %: the marker
# costs 13.5 us per step at 62k K = 1 and correction / density gain nothing
# measurable there (step trace; the voxel criterion reaches the outermost 2 of
# 11 wall layers only, in latency-bound kernels). Below 1 % the 3-D gain
# (~0.3-0.35 % fps per 1 % of candidates) is below the run-to-run spread.
_DEEP_WALL_AUTO_MINIMUM_CANDIDATE_FRACTION = 0.01
# E39 B1 (audit H01): correction and density in one neighbour traversal
# (shaders/correction_density.comp has the algebra). density reads one
# correction output, the self L_i, so the fused kernel accumulates correction's
# sums and the symmetric part of S_i = sum_j q_j (x) grad W_ij in one loop and
# contracts S_i with the L_i it has just written: L, the kernel sum and grad rho
# keep their bits, rho_{n+1} / P_{n+1} move by summation rounding. Every site
# where correction and density run back to back takes one fused dispatch: phase
# B (correction_interior + density_deep_interior), phase C's band (the band /
# boundary pipelines, the inner ghost column as self), the bootstrap (_all, and
# the band pass of the inner ghost column) and the single-cmd step (_all, or the
# split path). A slab whose correction and density particle sets differ keeps
# the separate kernels, recorded as before (_resolve_fused_correction_density):
# different band widths (V7_BAND_WIDTHS c != d, e.g. 2,3,4 and the compact band
# dispatch, which supports 2,3,4 only) or a V7_DIAG_GHOST_SELF that walks the
# inner ghost column in one band kernel only. B4 runs inside the fused interior
# and peerless full-domain kernels (their DEEP_WALL_SKIP variant). E39 B1 chain
# bench, interleaved 0 / 1 / 1 / 0, two RTX 5090: K = 1 2-D 1M 554.9 -> 729.4
# fps (+31.5 %), 3-D 1M 81.9 -> 106.1 (+29.5 %); K = 2 2-D 62k 1779 -> 2319
# (+30.4 %), 2-D 1M 832 -> 1102 (+32.4 %), 2-D 16M 69.1 -> 90.6 (+31.1 %), 3-D
# 8M (4-layer walls) 25.2 -> 32.7 (+29.8 %); K = 2 62k anatomy: phase B 261 ->
# 179 us, phase C 212 -> 150 us (band: correction 61 + density 69 -> 67 us).
# Registers (VK_KHR_pipeline_executable_properties): correction / density 40,
# fused 56 (2-D) / 64 (3-D); the variants with fewer registers (a 9 / 4-entry S,
# or a 3-entry gradient in 2-D: 48 / 56 + 2-5 KB shared memory) ran 1-3 % slower.
# B4 still pays on top: cavity3d_1m K = 1 +5.3 %, cavity3d_8m K = 1 +3.2 %.
# V7_FUSED_CORRECTION_DENSITY: 0 = the previous build command for command; 1 =
# fused wherever the slab allows it (default). Read once at import.
_FUSED_CORRECTION_DENSITY_ACCEPTED = ("0", "1")


def _parse_fused_correction_density(text: str) -> int:
    """V7_FUSED_CORRECTION_DENSITY: 0 or 1 (surrounding blanks ignored)."""
    value = text.strip()
    if value not in _FUSED_CORRECTION_DENSITY_ACCEPTED:
        raise ValueError(f"V7_FUSED_CORRECTION_DENSITY={text!r}: accepted values are 0, 1 (1 = correction and "
                         "density in one neighbour traversal where the slab's bands allow it; 0 = the separate "
                         "kernels, the previous build)")
    return int(value)


_FUSED_CORRECTION_DENSITY = _parse_fused_correction_density(os.environ.get("V7_FUSED_CORRECTION_DENSITY", "1"))
# E39 B3 (audit H03): phase C's band kernels run concurrently with phase B's
# cascade force, by command order in the one compute queue (no second queue).
# Every compute barrier is global, so a dispatch overlaps only a neighbour
# recorded with no barrier between them; the band chain must follow phase C's
# upload_done wait and the barriers after install / append_departed, so the
# overlap partner has to move there: force_deep_interior_scratch (phase B) is
# split into workgroup segments, each a base-0 dispatch of a pipeline whose
# thread t is thread first + t of the full dispatch (force.comp's FORCE_SEGMENT
# variant, spec constant 115; the same particles, the same code), and the
# segments are recorded next to the latency-bound band kernels, the barrier
# after each pair serving both (vkCmdDispatchBase would need no variant, but in
# this build on the RTX 5090 / driver 576.88 force_deep's base dispatch (146,
# 205) was observed to run only workgroups [146, 205), with or without a
# barrier around it, deterministically; an isolated probe with a simple kernel
# did not reproduce it, the cause is not located: logs/e39/b3/diag) - phase B:
# correction_density_interior [-> the first
# segment of force_deep] ; phase C: expand -> install -> append_departed ->
# {correction_density_boundary_band, segment} -> density copy ->
# {force_boundary_band, segment}. force_deep reads scratch rho / P of own
# columns >= the density band (>= 2), its own L / kernel sum (columns >= the
# force band, written by phase B) and the lists of columns >= 2; it writes
# acceleration / shift of columns >= the force band: no element a phase C kernel
# writes or reads-after-it-writes (_resolve_band_overlap has the table; install
# and the ghost kernels stay ordered before it by the barriers). Bit-identical:
# no kernel's inputs, spec constants or summation order change. The cost is the
# transfer-hiding window: phase B shrinks to correction_density_interior (+ the
# segments kept there), so the rule turns it on only where that still covers
# the transfer chain. V7_BAND_OVERLAP: 0 = the B1 build command for command;
# 1 = forced on wherever legal (a 2-D slab with peers that fuses and cascades
# force); auto = the rule below, one verdict for the whole chain (every slab
# or none, band_overlap_chain_verdict). Read once at import. A slab
# records the layout only while it fuses at recording time (a tool that turns
# fusion off after construction gets the B1 separate recording), and every
# phase C recording checks that phase B + C dispatch force_deep's workgroups
# exactly once (_check_force_deep_recorded_once raises otherwise).
_BAND_OVERLAP_ACCEPTED = ("0", "1", "auto")


def _parse_band_overlap(text: str) -> str:
    """V7_BAND_OVERLAP: one of _BAND_OVERLAP_ACCEPTED (case and surrounding blanks ignored)."""
    value = text.strip().lower()
    if value not in _BAND_OVERLAP_ACCEPTED:
        raise ValueError(f"V7_BAND_OVERLAP={text!r}: accepted values are {', '.join(_BAND_OVERLAP_ACCEPTED)} "
                         "(0 = off, the B1 build; 1 = forced on wherever legal (2-D, peers, fused, cascade force); "
                         "auto = one verdict per chain, all slabs or none: on for a 2-D chain of two slabs whose own "
                         "particle counts both lie in the measured window)")
    return value


_BAND_OVERLAP = _parse_band_overlap(os.environ.get("V7_BAND_OVERLAP", "auto"))
# The AUTO rule (band_overlap_chain_verdict), one verdict per chain: on for
# both slabs of a 2-D chain of two slabs whose initial own particle counts both
# lie in [minimum, maximum], off for every slab otherwise (chains of three slabs
# or more included). E39 B3 chain bench, 2-D, two RTX 5090, V7_BAND_OVERLAP 0
# vs 1 (the default layout), interleaved off on on off, paired on - off (mean
# +- std; own particles per slab): K = 2 62k (37k) -3.8 % (+-1.5 %), 250k
# (137k) -4.7 % (+-0.4 %), 1M (523k) +5.7 % (+-0.3 %), 2M (1.03M) +2.5 %
# (+-0.06 %), 4M (2.09M) +0.3 % (+-0.4 %, noise), 16M (8.1M) -1.4 % (+-0.3 %);
# the B3 review repeated it (three interleaved rounds, +- SE): 1M +6.34 +-
# 0.09 %, 2M +2.74 +- 0.11 %, 1M with both slabs on GPU 1 +5.56 +- 0.65 %, 4M
# +0.01 +- 0.12 %. Below: phase B without force_deep no longer covers the
# transfer chain (step trace 250k K = 2: phase B 237 -> 114-126 us against a
# 185-192 us chain, the receiver waits for the upload in 100 % of the steps,
# B -> C 94-100 us); above: the pairs hide little (4M K = 2: pair 743 / 768 us
# against 671 + 80 serial) and cost at 16M.
# Two slabs only: chains of three slabs or more ran on this rig only with GPUs
# shared between slabs (device maps 0,1,0 / 0,1,0,1), and there the sign of
# the effect changes with the size and the weights in ways the slab counts do
# not tell apart (every slab on; one or two pairs, or three rounds +- SE):
# K = 3 1M -5.6 %, -4.3 % (1,1,1), -4.17 +- 0.74 % (0.75,1,0.75), +2.59 +-
# 0.14 % (0.5,1,0.5); 2M +3.7 %, +4.38 +- 0.13 % (1,1,1), +4.80 +- 0.24 %
# (0.5,1,0.5); 4M +1.3 %, +0.56 +- 0.62 %, -0.56 +- 0.08 % (1,1,1, three
# sessions); K = 4 (0,1,0,1) 62k +17.5 %, 2M -7.32 +- 0.28 %, 4M -1.10 +-
# 0.11 % (step trace 2M K = 4: phase B 595-634 -> 290-308 us, the end slabs'
# B -> C upload wait 5-9 -> ~1050 us). V7_BAND_OVERLAP=1 forces it there.
# All slabs or none (E39 B3 review): a B3 slab that waits for the uploads of B1
# neighbours has lost the force_deep phase B ran during that wait and runs it
# in phase C, on the path to their next uploads. Decided per slab, 1M K = 3
# (0,1,0) with GPU-balanced weights turned on the middle slab only and lost
# against B1: 0.5,1,0.5 (268k / 512k / 267k) -2.10 +- 0.29 %, 0.75,1,0.75
# (319k / 409k / 318k) -2.1 % (E39 B3 fix 2 and review: three rounds of
# interleaved arms each, +- SE). Two-slab chains that straddle a threshold run
# B1 (1M K = 2 0.7,1.3 = 370k / 676k: +0.04 +- 0.19 %; 2M K = 2 0.5,1.5 =
# 527k / 1.54M: -0.07 +- 0.41 %) although both slabs on gain there (+2.76 +-
# 0.19 %, +0.99 +- 0.27 %): the rule keeps to the measured window. (A mixed
# chain loses only where its B3 slab waits: the per-slab rule's 1M K = 2
# 0.7,1.3, its larger slab on - the one the other waits for - gained +3.29 +-
# 0.03 %; 2M K = 2 0.5,1.5, its smaller slab on, -0.17 +- 0.24 %.)
_BAND_OVERLAP_AUTO_MINIMUM_OWN_PARTICLES = 400_000
_BAND_OVERLAP_AUTO_MAXIMUM_OWN_PARTICLES = 1_500_000
# Test hook, not a switch (logs/e39/b3/tools/variant_bench.py sets it before building the simulators): replace
# the layout of every slab that resolves on, a dict with _band_overlap_layout's keys. Production runs never set it.
_BAND_OVERLAP_LAYOUT_OVERRIDE: Optional[dict] = None
if _FAST_SUBMIT:
    from vulkan._vulkancache import ffi as _ffi
    from vulkan._vulkan import lib as _lib
_GLOBAL_SUBMIT_LOCK = threading.Lock()
_PER_DEVICE_SUBMIT_LOCKS: dict = {}
_PER_DEVICE_SUBMIT_LOCKS_GUARD = threading.Lock()


class _NoOpLock:
    """Context-manager stand-in for V7_SUBMIT_LOCK_SCOPE=none."""

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


_NO_OP_LOCK = _NoOpLock()


def configured_v7_switches() -> dict[str, int | str]:
    """E39: the v7 performance switches (one per audit item, each on by
    default; 0 = the previous build) with the values this process runs
    (module constants read at import), for the run headers and the step
    trace's run_meta. V7_DEEP_WALL_SKIP reads "auto" for its default (each
    slab prints what the rule decided); V7_DEEP_WALL_CHECK is its debug
    companion; V7_FUSED_CORRECTION_DENSITY = 1 fuses where a slab allows it
    (each slab prints whether it does and, if not, why); V7_BAND_OVERLAP reads
    "auto" for the rule (one verdict per chain; each slab prints it with its
    decision and layout)."""
    return {"V7_DENSITY_COPY_COMPUTE": int(_DENSITY_COPY_COMPUTE),
            "V7_GHOST_SEND_LANES": _GHOST_SEND_LANES,
            "V7_DEEP_WALL_SKIP": _DEEP_WALL_SKIP if _DEEP_WALL_SKIP == "auto" else int(_DEEP_WALL_SKIP),
            "V7_DEEP_WALL_CHECK": _DEEP_WALL_CHECK,
            "V7_FUSED_CORRECTION_DENSITY": _FUSED_CORRECTION_DENSITY,
            "V7_BAND_OVERLAP": _BAND_OVERLAP if _BAND_OVERLAP == "auto" else int(_BAND_OVERLAP)}


def band_overlap_chain_verdict(case: CaseV7) -> tuple[bool, str]:
    """E39 B3: (on, reason) of V7_BAND_OVERLAP for the whole chain that ``case``
    is a slab of. It reads only what compute_chain_partition gives every slab of
    a chain alike - the physics and case.chain_own_particle_counts - so every
    slab gets the same verdict, and _resolve_band_overlap turns a slab on only
    where this verdict is on and the slab is legal. 0: off; 1: on (forced, the
    legality still decides per slab). auto: on iff the chain is 2-D, has
    exactly two slabs and both slabs' initial own particle counts lie in
    [_BAND_OVERLAP_AUTO_MINIMUM_OWN_PARTICLES, _BAND_OVERLAP_AUTO_MAXIMUM_OWN_PARTICLES]
    (the measurements are at those constants; three slabs or more: off, see the
    AUTO rule); a case that carries no chain counts (not built by
    compute_chain_partition) is off."""
    if _BAND_OVERLAP == "0":
        return False, "V7_BAND_OVERLAP=0"
    if _BAND_OVERLAP == "1":
        return True, "forced (V7_BAND_OVERLAP=1)"
    counts = tuple(int(count) for count in case.chain_own_particle_counts)
    dimension = int(case.physics.dimension)
    if not counts:
        return False, "auto: no chain on this slab case (not built by partition_v7.compute_chain_partition)"
    if len(counts) < 2:
        return False, "auto: a chain of one slab (K = 1)"
    if dimension != 2:
        return False, f"auto: {dimension}-D chain (band kernels throughput-bound)"
    minimum, maximum = _BAND_OVERLAP_AUTO_MINIMUM_OWN_PARTICLES, _BAND_OVERLAP_AUTO_MAXIMUM_OWN_PARTICLES
    chain = f"2-D chain of {len(counts)} slabs, own particles {' / '.join(f'{count:,}' for count in counts)}"
    if len(counts) > 2:
        return False, (f"auto: {chain}: more than two slabs (measured only with GPUs shared between slabs, where "
                       "the sign changes with size and weights; V7_BAND_OVERLAP=1 forces it) - off on every slab")
    below = [index for index, count in enumerate(counts) if count < minimum]
    above = [index for index, count in enumerate(counts) if count > maximum]
    if below or above:
        outside = ([f"slab {', '.join(map(str, below))} < {minimum:,} (phase B without force_deep would expose the "
                    "transfer chain)"] if below else []) \
            + ([f"slab {', '.join(map(str, above))} > {maximum:,} (the pairs hide nothing measurable)"]
               if above else [])
        return False, f"auto: {chain}: {'; '.join(outside)} - off on every slab"
    return True, f"auto: {chain}, all in [{minimum:,}, {maximum:,}]"


def band_overlap_record(simulators) -> Optional[dict]:
    """E39 B3: V7_BAND_OVERLAP of a chain's simulators for run metadata (the
    chain bench's step-trace run_meta, canonical_dump's and fused_single_step's
    meta): {"chain": {"verdict", "reason", "every_slab_alike"} = the chain
    verdict (band_overlap_chain_verdict as slab 0 resolved it - a verdict, not
    the slabs' state: 1 says on also where no slab is legal; every_slab_alike:
    every slab resolved the same), "slabs": [[active, reason], ...] = each
    slab's band_overlap_resolution}; None for simulators without the switch."""
    if not simulators or not all(hasattr(simulator, "_band_overlap_active") for simulator in simulators):
        return None
    for simulator in simulators:
        simulator._band_overlap_active()            # resolves once (normally done in __init__)
    verdicts = [tuple(simulator.band_overlap_chain_verdict) for simulator in simulators]
    return {"chain": {"verdict": bool(verdicts[0][0]), "reason": verdicts[0][1],
                      "every_slab_alike": all(verdict == verdicts[0] for verdict in verdicts)},
            "slabs": [[bool(simulator.band_overlap_resolution[0]), simulator.band_overlap_resolution[1]]
                      for simulator in simulators]}


def deep_wall_candidate_count(case: CaseV7, skip_band_width: int) -> int:
    """E39 B4: the walls deep_wall_marker.comp + the skip decision would skip on
    the slab's initial particles (case.initial; the AUTO rule's input): bin with
    initialize_voxelization's float32 formula, presence = a non-BOUNDARY
    particle in an own voxel, deep = every in-grid 3^d neighbour own and without
    presence (z only in 3-D), skip = BOUNDARY in a deep voxel outside the band of
    width skip_band_width (the diagnostic fake band is ignored)."""
    grid, physics = case.grid, case.physics
    dimensions = np.array([grid.grid_dimension_x, grid.grid_dimension_y, grid.grid_dimension_z])
    origin = np.array([grid.origin_x, grid.origin_y, grid.origin_z], dtype=np.float32)
    smoothing_length = np.float32(physics.smoothing_length)
    positions = np.asarray(case.initial.positions, dtype=np.float32)
    if positions.shape[0] == 0:
        return 0
    coordinates = np.floor((positions - origin) / smoothing_length).astype(np.int32)
    kinds = np.array([material.kind for material in case.materials])[np.asarray(case.initial.material_group)]
    inside = np.all((coordinates >= 0) & (coordinates < dimensions), axis=1)
    face = int(dimensions[1] * dimensions[2])
    leading_x = case.ghost_grid.leading_ghost_voxel_count // face
    trailing_x = case.ghost_grid.trailing_ghost_voxel_count // face
    own = np.zeros(dimensions, dtype=bool)
    own[leading_x:int(dimensions[0]) - trailing_x] = True
    present = np.zeros(dimensions, dtype=bool)
    present[tuple(coordinates[inside & (kinds != KIND_BOUNDARY)].T)] = True
    blocked = np.pad(present | ~own, 1, constant_values=False)      # outside the grid: not visited
    neighborhood_blocked = np.zeros(dimensions, dtype=bool)
    z_range = int(physics.neighbor_z_range)
    for delta_x in (-1, 0, 1):
        for delta_y in (-1, 0, 1):
            for delta_z in range(-z_range, z_range + 1):
                neighborhood_blocked |= blocked[1 + delta_x:1 + delta_x + dimensions[0],
                                                 1 + delta_y:1 + delta_y + dimensions[1],
                                                 1 + delta_z:1 + delta_z + dimensions[2]]
    deep = own & ~neighborhood_blocked
    column = coordinates[:, 0]
    own_last_x = int(dimensions[0]) - 1 - trailing_x
    in_band = (((leading_x > 0) & (column < leading_x + skip_band_width))
               | ((trailing_x > 0) & (column > own_last_x - skip_band_width)))
    walls = np.flatnonzero(inside & (kinds == KIND_BOUNDARY) & ~in_band)
    return int(deep[tuple(coordinates[walls].T)].sum())


def _driver_submit_lock_for(physical_device_index: int):
    """Resolve the driver submit/signal lock for one physical GPU according
    to V7_SUBMIT_LOCK_SCOPE. Called once per simulator at init."""
    if _SUBMIT_LOCK_SCOPE == "global":
        return _GLOBAL_SUBMIT_LOCK
    if _SUBMIT_LOCK_SCOPE == "none":
        return _NO_OP_LOCK
    if _SUBMIT_LOCK_SCOPE != "device":
        raise ValueError(
            f"V7_SUBMIT_LOCK_SCOPE={_SUBMIT_LOCK_SCOPE!r} — "
            f"expected one of: device, global, none")
    with _PER_DEVICE_SUBMIT_LOCKS_GUARD:
        lock = _PER_DEVICE_SUBMIT_LOCKS.get(physical_device_index)
        if lock is None:
            lock = threading.Lock()
            _PER_DEVICE_SUBMIT_LOCKS[physical_device_index] = lock
        return lock


# ============================================================================
# Constants — descriptor binding layout (matches shaders/v4/common.glsl)
# ============================================================================

# Buffers that defrag.comp writes into a scratch SoA before copy-back.
# Same set as V1: set 0 bindings 0/1/3/4/5/6/7/8/9 (everything except scratch
# 2 which IS the scratch).
DEFRAG_SET0_BINDINGS = (0, 1, 3, 4, 5, 6, 7, 8, 9)

# Ghost transport: 9 SoA fields (same set as defrag — skip binding 2 scratch
# which is transient) + 2 set-1 buffers (inside_particle_count + index) + 1
# set-3 count field (ghost_send_*_count → ghost_recv_*_count). Total 12
# segments per direction; matches V1's transport_cpu_staging.py.
TRANSPORT_SET0_BINDINGS = DEFRAG_SET0_BINDINGS

# Buffer name → byte stride per particle (set 0 SoA).
_SET0_BYTE_STRIDES = {
    "position_voxel_id":           16,
    "density_pressure":             8,
    "velocity_mass":               16,
    "acceleration":                16,
    "shift":                       16,
    "material":                     4,
    "correction_inverse":          32,
    "density_gradient_kernel_sum": 16,
    "extension_fields":            16,
}
_SET0_BINDING_TO_NAME = {
    0: "position_voxel_id",
    1: "density_pressure",
    3: "velocity_mass",
    4: "acceleration",
    5: "shift",
    6: "material",
    7: "correction_inverse",
    8: "density_gradient_kernel_sum",
    9: "extension_fields",
}

# GlobalStatusBuffer field offsets (16 × uint, see common.glsl §3)
_OFFSET_GHOST_SEND_LEADING   = 32   # field [8]
_OFFSET_GHOST_SEND_TRAILING  = 36   # field [9]
_OFFSET_GHOST_RECV_LEADING   = 40   # field [10]
_OFFSET_GHOST_RECV_TRAILING  = 44   # field [11]
# Frame-stamp instrumentation fields (see common.glsl GlobalStatusBuffer).
_OFFSET_FRAME_STAMP          = 64   # field [16]
_OFFSET_GHOST_STAMP_LEADING  = 68   # field [17]
_OFFSET_GHOST_STAMP_TRAILING = 72   # field [18]
# V6 seam fields (common.glsl GlobalStatusBuffer, 40 uint = 160 B).
_GLOBAL_STATUS_BYTES                       = 160
_OFFSET_DEPARTED_COUNT                     = 84   # field [21], zeroed per frame
_OFFSET_OVERFLOW_DEPARTED                  = 88   # field [22], cumulative
_OFFSET_FAR_MIGRATION                      = 92   # field [23], cumulative
_OFFSET_REPLICA_INNER_SEND = {"leading": 96,  "trailing": 100}   # [24], [25]
_OFFSET_REPLICA_OUTER_SEND = {"leading": 104, "trailing": 108}   # [26], [27]
_OFFSET_REPLICA_INNER_RECV = {"leading": 112, "trailing": 116}   # [28], [29]
_OFFSET_REPLICA_OUTER_RECV = {"leading": 120, "trailing": 124}   # [30], [31]
_GLOBAL_STATUS_FIELD_NAMES = (
    "alive_particle_count", "maximum_velocity", "overflow_inside_count",
    "overflow_incoming_count", "first_overflow_voxel_inside",
    "first_overflow_voxel_incoming", "correction_fallback_count",
    "overflow_ghost_count", "ghost_send_leading_count",
    "ghost_send_trailing_count", "ghost_recv_leading_count",
    "ghost_recv_trailing_count", "migration_install_count",
    "overflow_install_tail", "overflow_install_inside",
    "first_overflow_voxel_install", "frame_stamp", "ghost_stamp_leading",
    "ghost_stamp_trailing", "stamp_error_count", "stamp_error_sample",
    "departed_count", "overflow_departed_count", "far_migration_count",
    "replica_inner_send_leading_count", "replica_inner_send_trailing_count",
    "replica_outer_send_leading_count", "replica_outer_send_trailing_count",
    "replica_inner_recv_leading_count", "replica_inner_recv_trailing_count",
    "replica_outer_recv_leading_count", "replica_outer_recv_trailing_count",
    "initialization_seam_clamp_count", "overflow_initialization_outside",
    # E39 B4 (V7_DEEP_WALL_CHECK=1): field [34], was status_reserved_2 (never written before)
    "overflow_deep_wall_skip_count",
)
# Replicas of the two-layer ghost (V7_GHOST_LAYERS = 2) carry only what the
# neighbour sweeps read (correction / density / force, incl. the inner layer
# recomputed as self): position_voxel_id, velocity_mass, density_pressure,
# material. Migrants still carry all 9 transported fields.
_REPLICA_TRANSPORT_FIELDS = ("position_voxel_id", "velocity_mass",
                             "density_pressure", "material")
# V7_PACKED_REPLICAS: words per packed replica record and layer (helpers.glsl
# packed_layer_base: direction * 16 + layer * 8): x y z rho | vx vy vz material-bits,
# the same record for G1 and G2.
_PACKED_REPLICA_WORDS = 8


# Path A+ (P2): buffer names that participate in cross-GPU transport and
# therefore must be created with SHARING_MODE_CONCURRENT across the compute
# and transfer queue families. Without this, vkCmdCopyBuffer on the transfer
# queue would require explicit queue family ownership transfer barriers in
# every readback / upload cmd buffer (~24 barriers per direction per frame).
# CONCURRENT trades a small driver-side optimization (~3-5% theoretical
# overhead, in practice within noise on our test setup) for skipping all
# that synchronization plumbing.
#
# Same name set as TRANSPORT_SET0_BINDINGS + set 1 inside_particle_count/
# index + set 3 global_status (the 12 segments listed in _compute_transport_
# segments). All other buffers stay SHARING_MODE_EXCLUSIVE (compute-queue-
# only).
_CONCURRENT_BUFFER_NAMES = frozenset({
    # Set 0 SoA (9)
    "position_voxel_id",
    "density_pressure",
    "velocity_mass",
    "acceleration",
    "shift",
    "material",
    "correction_inverse",
    "density_gradient_kernel_sum",
    "extension_fields",
    # Set 1 (3)
    "inside_particle_count",
    "inside_particle_index",
    "ghost_voxel_first_particle_id",   # V7_COMPACT_GHOST_LISTS
    "ghost_packed_words",              # V7_PACKED_REPLICAS
    # Set 3 (1)
    "global_status",
})


@dataclass
class _TransportSegment:
    """One vkCmdCopyBuffer region for ghost transport.

    For READBACK direction: src = device buffer, dst = sender_staging
    For UPLOAD direction:   src = recv_staging, dst = device buffer
    Both sides use the same staging_offset for a given segment index.
    """
    buffer_name: str        # 'position_voxel_id', 'inside_particle_count', 'global_status', etc.
    device_offset: int      # byte offset in the device buffer
    staging_offset: int     # byte offset in the staging buffer
    size: int               # byte count
    # V6 count-aware worker plan: a per-particle segment holds `stride` bytes
    # per slot and only the first `count` slots are live, where `count` is the
    # uint32 at `count_staging_offset` in the same staging buffer (the
    # sender's allocation counter for that region). None = copy in full.
    stride: int = 0
    count_staging_offset: Optional[int] = None
    # Pool region the slots belong to ("mixed" = the V5 pool; "inner" /
    # "outer" / "migrant" = the two-layer regions); None for voxel lists,
    # count words and the stamp. Used by the pool-peak recorder.
    region: Optional[str] = None


# ============================================================================
# Internal helpers: Buffer + BufferSpec
# ============================================================================

@dataclass
class _Buffer:
    handle: object
    memory: object
    size: int
    # Only populated for HOST_VISIBLE staging buffers (persistent map).
    mapped: Optional[object] = None
    mapped_view: Optional[np.ndarray] = None    # numpy uint8 view


@dataclass
class _BufferSpec:
    name: str
    set_index: int
    binding: int
    size: int
    usage: int


@dataclass(frozen=True)
class _BandOverlapPlan:
    """E39 B3: where force_deep_interior_scratch's workgroups run on a slab that resolves V7_BAND_OVERLAP on.
    Its per-own-particle dispatch (group_count workgroups, base 0) becomes the segment [0, phase_b_groups) in phase
    B (none when 0) and one (partner, first group, group count) segment per phase C pair, in recording order; the
    segments tile [0, group_count) exactly once. partner = the dispatch recorded next to the segment with no
    barrier between them: "correction_density" (correction_density_boundary_band), "copy" (the density copy pass)
    or "force" (force_boundary_band). force_deep_first: the segment is recorded before its partner."""
    group_count: int
    phase_b_groups: int
    segments: tuple
    force_deep_first: bool


# ============================================================================
# SphSimulatorV7
# ============================================================================

class SphSimulatorV7:
    """Per-GPU SPH simulator using timeline semaphore + 3-submit-per-frame pattern.

    Construction (Phase 2 scope): allocates buffers, builds descriptors,
    builds pipelines, creates timeline semaphore. Does NOT yet record cmd
    buffers, run bootstrap, or submit anything — those are Phase 3.

    Lifecycle:
        ctx = VulkanContextV7.create(device_index=...)
        case = load_case_v7("cases/lid_driven_cavity_2d/case.yaml")
        with SphSimulatorV7(ctx, case) as sim:
            sim.bootstrap()                  # single-GPU path
            sim.prepare_step_cmd_buffers()
            for n in range(max_steps):
                sim.submit_phase_a(n); sim.submit_phase_b(n); sim.submit_phase_c(n)
                sim.wait_frame_done(n)
    """

    # ========================================================================
    # Construction / destruction
    # ========================================================================

    def __init__(self, ctx: VulkanContextV7, case: CaseV7,
                 *, sync_scheme: str = "aggregated") -> None:
        self.ctx = ctx
        self.case = case
        # E37 wall boundary option (case.yaml numerics.wall_boundary, shaders/wall_boundary.glsl): simple = the v6
        # walls (default), adami = Adami et al. 2012 wall pressure + no-slip with the walls storing rho0 (E36's
        # adami_rho0, v7-wall-bc WALL_BC = 3). adami runs one slab only: wall_extrapolate.comp reads local
        # neighbours and the ghost transport does not carry wall_dummy_velocity.
        self.wall_boundary = case.numerics.wall_boundary
        if self._wall_adami and (case.transport.has_leading_peer or case.transport.has_trailing_peer):
            raise ValueError("wall_boundary adami supports one GPU (K = 1) only in this release; this slab has a "
                             f"{'leading' if case.transport.has_leading_peer else 'trailing'} peer")
        # V7_FAKE_BAND_TEST moves part of the density pass into phase C (density_boundary_band, after phase B's
        # wall pass), so the wall pass would read band fluid density before this step's band density exists.
        if self._wall_adami and _FAKE_BAND_COLUMN > 0:
            raise ValueError("V7_FAKE_BAND_TEST is not supported with wall_boundary adami: the phase B wall pass "
                             "would read the fake band's fluid density before density_boundary_band writes it")

        # Keep-alive bag for cffi cdata referenced by VkSpecializationInfo.
        # Python GC would otherwise free the cdata before pipeline creation.
        # Must be initialized BEFORE pipeline build (which calls _make_spec_info).
        self._spec_keepalive: list = []

        self._check_workgroup_limit()
        # V7_PACKED_REPLICAS ships no G1 pressure: the G1-as-self density pass and the C4 copy of the G1 region must
        # run. partition_v7.configured_packed_replicas() checks V7_DIAG_GHOST_SELF at call time, this module froze it
        # at import (_DIAG_GHOST_SELF_KERNELS): both must allow it.
        if configured_packed_replicas() and self.ghost_layers() >= 2 and "density" not in _DIAG_GHOST_SELF_KERNELS:
            raise ValueError("V7_PACKED_REPLICAS=1 ships no G1 pressure, but this process imported simulator_v7 with "
                             f"V7_DIAG_GHOST_SELF={','.join(_DIAG_GHOST_SELF_KERNELS)!r} (no 'density')")
        # V7_BAND_WIDTHS: boundary band widths (own voxel columns) of correction / density / force, read once
        # here; every spec 82 and band dispatch of this sim uses these values.
        self.band_widths = self._configured_band_widths()
        # E39 B4: V7_DEEP_WALL_SKIP resolved for this slab (buffers, pipelines and recordings follow it).
        if _DEEP_WALL_SKIP != "0":
            print(f"[SimV7] V7_DEEP_WALL_SKIP={_DEEP_WALL_SKIP}: "
                  f"{'on' if self._deep_wall_skip_active() else 'off'} ({self.deep_wall_skip_resolution[1]})"
                  + (f", V7_DEEP_WALL_CHECK=1" if _DEEP_WALL_CHECK else ""))
        # E39 B1: V7_FUSED_CORRECTION_DENSITY resolved for this slab (modules, pipelines and recordings follow it).
        if _FUSED_CORRECTION_DENSITY:
            print(f"[SimV7] V7_FUSED_CORRECTION_DENSITY=1: "
                  f"{'fused' if self._fused_correction_density_active() else 'separate kernels'} "
                  f"({self.fused_correction_density_resolution[1]})")
        # E39 B3: V7_BAND_OVERLAP resolved for this slab (the FORCE_SEGMENT module / pipelines and the phase B / C
        # recordings follow it).
        if _BAND_OVERLAP != "0":
            print(f"[SimV7] V7_BAND_OVERLAP={_BAND_OVERLAP}: "
                  f"{'on' if self._band_overlap_active() else 'off'} ({self.band_overlap_resolution[1]})")

        # Buffer allocation
        self._buffer_specs = self._build_buffer_specs()
        self.buffers = self._allocate_buffers()
        self.scratch_buffers = self._allocate_scratch_buffers()
        self.staging_buffers = self._allocate_staging_buffers()

        # Descriptors
        self.descriptor_layouts = self._build_descriptor_layouts()
        self.descriptor_pool = self._create_descriptor_pool()
        self.descriptor_sets = self._allocate_descriptor_sets()
        self._wire_descriptor_sets()

        # Pipelines
        self.pipeline_layout = self._build_pipeline_layout()
        self.shader_modules = self._load_shader_modules()
        self.pipelines = self._build_compute_pipelines()

        # Defrag has its own 5-set pipeline layout (set 4 = scratch SoA dst)
        # and its own descriptor pool + set 4 descriptor.
        self.defrag_set4_layout = self._build_defrag_set4_layout()
        self.defrag_descriptor_pool, self.defrag_set4 = self._allocate_defrag_set4()
        self._wire_defrag_set4()
        self.defrag_pipeline_layout = self._build_defrag_pipeline_layout()
        self.pipelines["defrag"] = self._build_defrag_pipeline()

        # Frame sync scheme: owns the timeline semaphore(s) and decides every
        # submit site's wait/signal (semaphore, value) pairs. "aggregated" =
        # historical 5N single timeline; "per-direction" = 3N main + 2N
        # transport timeline per peer direction (N-GPU chain safe). See
        # experiment/v7/utils/sync_scheme_v7.py + docs/sph_v5_design.md §3.1.
        peer_directions = tuple(
            direction for direction, has_peer in (
                ("leading", case.transport.has_leading_peer),
                ("trailing", case.transport.has_trailing_peer),
            ) if has_peer)
        self.sync = make_sync_scheme(sync_scheme, peer_directions)
        self.sync.create(ctx.device)

        # cmd buffer slots reserved for Phase 3
        self.phase_a_cmd: Any = None
        self.phase_b_cmd: Any = None
        self.phase_c_cmd: Any = None
        self.phase_c_cmd_odd: Any = None   # bench parity regions: odd-frame phase C
        self.defrag_cmd: Any = None
        # Path A+ (P4): per-direction transfer queue cmd buffers. Allocated
        # from ctx.transfer_command_pool and submitted on ctx.transfer_queue.
        # Each readback cmd runs in parallel with Phase B (correction_interior
        # + density_deep_interior) on the compute queue; each upload cmd runs
        # in parallel with Phase B's tail end (after worker memcpy) before
        # Phase C starts. Maps {direction → cmd}; only the directions in
        # _transport_segments have entries.
        self.transfer_readback_cmds: dict[str, Any] = {}
        self.transfer_upload_cmds: dict[str, Any] = {}
        # E29 step trace (phase_trace_v7.StepTracer): phase A / B and the transfer
        # cmds are ALSO recorded once per frame parity (each parity writes its own
        # timestamp slots), so frame n's ticks survive until the host reads them
        # before submitting frame n + 2 at depth 2. Off by default: the cmds and
        # submits are then exactly the ones above.
        self.step_trace_parity: bool = False
        self.phase_a_cmd_odd: Any = None
        self.phase_b_cmd_odd: Any = None
        self.transfer_readback_cmds_odd: dict[str, Any] = {}
        self.transfer_upload_cmds_odd: dict[str, Any] = {}
        # Single-GPU baseline path: one combined cmd buffer replacing the
        # 3-submit phase A/B/C pattern. Used only when this sim has no peer
        # (see prepare_step_single_cmd_buffer()). Cannot coexist with the
        # dual-GPU path on the same sim — the two recordings would clash on
        # SIMULTANEOUS_USE replay scheduling.
        self.step_single_cmd: Any = None
        # P3.C validation flag: when True, _record_step_single_cmd uses the
        # split pipeline variants (correction_interior + _boundary, density_
        # deep_interior + _boundary, force_deep_interior + _boundary) in
        # place of the *_all variants. Single-GPU mode's empty boundary band
        # makes the two paths bit-equivalent — any divergence is a shader
        # bug. Set BEFORE prepare_step_single_cmd_buffer() to take effect.
        self.step_single_use_split: bool = False

        # Optional GPU-timestamp collector. Attached by the benchmark runner
        # BEFORE prepare_step_cmd_buffers() (and BEFORE the first defrag) so
        # that ticks get baked into the pre-recorded SIMULTANEOUS_USE cmds.
        # When None, _bench_tick / _bench_reset_step / _bench_reset_defrag
        # are pure no-ops; the production runner pays zero per-frame cost.
        self.bench: Any = None
        # M5a: separate BenchTimer for the TRANSFER queue's readback/upload
        # cmds (own query pool — avoids cross-queue reset races). Same-pool
        # diffs are exact; cross-pool diffs vs self.bench are driver-
        # consistent but not spec-guaranteed (see bench_v7 audit notes).
        # Attach with queue_family_index=ctx.transfer_queue_family_index
        # BEFORE prepare_step_cmd_buffers.
        self.bench_transfer: Any = None

        # Driver submit/signal lock, scoped per V7_SUBMIT_LOCK_SCOPE
        # (default: one lock per physical GPU). See module-level comment.
        self._driver_submit_lock = _driver_submit_lock_for(
            ctx.physical_device_index)

        self._destroyed = False
        print(f"[SimV7] init complete on {ctx.device_name} "
              f"(own={case.capacities.own_pool_size}, "
              f"ghost L={case.capacities.leading_ghost_pool_size} "
              f"T={case.capacities.trailing_ghost_pool_size}, wall_boundary={self.wall_boundary})")

    @property
    def _wall_adami(self) -> bool:
        """E37: case.yaml numerics.wall_boundary == adami. Read from the case (not stored by __init__): the CPU
        tests call _build_buffer_specs / _global_entries on simulators built with object.__new__ + .case."""
        return self.case.numerics.wall_boundary == "adami"

    def destroy(self) -> None:
        if self._destroyed:
            return
        device = self.ctx.device
        vkDeviceWaitIdle(device)

        # cmd buffers (only if Phase 3 recorded them)
        cmd_pool = self.ctx.command_pool
        for cmd in (self.phase_a_cmd, self.phase_b_cmd,
                    self.phase_c_cmd, self.phase_c_cmd_odd,
                    self.phase_a_cmd_odd, self.phase_b_cmd_odd,
                    self.defrag_cmd, self.step_single_cmd):
            if cmd is not None:
                vkFreeCommandBuffers(device, cmd_pool, 1, [cmd])

        # Path A+ transfer queue cmd buffers (P4)
        transfer_pool = self.ctx.transfer_command_pool
        for cmds in (self.transfer_readback_cmds, self.transfer_upload_cmds,
                     self.transfer_readback_cmds_odd, self.transfer_upload_cmds_odd):
            for cmd in list(cmds.values()):
                vkFreeCommandBuffers(device, transfer_pool, 1, [cmd])
        self.transfer_readback_cmds = {}
        self.transfer_upload_cmds = {}
        self.transfer_readback_cmds_odd = {}
        self.transfer_upload_cmds_odd = {}

        self.sync.destroy(device)

        for pipeline in self.pipelines.values():
            vkDestroyPipeline(device, pipeline, None)
        self.pipelines = {}

        for module in self.shader_modules.values():
            vkDestroyShaderModule(device, module, None)
        self.shader_modules = {}

        if self.defrag_pipeline_layout is not None:
            vkDestroyPipelineLayout(device, self.defrag_pipeline_layout, None)
            self.defrag_pipeline_layout = None
        if self.pipeline_layout is not None:
            vkDestroyPipelineLayout(device, self.pipeline_layout, None)
            self.pipeline_layout = None

        if self.defrag_descriptor_pool is not None:
            vkDestroyDescriptorPool(device, self.defrag_descriptor_pool, None)
            self.defrag_descriptor_pool = None
        if self.descriptor_pool is not None:
            vkDestroyDescriptorPool(device, self.descriptor_pool, None)
            self.descriptor_pool = None

        if self.defrag_set4_layout is not None:
            vkDestroyDescriptorSetLayout(device, self.defrag_set4_layout, None)
            self.defrag_set4_layout = None
        for layout in self.descriptor_layouts:
            vkDestroyDescriptorSetLayout(device, layout, None)
        self.descriptor_layouts = []

        # Unmap + destroy staging
        for buf in self.staging_buffers.values():
            if buf.mapped is not None:
                vkUnmapMemory(device, buf.memory)
            vkDestroyBuffer(device, buf.handle, None)
            vkFreeMemory(device, buf.memory, None)
        self.staging_buffers = {}

        for buf in self.scratch_buffers.values():
            vkDestroyBuffer(device, buf.handle, None)
            vkFreeMemory(device, buf.memory, None)
        self.scratch_buffers = {}

        for buf in self.buffers.values():
            vkDestroyBuffer(device, buf.handle, None)
            vkFreeMemory(device, buf.memory, None)
        self.buffers = {}

        self._destroyed = True

    def __enter__(self) -> "SphSimulatorV7":
        return self

    def __exit__(self, *_: Any) -> None:
        self.destroy()

    # ========================================================================
    # Frame sync (delegated to self.sync — see sync_scheme_v7.py for the
    # timeline layouts: "aggregated" 5N single timeline vs "per-direction"
    # 3N main + 2N transport timeline per peer direction)
    # ========================================================================

    @property
    def timeline(self):
        """Primary timeline semaphore (carries frame_done). Kept for coarse
        progress introspection; per-direction transport semaphores are only
        reachable through self.sync."""
        return self.sync.primary_semaphore()

    def current_timeline_value(self) -> int:
        return vkGetSemaphoreCounterValue(
            self.ctx.device, self.sync.primary_semaphore())

    def semaphore_value(self, semaphore) -> int:
        if _FAST_SUBMIT:
            out = _ffi.new("uint64_t*")
            result = _lib.vkGetSemaphoreCounterValue(self.ctx.device, semaphore, out)
            if result != 0:
                raise RuntimeError(f"vkGetSemaphoreCounterValue failed: VkResult {result}")
            return int(out[0])
        return vkGetSemaphoreCounterValue(self.ctx.device, semaphore)

    def frame_done_reached(self, frame_n: int) -> bool:
        """Non-blocking: has this sim's frame_done(frame_n) been signaled?"""
        semaphore, value = self.sync.frame_done_op(frame_n)
        return self.semaphore_value(semaphore) >= value

    def sync_state(self) -> dict:
        """{semaphore_name: current value} across all sync semaphores —
        watchdog / debug prints."""
        return self.sync.state(self.ctx.device)

    # ========================================================================
    # High-level frame API
    # bootstrap() implemented in Section 8; submit_phase_* + wait_* in Section 10
    # ========================================================================

    def submit_defrag_and_wait(self) -> None:
        if self.defrag_cmd is None:
            self.defrag_cmd = self._record_defrag_cmd()
        self.ctx.submit_and_wait(self.defrag_cmd)

    # ========================================================================
    # Worker-facing accessors
    # ========================================================================

    def sender_staging_view(self, direction: str):
        """numpy.uint8 view over sender_staging_<direction>. Worker reads from
        this view via slice copy (read side of the worker memcpy)."""
        return self.staging_buffers[f"sender_staging_{direction}"].mapped_view

    def receiver_staging_view(self, direction: str):
        """numpy.uint8 view over receiver_staging_<direction>. Worker writes
        to this view via slice assignment (write side of the worker memcpy)."""
        return self.staging_buffers[f"receiver_staging_{direction}"].mapped_view

    def timeline_semaphore(self):
        return self.timeline

    def device(self):
        return self.ctx.device

    # ========================================================================
    # Readback (Phase 3 — implemented in Section 9 below)
    # ========================================================================

    def readback_positions(self):
        raise NotImplementedError("Phase 3 — TODO")

    def get_render_buffers(self) -> dict:
        """Buffer handles the renderer needs to bind as descriptor inputs.

        Vert shader reads:
          - position_voxel_id  (.xyz = position, .w = voxel_id_as_float)
          - velocity_mass      (.xyz = v_{n+1/2})
          - density_pressure   (.x = ρ, .y = P)
        Used by viewer pipeline; sim itself never reads these as descriptor
        bindings — they live in set 0 for the compute pipeline.
        """
        return {
            "position_voxel_id": self.buffers["position_voxel_id"].handle,
            "velocity_mass":     self.buffers["velocity_mass"].handle,
            "density_pressure":  self.buffers["density_pressure"].handle,
            "global_status":     self.buffers["global_status"].handle,
        }

    # ========================================================================
    # Section 1: Validation helpers
    # ========================================================================

    def _check_workgroup_limit(self) -> None:
        props = vkGetPhysicalDeviceProperties(self.ctx.physical_device)
        max_x = props.limits.maxComputeWorkGroupSize[0]
        wg = self.case.capacities.workgroup_size
        if wg > max_x:
            raise RuntimeError(
                f"WORKGROUP_SIZE {wg} > device limit {max_x} on {self.ctx.device_name}")
        if _GHOST_SEND_LANES:
            local_size = _GHOST_SEND_LOCAL_SIZE
            if local_size > min(max_x, props.limits.maxComputeWorkGroupInvocations):
                raise RuntimeError(
                    f"ghost_send_lanes.comp local size {local_size} > device limit "
                    f"{min(max_x, props.limits.maxComputeWorkGroupInvocations)} on {self.ctx.device_name}")
            self._ghost_send_groups_per_workgroup()

    @staticmethod
    def _ghost_send_groups_per_workgroup() -> int:
        """E39 B6: lane groups per workgroup of ghost_send_lanes.comp (local size
        / lanes); the local size must be a multiple of the lanes and the groups
        must fit the shader's shared arrays."""
        local_size, lanes = _GHOST_SEND_LOCAL_SIZE, _GHOST_SEND_LANES
        if lanes <= 0 or local_size % lanes or local_size // lanes > _GHOST_SEND_MAXIMUM_GROUPS_PER_WORKGROUP:
            raise ValueError(f"ghost_send_lanes.comp: local size {local_size} with {lanes} lanes per group "
                             f"(needs lanes > 0 dividing the local size, at most "
                             f"{_GHOST_SEND_MAXIMUM_GROUPS_PER_WORKGROUP} groups per workgroup)")
        return local_size // lanes

    # ========================================================================
    # Section 2: Buffer specs + allocation
    # ========================================================================

    def _build_buffer_specs(self) -> list[_BufferSpec]:
        case = self.case
        pool_capacity = case.capacities.total_pool_capacity()
        voxel_capacity = 1 + case.grid.total_voxel_count()
        cap_inside = case.capacities.max_particles_per_voxel
        cap_incoming = case.capacities.max_incoming_per_voxel
        n_materials = max(len(case.materials), 1)

        BSU = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
        TRANSFER = (VK_BUFFER_USAGE_TRANSFER_DST_BIT
                    | VK_BUFFER_USAGE_TRANSFER_SRC_BIT)
        VERT = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT

        # E39 B4 (only on a slab that skips deep walls, so V7_DEEP_WALL_SKIP=0 keeps the old layouts): set 1 binding 8
        # = per-voxel presence | deep marker (deep_wall_marker.comp), set 3 binding 10 = the decision record (two
        # stamps per particle; a 16-byte stub unless recording). Not in DEFRAG_SET0_BINDINGS / the transport.
        deep_wall_specs = [
            _BufferSpec("deep_wall_voxel_flag",         1, 8,  4 * 2 * voxel_capacity, BSU | TRANSFER),
            _BufferSpec("deep_wall_skip_record",        3, 10,
                        8 * pool_capacity if self._deep_wall_record() else 16, BSU | TRANSFER),
        ] if self._deep_wall_skip_active() else []

        return [
            # Set 0: particle SoA
            _BufferSpec("position_voxel_id",            0, 0, 16 * pool_capacity, BSU | TRANSFER | VERT),
            _BufferSpec("density_pressure",             0, 1,  8 * pool_capacity, BSU | TRANSFER | VERT),
            _BufferSpec("density_pressure_scratch",     0, 2,  8 * pool_capacity, BSU | TRANSFER | VERT),
            _BufferSpec("velocity_mass",                0, 3, 16 * pool_capacity, BSU | TRANSFER | VERT),
            _BufferSpec("acceleration",                 0, 4, 16 * pool_capacity, BSU | TRANSFER),
            _BufferSpec("shift",                        0, 5, 16 * pool_capacity, BSU | TRANSFER),
            _BufferSpec("material",                     0, 6,  4 * pool_capacity, BSU | TRANSFER),
            _BufferSpec("correction_inverse",           0, 7, 32 * pool_capacity, BSU | TRANSFER),
            _BufferSpec("density_gradient_kernel_sum",  0, 8, 16 * pool_capacity, BSU | TRANSFER),
            _BufferSpec("extension_fields",             0, 9, 16 * pool_capacity, BSU | TRANSFER),
            # E37 wall_boundary adami: wall dummy velocity + fluid kernel sum (wall_extrapolate.comp); transient,
            # not in DEFRAG_SET0_BINDINGS / the transport / the restart state. simple: a 16-byte stub (binding 10
            # is declared by force.comp, never read).
            _BufferSpec("wall_dummy_velocity",          0, 10, 16 * pool_capacity if self._wall_adami else 16,
                        BSU | TRANSFER),

            # Set 1: voxel cells
            _BufferSpec("inside_particle_count",        1, 0,  4 * voxel_capacity,                BSU | TRANSFER),
            _BufferSpec("incoming_particle_count",      1, 1,  4 * voxel_capacity,                BSU | TRANSFER),
            _BufferSpec("inside_particle_index",        1, 2,  4 * voxel_capacity * cap_inside,   BSU | TRANSFER),
            _BufferSpec("incoming_particle_index",      1, 3,  4 * voxel_capacity * cap_incoming, BSU | TRANSFER),
            _BufferSpec("voxel_base_offset",            1, 4,  4 * voxel_capacity,                BSU | TRANSFER),
            # V7_COMPACT_GHOST_LISTS: per-voxel first replica pid (ghost range used)
            _BufferSpec("ghost_voxel_first_particle_id", 1, 5, 4 * voxel_capacity,                BSU | TRANSFER),
            # V7_BAND_COMPACT_DISPATCH: compacted band pid list + per-voxel offsets
            _BufferSpec("band_compact_list",            1, 6,  self._band_compact_list_bytes(), BSU | TRANSFER),
            # V7_PACKED_REPLICAS: packed replica out/inbox, 2 layers x 8 words x R per direction
            _BufferSpec("ghost_packed_words",           1, 7,
                        max(4, 4 * 2 * 2 * _PACKED_REPLICA_WORDS * case.capacities.replica_region_size)
                        if configured_packed_replicas() else 4,                       BSU | TRANSFER),

            # Set 3: global / pool-health / materials
            _BufferSpec("global_status",                3, 0,  _GLOBAL_STATUS_BYTES,    BSU | TRANSFER),
            # binding 1: V5 pool-health watermark (4 uint = 16 B). Reclaims the
            # former unused overflow_log ring. See common.glsl PoolHealthBuffer.
            _BufferSpec("pool_health",                  3, 1,  16,                      BSU | TRANSFER),
            # V5 cleanup: bindings 2-6 (inlet_template / dispatch_indirect /
            # ghost_out_packet / ghost_in_staging / diagnostic) removed — they
            # were declared in V1/V2 but never read or written by any kernel.
            _BufferSpec("material_parameters",          3, 7,  48 * n_materials,        BSU | TRANSFER),
            _BufferSpec("defrag_scratch_counter",       3, 8,   4,                      BSU | TRANSFER),
            # V7_BAND_COMPACT_DISPATCH: indirect dispatch sizes + list column starts
            _BufferSpec("band_compact_meta",            3, 9, 128,
                        BSU | TRANSFER | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT),
        ] + deep_wall_specs

    def _allocate_buffer(
        self,
        size: int,
        usage: int,
        required_properties: int,
        preferred_properties: int = 0,
        shared_with_transfer: bool = False,
    ) -> _Buffer:
        """Allocate a device-local buffer.

        ``shared_with_transfer=True`` selects SHARING_MODE_CONCURRENT with
        both the compute and transfer queue family indices. Required for
        any buffer that the Path A+ transfer queue will vkCmdCopyBuffer
        (read or write). When the transfer family equals the compute family
        (fallback path in VulkanContextV7), CONCURRENT degenerates to a
        single-family case and the driver should treat it like EXCLUSIVE
        with no penalty.
        """
        if size == 0:
            raise ValueError("buffer size must be > 0")
        if (shared_with_transfer
                and self.ctx.transfer_queue_family_index
                    != self.ctx.compute_queue_family_index):
            family_indices = [
                self.ctx.compute_queue_family_index,
                self.ctx.transfer_queue_family_index,
            ]
            bci = VkBufferCreateInfo(
                size=size, usage=usage,
                sharingMode=VK_SHARING_MODE_CONCURRENT,
                queueFamilyIndexCount=len(family_indices),
                pQueueFamilyIndices=family_indices,
            )
        else:
            # EXCLUSIVE = single-queue-family ownership; no penalty if the
            # transfer family is the same as compute (no real cross-queue
            # access anyway).
            bci = VkBufferCreateInfo(
                size=size, usage=usage,
                sharingMode=VK_SHARING_MODE_EXCLUSIVE)
        handle = vkCreateBuffer(self.ctx.device, bci, None)
        reqs = vkGetBufferMemoryRequirements(self.ctx.device, handle)
        type_index = self.ctx.find_memory_type(
            reqs.memoryTypeBits, required_properties, preferred_properties)
        alloc_info = VkMemoryAllocateInfo(
            allocationSize=reqs.size, memoryTypeIndex=type_index)
        memory = vkAllocateMemory(self.ctx.device, alloc_info, None)
        vkBindBufferMemory(self.ctx.device, handle, memory, 0)
        return _Buffer(handle=handle, memory=memory, size=size)

    def _allocate_buffers(self) -> dict[str, _Buffer]:
        buffers: dict[str, _Buffer] = {}
        total = 0
        concurrent_count = 0
        # V7_CONCURRENT_BUFFERS=0 is a SINGLE-GPU diagnostic only: it keeps
        # every buffer EXCLUSIVE to the compute family so the cost of
        # CONCURRENT sharing can be measured against the V0 reference. A
        # multi-GPU run needs CONCURRENT for the transfer-queue DMA.
        concurrent_enabled = os.environ.get("V7_CONCURRENT_BUFFERS", "1") == "1"
        for spec in self._buffer_specs:
            shared = concurrent_enabled and spec.name in _CONCURRENT_BUFFER_NAMES
            buffers[spec.name] = self._allocate_buffer(
                spec.size, spec.usage,
                required_properties=VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                shared_with_transfer=shared,
            )
            total += spec.size
            if shared:
                concurrent_count += 1
        print(f"[SimV7] device-local buffers: {len(buffers)}, "
              f"{total / (1024 * 1024):.2f} MB "
              f"({concurrent_count} CONCURRENT for transfer queue)")
        return buffers

    def _allocate_scratch_buffers(self) -> dict[str, _Buffer]:
        """Per-set-0 SoA, allocate a scratch twin for defrag.comp's writes
        (copied back via vkCmdCopyBuffer in defrag cmd)."""
        scratch: dict[str, _Buffer] = {}
        total = 0
        for spec in self._buffer_specs:
            if spec.set_index != 0:
                continue
            if spec.binding not in DEFRAG_SET0_BINDINGS:
                continue
            scratch[spec.name] = self._allocate_buffer(
                spec.size,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                | VK_BUFFER_USAGE_TRANSFER_DST_BIT
                | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                required_properties=VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            )
            total += spec.size
        print(f"[SimV7] defrag scratch buffers: {len(scratch)}, "
              f"{total / (1024 * 1024):.2f} MB")
        return scratch

    def _compute_transport_segments(self, direction: str) -> tuple[list[_TransportSegment], int]:
        """Build the V1-equivalent 12-segment list for one direction. Returns
        (segments, total_staging_bytes). direction ∈ {'leading','trailing'}."""
        case = self.case
        cap_inside = case.capacities.max_particles_per_voxel

        if direction == "leading":
            pool_size = case.capacities.leading_ghost_pool_size
            ghost_voxel_count = case.ghost_grid.leading_ghost_voxel_count
            pid_first = 1                           # leading pid range starts at 1
            vid_first = 1                           # leading vid range starts at 1
            send_count_offset = _OFFSET_GHOST_SEND_LEADING
            recv_count_offset = _OFFSET_GHOST_RECV_LEADING
        elif direction == "trailing":
            pool_size = case.capacities.trailing_ghost_pool_size
            ghost_voxel_count = case.ghost_grid.trailing_ghost_voxel_count
            # Trailing pid range = [leading + own + 1, leading + own + trailing]
            pid_first = (case.capacities.leading_ghost_pool_size
                         + case.capacities.own_pool_size + 1)
            # Trailing vid range = [total - trailing + 1, total]
            vid_first = case.grid.total_voxel_count() - ghost_voxel_count + 1
            send_count_offset = _OFFSET_GHOST_SEND_TRAILING
            recv_count_offset = _OFFSET_GHOST_RECV_TRAILING
        else:
            raise ValueError(f"bad direction: {direction}")

        segments: list[_TransportSegment] = []
        staging_offset = 0
        if pool_size == 0 or ghost_voxel_count == 0:
            return segments, 0
        if case.capacities.replica_region_size > 0:
            return self._compute_two_layer_transport_segments(
                direction, pool_size, ghost_voxel_count, pid_first, vid_first,
                send_count_offset, recv_count_offset)

        # 1-9. Nine SoA fields × ghost-pid range (V7_LEAN_TRANSPORT: four,
        #      + extension_fields with V7_TRANSPORT_EXTENSION)
        particle_fields = transported_particle_fields()
        for name in particle_fields:
            stride = _SET0_BYTE_STRIDES[name]
            size = stride * pool_size
            device_offset = stride * pid_first
            segments.append(_TransportSegment(name, device_offset, staging_offset, size,
                                              stride=stride, region="mixed"))
            staging_offset += size

        # 10. set 1 inside_particle_count × ghost-vid range
        size = 4 * ghost_voxel_count
        segments.append(_TransportSegment(
            "inside_particle_count", 4 * vid_first, staging_offset, size))
        staging_offset += size

        # 11. set 1 inside_particle_index × ghost-vid range × MAX_PARTICLES_PER_VOXEL
        #     (V7_COMPACT_GHOST_LISTS: one first-pid word per ghost voxel)
        segments.append(self._ghost_list_segment(ghost_voxel_count, vid_first, staging_offset))
        staging_offset += segments[-1].size

        # 12. set 3 ghost_send_*_count (sender side: read device→staging at
        #     send_count_offset; receiver side: write staging→device at
        #     recv_count_offset). Buffer is global_status; both sides use the
        #     same staging slot but different device offsets. We store the
        #     SENDER's device offset here; recv side overrides at cmd record time.
        count_staging_offset = staging_offset
        segments.append(_TransportSegment(
            "global_status", send_count_offset, staging_offset, 4))
        staging_offset += 4
        # The nine per-particle segments are live up to the send count.
        for segment in segments[:len(particle_fields)]:
            segment.count_staging_offset = count_staging_offset

        # 13. set 3 frame stamp: sender's frame_stamp → receiver's
        #     ghost_stamp_<direction> (same send→recv override pattern as the
        #     count). LAST 4 bytes of the staging — the worker's host-side
        #     stamp check relies on that position.
        recv_stamp_offset = (_OFFSET_GHOST_STAMP_LEADING if direction == "leading"
                             else _OFFSET_GHOST_STAMP_TRAILING)
        segments.append(_TransportSegment(
            "global_status", _OFFSET_FRAME_STAMP, staging_offset, 4))
        staging_offset += 4

        # Receiver-side device-offset overrides for the two global_status
        # segments, keyed by the SENDER-side offset recorded in the segment.
        self._recv_status_overrides = getattr(self, "_recv_status_overrides", {})
        self._recv_status_overrides[direction] = {
            send_count_offset: recv_count_offset,
            _OFFSET_FRAME_STAMP: recv_stamp_offset,
        }
        # Legacy alias used by older code paths.
        self._recv_count_offsets = getattr(self, "_recv_count_offsets", {})
        self._recv_count_offsets[direction] = recv_count_offset
        return segments, staging_offset

    def _compute_two_layer_transport_segments(
        self, direction: str, pool_size: int, ghost_voxel_count: int,
        pid_first: int, vid_first: int, send_count_offset: int,
        recv_count_offset: int,
    ) -> tuple[list[_TransportSegment], int]:
        """V7_GHOST_LAYERS = 2 staging layout for one direction.

        Ghost pool = [inner replicas R | outer replicas R | migrants]. The
        replica regions travel with the 4 fields the neighbour sweeps read
        (_REPLICA_TRANSPORT_FIELDS), the migrant region with all 9. Then the
        two ghost columns' voxel lists, the three allocation counters (migrant
        counter -> the receiver's install count; replica counters -> diagnostic
        recv slots, they also bound the count-aware copy) and, LAST, the frame
        stamp (the worker's host-side stamp check reads the last 4 bytes)."""
        case = self.case
        cap_inside = case.capacities.max_particles_per_voxel
        replica_region = case.capacities.replica_region_size
        migrant_region = pool_size - 2 * replica_region
        if migrant_region <= 0:
            raise ValueError(f"{direction}: ghost pool {pool_size} leaves no migrant "
                             f"region after 2 x {replica_region} replica slots")
        segments: list[_TransportSegment] = []
        staging_offset = 0
        counted_segments: list[tuple[_TransportSegment, str]] = []

        def add_particle_segments(field_names, first_slot: int, slot_count: int,
                                  count_key: str) -> None:
            nonlocal staging_offset
            for name in field_names:
                stride = _SET0_BYTE_STRIDES[name]
                segment = _TransportSegment(
                    name, stride * (pid_first + first_slot), staging_offset,
                    stride * slot_count, stride=stride, region=count_key)
                segments.append(segment)
                counted_segments.append((segment, count_key))
                staging_offset += stride * slot_count

        if configured_packed_replicas():
            # V7_PACKED_REPLICAS: G1 and G2 blocks (x y z rho | vx vy vz
            # material-bits, the same record for both layers) of this direction's
            # packed region (common.glsl GhostPackedBuffer); stride = bytes per
            # replica of the block, live prefix = the layer's replica counter.
            if not configured_compact_ghost_lists():
                raise ValueError("V7_PACKED_REPLICAS=1 needs V7_COMPACT_GHOST_LISTS=1 "
                                 "(expand_ghost_lists unpacks the replicas)")
            region_bytes = 4 * replica_region
            direction_base = ((0 if direction == "leading" else 1)
                              * 2 * _PACKED_REPLICA_WORDS * region_bytes)
            for word_offset, stride, count_key in ((0, 16, "inner"), (4, 16, "inner"),
                                                    (8, 16, "outer"), (12, 16, "outer")):
                segment = _TransportSegment(
                    "ghost_packed_words", direction_base + word_offset * region_bytes,
                    staging_offset, stride * replica_region, stride=stride, region=count_key)
                segments.append(segment)
                counted_segments.append((segment, count_key))
                staging_offset += stride * replica_region
        else:
            add_particle_segments(_REPLICA_TRANSPORT_FIELDS, 0, replica_region, "inner")
            add_particle_segments(_REPLICA_TRANSPORT_FIELDS, replica_region, replica_region, "outer")
        add_particle_segments(transported_particle_fields(), 2 * replica_region,
                              migrant_region, "migrant")

        size = 4 * ghost_voxel_count
        segments.append(_TransportSegment(
            "inside_particle_count", 4 * vid_first, staging_offset, size))
        staging_offset += size
        segments.append(self._ghost_list_segment(ghost_voxel_count, vid_first, staging_offset))
        staging_offset += segments[-1].size

        count_words = (
            ("migrant", send_count_offset, recv_count_offset),
            ("inner", _OFFSET_REPLICA_INNER_SEND[direction], _OFFSET_REPLICA_INNER_RECV[direction]),
            ("outer", _OFFSET_REPLICA_OUTER_SEND[direction], _OFFSET_REPLICA_OUTER_RECV[direction]),
        )
        count_staging_offsets: dict[str, int] = {}
        overrides: dict[int, int] = {}
        for key, send_offset, recv_offset in count_words:
            count_staging_offsets[key] = staging_offset
            segments.append(_TransportSegment("global_status", send_offset, staging_offset, 4))
            overrides[send_offset] = recv_offset
            staging_offset += 4
        for segment, key in counted_segments:
            segment.count_staging_offset = count_staging_offsets[key]

        recv_stamp_offset = (_OFFSET_GHOST_STAMP_LEADING if direction == "leading"
                             else _OFFSET_GHOST_STAMP_TRAILING)
        segments.append(_TransportSegment(
            "global_status", _OFFSET_FRAME_STAMP, staging_offset, 4))
        staging_offset += 4
        overrides[_OFFSET_FRAME_STAMP] = recv_stamp_offset

        self._recv_status_overrides = getattr(self, "_recv_status_overrides", {})
        self._recv_status_overrides[direction] = overrides
        self._recv_count_offsets = getattr(self, "_recv_count_offsets", {})
        self._recv_count_offsets[direction] = recv_count_offset
        return segments, staging_offset

    def _ghost_list_segment(self, ghost_voxel_count: int, vid_first: int,
                            staging_offset: int) -> "_TransportSegment":
        """The ghost columns' inside lists: the MAX_PARTICLES_PER_VOXEL-wide
        inside_particle_index rows, or (V7_COMPACT_GHOST_LISTS) one
        ghost_voxel_first_particle_id word per voxel (expand_ghost_lists.comp
        rebuilds the rows on the receiver)."""
        if configured_compact_ghost_lists():
            return _TransportSegment("ghost_voxel_first_particle_id", 4 * vid_first,
                                     staging_offset, 4 * ghost_voxel_count)
        cap_inside = self.case.capacities.max_particles_per_voxel
        return _TransportSegment("inside_particle_index", 4 * vid_first * cap_inside,
                                 staging_offset, 4 * ghost_voxel_count * cap_inside)

    def transport_staging_bytes(self) -> dict[str, int]:
        """Bytes one readback (= one upload) DMA moves per direction per frame."""
        return dict(self._transport_total_bytes)

    def _allocate_staging_buffers(self) -> dict[str, _Buffer]:
        """Per-direction host-visible stagings (sender CACHED, receiver COHERENT)
        — see docs/sph_v5_design.md §14.5. Persistent-mapped at construction.

        Diagnostic print below shows the actual memory type the driver
        picked for sender and receiver, including whether DEVICE_LOCAL
        was granted (ReBAR-style VRAM exposed to CPU). Useful for
        future memory-type experiments.

        Attempted optimization 2026-05-21 (experiment B3): preferring
        DEVICE_LOCAL for sender and/or receiver. Result: receiver-as-
        ReBAR cut NV's install upload DMA 556→23 µs (24× faster) BUT
        broke worker memcpy — CPU write to ReBAR via numpy[:] runs at
        ~2.3 GB/s (vs theoretical 12 GB/s WC), so worker time grew
        220→1384 µs and stopped fitting inside correction_interior →
        sync hiding collapsed → net fps 228→175 (-23%). Sender-as-
        ReBAR was worse (CPU read of ReBAR is uncached PCIe BAR at
        ~10 MB/s, fps 228→2.7). The worker-bridge architecture
        requires sender_staging in cached host RAM, full stop.
        Real fix requires moving the worker memcpy off the CPU
        (Path A: cross-queue device→device transfer)."""
        case = self.case
        sender_required = (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                           | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
        sender_preferred = VK_MEMORY_PROPERTY_HOST_CACHED_BIT
        receiver_required = (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                             | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
        # 2026-09-15 (N56 8-GPU diagnosis): without a CACHED preference the NV
        # driver hands the receiver an UNCACHED write-combined host type and the
        # worker's numpy[:] copy into it runs at only ~1-2 GB/s (2.6 MB -> 1.5-2.8 ms
        # at 64M K=8). V7_RECEIVER_CACHED=1 prefers HOST_CACHED for the receiver
        # (A/B switch; default keeps the legacy choice until validated).
        receiver_preferred = (VK_MEMORY_PROPERTY_HOST_CACHED_BIT
                              if os.environ.get("V7_RECEIVER_CACHED", "0") == "1" else 0)
        usage = (VK_BUFFER_USAGE_TRANSFER_DST_BIT
                 | VK_BUFFER_USAGE_TRANSFER_SRC_BIT)

        # Compute segments + total size per direction; cache on self for cmd recording.
        self._transport_segments: dict[str, list[_TransportSegment]] = {}
        self._transport_total_bytes: dict[str, int] = {}
        self._recv_count_offsets = {}
        stagings: dict[str, _Buffer] = {}
        for direction_name, peer_attr in (
            ("leading",  "has_leading_peer"),
            ("trailing", "has_trailing_peer"),
        ):
            if not getattr(case.transport, peer_attr):
                continue
            segments, total = self._compute_transport_segments(direction_name)
            if total == 0:
                continue
            self._transport_segments[direction_name] = segments
            self._transport_total_bytes[direction_name] = total

            sender_buf = self._allocate_buffer(
                total, usage, sender_required, sender_preferred)
            mapped = vkMapMemory(self.ctx.device, sender_buf.memory, 0, total, 0)
            sender_buf.mapped = mapped
            sender_buf.mapped_view = np.frombuffer(mapped, dtype=np.uint8, count=total)
            stagings[f"sender_staging_{direction_name}"] = sender_buf

            recv_buf = self._allocate_buffer(
                total, usage, receiver_required, receiver_preferred)
            mapped_r = vkMapMemory(self.ctx.device, recv_buf.memory, 0, total, 0)
            recv_buf.mapped = mapped_r
            recv_buf.mapped_view = np.frombuffer(mapped_r, dtype=np.uint8, count=total)
            stagings[f"receiver_staging_{direction_name}"] = recv_buf

        total_bytes = sum(b.size for b in stagings.values())
        # Diagnostic: re-query the chosen memory type for one sender and one
        # receiver buffer. Sender and receiver have asymmetric preferred
        # properties (see docstring), so we report both. Same args →
        # find_memory_type returns the same index as the actual allocation.
        if stagings:
            def _flag_str(flags: int) -> str:
                names = []
                for bit, name in (
                    (VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,  "DEVICE_LOCAL"),
                    (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,  "HOST_VISIBLE"),
                    (VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, "HOST_COHERENT"),
                    (VK_MEMORY_PROPERTY_HOST_CACHED_BIT,   "HOST_CACHED"),
                ):
                    if flags & bit:
                        names.append(name)
                return "|".join(names) if names else "(none)"

            def _probe(buf, required, preferred) -> str:
                reqs = vkGetBufferMemoryRequirements(self.ctx.device, buf.handle)
                idx = self.ctx.find_memory_type(reqs.memoryTypeBits, required, preferred)
                flags = self.ctx._memory_properties.memoryTypes[idx].propertyFlags
                rebar = "[OK ReBAR]" if (flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) else "[host RAM]"
                return f"type[{idx}]={_flag_str(flags)} {rebar}"

            sender_example = next(b for k, b in stagings.items() if k.startswith("sender_"))
            recv_example   = next(b for k, b in stagings.items() if k.startswith("receiver_"))
            print(f"[SimV7] host staging buffers: {len(stagings)}, "
                  f"{total_bytes / 1024:.1f} KB (persistent-mapped)")
            print(f"  sender   {_probe(sender_example, sender_required, sender_preferred)}")
            print(f"  receiver {_probe(recv_example, receiver_required, receiver_preferred)}")
        else:
            print(f"[SimV7] host staging buffers: 0 (single-GPU mode, no peer)")
        return stagings

    # ========================================================================
    # Section 3: Descriptor sets
    # ========================================================================

    def _build_descriptor_layouts(self) -> list:
        """4 layouts: set 0/1/3 hold buffers; set 2 is empty (V5 merged-buffer
        scheme stores ghost in set 0/1)."""
        layouts = []
        for set_index in range(4):
            specs_in_set = [s for s in self._buffer_specs if s.set_index == set_index]
            bindings = [
                VkDescriptorSetLayoutBinding(
                    binding=spec.binding,
                    descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    descriptorCount=1,
                    stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
                )
                for spec in specs_in_set
            ]
            ci = VkDescriptorSetLayoutCreateInfo(
                bindingCount=len(bindings),
                pBindings=bindings if bindings else None,
            )
            layouts.append(vkCreateDescriptorSetLayout(self.ctx.device, ci, None))
        return layouts

    def _create_descriptor_pool(self):
        n_buffers = len(self._buffer_specs)
        pool_size = VkDescriptorPoolSize(
            type=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descriptorCount=n_buffers)
        ci = VkDescriptorPoolCreateInfo(
            maxSets=4, poolSizeCount=1, pPoolSizes=[pool_size])
        return vkCreateDescriptorPool(self.ctx.device, ci, None)

    def _allocate_descriptor_sets(self) -> list:
        alloc = VkDescriptorSetAllocateInfo(
            descriptorPool=self.descriptor_pool,
            descriptorSetCount=4,
            pSetLayouts=self.descriptor_layouts,
        )
        return vkAllocateDescriptorSets(self.ctx.device, alloc)

    def _wire_descriptor_sets(self) -> None:
        writes = []
        for spec in self._buffer_specs:
            buf_info = VkDescriptorBufferInfo(
                buffer=self.buffers[spec.name].handle,
                offset=0,
                range=spec.size,
            )
            writes.append(VkWriteDescriptorSet(
                dstSet=self.descriptor_sets[spec.set_index],
                dstBinding=spec.binding,
                dstArrayElement=0,
                descriptorCount=1,
                descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                pBufferInfo=[buf_info],
            ))
        vkUpdateDescriptorSets(self.ctx.device, len(writes), writes, 0, None)

    # ========================================================================
    # Section 4: Pipelines
    # ========================================================================

    def _build_pipeline_layout(self):
        ci = VkPipelineLayoutCreateInfo(
            setLayoutCount=4,
            pSetLayouts=self.descriptor_layouts,
            pushConstantRangeCount=0,
        )
        return vkCreatePipelineLayout(self.ctx.device, ci, None)

    def _load_shader_modules(self) -> dict[str, object]:
        shader_dir = (pathlib.Path(__file__).resolve().parents[1]
                      / "shaders" / "spv")
        # V7_SPV_DIR: load compiled shaders from another directory (A/B of
        # shader builds with the per-particle verifier, e.g. git HEAD vs tree).
        if os.environ.get("V7_SPV_DIR"):
            shader_dir = pathlib.Path(os.environ["V7_SPV_DIR"])
        modules: dict[str, object] = {}
        for shader_name in (
            "bootstrap_half_kick", "initialize_voxelization",
            # E39 B6: the lane-group ghost_send with V7_GHOST_SEND_LANES > 0, ghost_send.comp with 0 (so a pre-B6
            # V7_SPV_DIR runs with 0)
            "predict", "update_voxel", "ghost_send_lanes" if _GHOST_SEND_LANES else "ghost_send",
            "install_migrations",
            "correction", "density", "force", "defrag", "append_departed",
            "expand_ghost_lists", "band_compact",
            # E37: the wall pass only with wall_boundary adami (simple creates no wall pipeline, so it needs no
            # wall_extrapolate.comp.spv, e.g. in a V7_SPV_DIR A/B against a pre-E37 build)
            *(("wall_extrapolate",) if self._wall_adami else ()),
            # E39 B9: the density copy pass only with V7_DENSITY_COPY_COMPUTE=1 (likewise for a pre-B9 V7_SPV_DIR)
            *(("density_scratch_copy",) if _DENSITY_COPY_COMPUTE else ()),
            # E39 B4: the marker and the DEEP_WALL_SKIP variants only on a slab that skips deep walls
            *(("deep_wall_marker", "correction_deep_wall_skip", "density_deep_wall_skip")
              if self._deep_wall_skip_active() else ()),
            # E39 B1: the fused kernel only on a slab that fuses (likewise for a pre-B1 V7_SPV_DIR), its
            # DEEP_WALL_SKIP variant when that slab also skips deep walls
            *(("correction_density",) if self._fused_correction_density_active() else ()),
            *(("correction_density_deep_wall_skip",)
              if self._fused_correction_density_active() and self._deep_wall_skip_active() else ()),
            # E39 B3: force.comp's FORCE_SEGMENT variant only on a slab whose V7_BAND_OVERLAP plan has a force_deep
            # segment that does not start at workgroup 0 (likewise for a pre-B3 V7_SPV_DIR)
            *(("force_segment",) if self._force_deep_segment_first_groups() else ()),
        ):
            spv_path = shader_dir / f"{shader_name}.comp.spv"
            if not spv_path.exists():
                raise FileNotFoundError(
                    f"shader not compiled: {spv_path}. "
                    f"Run experiment/v7/compile_shaders_v7.py first.")
            code = spv_path.read_bytes()
            ci = VkShaderModuleCreateInfo(codeSize=len(code), pCode=code)
            modules[shader_name] = vkCreateShaderModule(self.ctx.device, ci, None)
        return modules

    # ----- Spec const blob helpers -----

    def _pack_spec(self, entries: list[tuple[int, str, Any]]
                   ) -> tuple[bytes, list[VkSpecializationMapEntry]]:
        """Pack a list of (constant_id, fmt, value) into a contiguous data blob
        + VkSpecializationMapEntry list. fmt is struct-style 'f'/'I'/'i'/'B'.
        All fields are 4 B in our use."""
        blob = bytearray()
        map_entries: list[VkSpecializationMapEntry] = []
        for const_id, fmt, value in entries:
            if fmt == 'B':
                packed = struct.pack('<I', 1 if value else 0)
            else:
                packed = struct.pack('<' + fmt, value)
            assert len(packed) == 4
            map_entries.append(VkSpecializationMapEntry(
                constantID=const_id, offset=len(blob), size=4))
            blob.extend(packed)
        return bytes(blob), map_entries

    def _make_spec_info(self, entries: list[tuple[int, str, Any]]
                        ) -> Optional[object]:
        if not entries:
            return None
        blob, map_entries = self._pack_spec(entries)
        # pData is const void *; python-vulkan can't auto-convert bytes,
        # need a typed cdata pointer. Keep blob + cdata + map_entries alive
        # until vkCreateComputePipelines returns.
        cdata = ffi.new("uint8_t[]", blob)
        info = VkSpecializationInfo(
            mapEntryCount=len(map_entries),
            pMapEntries=map_entries,
            dataSize=len(blob),
            pData=cdata,
        )
        self._spec_keepalive.append({
            "blob": blob, "cdata": cdata, "entries": map_entries, "info": info
        })
        return info

    def _global_entries(self) -> list[tuple[int, str, Any]]:
        p = self.case.physics
        n = self.case.numerics
        cap = self.case.capacities
        g = self.case.grid
        gh = self.case.ghost_grid
        return [
            (0,  'f', p.smoothing_length),
            (1,  'f', p.speed_of_sound),
            (2,  'f', p.delta_coefficient),
            (4,  'f', p.power_parameter),
            (5,  'f', p.cfl_number),
            (6,  'f', p.timestep),
            (7,  'f', g.origin_x),
            (8,  'f', g.origin_y),
            (9,  'f', g.origin_z),
            (10, 'B', 0),                       # STRICT_BIT_EXACT — 目前版本不再需要 bit-exact
            (11, 'I', g.grid_dimension_x),
            (12, 'I', g.grid_dimension_y),
            (13, 'I', g.grid_dimension_z),
            (14, 'f', n.regularization_xi),
            (15, 'f', n.regularization_determinant_threshold),
            (16, 'f', n.regularization_max_frobenius_norm),
            (17, 'f', p.gravity[0]),
            (18, 'f', p.gravity[1]),
            (19, 'f', p.gravity[2]),
            (20, 'I', g.voxel_order),
            (30, 'I', p.dimension),
            (31, 'I', p.neighbor_z_range),
            (32, 'f', p.kernel_coefficient),
            (33, 'f', p.kernel_gradient_coefficient),
            (40, 'f', n.eps_h_squared),
            (41, 'f', n.pst_main_shift_coefficient),
            (42, 'f', n.pst_anti_shift_coefficient),
            (43, 'B', int(n.use_kcg_correction)),
            (44, 'B', int(n.use_density_diffusion)),
            (45, 'B', int(n.use_pst)),
            (46, 'B', int(n.use_prefix_sum_defrag)),
            (50, 'I', cap.max_particles_per_voxel),
            (51, 'I', cap.workgroup_size),
            (52, 'I', cap.max_incoming_per_voxel),
            (53, 'I', cap.own_pool_size),
            (54, 'I', cap.leading_ghost_pool_size),
            (55, 'I', cap.trailing_ghost_pool_size),
            (80, 'I', gh.leading_ghost_voxel_count),
            (81, 'I', gh.trailing_ghost_voxel_count),
            # V6 seam / transport switches (common.glsl ids 83-88; defaults = V5).
            (83, 'I', gh.ghost_layers),
            (84, 'I', cap.departed_pool_size),
            (86, 'I', cap.replica_region_size),
            (87, 'B', int(configured_lean_transport())),
            (88, 'B', int(configured_transport_extension())),
            (89, 'B', int(configured_compact_ghost_lists())),
            (95, 'B', int(configured_delta_density())),
            (96, 'f', self.reference_density()),
            (98, 'B', int(configured_packed_replicas())),
            (99, 'I', configured_diagnostic_poison_inner_replica()),   # V7_DIAG_POISON_G1
            (97, 'B', int(configured_init_seam_clamp())),
            (100, 'I', 1 if self._wall_adami else 0),                    # E37 WALL_BOUNDARY (case.yaml)
            # NEIGHBOR_X_RANGE (id=82) is NOT global anymore — Path A+ needs
            # different widths per kernel (correction / density / force =
            # self.band_widths, V7_BAND_WIDTHS, default 2/2/3, for the
            # cascading interior/boundary split). Each split-kernel
            # pipeline appends its own (82, 'I', width) via the helpers
            # below. Non-split kernels (predict / update_voxel / ghost_send /
            # install_migrations / defrag) don't use in_boundary_band at all,
            # so they rely on the GLSL default (NEIGHBOR_X_RANGE = 0u).
        ]

    # ----- Per-pipeline mode entries (kernel-specific spec const overrides) ---

    def _correction_mode_entries(self, mode: int,
                                 band_dispatch: int = 0) -> list[tuple[int, str, Any]]:
        """CORRECTION_MODE (id=47) + NEIGHBOR_X_RANGE (id=82) + BAND_VOXEL_
        DISPATCH (id=57). Boundary band = band_widths[0] voxels (>= 2: column 0
        reaches ghost; column 1 reaches column 0 where migrants land after
        install_migration).
        V6: the band-dispatch boundary variant also walks the inner ghost
        column as self when V7_GHOST_LAYERS = 2 (GHOST_SELF_LAYER, id=85)."""
        return [(47, 'I', mode), (82, 'I', self.band_widths[0]), (57, 'I', band_dispatch),
                (58, 'I', _BAND_SLOT_LANES if band_dispatch == 1 else 0),
                (59, 'I', self._fake_band_column()),
                (85, 'I', self._ghost_self_layer(mode, band_dispatch, "correction"))]

    def _density_mode_entries(self, mode: int,
                              band_dispatch: int = 0) -> list[tuple[int, str, Any]]:
        """DENSITY_MODE (id=48) + NEIGHBOR_X_RANGE (id=82) + BAND_VOXEL_DISPATCH
        (id=57). Boundary band = band_widths[1] voxels (>= correction's band:
        density reads its own L and the neighbours' r, v, m, rho_n, material,
        never a neighbour's correction output; the historical default 3 added
        a column for the neighbour reach into stale correction that only the
        commented-out psi term needed). Used by Path A+ density split.
        V6: GHOST_SELF_LAYER (id=85) as in _correction_mode_entries."""
        return [(48, 'I', mode), (82, 'I', self.band_widths[1]), (57, 'I', band_dispatch),
                (58, 'I', _BAND_SLOT_LANES if band_dispatch == 1 else 0),
                (59, 'I', self._fake_band_column()),
                (85, 'I', self._ghost_self_layer(mode, band_dispatch, "density"))]

    def _ghost_self_layer(self, mode: int, band_dispatch: int, kernel: str = "") -> int:
        """1 for the band-dispatch BOUNDARY variants of correction / density
        when the ghost is two columns deep (V7_GHOST_LAYERS = 2): they then
        recompute the inner ghost column as self. 0 otherwise (V5).
        V7_DIAG_GHOST_SELF (diagnostic, default "correction,density") drops the
        inner-column self pass from the kernels it does not name, to time each
        kernel's share; without "density" the inner column keeps the uploaded
        rho_n (the density copy then skips it), i.e. V5's defect (1) returns."""
        if self.case.ghost_grid.ghost_layers >= 2 and band_dispatch in (1, 2) and mode == 2:
            if kernel and kernel not in _DIAG_GHOST_SELF_KERNELS:
                return 0
            return 1
        return 0

    def _ghost_self_density(self) -> bool:
        return self.ghost_layers() >= 2 and "density" in _DIAG_GHOST_SELF_KERNELS

    def reference_density(self) -> float:
        """rho_ref of V7_DELTA_DENSITY: the first fluid material's rest density
        (every material's rest density otherwise)."""
        fluids = [m for m in self.case.materials if m.kind == KIND_FLUID]
        chosen = fluids[0] if fluids else self.case.materials[0]
        return float(chosen.rest_density)

    def stored_density_offset(self) -> float:
        """Add to density_pressure.x read back from the GPU to get rho
        (rho_ref with V7_DELTA_DENSITY, 0 otherwise)."""
        return self.reference_density() if configured_delta_density() else 0.0

    def ghost_layers(self) -> int:
        return self.case.ghost_grid.ghost_layers

    def departed_pool_size(self) -> int:
        return self.case.capacities.departed_pool_size

    def _force_mode_entries(self, mode: int,
                            density_source: int = 0,
                            band_dispatch: int = 0) -> list[tuple[int, str, Any]]:
        """FORCE_MODE (id=49) + NEIGHBOR_X_RANGE (id=82) + FORCE_DENSITY_SOURCE
        (id=56; 0 = primary, 1 = scratch). Boundary band = band_widths[2]
        voxels (>= density's band + 1 for neighbor reach into stale-density).
        The Phase B cascading pipeline uses density_source=1 because the
        scratch->primary copy is only issued in Phase C."""
        return [(49, 'I', mode), (82, 'I', self.band_widths[2]), (56, 'I', density_source),
                (57, 'I', band_dispatch), (58, 'I', _BAND_SLOT_LANES if band_dispatch == 1 else 0),
                (59, 'I', self._fake_band_column())]

    def _configured_band_widths(self) -> tuple[int, int, int]:
        """V7_BAND_WIDTHS (partition_v7.configured_band_widths checks c >= 2,
        d >= c, f >= d + 1). V7_BAND_COMPACT_DISPATCH builds its band list for
        the 2/3/4 bands only (band_compact.comp), so other widths are rejected
        with it (unset, V7_BAND_WIDTHS then defaults to 2,3,4)."""
        widths = configured_band_widths()
        if _BAND_COMPACT and widths != COMPACT_DISPATCH_BAND_WIDTHS:
            raise ValueError("V7_BAND_COMPACT_DISPATCH=1 builds its band list for the 2/3/4 bands only; "
                             f"V7_BAND_WIDTHS={','.join(map(str, widths))} is not supported with it")
        return widths

    def _fake_band_column(self) -> int:
        """Spec const 59 value: the diagnostic band column for a sim without
        peers (V7_FAKE_BAND_TEST), 0 otherwise."""
        gh = self.case.ghost_grid
        if (_FAKE_BAND_COLUMN > 0 and gh.leading_ghost_voxel_count == 0
                and gh.trailing_ghost_voxel_count == 0):
            return _FAKE_BAND_COLUMN
        return 0

    def _ghost_direction_entries(
        self, direction: int
    ) -> list[tuple[int, str, Any]]:
        """Per-direction spec consts (ids 90-94) for ghost_send / install_migrations."""
        spec = (self.case.transport.leading if direction == 0
                else self.case.transport.trailing)
        # If direction has no peer, we still need to compile the pipeline
        # (descriptor wiring) but the spec consts can be defaults — the
        # pipeline will never actually be dispatched.
        if spec is None:
            return [
                (90, 'I', direction),
                (91, 'I', 0),
                (92, 'I', 0),
                (93, 'i', 0),
                (94, 'i', 0),
            ]
        return [
            (90, 'I', spec.direction),
            (91, 'I', spec.boundary_voxel_x_local),
            (92, 'I', spec.ghost_voxel_x_local),
            (93, 'i', spec.ghost_pid_offset_to_receiver),
            (94, 'i', spec.ghost_voxel_id_offset_to_receiver),
        ]

    def _ghost_send_lanes_entries(self) -> list[tuple[int, str, Any]]:
        """E39 B6: ghost_send_lanes.comp's own spec constants: 109 = lanes per
        (face voxel, layer) group (V7_GHOST_SEND_LANES), 110 = its workgroup
        size (_GHOST_SEND_LOCAL_SIZE, local_size_x_id)."""
        self._ghost_send_groups_per_workgroup()
        return [(109, 'I', _GHOST_SEND_LANES), (110, 'I', _GHOST_SEND_LOCAL_SIZE)]

    # ----- E39 B4: deep-wall skip (V7_DEEP_WALL_SKIP, shaders/deep_wall_skip.glsl) ----

    def _deep_wall_skip_active(self) -> bool:
        """Does this slab skip deep walls: V7_DEEP_WALL_SKIP resolved once per simulator (__init__; a CPU test's
        object.__new__ simulator on first use) into deep_wall_skip_resolution = (active, reason)."""
        resolution = self.__dict__.get("deep_wall_skip_resolution")
        if resolution is None:
            resolution = self.deep_wall_skip_resolution = self._resolve_deep_wall_skip()
        return resolution[0]

    def _resolve_deep_wall_skip(self) -> tuple[bool, str]:
        """(active, reason) of V7_DEEP_WALL_SKIP for this slab. 0: off; 1: on. auto: on iff the slab is 3-D and
        its initial particles hold deep-wall candidates (deep_wall_candidate_count, the walls the marker skips at
        step 0) of at least _DEEP_WALL_AUTO_MINIMUM_CANDIDATE_FRACTION of the slab's particles (the measurements
        behind the rule are at that constant). The rule reads case.initial, the state a bootstrap starts from; a
        restart decides from the case's initial state too (the walls do not move). It only picks the cheaper of two
        equal results: the marker is rebuilt every step, so any slab may run either way."""
        if _DEEP_WALL_SKIP == "0":
            return False, "V7_DEEP_WALL_SKIP=0"
        if _DEEP_WALL_SKIP == "1":
            return True, "forced (V7_DEEP_WALL_SKIP=1)"
        dimension = int(self.case.physics.dimension)
        if dimension != 3:
            return False, f"auto: {dimension}-D slab"
        band_widths = self.__dict__.get("band_widths") or self._configured_band_widths()
        candidates = deep_wall_candidate_count(self.case, band_widths[1])
        particles = int(self.case.initial.positions.shape[0])
        fraction = candidates / particles if particles else 0.0
        verdict = fraction >= _DEEP_WALL_AUTO_MINIMUM_CANDIDATE_FRACTION
        return verdict, (f"auto: 3-D, {candidates:,} deep-wall candidates at the initial state = "
                         f"{100.0 * fraction:.2f} % of {particles:,} particles "
                         f"{'>=' if verdict else '<'} {100.0 * _DEEP_WALL_AUTO_MINIMUM_CANDIDATE_FRACTION:g} %")

    def _deep_wall_skips_full_domain(self) -> bool:
        """The full-domain pipelines correction_all / density_all skip too only on a slab without ghost columns
        (their lists are then all this slab's own, final after update_voxel); with a peer they serve the bootstrap
        after the ghost round and stay the plain kernels."""
        ghost_grid = self.case.ghost_grid
        return ghost_grid.leading_ghost_voxel_count == 0 and ghost_grid.trailing_ghost_voxel_count == 0

    def _deep_wall_skip_pipeline_keys(self) -> tuple[str, ...]:
        """The pipelines built from the DEEP_WALL_SKIP variants (empty when the slab does not skip)."""
        if not self._deep_wall_skip_active():
            return ()
        keys = ("correction_interior", "density_deep_interior")
        if self._deep_wall_skips_full_domain():
            keys += ("correction_all", "density_all")
        return keys

    @staticmethod
    def _deep_wall_record() -> bool:
        """Record the skip decisions (spec 113): V7_DEEP_WALL_CHECK=1 or the canonical_dump hook."""
        return bool(_DEEP_WALL_CHECK or _DEEP_WALL_RECORD_DECISIONS)

    def _deep_wall_skip_entries(self) -> list[tuple[int, str, Any]]:
        """Spec constants of the variant pipelines (deep_wall_skip.glsl): 111 = density's band width (both
        kernels skip only outside it: correction_interior's set outside band c then equals density_deep_interior's
        outside band d >= c), 112 = V7_DEEP_WALL_CHECK, 113 = record."""
        return [(111, 'I', self.band_widths[1]), (112, 'B', _DEEP_WALL_CHECK), (113, 'B', self._deep_wall_record())]

    def _record_deep_wall_marker(self, cmd, tick: Optional[str] = None) -> None:
        """deep_wall_marker.comp's two passes over the extended voxels (presence, then deep), each followed by a
        compute barrier; the caller has already ordered this step's update_voxel (or the bootstrap lists) before
        it. Records nothing when the slab does not skip."""
        if not self._deep_wall_skip_active():
            return
        per_v = self._per_extended_voxel_dispatch_count()
        for pipeline_key in ("deep_wall_marker_presence", "deep_wall_marker_deep"):
            self._bind_pipeline_and_sets(cmd, pipeline_key)
            vkCmdDispatch(cmd, per_v, 1, 1)
            self._record_compute_barrier(cmd)
        if tick:
            self._bench_tick(cmd, tick)

    # ----- E39 B1: correction + density in one traversal (V7_FUSED_CORRECTION_DENSITY, correction_density.comp) --

    def _fused_correction_density_active(self) -> bool:
        """Does this slab run correction_density.comp: V7_FUSED_CORRECTION_DENSITY resolved once per simulator
        (__init__; a CPU test's object.__new__ simulator on first use) into fused_correction_density_resolution =
        (active, reason)."""
        resolution = self.__dict__.get("fused_correction_density_resolution")
        if resolution is None:
            resolution = self.fused_correction_density_resolution = self._resolve_fused_correction_density()
        return resolution[0]

    def _resolve_fused_correction_density(self) -> tuple[bool, str]:
        """(active, reason) of V7_FUSED_CORRECTION_DENSITY for this slab. The fused kernel takes correction's and
        density's place only where the two compute the same particles at every site: one band width (spec 82 of
        the split pipelines: V7_BAND_WIDTHS c == d; the compact band dispatch supports 2,3,4 only) and one inner
        ghost self layer (spec 85 of the band pipelines: V7_DIAG_GHOST_SELF naming both kernels or neither).
        Otherwise the whole slab keeps the separate kernels and records exactly what
        V7_FUSED_CORRECTION_DENSITY=0 records."""
        if not _FUSED_CORRECTION_DENSITY:
            return False, "V7_FUSED_CORRECTION_DENSITY=0"
        correction_band, density_band, force_band = (self.__dict__.get("band_widths")
                                                     or self._configured_band_widths())
        if correction_band != density_band:
            return False, (f"fallback: correction band {correction_band} != density band {density_band}, "
                           f"V7_BAND_WIDTHS={correction_band},{density_band},{force_band}"
                           + (" (V7_BAND_COMPACT_DISPATCH supports 2,3,4 only)" if _BAND_COMPACT else ""))
        if _BAND_COMPACT:
            # unreachable while the compact list serves 2,3,4 only; the fused band pass has no compact dispatch
            return False, "fallback: V7_BAND_COMPACT_DISPATCH=1 (the fused band pass has no compact-list dispatch)"
        correction_layer = self._ghost_self_layer(2, 1, "correction")
        density_layer = self._ghost_self_layer(2, 1, "density")
        if correction_layer != density_layer:
            return False, (f"fallback: V7_DIAG_GHOST_SELF={','.join(_DIAG_GHOST_SELF_KERNELS)} walks the inner "
                           f"ghost column as self in the {'correction' if correction_layer else 'density'} band "
                           "kernel only")
        # the band pipelines' self layer acts only where the slab has ghost columns (none without peers)
        has_ghost_columns = not self._deep_wall_skips_full_domain()
        return True, (f"one neighbour traversal for correction + density, band {correction_band}"
                      + (", inner ghost column as self" if correction_layer and has_ghost_columns else ""))

    def _fused_deep_wall_skip_pipeline_keys(self) -> tuple[str, ...]:
        """The fused pipelines built from correction_density.comp's DEEP_WALL_SKIP variant (B4's
        _deep_wall_skip_pipeline_keys for the fused kernel): the interior one and, on a slab without peers, the
        full-domain one; empty when the slab does not skip deep walls."""
        if not self._deep_wall_skip_active():
            return ()
        keys = ("correction_density_interior",)
        if self._deep_wall_skips_full_domain():
            keys += ("correction_density_all",)
        return keys

    # ----- E39 B3: the band chain overlapped with the cascade force (V7_BAND_OVERLAP) ------------------------------

    def _band_overlap_active(self) -> bool:
        """Does this slab record the B3 layout now: V7_BAND_OVERLAP resolved once per simulator (__init__; a CPU
        test's object.__new__ simulator on first use) into band_overlap_chain_verdict = (on, reason) of the whole
        chain (band_overlap_chain_verdict), band_overlap_resolution = (active, reason) of this slab and
        band_overlap_plan (a _BandOverlapPlan, None when off), and correction + density fused at recording time.
        The plan is made for the fused phase B / C only (phase B ends with correction_density_interior, phase C
        pairs the fused band kernel); a caller that turns fused_correction_density_resolution off after
        construction (seam_audit/fused_single_step.py records the separate kernels that way) gets the B1 separate
        recording, force_deep whole in phase B, at every site that asks here. _record_phase_c_cmd checks that
        phase B + phase C recorded force_deep's workgroups exactly once (_check_force_deep_recorded_once)."""
        resolution = self.__dict__.get("band_overlap_resolution")
        if resolution is None:
            self.band_overlap_chain_verdict = band_overlap_chain_verdict(self.case)
            plan, reason = self._resolve_band_overlap()
            self.band_overlap_plan = plan
            resolution = self.band_overlap_resolution = (plan is not None, reason)
        return resolution[0] and self._fused_correction_density_active()

    def _resolve_band_overlap(self) -> tuple[Optional[_BandOverlapPlan], str]:
        """(plan or None, reason) of V7_BAND_OVERLAP for this slab: on iff the slab is legal and the chain verdict
        (band_overlap_chain_verdict: 1 = on, auto = the rule over every slab of the chain) is on.

        Legal (1 and auto) only on a 2-D slab with peers whose phase B ends with force_deep_interior_scratch and
        whose phase C band pass is the fused band-voxel dispatch followed by the compute density copy: fused
        correction + density (B1; its fallbacks - V7_BAND_WIDTHS c != d, V7_BAND_COMPACT_DISPATCH, a
        V7_DIAG_GHOST_SELF naming one kernel - keep the B1 recording), V7_CASCADE_FORCE=1, V7_BAND_VOXEL_DISPATCH=1,
        V7_DENSITY_COPY_COMPUTE=1, wall_boundary simple (adami runs without peers anyway). 3-D band kernels are
        throughput-bound (nothing idle to fill) and K = 1 has no band chain: off. Every one of these tests gives
        the same answer on every slab of a chain built by compute_chain_partition (module switches, the band widths
        and ghost layers of the process, the case's dimension and wall option; every slab of a chain of K >= 2 has a
        peer and band voxels on that side), so with the chain verdict a chain runs the B3 layout on every slab or
        on none: a B3 slab waiting for the upload of a B1 neighbour has lost phase B's force_deep, which hid that
        wait (E39 B3 review, 2-D 1M K = 3: mixed chains ran 2 % slower than B1, see _BAND_OVERLAP_AUTO_*).

        Why the moved segments may run next to the band kernels (bands c / d / f = V7_BAND_WIDTHS, f >= d + 1 >= c
        + 1; columns are own voxel columns counted from a peer side, G1 the inner ghost column; every set below is
        decided per particle by the voxel id in position_voxel_id.w, which nothing in phase B / C rewrites for a
        particle that exists before install_migrations):
          force_deep_interior_scratch (self = own pids, returns unless fluid, alive and at column >= f):
            reads  position_voxel_id / velocity_mass / material of self and of the neighbours (columns >= f - 1),
                   density_pressure_scratch of self and neighbours (columns >= f - 1 >= d: correction_density_
                   interior's, phase B), correction_inverse / density_gradient_kernel_sum.w of self (columns >= f
                   >= c: phase B), inside_particle_count / _index of columns >= f - 1 >= 2 (update_voxel, phase A);
            writes acceleration / shift of self (columns >= f).
          correction_density_boundary_band (self = the band voxels' particles: columns < c, G1, departed copies):
            reads the same three fields + density_pressure (primary, rho_n) of G2 .. column c; writes
            correction_inverse, density_gradient_kernel_sum and density_pressure_scratch of self only.
          density copy pass: density_pressure_scratch -> density_pressure over the own range, G1 regions and the
            departed pool; no acceleration / shift, no read of what force_deep writes.
          force_boundary_band (self = columns < f): reads density_pressure (primary, after the copy) of G1 ..
            column f, its own L / kernel sum (columns < f); writes acceleration / shift of columns < f.
        So a segment and its partner write disjoint elements (acceleration / shift at columns >= f vs < f; L, kernel
        sum, scratch at columns < c / G1 / departed vs none) and neither reads an element the other writes (force_
        deep's scratch at columns >= d and L at >= f vs the band's writes at < c and G1; force_deep reads no primary
        rho and no acceleration / shift). install_migrations (own tail slots, lists of column 0 - column 1 for a far
        migration -, primary rho), expand_ghost_lists and append_departed (ghost lists and ghost SoA) are never
        paired: the global barriers after them order every segment behind them; a new migrant's tail slot then
        reads as column 0 / 1 < f (force_deep returns without a write) where phase B read it dead (also no write).
        Everything after a moved segment: the next barrier (pair 1, the copy), the end of phase C, whose
        frame_done signal (COMPUTE_SHADER) covers it, and phase A(n + 1), which opens with a global compute barrier
        and is queued after C(n) (V7_PHASE_A_NO_WAIT=1 adds no semaphore: queue order alone orders A(n + 1) after
        C(n)); readback(n + 1) waits phase_a_done(n + 1), upload(n + 1) through the worker on the receiver's
        readback_done(n + 1)."""
        if _BAND_OVERLAP == "0":
            return None, "V7_BAND_OVERLAP=0"
        ghost_grid = self.case.ghost_grid
        dimension = int(self.case.physics.dimension)
        if ghost_grid.leading_ghost_voxel_count == 0 and ghost_grid.trailing_ghost_voxel_count == 0:
            return None, "no peer (K = 1): no band chain"
        if dimension != 2:
            return None, f"{dimension}-D slab: band kernels throughput-bound"
        if self._wall_adami:
            return None, "wall_boundary adami"
        if not _CASCADE_FORCE:
            return None, "fallback: V7_CASCADE_FORCE=0 (no force_deep_interior to move)"
        if not self._fused_correction_density_active():
            return None, ("fallback: separate correction / density kernels "
                          f"({self.fused_correction_density_resolution[1]})")
        if not _BAND_VOXEL_DISPATCH:
            return None, "fallback: V7_BAND_VOXEL_DISPATCH=0 (band kernels over the own pool)"
        if not _DENSITY_COPY_COMPUTE:
            return None, "fallback: V7_DENSITY_COPY_COMPUTE=0 (transfer copy between the band kernels)"
        correction_band, _, force_band = self.__dict__.get("band_widths") or self._configured_band_widths()
        if self._per_band_dispatch_count(correction_band, self._ghost_self_layer(2, 1, "correction")) <= 0 \
                or self._per_band_dispatch_count(force_band) <= 0:
            return None, "no band workgroups"
        chain_active, verdict = band_overlap_chain_verdict(self.case)
        if not chain_active:
            return None, verdict
        plan = self._band_overlap_plan(self._band_overlap_layout())
        partners = {"correction_density": "correction_density_boundary_band", "copy": "the density copy",
                    "force": "force_boundary_band"}
        segments = ", ".join(f"{count:,} next to {partners[partner]}" for partner, _, count in plan.segments)
        return plan, (f"{verdict}; force_deep_interior {plan.group_count:,} workgroups: phase B "
                      f"{plan.phase_b_groups:,}, phase C {segments}, "
                      f"{'force_deep' if plan.force_deep_first else 'band kernel'} recorded first")

    def _band_overlap_layout(self) -> dict:
        """The layout of a slab that resolves on (_BAND_OVERLAP_LAYOUT_OVERRIDE replaces it in measurements):
        phase_b_fraction = the share of force_deep's populated workgroups (the first ceil(initial own particles /
        workgroup size)) kept in phase B, pairs = the phase C partners in recording order (each gets an equal
        share of the rest; the last also the empty tail), force_deep_first = the order inside a pair.

        The default - all of force_deep in phase C, half next to each band kernel, the band kernel recorded first -
        is the fastest of the measured layouts (E39 B3 step traces, 2-D 1M K = 2, period s0 / s1 in us: B1 865 /
        872; this layout 802 / 809; half kept in phase B 818 / 828, three quarters 835 / 841; force_deep recorded
        first 912 / 916, 941 / 953 and 968 / 977 for the full, half and three-quarter moves (a segment recorded
        first holds the SMs and the band kernel starts only in its tail: the pair is slower than the two kernels in
        a row); the copy as a third partner 876 / 885 (the copy pass grows 4 -> 135 us). The band kernel recorded
        first takes its few workgroups' slots at once and runs beside the segment: pair 1 = 207 us for
        correction_density_boundary_band (70 us alone) + half of force_deep (~170 us), pair 2 = 213 us for
        force_boundary_band (72) + the other half; 2M: full 1397 / 1410 vs half 1425 / 1440 (B1 1447 / 1453). At
        250k / 62k every layout that moves a share of force_deep loses or ties (the transfer window); at 4M / 16M
        keeping 80-95 % in phase B does not help either (4M: 2981 / 3017 us against B1 2949), so the rule switches
        those sizes off instead."""
        if _BAND_OVERLAP_LAYOUT_OVERRIDE is not None:
            return dict(_BAND_OVERLAP_LAYOUT_OVERRIDE)
        return {"phase_b_fraction": 0.0, "pairs": ("correction_density", "force"), "force_deep_first": False}

    def _band_overlap_plan(self, layout: dict) -> _BandOverlapPlan:
        """The workgroup segments of a layout: [0, phase_b_groups) in phase B, then one consecutive segment per
        pair (empty ones dropped) up to the full per-own-particle dispatch."""
        pairs = tuple(layout["pairs"])
        order = ("correction_density", "copy", "force")
        if not pairs or len(set(pairs)) != len(pairs) or any(pair not in order for pair in pairs) \
                or list(pairs) != sorted(pairs, key=order.index):
            raise ValueError(f"V7_BAND_OVERLAP layout pairs {pairs!r}: a non-empty subset of {order} in that order")
        fraction = float(layout["phase_b_fraction"])
        if not 0.0 <= fraction < 1.0:
            raise ValueError(f"V7_BAND_OVERLAP layout phase_b_fraction {fraction!r}: [0, 1)")
        group_count = self._per_own_particle_dispatch_count()
        workgroup = self.case.capacities.workgroup_size
        populated = min(group_count, max(1, -(-int(self.case.initial.positions.shape[0]) // workgroup)))
        phase_b_groups = min(int(round(fraction * populated)), group_count - 1)
        rest = max(0, populated - phase_b_groups)
        bounds = [phase_b_groups + rest * index // len(pairs) for index in range(len(pairs))] + [group_count]
        segments = tuple((pair, bounds[index], bounds[index + 1] - bounds[index])
                         for index, pair in enumerate(pairs) if bounds[index + 1] > bounds[index])
        return _BandOverlapPlan(group_count=group_count, phase_b_groups=phase_b_groups, segments=segments,
                                force_deep_first=bool(layout["force_deep_first"]))

    def _build_compute_pipelines(self) -> dict[str, object]:
        """Build the compute pipelines:

            1 × initialize_voxelization
            1 × bootstrap_half_kick
            1 × predict
            1 × update_voxel
            2 × ghost_send (leading, trailing)
            2 × install_migrations (leading, trailing)
            3 × correction (ALL, INTERIOR, BOUNDARY)             ← V5 split
            3 × density    (ALL, DEEP_INTERIOR, BOUNDARY)        ← Path A+ split
            3 × force      (ALL, DEEP_INTERIOR, BOUNDARY)        ← Path A+ split
            1 × defrag                                            (built later)

        Total = 16 (+ defrag built in Section 11 = 17).

        Naming convention: `<kernel>_all` for the V1-equivalent single-
        pipeline variant (used by bootstrap + single-GPU step + dual Phase
        C while Path A+ wiring is pending); `<kernel>_interior` and
        `<kernel>_boundary` for correction's split; density and
        force use `_deep_interior` (historically the larger boundary band).
        Band widths = self.band_widths (V7_BAND_WIDTHS, default 2/2/3 voxels
        for correction / density / force).

        ghost_send + install_migrations are always built for BOTH directions
        even if this GPU has no peer on that side; phase A/C cmd recording
        skips dispatch on the unused direction (cf. V1).

        E39 B1: a slab that fuses correction and density also builds
        correction_density_all / _interior / _boundary / _boundary_band
        (correction_density.comp); the recordings then bind those instead of
        the correction / density pairs."""
        pipelines: dict[str, object] = {}

        # Pipelines that only need global spec consts (and the shared 4-set
        # pipeline layout). These kernels don't use in_boundary_band, so they
        # rely on the GLSL default NEIGHBOR_X_RANGE = 0u. defrag is excluded
        # here — it uses a 5-set layout (set 4 = destination scratch SoA) and
        # is built in Section 11 alongside its cmd buffer.
        if self.ghost_layers() >= 2 and not _BAND_VOXEL_DISPATCH:
            raise ValueError("V7_GHOST_LAYERS=2 recomputes the inner ghost column inside the "
                             "band-voxel dispatch of correction/density_boundary; it needs "
                             "V7_BAND_VOXEL_DISPATCH=1")
        for key in ("initialize_voxelization", "bootstrap_half_kick",
                    "predict", "update_voxel", "append_departed", "expand_ghost_lists"):
            pipelines[key] = self._create_pipeline(
                shader=self.shader_modules[key],
                entries=self._global_entries(),
            )

        # ghost_send per direction (E39 B6: ghost_send_lanes_<direction> with V7_GHOST_SEND_LANES > 0 — no
        # ghost_send_<direction> pipeline then, so a recording that binds the old kernel fails loudly)
        for direction, dir_name in ((0, "leading"), (1, "trailing")):
            if _GHOST_SEND_LANES:
                pipelines[self._ghost_send_pipeline_key(dir_name)] = self._create_pipeline(
                    shader=self.shader_modules["ghost_send_lanes"],
                    entries=(self._global_entries() + self._ghost_direction_entries(direction)
                             + self._ghost_send_lanes_entries()),
                )
                continue
            pipelines[f"ghost_send_{dir_name}"] = self._create_pipeline(
                shader=self.shader_modules["ghost_send"],
                entries=self._global_entries() + self._ghost_direction_entries(direction),
            )

        # install_migrations per direction
        for direction, dir_name in ((0, "leading"), (1, "trailing")):
            pipelines[f"install_migrations_{dir_name}"] = self._create_pipeline(
                shader=self.shader_modules["install_migrations"],
                entries=self._global_entries() + self._ghost_direction_entries(direction),
            )

        # E39 B4: on a slab that skips deep walls these keys come from the DEEP_WALL_SKIP variants of
        # correction.comp / density.comp (same spec constants + 111-113); every other pipeline is unchanged.
        deep_wall_keys = self._deep_wall_skip_pipeline_keys()

        # correction × 3 modes (V5 #1 — boundary band = band_widths[0] voxels)
        for mode, mode_name in ((0, "all"), (1, "interior"), (2, "boundary")):
            key = f"correction_{mode_name}"
            if key in deep_wall_keys:
                pipelines[key] = self._create_pipeline(
                    shader=self.shader_modules["correction_deep_wall_skip"],
                    entries=(self._global_entries() + self._correction_mode_entries(mode)
                             + self._deep_wall_skip_entries()))
                continue
            pipelines[key] = self._create_pipeline(
                shader=self.shader_modules["correction"],
                entries=self._global_entries() + self._correction_mode_entries(mode),
            )

        # density × 3 modes (Path A+ — boundary band = band_widths[1] voxels)
        for mode, mode_name in ((0, "all"), (1, "deep_interior"), (2, "boundary")):
            key = f"density_{mode_name}"
            if key in deep_wall_keys:
                pipelines[key] = self._create_pipeline(
                    shader=self.shader_modules["density_deep_wall_skip"],
                    entries=(self._global_entries() + self._density_mode_entries(mode)
                             + self._deep_wall_skip_entries()))
                continue
            pipelines[key] = self._create_pipeline(
                shader=self.shader_modules["density"],
                entries=self._global_entries() + self._density_mode_entries(mode),
            )
        # E39 B4: the marker's two passes (spec 114 = pass).
        if deep_wall_keys:
            for key, marker_pass in (("deep_wall_marker_presence", 0), ("deep_wall_marker_deep", 1)):
                pipelines[key] = self._create_pipeline(
                    shader=self.shader_modules["deep_wall_marker"],
                    entries=self._global_entries() + [(114, 'I', marker_pass)])

        # force × 3 modes (Path A+ — boundary band = band_widths[2] voxels)
        for mode, mode_name in ((0, "all"), (1, "deep_interior"), (2, "boundary")):
            pipelines[f"force_{mode_name}"] = self._create_pipeline(
                shader=self.shader_modules["force"],
                entries=self._global_entries() + self._force_mode_entries(mode),
            )
        # V3.3: Phase B variant of force_deep_interior reading rho/P from scratch.
        pipelines["force_deep_interior_scratch"] = self._create_pipeline(
            shader=self.shader_modules["force"],
            entries=self._global_entries() + self._force_mode_entries(1, density_source=1),
        )
        # E39 B3: on a slab that resolves V7_BAND_OVERLAP on, every force_deep segment that does not start at
        # workgroup 0 gets force.comp's FORCE_SEGMENT variant with the same spec constants + 115 = its first thread.
        for first_group in self._force_deep_segment_first_groups():
            pipelines[self._force_deep_segment_pipeline_key(first_group)] = self._create_pipeline(
                shader=self.shader_modules["force_segment"],
                entries=(self._global_entries() + self._force_mode_entries(1, density_source=1)
                         + [(115, 'I', first_group * self.case.capacities.workgroup_size)]),
            )
        # E37 wall_boundary adami: the wall pass, reading the fluid rho/P from primary (after the scratch -> primary
        # copy) or from scratch (phase B, before the copy, in front of force_deep_interior_scratch).
        if self._wall_adami:
            for key, source in (("wall_extrapolate", 0), ("wall_extrapolate_scratch", 1)):
                pipelines[key] = self._create_pipeline(
                    shader=self.shader_modules["wall_extrapolate"],
                    entries=self._global_entries() + [(56, 'I', source)])
        # E39 B9: the density scratch -> primary copy pass (its own spec constants only: the slab's regions).
        if _DENSITY_COPY_COMPUTE:
            pipelines["density_scratch_copy"] = self._create_pipeline(
                shader=self.shader_modules["density_scratch_copy"],
                entries=self._density_scratch_copy_entries())
        # V3.4: band-voxel dispatch variants of the three Phase C boundary
        # pipelines (thread = (band voxel, slot); see helpers.glsl).
        pipelines["correction_boundary_band"] = self._create_pipeline(
            shader=self.shader_modules["correction"],
            entries=self._global_entries() + self._correction_mode_entries(2, band_dispatch=1),
        )
        pipelines["density_boundary_band"] = self._create_pipeline(
            shader=self.shader_modules["density"],
            entries=self._global_entries() + self._density_mode_entries(2, band_dispatch=1),
        )
        pipelines["force_boundary_band"] = self._create_pipeline(
            shader=self.shader_modules["force"],
            entries=self._global_entries() + self._force_mode_entries(2, band_dispatch=1),
        )

        if _BAND_COMPACT:
            if not _BAND_VOXEL_DISPATCH:
                raise ValueError("V7_BAND_COMPACT_DISPATCH=1 needs V7_BAND_VOXEL_DISPATCH=1")
            pipelines["correction_boundary_compact"] = self._create_pipeline(
                shader=self.shader_modules["correction"],
                entries=self._global_entries() + self._correction_mode_entries(2, band_dispatch=2))
            pipelines["density_boundary_compact"] = self._create_pipeline(
                shader=self.shader_modules["density"],
                entries=self._global_entries() + self._density_mode_entries(2, band_dispatch=2))
            pipelines["force_boundary_compact"] = self._create_pipeline(
                shader=self.shader_modules["force"],
                entries=self._global_entries() + self._force_mode_entries(2, band_dispatch=2))
            list_self_layer = 1 if self.ghost_layers() >= 2 else 0
            compact_entries = self._global_entries() + [
                (85, 'I', list_self_layer),
                (61, 'I', self._ghost_self_layer(2, 2, "correction")),
                (62, 'I', self._ghost_self_layer(2, 2, "density"))]
            pipelines["band_compact_scan"] = self._create_pipeline(
                shader=self.shader_modules["band_compact"], entries=compact_entries + [(60, 'I', 0)])
            pipelines["band_compact_scatter"] = self._create_pipeline(
                shader=self.shader_modules["band_compact"], entries=compact_entries + [(60, 'I', 1)])

        # E39 B1: on a slab that fuses, correction_density.comp with correction's spec constants (the slab's
        # correction and density bands and inner-ghost self layers agree) for every mode / dispatch the
        # recordings bind; the interior pipeline (and the full-domain one without peers) from its DEEP_WALL_SKIP
        # variant with B4's spec constants when the slab skips deep walls. The separate correction / density
        # pipelines above stay built: experiment/seam_audit/fused_single_step.py records both on one slab.
        fused_deep_wall_keys = self._fused_deep_wall_skip_pipeline_keys()
        if self._fused_correction_density_active():
            for key, mode, band_dispatch in (("correction_density_all", 0, 0), ("correction_density_interior", 1, 0),
                                             ("correction_density_boundary", 2, 0),
                                             ("correction_density_boundary_band", 2, 1)):
                entries = self._global_entries() + self._correction_mode_entries(mode, band_dispatch)
                if key in fused_deep_wall_keys:
                    pipelines[key] = self._create_pipeline(
                        shader=self.shader_modules["correction_density_deep_wall_skip"],
                        entries=entries + self._deep_wall_skip_entries())
                    continue
                pipelines[key] = self._create_pipeline(shader=self.shader_modules["correction_density"],
                                                       entries=entries)

        cascade_note = (" (V7_CASCADE_FORCE=1: force_deep_interior in Phase B)"
                        if _CASCADE_FORCE else "")
        ghost_self = 1 if self.ghost_layers() >= 2 else 0
        correction_band, density_band, force_band = self.band_widths
        if _BAND_VOXEL_DISPATCH:
            cascade_note += (f" (V7_BAND_VOXEL_DISPATCH=1, lanes={_BAND_SLOT_LANES or 'slots'}, bands "
                             f"{correction_band}/{density_band}/{force_band}: boundary kernels over "
                             f"band voxels: {self._band_thread_count(correction_band, ghost_self):,}/"
                             f"{self._band_thread_count(density_band, ghost_self):,}/"
                             f"{self._band_thread_count(force_band):,} "
                             f"threads vs {self.case.capacities.own_pool_size:,})")
        if self._transport_segments:
            cascade_note += (f" (seam: ghost_layers={self.ghost_layers()}, departed pool="
                             f"{self.departed_pool_size()})")
        if _DENSITY_COPY_COMPUTE:
            copy_regions = self._density_copy_slot_regions()
            cascade_note += (f" (V7_DENSITY_COPY_COMPUTE=1: density copy pass over {len(copy_regions)} "
                             f"region(s), {sum(slot_count for _, slot_count in copy_regions):,} slots)")
        else:
            cascade_note += " (V7_DENSITY_COPY_COMPUTE=0: density copy by vkCmdCopyBuffer)"
        if self._transport_segments:
            if _GHOST_SEND_LANES:
                cascade_note += (f" (V7_GHOST_SEND_LANES={_GHOST_SEND_LANES}: ghost_send over "
                                 f"{self._ghost_send_group_total():,} lane groups, local size {_GHOST_SEND_LOCAL_SIZE}, "
                                 f"{self._ghost_send_group_count():,} workgroups per direction)")
            else:
                cascade_note += (f" (V7_GHOST_SEND_LANES=0: ghost_send one thread per face voxel, "
                                 f"{self._ghost_send_group_count():,} workgroups per direction)")
        if deep_wall_keys:
            cascade_note += (f" (V7_DEEP_WALL_SKIP: {', '.join(deep_wall_keys)} skip deep walls outside band "
                             f"{self.band_widths[1]}, marker 2 x {self._per_extended_voxel_dispatch_count():,} "
                             f"workgroups{', check' if _DEEP_WALL_CHECK else ''}"
                             f"{', record' if self._deep_wall_record() else ''})")
        if self._fused_correction_density_active():
            cascade_note += (f" (V7_FUSED_CORRECTION_DENSITY: correction_density_all / _interior / _boundary / "
                             f"_boundary_band replace correction + density at band {self.band_widths[0]}"
                             + (f"; deep-wall skip in {', '.join(fused_deep_wall_keys)}" if fused_deep_wall_keys
                                else "") + ")")
        if self._band_overlap_active():
            plan = self.band_overlap_plan
            cascade_note += (f" (V7_BAND_OVERLAP: force_deep_interior_scratch {plan.phase_b_groups:,} of "
                             f"{plan.group_count:,} workgroups in phase B, {len(plan.segments)} phase C segment(s), "
                             f"{len(self._force_deep_segment_first_groups())} force_segment pipeline(s))")
        print(f"[SimV7] compute pipelines: {len(pipelines)}{cascade_note}")
        return pipelines

    def _create_pipeline(
        self,
        shader,
        entries: list[tuple[int, str, Any]],
    ):
        spec_info = self._make_spec_info(entries)
        stage = VkPipelineShaderStageCreateInfo(
            stage=VK_SHADER_STAGE_COMPUTE_BIT,
            module=shader,
            pName="main",
            pSpecializationInfo=spec_info,
        )
        ci = VkComputePipelineCreateInfo(
            stage=stage,
            layout=self.pipeline_layout,
        )
        return vkCreateComputePipelines(
            self.ctx.device, VK_NULL_HANDLE, 1, [ci], None)[0]

    # ========================================================================
    # Section 5: Timeline semaphore
    # ========================================================================

    # ========================================================================
    # Section 6: Cmd buffer helpers (Phase 3)
    # ========================================================================

    def _allocate_oneshot_cmd(self):
        info = VkCommandBufferAllocateInfo(
            commandPool=self.ctx.command_pool,
            level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=1,
        )
        return vkAllocateCommandBuffers(self.ctx.device, info)[0]

    def _allocate_transfer_oneshot_cmd(self):
        """Allocate a cmd buffer from ctx.transfer_command_pool — required for
        cmd buffers that get submitted on ctx.transfer_queue (Vulkan binds
        a pool to a specific queue family at creation). Used by Path A+
        readback / upload cmds."""
        info = VkCommandBufferAllocateInfo(
            commandPool=self.ctx.transfer_command_pool,
            level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=1,
        )
        return vkAllocateCommandBuffers(self.ctx.device, info)[0]

    def _per_own_particle_dispatch_count(self) -> int:
        wg = self.case.capacities.workgroup_size
        own = self.case.capacities.own_pool_size
        return (own + wg - 1) // wg

    def _per_extended_voxel_dispatch_count(self) -> int:
        wg = self.case.capacities.workgroup_size
        v = self.case.grid.total_voxel_count()
        return (v + wg - 1) // wg

    def _band_thread_count(self, band_range: int, ghost_self_layer: int = 0) -> int:
        """V3.4: threads for a band-voxel dispatch = band voxels * slots.
        Mirrors helpers.glsl band_voxel_count(): `range` own columns (+ the
        inner ghost column when ghost_self_layer = 1, V7_GHOST_LAYERS = 2) on
        every side that has a peer (ghost voxel count > 0), times NY*NZ, times
        MAX_PARTICLES_PER_VOXEL."""
        gh = self.case.ghost_grid
        face = self.case.grid.grid_dimension_y * self.case.grid.grid_dimension_z
        side_columns = band_range + ghost_self_layer
        columns = ((side_columns if gh.leading_ghost_voxel_count > 0 else 0)
                   + (side_columns if gh.trailing_ghost_voxel_count > 0 else 0))
        if self._fake_band_column() > 0:
            columns = band_range          # diagnostic band, one side
        lanes = _BAND_SLOT_LANES if _BAND_SLOT_LANES > 0 else self.case.capacities.max_particles_per_voxel
        return columns * face * lanes

    def _band_compact_voxel_count(self) -> int:
        """Band voxels of the compacted list: 4 own columns + the inner ghost
        column (V7_GHOST_LAYERS = 2) per side with a peer, times NY*NZ. The
        list serves the default 2/3/4 bands only (other V7_BAND_WIDTHS are
        rejected with V7_BAND_COMPACT_DISPATCH in the constructor)."""
        gh = self.case.ghost_grid
        face = self.case.grid.grid_dimension_y * self.case.grid.grid_dimension_z
        side_columns = 4 + (1 if gh.ghost_layers >= 2 else 0)
        columns = ((side_columns if gh.leading_ghost_voxel_count > 0 else 0)
                   + (side_columns if gh.trailing_ghost_voxel_count > 0 else 0))
        return columns * face

    def _band_compact_list_bytes(self) -> int:
        """Worst-case list (every band voxel full) + one offset per band voxel."""
        if not _BAND_COMPACT:
            return 4
        voxels = self._band_compact_voxel_count()
        return max(4, 4 * voxels * (self.case.capacities.max_particles_per_voxel + 1))

    def _record_indirect_barrier(self, cmd) -> None:
        """Compute writes -> indirect-command read + compute access (the band
        compaction scan writes the dispatch sizes the band kernels use)."""
        mb = VkMemoryBarrier2(
            sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
            srcStageMask=VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            srcAccessMask=(VK_ACCESS_2_SHADER_STORAGE_READ_BIT
                           | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT),
            dstStageMask=(VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT
                          | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT),
            dstAccessMask=(VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT
                           | VK_ACCESS_2_SHADER_STORAGE_READ_BIT
                           | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT),
        )
        info = VkDependencyInfo(
            sType=VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            memoryBarrierCount=1, pMemoryBarriers=[mb])
        vkCmdPipelineBarrier2(cmd, info)

    def _per_band_dispatch_count(self, band_range: int, ghost_self_layer: int = 0) -> int:
        wg = self.case.capacities.workgroup_size
        return (self._band_thread_count(band_range, ghost_self_layer) + wg - 1) // wg

    def _per_yz_face_dispatch_count(self) -> int:
        """ghost_send dispatches one thread per (y,z) face slot = NY*NZ threads."""
        wg = self.case.capacities.workgroup_size
        face = self.case.grid.grid_dimension_y * self.case.grid.grid_dimension_z
        return (face + wg - 1) // wg

    def _ghost_send_group_total(self) -> int:
        """E39 B6: lane groups of ghost_send_lanes.comp = face voxels x layers
        (2 with V7_GHOST_LAYERS = 2, else 1: the shader's GHOST_LAYERS >= 2u
        test on spec constant 83), one per (face voxel, layer) pair."""
        face = self.case.grid.grid_dimension_y * self.case.grid.grid_dimension_z
        layer_count = 2 if self.case.ghost_grid.ghost_layers >= 2 else 1
        return face * layer_count

    def _ghost_send_group_count(self) -> int:
        """Workgroups of one ghost_send dispatch (one direction): with
        V7_GHOST_SEND_LANES = 0 ghost_send.comp's ceil(NY*NZ / WORKGROUP_SIZE)
        (_per_yz_face_dispatch_count); with L > 0 ghost_send_lanes.comp's
        ceil(groups / (local size / L)) — the shader's tail invocations past the
        last group only take part in its barrier."""
        if not _GHOST_SEND_LANES:
            return self._per_yz_face_dispatch_count()
        groups_per_workgroup = self._ghost_send_groups_per_workgroup()
        return (self._ghost_send_group_total() + groups_per_workgroup - 1) // groups_per_workgroup

    @staticmethod
    def _ghost_send_pipeline_key(direction: str) -> str:
        return f"ghost_send_lanes_{direction}" if _GHOST_SEND_LANES else f"ghost_send_{direction}"

    def _record_ghost_send_dispatch(self, cmd, direction: str) -> None:
        """Bind this direction's ghost_send pipeline and dispatch it: phase A,
        the bootstrap ghost round and experiment/seam_audit/canonical_dump.py's
        canonical bootstrap. V7_GHOST_SEND_LANES = 0 records exactly the v6
        pair (bind ghost_send_<direction>, dispatch per (y, z) face voxel)."""
        self._bind_pipeline_and_sets(cmd, self._ghost_send_pipeline_key(direction))
        vkCmdDispatch(cmd, self._ghost_send_group_count(), 1, 1)

    def _per_ghost_pid_dispatch_count(self, direction: str) -> int:
        """install_migrations threads: the direction's migrant slots (the whole
        V5 mixed pool, or the migrant region after the two replica regions)."""
        wg = self.case.capacities.workgroup_size
        if direction == "leading":
            pool = self.case.capacities.leading_ghost_pool_size
        elif direction == "trailing":
            pool = self.case.capacities.trailing_ghost_pool_size
        else:
            raise ValueError(direction)
        if pool > 0:
            pool -= 2 * self.case.capacities.replica_region_size
        return (pool + wg - 1) // wg if pool > 0 else 0

    def _per_expand_dispatch_count(self) -> int:
        """expand_ghost_lists threads: (inbound ghost voxel, slot) pairs; 0 when
        V7_COMPACT_GHOST_LISTS is off or the slab has no peer."""
        if not (configured_compact_ghost_lists() and self._transport_segments):
            return 0
        ghost_grid = self.case.ghost_grid
        voxels = ghost_grid.leading_ghost_voxel_count + ghost_grid.trailing_ghost_voxel_count
        threads = voxels * self.case.capacities.max_particles_per_voxel
        wg = self.case.capacities.workgroup_size
        return (threads + wg - 1) // wg

    def _record_expand_ghost_lists(self, cmd) -> bool:
        """V7_COMPACT_GHOST_LISTS: rebuild the inbound ghost voxel rows from
        (count, first pid) after the upload, before append_departed / sweeps."""
        groups = self._per_expand_dispatch_count()
        if groups == 0:
            return False
        self._record_transfer_to_compute_barrier(cmd)
        self._bind_pipeline_and_sets(cmd, "expand_ghost_lists")
        vkCmdDispatch(cmd, groups, 1, 1)
        self._record_compute_barrier(cmd)
        return True

    def _per_departed_dispatch_count(self) -> int:
        wg = self.case.capacities.workgroup_size
        departed = self.case.capacities.departed_pool_size
        return (departed + wg - 1) // wg if departed > 0 else 0

    def departed_first_pid(self) -> int:
        capacities = self.case.capacities
        return (capacities.leading_ghost_pool_size + capacities.own_pool_size
                + capacities.trailing_ghost_pool_size + 1)

    # ----- bench timestamp helpers (no-op when self.bench is None) ----------

    def _bench_tick_transfer(self, cmd, label: str) -> None:
        """Transfer-pool tick (vkCmdWriteTimestamp IS legal on transfer-only
        queues). The pool RESET is NOT recorded here — vkCmdResetQueryPool
        is not supported on transfer-only queues (vk.xml queues list; the
        old embedded reset ran only because validation was off and the NV
        driver tolerated it). The reset lives in phase_a_cmd on the compute
        queue instead; see _record_phase_a_cmd."""
        if self.bench_transfer is not None:
            self.bench_transfer.tick(cmd, label)

    def _bench_tick(self, cmd, label: str) -> None:
        """Insert vkCmdWriteTimestamp into ``cmd`` if a BenchTimer is attached."""
        if self.bench is not None:
            self.bench.tick(cmd, label)

    # ----- V3.5 fast submit: cached cffi batches + raw entry points ----------

    def _fast_build_batch(self, sites: list, queue_stage: int):
        """sites = [(cmd, waits, signals)] for ONE queue, in submission order.
        Returns (VkSubmitInfo2[n], keepalive list). Wait/signal semaphore
        handles and counts are fixed per site; only the values change per
        frame (rewritten by _fast_set_values)."""
        infos = _ffi.new("VkSubmitInfo2[%d]" % len(sites))
        keep = [infos]
        for index, (cmd, waits, signals) in enumerate(sites):
            info = infos[index]
            info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2
            cmd_info = _ffi.new("VkCommandBufferSubmitInfo[1]")
            cmd_info[0].sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO
            cmd_info[0].commandBuffer = cmd
            info.commandBufferInfoCount = 1
            info.pCommandBufferInfos = cmd_info
            keep.append(cmd_info)
            for count_field, ptr_field, ops in (
                    ("waitSemaphoreInfoCount", "pWaitSemaphoreInfos", waits),
                    ("signalSemaphoreInfoCount", "pSignalSemaphoreInfos", signals)):
                if not ops:
                    continue
                arr = _ffi.new("VkSemaphoreSubmitInfo[%d]" % len(ops))
                for k, (semaphore, value) in enumerate(ops):
                    arr[k].sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO
                    arr[k].semaphore = semaphore
                    arr[k].value = value
                    arr[k].stageMask = queue_stage
                setattr(info, count_field, len(ops))
                setattr(info, ptr_field, arr)
                keep.append(arr)
        return infos, keep

    @staticmethod
    def _fast_set_values(info, waits: list, signals: list) -> None:
        if waits:
            assert info.waitSemaphoreInfoCount == len(waits)
            for k, (_, value) in enumerate(waits):
                info.pWaitSemaphoreInfos[k].value = value
        if signals:
            assert info.signalSemaphoreInfoCount == len(signals)
            for k, (_, value) in enumerate(signals):
                info.pSignalSemaphoreInfos[k].value = value

    def _fast_submit_prepare(self) -> None:
        """Build the cached batches (call after prepare_step_cmd_buffers)."""
        if not _FAST_SUBMIT:
            return
        s = self.sync
        compute_sites = [
            (self.phase_a_cmd, [] if _PHASE_A_NO_WAIT else s.phase_a_waits(1),
             s.phase_a_signals(1)),
            (self.phase_b_cmd, [], []),
            (self.phase_c_cmd, s.phase_c_waits(1), s.phase_c_signals(1)),
        ]
        self._fast_compute, self._fast_compute_keep = self._fast_build_batch(
            compute_sites, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT)
        self._fast_readback_dirs = list(self.transfer_readback_cmds)
        readback_sites = [
            (self.transfer_readback_cmds[d], s.readback_waits(d, 1),
             s.readback_signals(d, 1, i == len(self._fast_readback_dirs) - 1))
            for i, d in enumerate(self._fast_readback_dirs)]
        self._fast_readback, self._fast_readback_keep = (
            self._fast_build_batch(readback_sites, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT)
            if readback_sites else (None, []))
        self._fast_upload_dirs = list(self.transfer_upload_cmds)
        upload_sites = [
            (self.transfer_upload_cmds[d], s.upload_waits(d, 1),
             s.upload_signals(d, 1, i == len(self._fast_upload_dirs) - 1))
            for i, d in enumerate(self._fast_upload_dirs)]
        self._fast_upload, self._fast_upload_keep = (
            self._fast_build_batch(upload_sites, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT)
            if upload_sites else (None, []))
        self._fast_ready = True

    def _fast_queue_submit(self, queue, count: int, infos) -> None:
        with self._driver_submit_lock:
            result = _lib.vkQueueSubmit2(queue, count, infos, _ffi.NULL)
        if result != 0:
            raise RuntimeError(f"vkQueueSubmit2 (fast path) failed: VkResult {result}")

    def submit_frame_compute_fast(self, frame_n: int) -> None:
        """Phase A + B + C of frame n in ONE vkQueueSubmit2 (3 in-order
        VkSubmitInfo2). Wait/signal ops identical to submit_phase_a/b/c."""
        s = self.sync
        infos = self._fast_compute
        self._fast_set_values(infos[0], [] if _PHASE_A_NO_WAIT else s.phase_a_waits(frame_n),
                              s.phase_a_signals(frame_n))
        self._fast_set_values(infos[2], s.phase_c_waits(frame_n), s.phase_c_signals(frame_n))
        phase_c_cmd = (self.phase_c_cmd_odd
                       if (self.phase_c_cmd_odd is not None and frame_n % 2 == 1)
                       else self.phase_c_cmd)
        infos[2].pCommandBufferInfos[0].commandBuffer = phase_c_cmd
        if self.phase_a_cmd_odd is not None:   # E29 step trace
            odd = frame_n % 2 == 1
            infos[0].pCommandBufferInfos[0].commandBuffer = self.phase_a_cmd_odd if odd else self.phase_a_cmd
            infos[1].pCommandBufferInfos[0].commandBuffer = self.phase_b_cmd_odd if odd else self.phase_b_cmd
        self._fast_queue_submit(self.ctx.compute_queue, 3, infos)

    def submit_transfer_readback_fast(self, frame_n: int) -> None:
        if self._fast_readback is None:
            return
        s = self.sync
        last = len(self._fast_readback_dirs) - 1
        for i, d in enumerate(self._fast_readback_dirs):
            self._fast_set_values(self._fast_readback[i], s.readback_waits(d, frame_n),
                                  s.readback_signals(d, frame_n, i == last))
            if self.transfer_readback_cmds_odd:   # E29 step trace
                self._fast_readback[i].pCommandBufferInfos[0].commandBuffer = (
                    self.transfer_readback_cmds_odd[d] if frame_n % 2 == 1
                    else self.transfer_readback_cmds[d])
        self._fast_queue_submit(self.ctx.transfer_queue, last + 1, self._fast_readback)

    def submit_transfer_upload_fast(self, frame_n: int) -> None:
        if self._fast_upload is None:
            return
        s = self.sync
        last = len(self._fast_upload_dirs) - 1
        for i, d in enumerate(self._fast_upload_dirs):
            self._fast_set_values(self._fast_upload[i], s.upload_waits(d, frame_n),
                                  s.upload_signals(d, frame_n, i == last))
            if self.transfer_upload_cmds_odd:   # E29 step trace
                self._fast_upload[i].pCommandBufferInfos[0].commandBuffer = (
                    self.transfer_upload_cmds_odd[d] if frame_n % 2 == 1
                    else self.transfer_upload_cmds[d])
        queue = getattr(self.ctx, "transfer_queue_upload", None) or self.ctx.transfer_queue
        self._fast_queue_submit(queue, last + 1, self._fast_upload)

    def _set_step_trace_parity(self, parity: int) -> None:
        """E29: frame-parity timers (phase_trace_v7.StepTraceTimer) place the
        ticks of the cmd being recorded in this parity's slot block."""
        for timer in (self.bench, self.bench_transfer):
            if timer is not None and hasattr(timer, "set_recording_parity"):
                timer.set_recording_parity(parity)

    def _bench_reset_step(self, cmd, start_label: str) -> None:
        """First action of phase_a_cmd: reset step query slots + first tick."""
        if self.bench is not None:
            self.bench.record_step_reset_and_start(cmd, start_label)

    def _bench_reset_defrag(self, cmd, start_label: str) -> None:
        """First action of defrag_cmd: reset defrag slots + defrag start tick."""
        if self.bench is not None:
            self.bench.record_defrag_reset_and_start(cmd, start_label)

    # ----- sync2 barriers ---------------------------------------------------

    def _record_compute_barrier(self, cmd) -> None:
        """Global compute→compute memory barrier (sync2)."""
        mb = VkMemoryBarrier2(
            sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
            srcStageMask=VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            srcAccessMask=(VK_ACCESS_2_SHADER_STORAGE_READ_BIT
                           | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT),
            dstStageMask=VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            dstAccessMask=(VK_ACCESS_2_SHADER_STORAGE_READ_BIT
                           | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT),
        )
        info = VkDependencyInfo(
            sType=VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            memoryBarrierCount=1,
            pMemoryBarriers=[mb],
        )
        vkCmdPipelineBarrier2(cmd, info)

    def _record_transfer_to_compute_barrier(self, cmd) -> None:
        mb = VkMemoryBarrier2(
            sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
            srcStageMask=VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            srcAccessMask=VK_ACCESS_2_TRANSFER_WRITE_BIT,
            dstStageMask=VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            dstAccessMask=(VK_ACCESS_2_SHADER_STORAGE_READ_BIT
                           | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT),
        )
        info = VkDependencyInfo(
            sType=VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            memoryBarrierCount=1,
            pMemoryBarriers=[mb],
        )
        vkCmdPipelineBarrier2(cmd, info)

    def _record_compute_to_transfer_barrier(self, cmd) -> None:
        mb = VkMemoryBarrier2(
            sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
            srcStageMask=VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            srcAccessMask=VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            dstStageMask=VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            dstAccessMask=VK_ACCESS_2_TRANSFER_READ_BIT,
        )
        info = VkDependencyInfo(
            sType=VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            memoryBarrierCount=1,
            pMemoryBarriers=[mb],
        )
        vkCmdPipelineBarrier2(cmd, info)

    def _record_compute_to_clear_barrier(self, cmd) -> None:
        """V6: earlier compute reads/writes (atomics) -> a vkCmdFillBuffer that
        rewrites the same counters. The V5 counter reset is preceded only by a
        compute->compute barrier, whose second scope does not formally include
        the transfer (clear) stage; the new V6 counters get the proper one."""
        mb = VkMemoryBarrier2(
            sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
            srcStageMask=VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            srcAccessMask=(VK_ACCESS_2_SHADER_STORAGE_READ_BIT
                           | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT),
            dstStageMask=VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            dstAccessMask=VK_ACCESS_2_TRANSFER_WRITE_BIT,
        )
        info = VkDependencyInfo(
            sType=VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            memoryBarrierCount=1,
            pMemoryBarriers=[mb],
        )
        vkCmdPipelineBarrier2(cmd, info)

    def _record_compute_to_host_barrier(self, cmd) -> None:
        """End of Phase A: GPU finished writing sender_staging; host (worker
        thread) about to read it via mapped pointer. HOST_COHERENT alone is
        insufficient — need an explicit access-scope barrier per spec."""
        mb = VkMemoryBarrier2(
            sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
            srcStageMask=VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            srcAccessMask=VK_ACCESS_2_TRANSFER_WRITE_BIT,
            dstStageMask=VK_PIPELINE_STAGE_2_HOST_BIT,
            dstAccessMask=VK_ACCESS_2_HOST_READ_BIT,
        )
        info = VkDependencyInfo(
            sType=VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            memoryBarrierCount=1,
            pMemoryBarriers=[mb],
        )
        vkCmdPipelineBarrier2(cmd, info)

    def _record_reset_ghost_send_count(self, cmd, direction: str) -> None:
        """Zero the ghost_send_<direction>_count field of GlobalStatusBuffer
        before ghost_send.comp's atomicAdd. Without reset, the previous step's
        count corrupts slot allocation."""
        offset = (_OFFSET_GHOST_SEND_LEADING if direction == "leading"
                  else _OFFSET_GHOST_SEND_TRAILING)
        vkCmdFillBuffer(
            cmd, self.buffers["global_status"].handle, offset, 4, 0)
        if self.case.capacities.replica_region_size > 0:
            # V7_GHOST_LAYERS = 2: the inner / outer replica allocation counters.
            self._record_compute_to_clear_barrier(cmd)
            for counter_offset in (_OFFSET_REPLICA_INNER_SEND[direction],
                                   _OFFSET_REPLICA_OUTER_SEND[direction]):
                vkCmdFillBuffer(
                    cmd, self.buffers["global_status"].handle, counter_offset, 4, 0)

    def _record_reset_departed_count(self, cmd) -> None:
        """V7_KEEP_DEPARTED: zero the departed-pool allocation counter before
        this frame's ghost_send dispatches (both directions share the pool)."""
        vkCmdFillBuffer(
            cmd, self.buffers["global_status"].handle, _OFFSET_DEPARTED_COUNT, 4, 0)

    def _record_readback_for_direction(self, cmd, direction: str) -> None:
        """Scatter device-local ghost bytes → sender_staging_<direction>."""
        staging = self.staging_buffers[f"sender_staging_{direction}"]
        for seg in self._transport_segments[direction]:
            src_buf = self.buffers[seg.buffer_name]
            region = VkBufferCopy(
                srcOffset=seg.device_offset,
                dstOffset=seg.staging_offset,
                size=seg.size,
            )
            vkCmdCopyBuffer(cmd, src_buf.handle, staging.handle, 1, [region])

    def _record_upload_for_direction(self, cmd, direction: str) -> None:
        """Gather receiver_staging_<direction> → device-local ghost bytes.
        For the count segment specifically, dst is global_status[recv_count_offset]
        not [send_count_offset] — sender's count slot becomes receiver's
        ghost_recv_*_count slot."""
        staging = self.staging_buffers[f"receiver_staging_{direction}"]
        recv_overrides = self._recv_status_overrides[direction]
        for seg in self._transport_segments[direction]:
            dst_buf = self.buffers[seg.buffer_name]
            if seg.buffer_name == "global_status":
                dst_offset = recv_overrides[seg.device_offset]
            else:
                dst_offset = seg.device_offset
            region = VkBufferCopy(
                srcOffset=seg.staging_offset,
                dstOffset=dst_offset,
                size=seg.size,
            )
            vkCmdCopyBuffer(cmd, staging.handle, dst_buf.handle, 1, [region])

    # ----- Pipeline binding -------------------------------------------------

    def _bind_pipeline_and_sets(self, cmd, pipeline_key: str) -> None:
        vkCmdBindPipeline(
            cmd, VK_PIPELINE_BIND_POINT_COMPUTE, self.pipelines[pipeline_key])
        vkCmdBindDescriptorSets(
            cmd, VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline_layout,
            0, 4, self.descriptor_sets, 0, None)

    # ----- Density scratch copy-back (inside step cmd) ---------------------

    def _record_density_scratch_to_primary_copy(self, cmd, overlap_pair: bool = False) -> None:
        """After density.comp writes density_pressure_scratch, copy that back
        to density_pressure (primary) inside the same submit, over the regions
        of _density_copy_buffer_regions. Force.comp's next dispatch reads
        primary.

        E39 B9 (V7_DENSITY_COPY_COMPUTE, default 1): the copy is the compute
        pass density_scratch_copy.comp (a bit copy of 32-bit words over the
        same slots, _density_copy_slot_regions) between two compute -> compute
        barriers; 0 records the v6 vkCmdCopyBuffer
        (_record_density_scratch_to_primary_transfer_copy).

        Before the pass, the barrier orders the copy after every earlier
        compute access of this queue: the density (and adami wall) passes'
        scratch writes it reads; the correction / density reads of primary
        (rho_n) and the writes to primary (expand_ghost_lists,
        install_migrations, the wall pass) its writes follow. The transfer
        queue's upload into the ghost regions is behind phase C's semaphore
        wait (second scope COMPUTE_SHADER, which holds this dispatch).
        After it, every reader / writer of the two buffers is a compute
        shader of this queue (force, the wall pass, the next frame's
        ghost_send / correction / density: primary read and written, scratch
        overwritten) or ordered behind one: the next frame's readback and
        upload on the transfer queue through phase_a_done (signalled at
        COMPUTE_SHADER by the next phase A, queued after this cmd), defrag's
        vkCmdCopyBuffer into primary through its compute -> transfer barrier
        (source access SHADER_STORAGE_WRITE), host readbacks through the
        fence / semaphore wait after the frame, as for every field a kernel
        writes. The step trace keeps bracketing the copy: the caller's
        c_density_boundary_end / c_density_end ticks (BOTTOM_OF_PIPE) sit
        before and after this recording.

        E39 B3: phase C passes overlap_pair=True, which records the force_deep
        segment its V7_BAND_OVERLAP plan pairs with the copy (if any) between
        the copy dispatch and the closing barrier (_record_band_overlap_pair);
        every other call site and every slab without such a segment record
        the copy as above."""
        if not _DENSITY_COPY_COMPUTE:
            self._record_density_scratch_to_primary_transfer_copy(cmd)
            return
        # compute→compute (density wrote scratch; correction / density read primary)
        self._record_compute_barrier(cmd)

        def record_copy() -> None:
            self._bind_pipeline_and_sets(cmd, "density_scratch_copy")
            vkCmdDispatch(cmd, self._density_scratch_copy_group_count(), 1, 1)
        if overlap_pair:
            self._record_band_overlap_pair(cmd, "copy", record_copy)
        else:
            record_copy()
        # compute→compute (force will read primary; the next density writes scratch)
        self._record_compute_barrier(cmd)

    def _force_deep_segment_first_groups(self) -> tuple[int, ...]:
        """E39 B3: the first workgroups of this slab's force_deep segments that do not start at 0 (each needs a
        FORCE_SEGMENT pipeline); empty when V7_BAND_OVERLAP is off for the slab."""
        if not self._band_overlap_active():
            return ()
        return tuple(sorted({first for _, first, _ in self.band_overlap_plan.segments if first > 0}))

    @staticmethod
    def _force_deep_segment_pipeline_key(first_group: int) -> str:
        """E39 B3: the pipeline of the force_deep segment starting at ``first_group`` (0: the plain
        force_deep_interior_scratch)."""
        return "force_deep_interior_scratch" if first_group == 0 else f"force_deep_interior_scratch_from_{first_group}"

    def _record_force_deep_segment(self, cmd, first_group: int, group_count: int) -> None:
        """E39 B3: force_deep_interior_scratch over workgroups [first_group, first_group + group_count) of its
        per-own-particle dispatch, as a base-0 dispatch of group_count workgroups: from workgroup 0 the plain
        pipeline, otherwise its FORCE_SEGMENT pipeline, whose thread t runs force.comp's main() as thread
        first_group * workgroup size + t (spec constant 115), so the segments of a plan run exactly the threads of
        the one dispatch. Phase B's whole dispatch (B1, a slab without the B3 layout) is the segment (0, all).
        Every call adds group_count to the current phase recording's count (_check_force_deep_recorded_once)."""
        self._bind_pipeline_and_sets(cmd, self._force_deep_segment_pipeline_key(first_group))
        vkCmdDispatch(cmd, group_count, 1, 1)
        self._force_deep_recorded_groups = self.__dict__.get("_force_deep_recorded_groups", 0) + group_count

    def _check_force_deep_recorded_once(self) -> None:
        """E39 B3 recording invariant, at the end of every phase C recording: with cascading force the phase B
        recording (the last one of this simulator) + this phase C recording dispatch force_deep_interior_scratch's
        per-own-particle workgroups exactly once in all - whole in phase B (B1) or as the plan's segments (B3) -
        or the step would leave last step's acceleration / shift on the particles of a dropped segment (or run a
        segment twice) without any other sign. The counts are workgroups (_record_force_deep_segment); the
        segments' disjointness is _band_overlap_plan's and the CPU test's. Raises RuntimeError. A phase C recorded
        without a phase B (CPU tests of phase C alone) is checked only on a slab that records the B3 layout, where
        it raises."""
        if not _CASCADE_FORCE:
            return
        phase_b_groups = self.__dict__.get("_force_deep_phase_b_groups")
        phase_c_groups = self.__dict__.get("_force_deep_recorded_groups", 0)
        band_overlap = self._band_overlap_active()
        if phase_b_groups is None and phase_c_groups == 0 and not band_overlap:
            return
        expected = self._per_own_particle_dispatch_count()
        if phase_b_groups is None or phase_b_groups + phase_c_groups != expected:
            phase_b_text = ("no phase B recording" if phase_b_groups is None
                            else f"{phase_b_groups:,} workgroups in phase B")
            raise RuntimeError(
                f"force_deep_interior_scratch recorded with {phase_b_text} + {phase_c_groups:,} workgroups in phase "
                f"C, its dispatch has {expected:,}: V7_BAND_OVERLAP {'on' if band_overlap else 'off'} at this phase C "
                f"recording (resolution {self.__dict__.get('band_overlap_resolution')}, fused correction + density "
                f"{self._fused_correction_density_active()}); phase B and phase C must be recorded from one state "
                "(prepare_step_cmd_buffers)")

    def _record_band_overlap_pair(self, cmd, partner: str, record_partner) -> None:
        """E39 B3: record_partner() (bind + dispatch of a phase C band kernel / the density copy pass) and, on a
        slab that resolves V7_BAND_OVERLAP on, the force_deep segment its plan pairs with ``partner``, with NO
        barrier between the two: the caller's next barrier (or the end of phase C: the frame_done signal and phase
        A(n + 1)'s opening barrier) serves both (_resolve_band_overlap has the dependency table). Without a
        segment for ``partner``: record_partner() alone, the B1 recording."""
        segment = None
        if self._band_overlap_active():
            segment = next((item for item in self.band_overlap_plan.segments if item[0] == partner), None)
        if segment is None:
            record_partner()
            return
        _, first_group, group_count = segment
        if self.band_overlap_plan.force_deep_first:
            self._record_force_deep_segment(cmd, first_group, group_count)
            record_partner()
        else:
            record_partner()
            self._record_force_deep_segment(cmd, first_group, group_count)

    def _record_density_scratch_to_primary_transfer_copy(self, cmd) -> None:
        """V7_DENSITY_COPY_COMPUTE=0: the v6 recording of the copy."""
        scratch = self.buffers["density_pressure_scratch"]
        primary = self.buffers["density_pressure"]
        regions = self._density_copy_buffer_regions()
        # compute→transfer (density wrote scratch)
        self._record_compute_to_transfer_barrier(cmd)
        vkCmdCopyBuffer(cmd, scratch.handle, primary.handle, len(regions), regions)
        # transfer→compute (force will read primary)
        self._record_transfer_to_compute_barrier(cmd)

    def _density_copy_buffer_regions(self) -> list:
        """VkBufferCopy regions (source offset = destination offset) of the
        scratch -> primary copy, in recording order.

        Copy ONLY the own pid range (+ the V7_GHOST_LAYERS = 2 self regions,
        _ghost_self_density_copy_regions). The ghost-pid range of primary holds
        ρ values uploaded from the peer GPU this step; density.comp doesn't
        dispatch on ghost pids so scratch's ghost range is stale zero — a
        full-buffer copy would zero out the uploaded ghost density and make
        force.comp read ρ=0 for ghost neighbours (→ NaN pressure)."""
        density_stride = 8  # vec2 floats
        own_first = self.own_first_pid()
        own_pool = self.case.capacities.own_pool_size
        own_byte_offset = own_first * density_stride
        own_byte_size = own_pool * density_stride
        regions = [VkBufferCopy(srcOffset=own_byte_offset,
                                dstOffset=own_byte_offset,
                                size=own_byte_size)]
        regions += self._ghost_self_density_copy_regions(density_stride)
        return regions

    def _density_copy_slot_regions(self) -> list[tuple[int, int]]:
        """E39 B9: (first slot, slot count) of every region of
        _density_copy_buffer_regions, same order, one slot = one (rho, P) vec2
        = 8 bytes; the spec constants of density_scratch_copy.comp."""
        density_stride = 8
        slot_regions = []
        for region in self._density_copy_buffer_regions():
            byte_offset, byte_size = int(region.srcOffset), int(region.size)
            if (int(region.dstOffset) != byte_offset or byte_offset % density_stride
                    or byte_size % density_stride or byte_size <= 0):
                raise ValueError(f"density copy region (src {byte_offset}, dst {int(region.dstOffset)}, "
                                 f"{byte_size} B) is not a slot range at equal offsets")
            slot_regions.append((byte_offset // density_stride, byte_size // density_stride))
        if len(slot_regions) > _DENSITY_COPY_REGION_LIMIT:
            raise ValueError(f"{len(slot_regions)} density copy regions; density_scratch_copy.comp "
                             f"takes {_DENSITY_COPY_REGION_LIMIT}")
        return slot_regions

    def _density_scratch_copy_entries(self) -> list[tuple[int, str, Any]]:
        """Spec constants of density_scratch_copy.comp: ids 101 + 2 r /
        102 + 2 r = first slot / slot count of region r (slot count 0:
        unused)."""
        slot_regions = self._density_copy_slot_regions()
        slot_regions += [(0, 0)] * (_DENSITY_COPY_REGION_LIMIT - len(slot_regions))
        entries = []
        for region_index, (first_slot, slot_count) in enumerate(slot_regions):
            entries.append((101 + 2 * region_index, 'I', first_slot))
            entries.append((102 + 2 * region_index, 'I', slot_count))
        return entries

    def _density_scratch_copy_group_count(self) -> int:
        """Workgroups of the copy pass: one invocation per copied slot,
        _DENSITY_COPY_LOCAL_SIZE invocations per workgroup."""
        slot_total = sum(slot_count for _, slot_count in self._density_copy_slot_regions())
        return (slot_total + _DENSITY_COPY_LOCAL_SIZE - 1) // _DENSITY_COPY_LOCAL_SIZE

    def _record_wall_extrapolate(self, cmd, density_source: str = "primary",
                                 tick: Optional[str] = None) -> None:
        """E37 wall_boundary adami: wall_extrapolate.comp over the own pid range, then a compute barrier
        (force, or the next pass, reads the walls' (rho0, p_w) and dummy velocity). ``density_source`` = where
        the fluid's rho/P of this step are: "primary" after the scratch -> primary copy, "scratch" in phase B
        before it. Records nothing with simple (the v6 recording)."""
        if not self._wall_adami:
            return
        self._bind_pipeline_and_sets(
            cmd, "wall_extrapolate_scratch" if density_source == "scratch" else "wall_extrapolate")
        vkCmdDispatch(cmd, self._per_own_particle_dispatch_count(), 1, 1)
        if tick:
            self._bench_tick(cmd, tick)
        self._record_compute_barrier(cmd)

    def _ghost_self_density_copy_regions(self, density_stride: int) -> list:
        """V7_GHOST_LAYERS = 2: density_boundary recomputed the inner ghost
        column as self -- the inner replica region of every peer direction and
        the departed pool (departed migrants sit in the inner ghost column's
        voxel lists) -- into scratch; publish those rho_{n+1}/P_{n+1} to
        primary so this step's force reads them. Outer replicas are only read
        before this copy (as correction/density neighbours of the inner
        column), so they keep the uploaded rho_n. Empty for one ghost layer."""
        capacities = self.case.capacities
        if not self._ghost_self_density():
            return []
        regions = []
        first_slots = []
        if capacities.leading_ghost_pool_size > 0:
            first_slots.append(1)
        if capacities.trailing_ghost_pool_size > 0:
            first_slots.append(capacities.leading_ghost_pool_size
                               + capacities.own_pool_size + 1)
        for first_slot in first_slots:
            regions.append(VkBufferCopy(srcOffset=first_slot * density_stride,
                                        dstOffset=first_slot * density_stride,
                                        size=capacities.replica_region_size * density_stride))
        if capacities.departed_pool_size > 0:
            departed_offset = self.departed_first_pid() * density_stride
            regions.append(VkBufferCopy(srcOffset=departed_offset, dstOffset=departed_offset,
                                        size=capacities.departed_pool_size * density_stride))
        return regions

    # ========================================================================
    # Section 7: Initial data upload (Phase 3)
    # ========================================================================

    def own_first_pid(self) -> int:
        return self.case.capacities.leading_ghost_pool_size + 1

    def own_last_pid(self) -> int:
        return (self.case.capacities.leading_ghost_pool_size
                + self.case.capacities.own_pool_size)

    def _build_initial_data(self) -> dict[str, bytes]:
        """Build CPU-side payloads keyed by buffer name. Caller uploads each
        via _staging_upload + ctx.submit_and_wait. Buffers not in this dict
        get zeroed via _zero_buffer.

        V5 layout: own particles start at own_first_pid (V1.0a merged-buffer
        scheme); ghost-pid slots stay zero until next step's ghost_send fills
        them.
        """
        case = self.case
        pool_capacity = case.capacities.total_pool_capacity()
        own_first = self.own_first_pid()
        n_initial = case.initial.positions.shape[0]
        data: dict[str, bytes] = {}

        # position_voxel_id (vec4: xyz, voxel_id_as_float=0 initially)
        position_voxel_id = np.zeros((pool_capacity, 4), dtype=np.float32)
        position_voxel_id[own_first:own_first + n_initial, 0:3] = case.initial.positions
        data["position_voxel_id"] = position_voxel_id.tobytes()

        # velocity_mass (vec4: vx, vy, vz, mass)
        velocity_mass = np.zeros((pool_capacity, 4), dtype=np.float32)
        velocity_mass[own_first:own_first + n_initial, 0:3] = case.initial.velocities
        for i in range(n_initial):
            group = int(case.initial.material_group[i])
            mat = case.materials[group]
            velocity_mass[own_first + i, 3] = mat.rest_density * mat.volume
        data["velocity_mass"] = velocity_mass.tobytes()

        # density_pressure (vec2: ρ₀, 0); V7_DELTA_DENSITY stores ρ₀ - ρ_ref
        density_pressure = np.zeros((pool_capacity, 2), dtype=np.float32)
        stored_offset = self.stored_density_offset()
        for i in range(n_initial):
            group = int(case.initial.material_group[i])
            mat = case.materials[group]
            density_pressure[own_first + i, 0] = mat.rest_density - stored_offset
        data["density_pressure"] = density_pressure.tobytes()

        # material (uint group_id, 0 for empty slots)
        material_arr = np.zeros(pool_capacity, dtype=np.uint32)
        if n_initial > 0:
            material_arr[own_first:own_first + n_initial] = case.initial.material_group
        data["material"] = material_arr.tobytes()

        data["material_parameters"] = self._material_parameters_payload()
        return data

    def _material_parameters_payload(self) -> bytes:
        """material_parameters (48 B per row), shared by bootstrap and restart."""
        case = self.case
        mp_blob = bytearray()
        for mat in case.materials:
            # particle_mass (V6, was reserved_material_0): the float32 of
            # rest_density * volume, the same value _build_initial_data uploads
            # as every particle's mass (V7_PACKED_REPLICAS rebuilds it from here).
            row = struct.pack(
                "<I f f f f f f f f f f I",
                mat.kind, mat.rest_density, mat.viscosity, mat.eos_constant,
                mat.smoothing_length, mat.radius, mat.volume,
                mat.rotor_angular_velocity,
                mat.viscosity_transfer, mat.viscosity_rotation,
                mat.rest_density * mat.volume, mat.reserved_material_1,
            )
            assert len(row) == 48
            mp_blob.extend(row)
        if not mp_blob:
            mp_blob = b"\x00" * 48
        return bytes(mp_blob)

    def _staging_upload(self, dest: _Buffer, payload: bytes) -> None:
        if len(payload) > dest.size:
            raise ValueError(
                f"payload {len(payload)} > buffer {dest.size}")
        staging = self._allocate_buffer(
            size=len(payload),
            usage=VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            required_properties=(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                                 | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
        )
        try:
            mapped = vkMapMemory(self.ctx.device, staging.memory, 0, len(payload), 0)
            view = np.frombuffer(mapped, dtype=np.uint8, count=len(payload))
            view[:] = np.frombuffer(payload, dtype=np.uint8)
            vkUnmapMemory(self.ctx.device, staging.memory)
            cmd = self._allocate_oneshot_cmd()
            vkBeginCommandBuffer(cmd, VkCommandBufferBeginInfo(
                flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT))
            vkCmdCopyBuffer(cmd, staging.handle, dest.handle, 1, [
                VkBufferCopy(srcOffset=0, dstOffset=0, size=len(payload))
            ])
            vkEndCommandBuffer(cmd)
            self.ctx.submit_and_wait(cmd)
            vkFreeCommandBuffers(self.ctx.device, self.ctx.command_pool, 1, [cmd])
        finally:
            vkDestroyBuffer(self.ctx.device, staging.handle, None)
            vkFreeMemory(self.ctx.device, staging.memory, None)

    def _zero_buffer(self, dest: _Buffer) -> None:
        cmd = self._allocate_oneshot_cmd()
        vkBeginCommandBuffer(cmd, VkCommandBufferBeginInfo(
            flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT))
        vkCmdFillBuffer(cmd, dest.handle, 0, dest.size, 0)
        vkEndCommandBuffer(cmd)
        self.ctx.submit_and_wait(cmd)
        vkFreeCommandBuffers(self.ctx.device, self.ctx.command_pool, 1, [cmd])

    def _upload_initial_state(self, initial: Optional[dict[str, bytes]] = None) -> None:
        if initial is None:
            initial = self._build_initial_data()
        for name, buf in self.buffers.items():
            if name in initial:
                payload = initial[name]
                if len(payload) < buf.size:
                    payload = payload + b"\x00" * (buf.size - len(payload))
                self._staging_upload(buf, payload)
            else:
                self._zero_buffer(buf)
        for buf in self.scratch_buffers.values():
            self._zero_buffer(buf)
        print(f"[SimV7] uploaded initial state ({len(initial)} payload buffers)")

    # ========================================================================
    # Section 8: Bootstrap (Phase 3)
    # ========================================================================

    def _record_bootstrap_init_cmd(self):
        """Bootstrap stage 1: initialize_voxelization + (if peer) ghost_send +
        readback. Outbox staging ready after this submit. Fence-wait submit.
        For sims with no peer this is just init_voxelization."""
        cmd = self._allocate_oneshot_cmd()
        vkBeginCommandBuffer(cmd, VkCommandBufferBeginInfo(
            flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT))
        self._record_compute_barrier(cmd)

        per_p = self._per_own_particle_dispatch_count()

        self._bind_pipeline_and_sets(cmd, "initialize_voxelization")
        vkCmdDispatch(cmd, per_p, 1, 1)

        for direction in ("leading", "trailing"):
            if direction not in self._transport_segments:
                continue
            self._record_compute_barrier(cmd)
            self._record_reset_ghost_send_count(cmd, direction)
            self._record_transfer_to_compute_barrier(cmd)
            self._record_ghost_send_dispatch(cmd, direction)
            self._record_compute_to_transfer_barrier(cmd)
            self._record_readback_for_direction(cmd, direction)
            self._record_compute_to_host_barrier(cmd)

        vkEndCommandBuffer(cmd)
        return cmd

    def _record_bootstrap_compute_cmd(self):
        """Bootstrap stage 2: (if peer) upload + install_migrations →
        correction(ALL) → density → force → bootstrap_half_kick. Runs after
        host memcpy completes the cross-GPU ghost transport."""
        cmd = self._allocate_oneshot_cmd()
        vkBeginCommandBuffer(cmd, VkCommandBufferBeginInfo(
            flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT))

        per_p = self._per_own_particle_dispatch_count()

        for direction in ("leading", "trailing"):
            if direction not in self._transport_segments:
                continue
            self._record_upload_for_direction(cmd, direction)
            self._record_transfer_to_compute_barrier(cmd)
            self._bind_pipeline_and_sets(cmd, f"install_migrations_{direction}")
            per_ghost_pid = self._per_ghost_pid_dispatch_count(direction)
            vkCmdDispatch(cmd, per_ghost_pid, 1, 1)
            self._record_compute_barrier(cmd)
        self._record_expand_ghost_lists(cmd)

        # E37 wall_boundary adami: the wall pass once before step 0, on the initial state (the bootstrap correction
        # and density read its (rho0, p_w) as every step's do), and again after density + copy below, before force.
        if self._wall_adami:
            self._record_compute_barrier(cmd)
            self._record_wall_extrapolate(cmd)

        # E39 B4 (V7_DEEP_WALL_SKIP): on a slab without peers correction_all / density_all skip deep walls, so the
        # marker of the bootstrap lists goes first (the barrier orders it after the voxelization submit).
        if self._deep_wall_skip_active() and self._deep_wall_skips_full_domain():
            self._record_compute_barrier(cmd)
            self._record_deep_wall_marker(cmd)

        ghost_self = 1 if (self.ghost_layers() >= 2 and self._transport_segments) else 0
        if self._fused_correction_density_active():
            # E39 B1: correction_all + density_all in one traversal, then (two ghost layers) the band pass of both
            # (the own band again, same inputs, same result, plus the inner ghost column as self) in one; the
            # barrier orders the pass's rewrite of the own band after the first one's.
            self._bind_pipeline_and_sets(cmd, "correction_density_all")
            vkCmdDispatch(cmd, per_p, 1, 1)
            if ghost_self:
                self._record_compute_barrier(cmd)
                self._bind_pipeline_and_sets(cmd, "correction_density_boundary_band")
                vkCmdDispatch(cmd, self._per_band_dispatch_count(
                    self.band_widths[0], self._ghost_self_layer(2, 1, "correction")), 1, 1)
        else:
            self._bind_pipeline_and_sets(cmd, "correction_all")
            vkCmdDispatch(cmd, per_p, 1, 1)
            self._record_compute_barrier(cmd)
            if ghost_self:
                # V7_GHOST_LAYERS = 2: the band pipeline re-runs the own band
                # (same inputs, same result) plus the inner ghost column as self,
                # so the bootstrap force reads this step's rho/P at the seam too.
                self._bind_pipeline_and_sets(cmd, "correction_boundary_band")
                vkCmdDispatch(cmd, self._per_band_dispatch_count(
                    self.band_widths[0], self._ghost_self_layer(2, 1, "correction")), 1, 1)
                self._record_compute_barrier(cmd)

            self._bind_pipeline_and_sets(cmd, "density_all")
            vkCmdDispatch(cmd, per_p, 1, 1)
            if ghost_self:
                self._record_compute_barrier(cmd)
                self._bind_pipeline_and_sets(cmd, "density_boundary_band")
                vkCmdDispatch(cmd, self._per_band_dispatch_count(
                    self.band_widths[1], self._ghost_self_layer(2, 1, "density")), 1, 1)
        self._record_density_scratch_to_primary_copy(cmd)
        self._record_wall_extrapolate(cmd)

        self._bind_pipeline_and_sets(cmd, "force_all")
        vkCmdDispatch(cmd, per_p, 1, 1)
        self._record_compute_barrier(cmd)

        self._bind_pipeline_and_sets(cmd, "bootstrap_half_kick")
        vkCmdDispatch(cmd, per_p, 1, 1)

        vkEndCommandBuffer(cmd)
        return cmd

    def bootstrap_init(self) -> None:
        """First half of split bootstrap: upload initial state + run
        initialize_voxelization + ghost_send + readback. After this returns,
        sender_staging_view(<dir>) holds the boundary replicas ready for
        host memcpy. Orchestrator calls this on both sims, then bridges via
        memcpy, then calls bootstrap_compute on both sims."""
        self._upload_initial_state()
        cmd = self._record_bootstrap_init_cmd()
        self.ctx.submit_and_wait(cmd)
        vkFreeCommandBuffers(self.ctx.device, self.ctx.command_pool, 1, [cmd])

    def bootstrap_compute(self) -> None:
        """Second half of split bootstrap: upload ghost (host memcpy already
        landed in receiver_staging by caller) + install_migrations + correction
        + density + force + bootstrap_half_kick. After this returns, a_0 and
        v_{-1/2} are set with valid ghost-neighbor SPH contributions."""
        cmd = self._record_bootstrap_compute_cmd()
        self.ctx.submit_and_wait(cmd)
        vkFreeCommandBuffers(self.ctx.device, self.ctx.command_pool, 1, [cmd])
        status = self.readback_global_status()
        print(f"[SimV7] bootstrap done: alive={status['alive_particle_count']} "
              f"overflow_inside={status['overflow_inside_count']} "
              f"overflow_incoming={status['overflow_incoming_count']}")

    def bootstrap(self) -> None:
        """Single-GPU bootstrap convenience: combine init + compute. No ghost
        transport needed when this sim has no peer."""
        if self._transport_segments:
            raise RuntimeError(
                "bootstrap() is single-GPU only; for dual-GPU call "
                "bootstrap_init() + (host memcpy) + bootstrap_compute() via "
                "DualGpuOrchestratorV7.bootstrap_all()")
        self.bootstrap_init()
        self.bootstrap_compute()

    # ========================================================================
    # Section 8b: Restart from a saved step-boundary state
    # ========================================================================

    # The set-0 fields of one own particle at a step boundary, as (components
    # per pool slot, dtype). Everything a step reads from the previous one:
    # predict reads position_voxel_id (x_n), velocity_mass (v_{n-1/2}, m),
    # acceleration (a_n), shift (shift_n) and material; density integrates
    # density_pressure.x (rho_n) and a ghost replica packed in phase A carries
    # rho_n, P_n. correction_inverse and density_gradient_kernel_sum are
    # recomputed before every read but are restored too (the 1-layer replica
    # carries them over the link). extension_fields carries no physics.
    RESTART_FIELD_LAYOUT = {
        "position_voxel_id":           (4, np.float32),
        "velocity_mass":               (4, np.float32),
        "density_pressure":            (2, np.float32),
        "acceleration":                (4, np.float32),
        "shift":                       (4, np.float32),
        "material":                    (1, np.uint32),
        "correction_inverse":          (8, np.float32),
        "density_gradient_kernel_sum": (4, np.float32),
        "extension_fields":            (4, np.float32),
    }

    def restart_init(self, state: dict, stored_density_offset: float = 0.0) -> None:
        """Restart instead of bootstrap, from a saved step-boundary state.

        ``state[name]`` holds this slab's own particles, one row each, for
        every field of RESTART_FIELD_LAYOUT (same row order in every field;
        the row order becomes the initial pid order). Rows go to own pids
        own_first_pid() + k, every other slot and buffer is zeroed, and
        initialize_voxelization rebuilds the voxel lists and voxel ids.
        No correction / density / force pass and no backward half kick:
        a_n, shift_n, rho_n, P_n and v_{n-1/2} come from the state, so the
        next submitted frame is step n + 1 of the run the state came from.
        Ghost slots stay empty; the frame's own phase A (ghost_send) and the
        transport fill them before any kernel reads them, as in every step.
        Callers then record the step cmd buffers (ChainOrchestratorV7.
        restart_all, which also runs the bootstrap defrag)."""
        layout = self.RESTART_FIELD_LAYOUT
        missing = [name for name in layout if name not in state]
        if missing:
            raise ValueError(f"restart state lacks {missing}")
        row_count = int(np.asarray(state["position_voxel_id"]).shape[0])
        own_pool = self.case.capacities.own_pool_size
        if row_count > own_pool:
            raise ValueError(f"restart state has {row_count:,} particles for an own "
                             f"pool of {own_pool:,}")
        pool_capacity = self.case.capacities.total_pool_capacity()
        own_first = self.own_first_pid()
        payload: dict[str, bytes] = {}
        for name, (component_count, element_type) in layout.items():
            rows = np.asarray(state[name])
            expected_shape = ((row_count,) if component_count == 1
                              else (row_count, component_count))
            if rows.shape != expected_shape or rows.dtype != element_type:
                raise ValueError(f"restart field {name}: {rows.dtype} {rows.shape}, "
                                 f"expected {np.dtype(element_type)} {expected_shape}")
            full = np.zeros((pool_capacity,) + expected_shape[1:], dtype=element_type)
            full[own_first:own_first + row_count] = rows
            shift = float(stored_density_offset) - self.stored_density_offset()
            if name == "density_pressure" and shift != 0.0:
                # the state holds stored + stored_density_offset = rho (0: plain rho);
                # convert to this sim's representation in float64 (exact when the
                # two offsets are equal: the shift is then 0 and nothing changes)
                rows64 = full[own_first:own_first + row_count, 0].astype(np.float64) + shift
                full[own_first:own_first + row_count, 0] = rows64.astype(np.float32)
            payload[name] = full.tobytes()
        if row_count and float(np.asarray(state["velocity_mass"])[:, 3].min()) <= 0.0:
            raise ValueError("restart state has rows with mass <= 0 "
                             "(initialize_voxelization would skip them)")
        payload["material_parameters"] = self._material_parameters_payload()
        self._upload_initial_state(payload)

        cmd = self._allocate_oneshot_cmd()
        vkBeginCommandBuffer(cmd, VkCommandBufferBeginInfo(
            flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT))
        self._record_compute_barrier(cmd)
        self._bind_pipeline_and_sets(cmd, "initialize_voxelization")
        vkCmdDispatch(cmd, self._per_own_particle_dispatch_count(), 1, 1)
        self._record_compute_barrier(cmd)
        # E37 wall_boundary adami: the wall pass once before the first step, from the restored fluid state (the
        # dummy velocity is not part of the state).
        self._record_wall_extrapolate(cmd)
        vkEndCommandBuffer(cmd)
        self.ctx.submit_and_wait(cmd)
        vkFreeCommandBuffers(self.ctx.device, self.ctx.command_pool, 1, [cmd])

        # Every row must land in an OWN voxel: a particle exactly on a cut
        # line could round into a ghost column on the GPU, where the next
        # upload would overwrite it.
        status = self.readback_global_status()
        positions = np.frombuffer(self.readback_buffer_by_name("position_voxel_id"),
                                  dtype=np.float32)[:pool_capacity * 4].reshape(pool_capacity, 4)
        voxel_ids = np.rint(positions[own_first:own_first + row_count, 3]).astype(np.int64)
        first_own_voxel = self.case.ghost_grid.leading_ghost_voxel_count + 1
        last_own_voxel = (self.case.grid.total_voxel_count()
                          - self.case.ghost_grid.trailing_ghost_voxel_count)
        outside = int(((voxel_ids < first_own_voxel) | (voxel_ids > last_own_voxel)).sum())
        if (status["alive_particle_count"] != row_count
                or status["overflow_inside_count"] or outside):
            raise RuntimeError(
                f"restart: {status['alive_particle_count']:,} of {row_count:,} particles "
                f"placed, overflow_inside={status['overflow_inside_count']}, {outside} "
                f"outside the own voxels")
        print(f"[SimV7] restart state loaded: {row_count:,} own particles, voxel lists "
              f"rebuilt (no bootstrap correction/density/force/half kick)")

    # ========================================================================
    # Section 9: Readback (Phase 3)
    # ========================================================================

    def _readback_buffer(self, buf: _Buffer) -> bytes:
        staging = self._allocate_buffer(
            size=buf.size,
            usage=VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            required_properties=(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                                 | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
        )
        try:
            cmd = self._allocate_oneshot_cmd()
            vkBeginCommandBuffer(cmd, VkCommandBufferBeginInfo(
                flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT))
            vkCmdCopyBuffer(cmd, buf.handle, staging.handle, 1, [
                VkBufferCopy(srcOffset=0, dstOffset=0, size=buf.size)
            ])
            vkEndCommandBuffer(cmd)
            self.ctx.submit_and_wait(cmd)
            vkFreeCommandBuffers(self.ctx.device, self.ctx.command_pool, 1, [cmd])
            mapped = vkMapMemory(self.ctx.device, staging.memory, 0, buf.size, 0)
            payload = bytes(np.frombuffer(mapped, dtype=np.uint8, count=buf.size))
            vkUnmapMemory(self.ctx.device, staging.memory)
            return payload
        finally:
            vkDestroyBuffer(self.ctx.device, staging.handle, None)
            vkFreeMemory(self.ctx.device, staging.memory, None)

    def readback_global_status(self) -> dict:
        """GlobalStatusBuffer is 40 x 4 B = 160 B in V6 (V5: 24). Layout per
        common.glsl. Every key starting with "overflow_" is a must-be-zero
        invariant (V6 adds overflow_departed_count)."""
        payload = self._readback_buffer(self.buffers["global_status"])
        fields = struct.unpack("<I f " + "I " * 38, payload)
        return {name: fields[index]
                for index, name in enumerate(_GLOBAL_STATUS_FIELD_NAMES)}

    def readback_pool_health(self) -> dict:
        """PoolHealthBuffer (set 3 binding 1) = 4 uint = 16 B. atomicMax
        watermarks written by install_migrations.comp, NEVER reset (survive
        defrag). Use to size own_pool_size from data and to warn before the
        install tail overflows (which would silently drop a migrant).

        Derived fields:
          free_margin    = own_pool_size - peak_tail_high_water
                           (<= 0 means the pool overflowed; magnitude = deficit)
          used_fraction  = peak_tail_high_water / own_pool_size
          sizing_target  = peak_tail_high_water (minimum own_pool_size that would
                           have avoided overflow this run; add margin on top)
        """
        payload = self._readback_buffer(self.buffers["pool_health"])
        peak_tail, peak_migration, peak_departed, _r1 = struct.unpack("<I I I I", payload)
        own_pool = self.case.capacities.own_pool_size
        return {
            "peak_tail_high_water": peak_tail,
            "peak_migration_count": peak_migration,
            # V6: most migrants sent away in one frame (departed pool demand).
            "peak_departed_count":  peak_departed,
            "departed_pool_size":   self.case.capacities.departed_pool_size,
            "own_pool_size":        own_pool,
            "free_margin":          own_pool - peak_tail,
            "used_fraction":        (peak_tail / own_pool) if own_pool else 0.0,
            "sizing_target":        peak_tail,
        }

    def readback_buffer_by_name(self, name: str) -> bytes:
        """Public wrapper of _readback_buffer for debug-log helpers.

        Issues one vkCmdCopyBuffer device→staging + fence-wait. Use for ad-hoc
        single-buffer inspection. For multi-buffer snapshot, prefer
        readback_buffers_batch() to amortize the fence cost.
        """
        if name not in self.buffers:
            raise KeyError(f"unknown buffer: {name!r}")
        return self._readback_buffer(self.buffers[name])

    def readback_buffers_batch(self, names: list[str], density: str = "absolute") -> dict[str, bytes]:
        """Read N buffers via a single cmd buffer + fence wait.

        density = "absolute" (default): density_pressure / density_pressure_scratch
        come back with .x = rho also under V7_DELTA_DENSITY (rho_ref added, float32);
        "stored": the raw stored value (rho - rho_ref under the switch) for tools
        that need the exact representation (delta_density_eval).

        Per-buffer fence overhead is the dominant cost when reading many small
        buffers; batching N copies into one submit drops 22 fence waits to 1
        for a 23-buffer snapshot (~100-200ms saved on cavity scale). Stagings
        are allocated/destroyed per call (debug-only path; not hot).
        """
        if not names:
            return {}
        for name in names:
            if name not in self.buffers:
                raise KeyError(f"unknown buffer: {name!r}")

        device = self.ctx.device
        # 1. Allocate one staging per source buffer
        stagings: dict[str, _Buffer] = {}
        for name in names:
            buf = self.buffers[name]
            stagings[name] = self._allocate_buffer(
                size=buf.size,
                usage=VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                required_properties=(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                                     | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
            )

        try:
            # 2. Record one cmd buffer with all copies
            cmd = self._allocate_oneshot_cmd()
            vkBeginCommandBuffer(cmd, VkCommandBufferBeginInfo(
                flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT))
            for name in names:
                buf = self.buffers[name]
                staging = stagings[name]
                vkCmdCopyBuffer(cmd, buf.handle, staging.handle, 1, [
                    VkBufferCopy(srcOffset=0, dstOffset=0, size=buf.size)
                ])
            vkEndCommandBuffer(cmd)

            # 3. Single fence-wait submit
            self.ctx.submit_and_wait(cmd)
            vkFreeCommandBuffers(device, self.ctx.command_pool, 1, [cmd])

            # 4. Map each staging, copy bytes out
            result: dict[str, bytes] = {}
            for name in names:
                staging = stagings[name]
                mapped = vkMapMemory(device, staging.memory, 0, staging.size, 0)
                result[name] = bytes(np.frombuffer(mapped, dtype=np.uint8,
                                                   count=staging.size))
                vkUnmapMemory(device, staging.memory)
                offset = self.stored_density_offset()
                if (density == "absolute" and offset != 0.0
                        and name in ("density_pressure", "density_pressure_scratch")):
                    values = np.frombuffer(result[name], dtype=np.float32).copy().reshape(-1, 2)
                    values[:, 0] += np.float32(offset)
                    result[name] = values.tobytes()
            return result
        finally:
            for staging in stagings.values():
                vkDestroyBuffer(device, staging.handle, None)
                vkFreeMemory(device, staging.memory, None)

    # ========================================================================
    # Section 10: Step cmd buffers + sync2 timeline submit/wait (Phase 3b)
    #
    # V5 v1.0 single-GPU simplification: phase A only does predict + update_voxel;
    # phase B does correction(ALL); phase C does density + force. ghost_send /
    # install_migration / readback / upload are skipped (no peer; ghost-pid pool
    # is sized 0 in case construction). Phase 4 wires per-direction ghost flow.
    # ========================================================================

    def _record_phase_a_cmd(self):
        """V5 phase A: predict + update_voxel + (per-direction: reset ghost_send_count
        + ghost_send dispatch + readback vkCmdCopyBuffer + compute→host barrier).
        signal 3N+1 at submit."""
        cmd = self._allocate_oneshot_cmd()
        vkBeginCommandBuffer(cmd, VkCommandBufferBeginInfo(
            flags=VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT))
        # Bench: phase A is the first cmd of the frame, so it owns the
        # per-frame step-slot reset. No-op when bench is unattached.
        self._bench_reset_step(cmd, "a_start")
        # The TRANSFER pool's per-frame reset also lives here, on the
        # compute queue (2026-07-22 audit fix: vkCmdResetQueryPool is
        # invalid on transfer-only queues). This cmd signals phase_a_done,
        # which every readback cmd of frame N waits on, so the reset precedes
        # frame N's transfer writes. It follows frame N-1's transfer writes
        # through the compute queue: they all happen-before upload_done(N-1),
        # which C(N-1) waits on, and this cmd is submitted after C(N-1). With
        # the default V7_PHASE_A_NO_WAIT=1 (no frame_done(N-1) wait) that
        # second edge is queue order — the queue does not start A(N) before
        # C(N-1)'s wait is satisfied — not a semaphore dependency on this
        # reset; instrumentation only (no physics reads the pool).
        if self.bench_transfer is not None:
            self.bench_transfer.record_external_reset(cmd)
        self._record_compute_barrier(cmd)

        per_p = self._per_own_particle_dispatch_count()
        per_v = self._per_extended_voxel_dispatch_count()

        self._bind_pipeline_and_sets(cmd, "predict")
        vkCmdDispatch(cmd, per_p, 1, 1)
        self._bench_tick(cmd, "a_predict_end")
        self._record_compute_barrier(cmd)

        self._bind_pipeline_and_sets(cmd, "update_voxel")
        vkCmdDispatch(cmd, per_v, 1, 1)
        self._bench_tick(cmd, "a_voxel_end")

        # V7_KEEP_DEPARTED: zero the departed-pool counter once per frame, with
        # the same barrier pattern as the per-direction counter reset below.
        if self.case.capacities.departed_pool_size > 0 and self._transport_segments:
            self._record_compute_to_clear_barrier(cmd)
            self._record_reset_departed_count(cmd)
            self._record_transfer_to_compute_barrier(cmd)

        # Per-direction ghost flow (skipped if this GPU has no peer on that side).
        # Path A+: ghost_send.comp dispatch only. The readback DMA + host
        # coherence barrier have been MOVED to the transfer queue (see
        # _record_transfer_readback_cmd). Phase A's phase_a_done signal marks
        # "ghost_send has written device buffers"; the transfer queue waits on
        # it, then runs the readback DMA in parallel with Phase B
        # (correction_interior + density_deep_interior) on the compute queue.
        for direction in ("leading", "trailing"):
            if direction not in self._transport_segments:
                continue
            self._record_compute_barrier(cmd)
            self._record_reset_ghost_send_count(cmd, direction)
            self._record_transfer_to_compute_barrier(cmd)
            self._record_ghost_send_dispatch(cmd, direction)
            self._bench_tick(cmd, f"a_ghost_{direction}_end")

        vkEndCommandBuffer(cmd)
        return cmd

    def _record_phase_b_cmd(self):
        """V5 Path A+ Phase B: correction_interior + density_deep_interior over
        own pid range. Both kernels skip their respective boundary bands
        (self.band_widths, default correction 2 / density_deep 2 voxels) so
        they only touch particles whose inputs are all final for this frame.

        Runs in parallel with the transfer chain on the transfer queue
        (readback DMA + worker memcpy + upload DMA). Phase B's total work
        must be ≥ transfer chain length to fully hide it — on cross-vendor
        cavity 1M, Phase B ≈ 1.9 ms vs transfer chain ≈ 1.3 ms, so fully
        hidden. See docs/sph_v5_design.md Path A+ section for the cascading-
        split data dependency analysis.

        density_deep_interior writes density_pressure_scratch[deep_interior
        pids]; the scratch→primary copy is deferred until Phase C (after
        density_boundary also writes its scratch slice). This means the
        primary density_pressure buffer holds previous-frame values during
        Phase B — that's fine because Phase B's only consumer of density is
        density_deep_interior itself, which reads primary for ρ_n (the
        SPH continuity equation uses LAST-frame density). force_deep_interior
        would need to read scratch (this frame's value); since we don't
        cascade force in this design, this is a non-issue."""
        cmd = self._allocate_oneshot_cmd()
        vkBeginCommandBuffer(cmd, VkCommandBufferBeginInfo(
            flags=VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT))
        # E39 B3: force_deep workgroups of this recording (_record_force_deep_segment counts; phase C checks the sum)
        self._force_deep_recorded_groups = 0
        self._bench_tick(cmd, "b_start")
        # Entry: cross-submit memory visibility (Phase A writes → here reads)
        self._record_compute_barrier(cmd)
        # E39 B4 (V7_DEEP_WALL_SKIP): the deep-wall marker of this step's lists (update_voxel, phase A), for the
        # two interior kernels below. Here rather than in phase A: the transfer chain starts at phase_a_done, and
        # the marker reads only own voxels' lists (the upload writes the ghost range).
        self._record_deep_wall_marker(cmd, tick="b_deep_wall_marker_end")

        per_p = self._per_own_particle_dispatch_count()

        if self._fused_correction_density_active():
            # E39 B1: correction_interior + density_deep_interior in one traversal (one band width, so the same
            # particles; a particle's density reads only its own L, computed in the same invocation). Exit
            # barrier: phase C's band pass and force read L / scratch.
            self._bind_pipeline_and_sets(cmd, "correction_density_interior")
            vkCmdDispatch(cmd, per_p, 1, 1)
            self._record_compute_barrier(cmd)
            self._bench_tick(cmd, "b_correction_density_interior_end")
        else:
            self._bind_pipeline_and_sets(cmd, "correction_interior")
            vkCmdDispatch(cmd, per_p, 1, 1)
            # Barrier between correction_interior and density_deep_interior:
            # density reads correction_inverse just written.
            self._record_compute_barrier(cmd)
            self._bench_tick(cmd, "b_correction_interior_end")

            # Path A+ P5: density_deep_interior in Phase B. Boundary band =
            # band_widths[1] voxels (>= correction's: density reads its own L and
            # no neighbour correction output). Writes to scratch;
            # scratch→primary copy happens in Phase C after density_boundary.
            self._bind_pipeline_and_sets(cmd, "density_deep_interior")
            vkCmdDispatch(cmd, per_p, 1, 1)
            # Exit barrier — Phase C's density_boundary reads scratch, force_all
            # reads primary (after copy). Cross-submit visibility required.
            self._record_compute_barrier(cmd)
            self._bench_tick(cmd, "b_density_deep_interior_end")

        if _CASCADE_FORCE and self._band_overlap_active():
            # E39 B3 (V7_BAND_OVERLAP): only the plan's phase B segment of force_deep_interior_scratch (none when it
            # has 0 workgroups); the other segments run in phase C next to the band kernels. Phase B then ends with
            # correction_density_interior's exit barrier above.
            phase_b_groups = self.band_overlap_plan.phase_b_groups
            if phase_b_groups > 0:
                self._record_force_deep_segment(cmd, 0, phase_b_groups)
                self._record_compute_barrier(cmd)
                self._bench_tick(cmd, "b_force_deep_interior_end")
        elif _CASCADE_FORCE:
            # V3.3 cascading force: force on the deep interior (band =
            # band_widths[2] voxel columns) reads rho/P from SCRATCH (this
            # frame's values for columns >= band_widths[1], all written by
            # density_deep_interior above), self correction_inverse /
            # kernel_sum from correction_interior above, and positions /
            # velocities from Phase A. Nothing it touches is modified by
            # Phase C's install_migrations (migrants land in column 0) or
            # density_boundary (columns below band_widths[1]), so it is
            # safe here and widens the transfer-hiding window to B + force.
            # E37 wall_boundary adami: the wall pass reads the fluid's rho/P of this frame from scratch too and
            # writes (rho0, p_w) to scratch (read by the force below) and primary.
            self._record_wall_extrapolate(cmd, "scratch", tick="b_wall_extrapolate_end")
            # bind force_deep_interior_scratch + dispatch per_p (E39 B3: counted as the segment (0, all))
            self._record_force_deep_segment(cmd, 0, per_p)
            self._record_compute_barrier(cmd)
            self._bench_tick(cmd, "b_force_deep_interior_end")

        vkEndCommandBuffer(cmd)
        self._force_deep_phase_b_groups = self._force_deep_recorded_groups
        return cmd

    def _record_phase_c_cmd(self, parity: int = 0):
        """V5 Phase C: per-direction install_migration → correction
        (BOUNDARY) → density → force. Submitted waiting upload_done, signals
        frame_done (values per sync scheme).

        correction_boundary runs ONLY on boundary-band particles (interior
        already covered by Phase B). It runs AFTER install_migrations so that:
          (a) newly arrived migrants (now own pids in the boundary column)
              get their M⁻¹ + density_gradient_kernel_sum computed this frame
          (b) all boundary particles see *uploaded* ghost data (set 1 ghost-vid
              inside_particle_index now has own-frame pid values, not the
              peer-frame scratch left over from Phase A's ghost_send).
        See docs/sph_v5_design.md §5.3 + §7."""
        cmd = self._allocate_oneshot_cmd()
        vkBeginCommandBuffer(cmd, VkCommandBufferBeginInfo(
            flags=VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT))
        # E39 B3: force_deep workgroups of this recording (the plan's phase C segments; checked at the end)
        self._force_deep_recorded_groups = 0
        if self.bench is not None:
            self.bench.begin_phase_c_region(cmd, parity)
        self._bench_tick(cmd, "c_start")

        per_p = self._per_own_particle_dispatch_count()

        # V7_COMPACT_GHOST_LISTS: the inbound ghost voxel rows first.
        if self._record_expand_ghost_lists(cmd):
            self._bench_tick(cmd, "c_expand_end")

        # Per-direction install_migration (skipped if no peer). Path A+:
        # upload DMA has been MOVED to the transfer queue (see
        # _record_transfer_upload_cmd). Phase C's wait on upload_done
        # guarantees receiver_staging→device DMA is complete before install
        # starts, so install can read the ghost-pid range directly.
        for direction in ("leading", "trailing"):
            if direction not in self._transport_segments:
                continue
            # Cross-queue handover barrier: transfer queue wrote our ghost-
            # pid SoA via vkCmdCopyBuffer; we now read it via install_
            # migrations.comp. CONCURRENT sharing (P2) means no queue family
            # ownership transfer needed, but visibility barrier still required.
            self._record_transfer_to_compute_barrier(cmd)
            self._bind_pipeline_and_sets(cmd, f"install_migrations_{direction}")
            per_ghost_pid = self._per_ghost_pid_dispatch_count(direction)
            vkCmdDispatch(cmd, per_ghost_pid, 1, 1)
            self._record_compute_barrier(cmd)
            self._bench_tick(cmd, f"c_install_{direction}_end")

        # V7_KEEP_DEPARTED: re-register this frame's outgoing migrants in the
        # (just uploaded) ghost voxel lists before any seam sweep reads them.
        if self.case.capacities.departed_pool_size > 0 and self._transport_segments:
            self._bind_pipeline_and_sets(cmd, "append_departed")
            vkCmdDispatch(cmd, self._per_departed_dispatch_count(), 1, 1)
            self._record_compute_barrier(cmd)
            self._bench_tick(cmd, "c_append_departed_end")
        ghost_self = 1 if self.ghost_layers() >= 2 else 0

        # Correction on boundary band only (interior was covered by Phase B).
        # Includes any new migrants installed above (they land in boundary
        # columns by construction).
        # V3.4: with band-voxel dispatch the three boundary pipelines launch
        # only (band voxel, slot) threads; a slab without peers has no band
        # and skips the dispatch (the full-range path early-returned anyway).
        # Band widths: self.band_widths (V7_BAND_WIDTHS, default 2/2/3).
        correction_band, density_band, force_band = self.band_widths
        compact = _BAND_COMPACT and self._band_compact_voxel_count() > 0
        if self._fused_correction_density_active():
            # E39 B1: the correction and density band passes in one traversal (one band width and inner-ghost self
            # layer; a slab with the compact band dispatch never fuses: its widths are 2,3,4). No barrier before
            # the tick: the density copy below opens with one.
            if _BAND_VOXEL_DISPATCH:
                per_band_fused = self._per_band_dispatch_count(
                    correction_band, self._ghost_self_layer(2, 1, "correction"))
                if per_band_fused > 0:
                    def record_band() -> None:
                        self._bind_pipeline_and_sets(cmd, "correction_density_boundary_band")
                        vkCmdDispatch(cmd, per_band_fused, 1, 1)
                    # E39 B3: + the force_deep segment of V7_BAND_OVERLAP's plan, no barrier between them
                    self._record_band_overlap_pair(cmd, "correction_density", record_band)
            else:
                self._bind_pipeline_and_sets(cmd, "correction_density_boundary")
                vkCmdDispatch(cmd, per_p, 1, 1)
            self._bench_tick(cmd, "c_correction_density_boundary_end")
        else:
            if compact:
                # V7_BAND_COMPACT_DISPATCH: scan (1 workgroup) + scatter build the
                # band pid list; the band kernels then run one thread per entry.
                self._bind_pipeline_and_sets(cmd, "band_compact_scan")
                vkCmdDispatch(cmd, 1, 1, 1)
                self._record_compute_barrier(cmd)
                self._bind_pipeline_and_sets(cmd, "band_compact_scatter")
                scatter_threads = (self._band_compact_voxel_count()
                                   * self.case.capacities.max_particles_per_voxel)
                # band_compact.comp has a fixed local size of 1024 (the scan's)
                vkCmdDispatch(cmd, (scatter_threads + 1023) // 1024, 1, 1)
                self._record_indirect_barrier(cmd)
                self._bench_tick(cmd, "c_band_compact_end")
                self._bind_pipeline_and_sets(cmd, "correction_boundary_compact")
                vkCmdDispatchIndirect(cmd, self.buffers["band_compact_meta"].handle, 0)
            elif _BAND_VOXEL_DISPATCH:
                per_band_correction = self._per_band_dispatch_count(
                    correction_band, self._ghost_self_layer(2, 1, "correction"))
                if per_band_correction > 0:
                    self._bind_pipeline_and_sets(cmd, "correction_boundary_band")
                    vkCmdDispatch(cmd, per_band_correction, 1, 1)
            else:
                self._bind_pipeline_and_sets(cmd, "correction_boundary")
                vkCmdDispatch(cmd, per_p, 1, 1)
            self._record_compute_barrier(cmd)
            self._bench_tick(cmd, "c_correction_boundary_end")

            # Path A+ P5: density_boundary covers only the density boundary band;
            # density_deep_interior in Phase B already wrote scratch[deep_interior
            # pids]. Together they cover the full own pid range. The scratch→primary
            # copy below transfers the union to primary in one shot, so force_all
            # below reads fresh ρ_{n+1} for every neighbor.
            if compact:
                self._bind_pipeline_and_sets(cmd, "density_boundary_compact")
                vkCmdDispatchIndirect(cmd, self.buffers["band_compact_meta"].handle, 16)
            elif _BAND_VOXEL_DISPATCH:
                per_band_density = self._per_band_dispatch_count(
                    density_band, self._ghost_self_layer(2, 1, "density"))
                if per_band_density > 0:
                    self._bind_pipeline_and_sets(cmd, "density_boundary_band")
                    vkCmdDispatch(cmd, per_band_density, 1, 1)
            else:
                self._bind_pipeline_and_sets(cmd, "density_boundary")
                vkCmdDispatch(cmd, per_p, 1, 1)
            self._bench_tick(cmd, "c_density_boundary_end")   # kernel vs copy split
        # E39 B3: a V7_BAND_OVERLAP plan may pair a force_deep segment with the copy pass
        self._record_density_scratch_to_primary_copy(cmd, overlap_pair=True)
        self._bench_tick(cmd, "c_density_end")
        if not _CASCADE_FORCE:
            # E37 wall_boundary adami without cascading force: force_all below is the first force of the frame
            # (with it, phase B ran the wall pass).
            self._record_wall_extrapolate(cmd, "primary", tick="c_wall_extrapolate_end")

        # V3.3: with cascading force, Phase B already covered the deep
        # interior; only the force boundary band (incl. this frame's
        # migrants) remains, reading primary (fresh for every own column
        # after the copy above; ghost slots stale by one step as before).
        if _CASCADE_FORCE and compact:
            self._bind_pipeline_and_sets(cmd, "force_boundary_compact")
            vkCmdDispatchIndirect(cmd, self.buffers["band_compact_meta"].handle, 32)
        elif _CASCADE_FORCE and _BAND_VOXEL_DISPATCH:
            per_band_force = self._per_band_dispatch_count(force_band)
            if per_band_force > 0:
                def record_band() -> None:
                    self._bind_pipeline_and_sets(cmd, "force_boundary_band")
                    vkCmdDispatch(cmd, per_band_force, 1, 1)
                # E39 B3: + the force_deep segment of V7_BAND_OVERLAP's plan, no barrier between them (the end of
                # phase C closes the pair)
                self._record_band_overlap_pair(cmd, "force", record_band)
        else:
            self._bind_pipeline_and_sets(
                cmd, "force_boundary" if _CASCADE_FORCE else "force_all")
            vkCmdDispatch(cmd, per_p, 1, 1)
        self._bench_tick(cmd, "c_force_end")
        # E39 B3: phase B + this recording dispatch force_deep's workgroups exactly once (raises otherwise)
        self._check_force_deep_recorded_once()

        vkEndCommandBuffer(cmd)
        if self.bench is not None:
            self.bench.end_phase_c_region()
        return cmd

    def _record_transfer_readback_cmd(self, direction: str):
        """Path A+ transfer queue cmd: device→sender_staging DMA + host
        coherence barrier for one direction. Submitted on ctx.transfer_queue
        in parallel with Phase B on the compute queue.

        SIMULTANEOUS_USE so the same cmd can be in flight across frames if
        we ever go to depth>1 (current depth=1 still benefits from skipping
        per-frame re-recording)."""
        cmd = self._allocate_transfer_oneshot_cmd()
        vkBeginCommandBuffer(cmd, VkCommandBufferBeginInfo(
            flags=VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT))
        # Note: no compute→transfer barrier needed here. The phase_a_done
        # timeline wait already ensures ghost_send.comp's writes have
        # happened-before this submit; CONCURRENT buffer sharing (P2)
        # eliminates the queue family ownership transfer cost.
        self._bench_tick_transfer(cmd, f"t_rb_{direction}_start")
        self._record_readback_for_direction(cmd, direction)
        self._bench_tick_transfer(cmd, f"t_rb_{direction}_copy_end")
        # Compute→host barrier so worker's CPU read sees the just-written
        # sender_staging bytes. Issued on transfer queue (legal — barriers
        # can be issued on any queue including transfer).
        self._record_compute_to_host_barrier(cmd)
        self._bench_tick_transfer(cmd, f"t_rb_{direction}_end")
        vkEndCommandBuffer(cmd)
        return cmd

    def _record_transfer_upload_cmd(self, direction: str):
        """Path A+ transfer queue cmd: receiver_staging→device DMA for one
        direction. Submitted on ctx.transfer_queue gated on the worker's
        worker_done host-signal (memcpy done). Phase C waits for upload_done
        (signaled at end of the last direction's cmd) before
        install_migrations dispatches."""
        cmd = self._allocate_transfer_oneshot_cmd()
        vkBeginCommandBuffer(cmd, VkCommandBufferBeginInfo(
            flags=VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT))
        self._bench_tick_transfer(cmd, f"t_up_{direction}_start")
        self._record_upload_for_direction(cmd, direction)
        self._bench_tick_transfer(cmd, f"t_up_{direction}_end")
        vkEndCommandBuffer(cmd)
        return cmd

    def prepare_step_cmd_buffers(self) -> None:
        """Record phase A/B/C cmd buffers + Path A+ transfer queue cmd
        buffers once. Caller invokes after bootstrap. Re-call to re-record
        (e.g. when switching CORRECTION_MODE_ALL → split in Phase 6)."""
        device = self.ctx.device
        pool = self.ctx.command_pool
        for old in (self.phase_a_cmd, self.phase_b_cmd, self.phase_c_cmd,
                    self.phase_c_cmd_odd, self.phase_a_cmd_odd, self.phase_b_cmd_odd):
            if old is not None:
                vkFreeCommandBuffers(device, pool, 1, [old])
        self.phase_c_cmd_odd = None
        self.phase_a_cmd_odd = None
        self.phase_b_cmd_odd = None
        if self.bench is not None and os.environ.get("V7_BENCH_PARITY", "1") == "1":
            self.bench.enable_parity_regions()
        for direction, old in list(self.transfer_readback_cmds.items()):
            vkFreeCommandBuffers(
                device, self.ctx.transfer_command_pool, 1, [old])
        for direction, old in list(self.transfer_upload_cmds.items()):
            vkFreeCommandBuffers(
                device, self.ctx.transfer_command_pool, 1, [old])
        for old in (list(self.transfer_readback_cmds_odd.values())
                    + list(self.transfer_upload_cmds_odd.values())):
            vkFreeCommandBuffers(device, self.ctx.transfer_command_pool, 1, [old])
        self.transfer_readback_cmds = {}
        self.transfer_upload_cmds = {}
        self.transfer_readback_cmds_odd = {}
        self.transfer_upload_cmds_odd = {}
        self._fast_ready = False

        self._set_step_trace_parity(0)
        self.phase_a_cmd = self._record_phase_a_cmd()
        self.phase_b_cmd = self._record_phase_b_cmd()
        self.phase_c_cmd = self._record_phase_c_cmd(0)
        if self.step_trace_parity:
            # E29: identical work, odd-frame timestamp slots.
            self._set_step_trace_parity(1)
            self.phase_a_cmd_odd = self._record_phase_a_cmd()
            self.phase_b_cmd_odd = self._record_phase_b_cmd()
        if self.bench is not None and self.bench.parity_regions:
            # Identical work; only the timestamp slots differ (odd region).
            self.phase_c_cmd_odd = self._record_phase_c_cmd(1)

        # (Transfer-pool reset is recorded inside phase_a_cmd above —
        # compute queue — not in the transfer cmds; see _bench_tick_transfer.)
        for direction in self._transport_segments:
            self._set_step_trace_parity(0)
            self.transfer_readback_cmds[direction] = (
                self._record_transfer_readback_cmd(direction))
            self.transfer_upload_cmds[direction] = (
                self._record_transfer_upload_cmd(direction))
            if self.step_trace_parity:
                self._set_step_trace_parity(1)
                self.transfer_readback_cmds_odd[direction] = (
                    self._record_transfer_readback_cmd(direction))
                self.transfer_upload_cmds_odd[direction] = (
                    self._record_transfer_upload_cmd(direction))
        self._set_step_trace_parity(0)

        n_dir = len(self._transport_segments)
        print(f"[SimV7] step cmd buffers recorded "
              f"(phase A/B/C + {n_dir} readback + {n_dir} upload on transfer Q)")
        self._fast_submit_prepare()
        if _FAST_SUBMIT:
            print("[SimV7] V7_FAST_SUBMIT=1: cached cffi submit batches "
                  f"(compute A+B+C, {n_dir} readback, {n_dir} upload), raw waits/signals")

    def _record_step_single_cmd(self):
        """Single-GPU combined cmd: predict + update_voxel + correction_all
        + density + force. No ghost flow, no timeline semaphores — caller
        submits with a plain fence wait per frame.

        Skips ghost_send and install_migrations because they are cross-GPU
        exclusive: no peer means nothing to replicate and nothing to install
        (predict's drift-to-ghost branch never fires when both ghost x
        thicknesses are 0; particles that leave the domain hit !in_own_grid
        and are killed locally). Uses correction_all instead of the
        interior/boundary split because there is no sync window to hide and
        in_boundary_band is empty when both ghost pools = 0 (split would
        run interior over all particles + boundary as a per-particle
        early-return no-op, equivalent but with one extra dispatch)."""
        cmd = self._allocate_oneshot_cmd()
        vkBeginCommandBuffer(cmd, VkCommandBufferBeginInfo(
            flags=VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT))
        # Bench: this is the only step cmd on this path, so it owns the
        # per-frame step-slot reset. No-op when bench is unattached.
        self._bench_reset_step(cmd, "step_start")
        self._record_compute_barrier(cmd)

        per_p = self._per_own_particle_dispatch_count()
        per_v = self._per_extended_voxel_dispatch_count()

        self._bind_pipeline_and_sets(cmd, "predict")
        vkCmdDispatch(cmd, per_p, 1, 1)
        self._bench_tick(cmd, "predict_end")
        self._record_compute_barrier(cmd)

        self._bind_pipeline_and_sets(cmd, "update_voxel")
        vkCmdDispatch(cmd, per_v, 1, 1)
        self._bench_tick(cmd, "voxel_end")
        self._record_compute_barrier(cmd)
        # E39 B4 (V7_DEEP_WALL_SKIP): the deep-wall marker of the lists just rebuilt, for correction / density.
        self._record_deep_wall_marker(cmd, tick="deep_wall_marker_end")

        # V7_FAKE_BAND_TEST (diagnostic): a non-empty band inside a single-GPU
        # domain forces the split path, with the boundary pipelines dispatched
        # over the band voxels (V3.4 band dispatch) exactly as in Phase C of a
        # multi-GPU run, and one tick per kernel so the band kernels' cost can
        # be read per particle.
        use_split = self.step_single_use_split or self._fake_band_column() > 0
        correction_band, density_band, force_band = self.band_widths

        def dispatch_boundary(name: str, band_range: int) -> None:
            if _BAND_VOXEL_DISPATCH and self._band_thread_count(band_range) > 0:
                self._bind_pipeline_and_sets(cmd, name + "_band")
                vkCmdDispatch(cmd, self._per_band_dispatch_count(band_range), 1, 1)
            else:
                self._bind_pipeline_and_sets(cmd, name)
                vkCmdDispatch(cmd, per_p, 1, 1)

        if self._fused_correction_density_active():
            # E39 B1: correction + density in one traversal (the split path: interior, then the boundary pass);
            # the density copy below opens with a barrier.
            if use_split:
                self._bind_pipeline_and_sets(cmd, "correction_density_interior")
                vkCmdDispatch(cmd, per_p, 1, 1)
                self._bench_tick(cmd, "correction_density_interior_end")
                self._record_compute_barrier(cmd)
                dispatch_boundary("correction_density_boundary", correction_band)
            else:
                self._bind_pipeline_and_sets(cmd, "correction_density_all")
                vkCmdDispatch(cmd, per_p, 1, 1)
            self._bench_tick(cmd, "correction_density_end")
        else:
            if use_split:
                # P3.C validation: substitute _all variants with their split
                # equivalents. In single-GPU mode in_boundary_band always returns
                # false (LEADING/TRAILING_GHOST_VOXEL_COUNT = 0), so:
                #   _interior / _deep_interior cover ALL particles (identical to _all)
                #   _boundary covers ZERO particles (every thread early-returns)
                # Output must therefore be bit-identical to the non-split path —
                # any divergence in alive count or per-buffer state proves a
                # shader-side bug. (With a fake band the boundary pipelines cover
                # the band and the interior ones the rest; still equivalent.)
                self._bind_pipeline_and_sets(cmd, "correction_interior")
                vkCmdDispatch(cmd, per_p, 1, 1)
                self._bench_tick(cmd, "correction_interior_end")
                self._record_compute_barrier(cmd)
                dispatch_boundary("correction_boundary", correction_band)
            else:
                self._bind_pipeline_and_sets(cmd, "correction_all")
                vkCmdDispatch(cmd, per_p, 1, 1)
            self._bench_tick(cmd, "correction_end")
            self._record_compute_barrier(cmd)

            if use_split:
                self._bind_pipeline_and_sets(cmd, "density_deep_interior")
                vkCmdDispatch(cmd, per_p, 1, 1)
                self._bench_tick(cmd, "density_deep_interior_end")
                self._record_compute_barrier(cmd)
                dispatch_boundary("density_boundary", density_band)
                self._bench_tick(cmd, "density_boundary_end")
                self._record_compute_barrier(cmd)
            else:
                self._bind_pipeline_and_sets(cmd, "density_all")
                vkCmdDispatch(cmd, per_p, 1, 1)
        self._record_density_scratch_to_primary_copy(cmd)
        self._bench_tick(cmd, "density_end")
        self._record_compute_barrier(cmd)
        self._record_wall_extrapolate(cmd, "primary", tick="wall_extrapolate_end")

        if use_split:
            self._bind_pipeline_and_sets(cmd, "force_deep_interior")
            vkCmdDispatch(cmd, per_p, 1, 1)
            self._bench_tick(cmd, "force_deep_interior_end")
            self._record_compute_barrier(cmd)
            dispatch_boundary("force_boundary", force_band)
        else:
            self._bind_pipeline_and_sets(cmd, "force_all")
            vkCmdDispatch(cmd, per_p, 1, 1)
        self._bench_tick(cmd, "force_end")

        vkEndCommandBuffer(cmd)
        return cmd

    def prepare_step_single_cmd_buffer(self) -> None:
        """Record the single-GPU combined step cmd buffer once. Caller invokes
        after bootstrap. Mutually exclusive with prepare_step_cmd_buffers()
        — the dual-GPU 3-submit path requires this sim to have at least one
        peer direction, which conflicts with the single-GPU assumption."""
        if self._transport_segments:
            raise RuntimeError(
                "prepare_step_single_cmd_buffer() requires a no-peer sim; "
                "this sim has transport segments for "
                f"{sorted(self._transport_segments)}. Use "
                "prepare_step_cmd_buffers() (dual-GPU 3-submit path) instead.")
        device = self.ctx.device
        pool = self.ctx.command_pool
        if self.step_single_cmd is not None:
            vkFreeCommandBuffers(device, pool, 1, [self.step_single_cmd])
        self.step_single_cmd = self._record_step_single_cmd()
        print(f"[SimV7] step cmd buffer recorded (single-GPU combined)")

    def submit_step_single_and_wait(self) -> None:
        """Submit the single-GPU step cmd buffer and block on its fence.
        One submit + one wait per frame — no timeline semaphores, no peer
        sync. Used by _run_v7_single_bench.py and any future single-GPU
        runner."""
        if self.step_single_cmd is None:
            raise RuntimeError(
                "step_single_cmd not recorded; "
                "call prepare_step_single_cmd_buffer()")
        self.ctx.submit_and_wait(self.step_single_cmd)

    # ----- sync2 timeline submit / wait / host signal -----------------------

    def submit_with_timeline(
        self,
        cmd,
        *,
        waits: list,
        signals: list,
        wait_stage: int = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        queue: Optional[object] = None,
        signal_stage: int = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
    ) -> None:
        """Submit ``cmd`` with timeline-semaphore wait/signal ops. ``waits`` /
        ``signals`` are lists of (semaphore, value) pairs — produced by
        self.sync's per-site getters. Defaults to compute_queue +
        compute_shader stage mask; pass ``queue=ctx.transfer_queue`` +
        ``wait_stage=VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT`` for Path A+
        readback/upload submits on the transfer queue."""
        if queue is None:
            queue = self.ctx.compute_queue
        wait_infos = [
            VkSemaphoreSubmitInfo(
                sType=VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                semaphore=semaphore,
                value=value,
                stageMask=wait_stage,
            ) for semaphore, value in waits
        ]
        signal_infos = [
            VkSemaphoreSubmitInfo(
                sType=VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                semaphore=semaphore,
                value=value,
                stageMask=signal_stage,
            ) for semaphore, value in signals
        ]
        cmd_info = VkCommandBufferSubmitInfo(
            sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
            commandBuffer=cmd,
        )
        submit_info = VkSubmitInfo2(
            sType=VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
            waitSemaphoreInfoCount=len(wait_infos),
            pWaitSemaphoreInfos=wait_infos if wait_infos else None,
            commandBufferInfoCount=1,
            pCommandBufferInfos=[cmd_info],
            signalSemaphoreInfoCount=len(signal_infos),
            pSignalSemaphoreInfos=signal_infos if signal_infos else None,
        )
        with self._driver_submit_lock:
            vkQueueSubmit2(queue, 1, [submit_info], VK_NULL_HANDLE)

    def wait_semaphore(self, semaphore, value: int,
                       timeout_ns: int = 0xFFFFFFFFFFFFFFFF) -> bool:
        """vkWaitSemaphores with optional timeout. Default INFINITE.
        Returns True if value was reached, False if timed out.

        python-vulkan raises an exception on VK_TIMEOUT rather than returning
        it as a value, so we catch the timeout case explicitly. Other
        VkResult errors (DEVICE_LOST etc.) propagate.
        """
        if _FAST_SUBMIT:
            # raw cffi (thread-safe: fresh small structs per call, ~3 us vs
            # ~40 us through the wrapper; workers call this too).
            semaphores = _ffi.new("VkSemaphore[1]", [semaphore])
            values = _ffi.new("uint64_t[1]", [value])
            info = _ffi.new("VkSemaphoreWaitInfo*")
            info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO
            info.semaphoreCount = 1
            info.pSemaphores = semaphores
            info.pValues = values
            result = _lib.vkWaitSemaphores(self.ctx.device, info, timeout_ns)
            if result == 0:
                return True
            if result == 2:   # VK_TIMEOUT
                return False
            raise RuntimeError(f"vkWaitSemaphores (fast path) failed: VkResult {result}")
        info = VkSemaphoreWaitInfo(
            sType=VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
            semaphoreCount=1,
            pSemaphores=[semaphore],
            pValues=[value],
        )
        try:
            vkWaitSemaphores(self.ctx.device, info, timeout_ns)
            return True
        except Exception as e:
            # python-vulkan maps VK_TIMEOUT (=2) to VkTimeout class
            if type(e).__name__ == "VkTimeout":
                return False
            raise

    def wait_timeline(self, value: int, timeout_ns: int = 0xFFFFFFFFFFFFFFFF) -> bool:
        """Wait on the PRIMARY timeline (frame_done carrier). Prefer the
        (semaphore, value) ops from self.sync for anything scheme-dependent."""
        return self.wait_semaphore(self.sync.primary_semaphore(), value,
                                   timeout_ns=timeout_ns)

    def host_signal_semaphore(self, semaphore, value: int) -> None:
        """vkSignalSemaphore for host-side timeline advance. Ghost workers
        call this on the *destination* sim with the (semaphore, value) from
        dest.sync.worker_signal_op()."""
        if _FAST_SUBMIT:
            info = _ffi.new("VkSemaphoreSignalInfo*")
            info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO
            info.semaphore = semaphore
            info.value = value
            result = _lib.vkSignalSemaphore(self.ctx.device, info)
            if result != 0:
                raise RuntimeError(f"vkSignalSemaphore (fast path) failed: VkResult {result}")
            return
        info = VkSemaphoreSignalInfo(
            sType=VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO,
            semaphore=semaphore,
            value=value,
        )
        with self._driver_submit_lock:
            vkSignalSemaphore(self.ctx.device, info)

    def host_signal_timeline(self, value: int) -> None:
        """Legacy: host-signal the PRIMARY timeline (single-GPU smoke tests
        substitute for the absent worker this way, aggregated scheme only)."""
        self.host_signal_semaphore(self.sync.primary_semaphore(), value)

    # ----- High-level frame API (replaces earlier stubs) --------------------

    def submit_phase_a(self, frame_n: int) -> None:
        """Compute Q: predict + update_voxel + ghost_send. Signals
        phase_a_done; waits for the previous frame's frame_done only with
        V7_PHASE_A_NO_WAIT=0 (by default it follows C(n-1) by queue order)."""
        if self.phase_a_cmd is None:
            raise RuntimeError("phase_a_cmd not recorded; call prepare_step_cmd_buffers()")
        self.submit_with_timeline(
            (self.phase_a_cmd_odd if (self.phase_a_cmd_odd is not None and frame_n % 2 == 1)
             else self.phase_a_cmd),
            waits=[] if _PHASE_A_NO_WAIT else self.sync.phase_a_waits(frame_n),
            signals=self.sync.phase_a_signals(frame_n),
        )

    def submit_phase_b(self, frame_n: int) -> None:
        """Compute Q: correction_interior (+ density_deep_interior in P5).
        Queue-ordered after phase A on compute queue; no semaphore wait/signal
        in ANY sync scheme — this is the transfer-hiding window.
        Runs in parallel with transfer Q's readback + worker memcpy + upload."""
        if self.phase_b_cmd is None:
            raise RuntimeError("phase_b_cmd not recorded; call prepare_step_cmd_buffers()")
        self.submit_with_timeline(
            (self.phase_b_cmd_odd if (self.phase_b_cmd_odd is not None and frame_n % 2 == 1)
             else self.phase_b_cmd), waits=[], signals=[])

    def submit_phase_c(self, frame_n: int) -> None:
        """Compute Q: install + correction_boundary + density_boundary + copy
        + force. Waits upload_done (from transfer Q; scheme decides the
        semaphore/value — no-peer sims wait phase_a_done instead, a valid
        monotonically advancing stand-in). Signals frame_done."""
        if self.phase_c_cmd is None:
            raise RuntimeError("phase_c_cmd not recorded; call prepare_step_cmd_buffers()")
        phase_c_cmd = (self.phase_c_cmd_odd
                       if (self.phase_c_cmd_odd is not None and frame_n % 2 == 1)
                       else self.phase_c_cmd)
        self.submit_with_timeline(
            phase_c_cmd,
            waits=self.sync.phase_c_waits(frame_n),
            signals=self.sync.phase_c_signals(frame_n),
        )

    def submit_transfer_readback(self, frame_n: int) -> None:
        """Transfer Q: device→sender_staging DMA for all active directions.
        Each direction's cmd waits phase_a_done. Signal placement is
        scheme-dependent: aggregated — only the LAST direction signals the
        shared readback_done (queue FIFO makes it govern all directions);
        per-direction — every direction signals its own transport timeline.
        No-op when this sim has no peer (single-GPU-style slab)."""
        if not self.transfer_readback_cmds:
            # No peers → no readback. Worker won't wait on this either.
            return
        directions = list(self.transfer_readback_cmds)
        for index, direction in enumerate(directions):
            is_last = index == len(directions) - 1
            self.submit_with_timeline(
                (self.transfer_readback_cmds_odd[direction]
                 if (self.transfer_readback_cmds_odd and frame_n % 2 == 1)
                 else self.transfer_readback_cmds[direction]),
                waits=self.sync.readback_waits(direction, frame_n),
                signals=self.sync.readback_signals(direction, frame_n, is_last),
                wait_stage=VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
                signal_stage=VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
                queue=self.ctx.transfer_queue,
            )

    def submit_transfer_upload(self, frame_n: int) -> None:
        """Transfer Q: receiver_staging→device DMA for all active directions.
        Each direction's cmd waits its worker memcpy (host-signaled by the
        worker that wrote our receiver_staging; scheme decides whether that
        is the shared worker_done slot or the direction's own transport
        timeline). Last cmd signals upload_done; Phase C waits on this.
        No-op when no peer."""
        if not self.transfer_upload_cmds:
            return
        directions = list(self.transfer_upload_cmds)
        for index, direction in enumerate(directions):
            is_last = index == len(directions) - 1
            self.submit_with_timeline(
                (self.transfer_upload_cmds_odd[direction]
                 if (self.transfer_upload_cmds_odd and frame_n % 2 == 1)
                 else self.transfer_upload_cmds[direction]),
                waits=self.sync.upload_waits(direction, frame_n),
                signals=self.sync.upload_signals(direction, frame_n, is_last),
                wait_stage=VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
                signal_stage=VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
                queue=(getattr(self.ctx, "transfer_queue_upload", None)
                       or self.ctx.transfer_queue),
            )

    def wait_frame_done(self, frame_n: int) -> None:
        semaphore, value = self.sync.frame_done_op(frame_n)
        self.wait_semaphore(semaphore, value)

    # ========================================================================
    # Section 11: Defrag pipeline + cmd (Phase 3c)
    # 5-set pipeline layout: sets 0..3 reused + set 4 for scratch destination
    # SoA. Skips binding 2 (density_pressure_scratch) — transient, no need
    # to migrate.
    # ========================================================================

    def _build_defrag_set4_layout(self):
        bindings = [
            VkDescriptorSetLayoutBinding(
                binding=b,
                descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1,
                stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,
            )
            for b in DEFRAG_SET0_BINDINGS
        ]
        ci = VkDescriptorSetLayoutCreateInfo(
            bindingCount=len(bindings), pBindings=bindings)
        return vkCreateDescriptorSetLayout(self.ctx.device, ci, None)

    def _allocate_defrag_set4(self):
        pool_size = VkDescriptorPoolSize(
            type=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            descriptorCount=len(DEFRAG_SET0_BINDINGS),
        )
        pool_ci = VkDescriptorPoolCreateInfo(
            maxSets=1, poolSizeCount=1, pPoolSizes=[pool_size])
        pool = vkCreateDescriptorPool(self.ctx.device, pool_ci, None)
        alloc = VkDescriptorSetAllocateInfo(
            descriptorPool=pool,
            descriptorSetCount=1,
            pSetLayouts=[self.defrag_set4_layout],
        )
        descriptor_set = vkAllocateDescriptorSets(self.ctx.device, alloc)[0]
        return pool, descriptor_set

    def _wire_defrag_set4(self) -> None:
        # Names match set 0 SoA — same buffer keys for scratch and primary.
        # Skip binding 2 (density_pressure_scratch); it's not in
        # DEFRAG_SET0_BINDINGS so no scratch twin exists.
        writes = []
        for spec in self._buffer_specs:
            if spec.set_index != 0:
                continue
            if spec.binding not in DEFRAG_SET0_BINDINGS:
                continue
            scratch_buf = self.scratch_buffers[spec.name]
            buf_info = VkDescriptorBufferInfo(
                buffer=scratch_buf.handle, offset=0, range=scratch_buf.size)
            writes.append(VkWriteDescriptorSet(
                dstSet=self.defrag_set4,
                dstBinding=spec.binding,
                dstArrayElement=0,
                descriptorCount=1,
                descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                pBufferInfo=[buf_info],
            ))
        vkUpdateDescriptorSets(self.ctx.device, len(writes), writes, 0, None)

    def _build_defrag_pipeline_layout(self):
        # 5 sets: existing 4 (0..3) + defrag's set 4
        layouts = list(self.descriptor_layouts) + [self.defrag_set4_layout]
        ci = VkPipelineLayoutCreateInfo(
            setLayoutCount=5, pSetLayouts=layouts, pushConstantRangeCount=0)
        return vkCreatePipelineLayout(self.ctx.device, ci, None)

    def _build_defrag_pipeline(self):
        spec_info = self._make_spec_info(self._global_entries())
        stage = VkPipelineShaderStageCreateInfo(
            stage=VK_SHADER_STAGE_COMPUTE_BIT,
            module=self.shader_modules["defrag"],
            pName="main",
            pSpecializationInfo=spec_info,
        )
        ci = VkComputePipelineCreateInfo(
            stage=stage, layout=self.defrag_pipeline_layout)
        return vkCreateComputePipelines(
            self.ctx.device, VK_NULL_HANDLE, 1, [ci], None)[0]

    def _record_defrag_cmd(self):
        """V5 defrag cmd buffer (mirrors V1 _record_defrag_cmd):
            1. fill defrag_scratch_counter = 0
            2. transfer→compute barrier
            3. defrag dispatch (per extended voxel)
            4. compute→transfer barrier
            5. copy each scratch SoA → primary (9 copies, set 0 except scratch)
            6. copy defrag_scratch_counter → global_status.alive_particle_count
            7. fill migration_install_count = 0
            8. transfer→compute barrier (next step's predict reads set 0)
        """
        cmd = self._allocate_oneshot_cmd()
        vkBeginCommandBuffer(cmd, VkCommandBufferBeginInfo(
            flags=VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT))
        # Bench: defrag owns its own slot-range reset (the step reset in
        # phase_a_cmd doesn't cover defrag slots — defrag may not run that
        # frame, and we don't want stale defrag readings).
        self._bench_reset_defrag(cmd, "defrag_start")
        self._record_compute_barrier(cmd)

        # 1. Reset scratch counter
        scratch_counter = self.buffers["defrag_scratch_counter"]
        vkCmdFillBuffer(cmd, scratch_counter.handle, 0, scratch_counter.size, 0)

        # 1b. Zero each defrag scratch SoA before dispatch.
        # defrag.comp only writes scratch[own_first_pid .. own_first_pid + alive - 1].
        # Without this fill, the dead-tail (slots beyond alive) keeps stale bytes
        # from PREVIOUS defrag runs — when vkCmdCopyBuffer scratch→primary
        # overwrites primary wholesale, those stale bytes resurrect "ghost
        # particles" with mass>0 and valid voxel_id but no inside_particle_index
        # reference (orphans). Investigation 2026-05-19 traced the alive-count
        # drift bug to exactly this; see logs/test_run/snapshots analysis.
        # Cost: ~167 MB total fill per defrag → ~3 ms at VRAM bandwidth, runs
        # every defrag_cadence (1000) frames → <0.05% wall-clock impact.
        for spec in self._buffer_specs:
            if spec.set_index != 0:
                continue
            if spec.binding not in DEFRAG_SET0_BINDINGS:
                continue
            scratch_buf = self.scratch_buffers[spec.name]
            vkCmdFillBuffer(cmd, scratch_buf.handle, 0, scratch_buf.size, 0)

        # 2. transfer→compute (defrag dispatch will atomicAdd the counter)
        self._record_transfer_to_compute_barrier(cmd)

        # 3. defrag dispatch — uses defrag_pipeline_layout (5 sets)
        vkCmdBindPipeline(
            cmd, VK_PIPELINE_BIND_POINT_COMPUTE, self.pipelines["defrag"])
        all_sets = list(self.descriptor_sets) + [self.defrag_set4]
        vkCmdBindDescriptorSets(
            cmd, VK_PIPELINE_BIND_POINT_COMPUTE, self.defrag_pipeline_layout,
            0, 5, all_sets, 0, None)
        vkCmdDispatch(cmd, self._per_extended_voxel_dispatch_count(), 1, 1)

        # 4. compute→transfer (copy scratch→primary)
        self._record_compute_to_transfer_barrier(cmd)

        # 5. Copy each scratch SoA back to primary
        for spec in self._buffer_specs:
            if spec.set_index != 0:
                continue
            if spec.binding not in DEFRAG_SET0_BINDINGS:
                continue
            src = self.scratch_buffers[spec.name].handle
            dst = self.buffers[spec.name].handle
            vkCmdCopyBuffer(cmd, src, dst, 1, [
                VkBufferCopy(srcOffset=0, dstOffset=0, size=spec.size)
            ])

        # 6. Refresh alive_particle_count from defrag_scratch_counter.
        # GlobalStatusBuffer layout: alive_particle_count at offset 0, 4 B.
        vkCmdCopyBuffer(
            cmd, scratch_counter.handle,
            self.buffers["global_status"].handle, 1, [
                VkBufferCopy(srcOffset=0, dstOffset=0, size=4)
            ])

        # 7. Reset migration_install_count (offset 12 × 4 = 48 in global_status,
        # cf. common.glsl GlobalStatusBuffer field order).
        vkCmdFillBuffer(cmd, self.buffers["global_status"].handle, 48, 4, 0)

        # 8. transfer→compute (next step's predict reads set 0)
        self._record_transfer_to_compute_barrier(cmd)
        self._bench_tick(cmd, "defrag_end")

        vkEndCommandBuffer(cmd)
        return cmd
