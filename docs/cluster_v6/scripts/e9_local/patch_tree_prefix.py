"""E9 code change (user 2026-10-27, item 1): the prefix defrag order becomes the production path of experiment/v7.
Exact-string edits (each asserted unique) in the worktree given as argv[1] (default: the e8 worktree); the same edits
apply to the rc1 copies in the v7-single worktree (argv[1] = that worktree, --single adds simulator_single.py).
Kernels / SPIR-V untouched: only case_v7.py, case_loader_v7.py, simulator_v7.py (+ the new utils/defrag_prefix.py),
the harness print, the three case generators (and, --single, experiment/v7_single/utils/simulator_single.py)."""
import pathlib
import sys

ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "C:/Users/cchen/PycharmProjects/vulkan-demo-e8")
SINGLE = "--single" in sys.argv


def edit(relative, replacements):
    path = ROOT / relative
    text = path.read_text(encoding="utf-8")
    for old, new in replacements:
        count = text.count(old)
        assert count == 1, f"{relative}: {count} occurrences of {old[:70]!r}"
        text = text.replace(old, new)
    path.write_text(text, encoding="utf-8", newline="\n")
    print(f"patched {relative} ({len(replacements)} edits)")


# ---------------------------------------------------------------- case_v7.py: the default
edit("experiment/v7/utils/case_v7.py", [(
    "    use_pst: bool = True\n    use_prefix_sum_defrag: bool = False\n    defrag_cadence: int = 1000\n",
    "    use_pst: bool = True\n"
    "    # Defrag destination order (spec const 46 USE_PREFIX_SUM_DEFRAG; case.yaml numerics.defrag_order, E9 2026-10-27).\n"
    "    # True = \"prefix\": the host scans inside_particle_count over the own voxels before every defrag\n"
    "    # (utils/defrag_prefix.py) and defrag.comp's prefix branch packs voxel v's particles at the exclusive prefix sum,\n"
    "    # i.e. the pid order is the voxel-id order. False = \"atomic\": base = atomicAdd(defrag_scratch_counter, count),\n"
    "    # voxel blocks in the workgroups' arrival order (the dispatch's first wave, ~16M particles, lands scrambled:\n"
    "    # +12 % per particle on an 8M slab, E8 report 7.18). The SPIR-V is the same either way.\n"
    "    use_prefix_sum_defrag: bool = True\n    defrag_cadence: int = 1000\n")])

# ---------------------------------------------------------------- case_loader_v7.py: numerics.defrag_order
edit("experiment/v7/utils/case_loader_v7.py", [
    ("def load_case_v7(case_yaml_path: str | pathlib.Path) -> CaseV7:\n",
     "DEFRAG_ORDER_CHOICES = (\"prefix\", \"atomic\")     # case.yaml numerics.defrag_order (E9, 2026-10-27)\n"
     "\n"
     "\n"
     "def _defrag_order_is_prefix(numerics_mapping: dict, case_path: pathlib.Path) -> bool:\n"
     "    \"\"\"numerics.defrag_order: \"prefix\" (default since E9: the host scan of utils/defrag_prefix.py + defrag.comp's\n"
     "    prefix branch, pid order = voxel-id order) or \"atomic\" (the pre-E9 atomicAdd arrival order, kept for A/B runs).\n"
     "    The legacy key use_prefix_sum_defrag is ignored: the generators wrote 'false' into every case before 2026-10-27\n"
     "    without anybody choosing it, and 'true' never had a producer of voxel_base_offset (the switch alone leaves the\n"
     "    buffer unwritten and the alive count 0).\"\"\"\n"
     "    order = str(numerics_mapping.get(\"defrag_order\", \"prefix\"))\n"
     "    if order not in DEFRAG_ORDER_CHOICES:\n"
     "        raise ValueError(f\"numerics.defrag_order: {order!r}, expected one of {', '.join(DEFRAG_ORDER_CHOICES)}\")\n"
     "    if \"use_prefix_sum_defrag\" in numerics_mapping:\n"
     "        print(f\"[case_loader_v7] {case_path.parent.name}: numerics.use_prefix_sum_defrag is a legacy key and is ignored \"\n"
     "              f\"(defrag_order = {order}; set numerics.defrag_order to choose)\", flush=True)\n"
     "    return order == \"prefix\"\n"
     "\n"
     "\n"
     "def load_case_v7(case_yaml_path: str | pathlib.Path) -> CaseV7:\n"),
    ("        use_prefix_sum_defrag=bool(nm.get(\"use_prefix_sum_defrag\", False)),\n",
     "        use_prefix_sum_defrag=_defrag_order_is_prefix(nm, case_path),\n"),
])

# ---------------------------------------------------------------- simulator_v7.py: the host scan before every defrag
edit("experiment/v7/utils/simulator_v7.py", [
    ("from experiment.v7.utils.vulkan_context_v7 import VulkanContextV7\n",
     "from experiment.v7.utils.vulkan_context_v7 import VulkanContextV7\n"
     "from experiment.v7.utils.defrag_prefix import upload_defrag_prefix\n"),
    ("    def submit_defrag_and_wait(self) -> None:\n"
     "        if self.defrag_cmd is None:\n"
     "            self.defrag_cmd = self._record_defrag_cmd()\n"
     "        self.ctx.submit_and_wait(self.defrag_cmd)\n",
     "    def submit_defrag_and_wait(self) -> None:\n"
     "        \"\"\"One defrag, queue drained (bootstrap and every defrag_cadence frames). Prefix order (E9, the default):\n"
     "        the host scan of utils/defrag_prefix.py first - voxel_base_offset in voxel-id order + the own total in\n"
     "        defrag_scratch_counter - then the recorded command; atomic order: the command alone.\"\"\"\n"
     "        if self.defrag_cmd is None:\n"
     "            self.defrag_cmd = self._record_defrag_cmd()\n"
     "        if self.case.numerics.use_prefix_sum_defrag:\n"
     "            self._defrag_prefix_records.append(upload_defrag_prefix(self, verbose=len(self._defrag_prefix_records) < 2))\n"
     "        self.ctx.submit_and_wait(self.defrag_cmd)\n"),
    ("        \"\"\"V5 defrag cmd buffer (mirrors V1 _record_defrag_cmd):\n"
     "            1. fill defrag_scratch_counter = 0\n",
     "        \"\"\"V5 defrag cmd buffer (mirrors V1 _record_defrag_cmd):\n"
     "            1. fill defrag_scratch_counter = 0   (atomic order only; prefix order: the host wrote the own total\n"
     "               into it and voxel_base_offset before the submit, utils/defrag_prefix.py)\n"),
    ("        # 1. Reset scratch counter\n"
     "        scratch_counter = self.buffers[\"defrag_scratch_counter\"]\n"
     "        vkCmdFillBuffer(cmd, scratch_counter.handle, 0, scratch_counter.size, 0)\n",
     "        # 1. Reset scratch counter (atomic order: defrag.comp's atomicAdd base). Prefix order (E9 default): the\n"
     "        # kernel reads voxel_base_offset instead and never touches the counter; the host put the own-particle\n"
     "        # total into it right before this submit, so step 6 refreshes alive_particle_count exactly as before.\n"
     "        scratch_counter = self.buffers[\"defrag_scratch_counter\"]\n"
     "        if not self.case.numerics.use_prefix_sum_defrag:\n"
     "            vkCmdFillBuffer(cmd, scratch_counter.handle, 0, scratch_counter.size, 0)\n"),
    ("        self._destroyed = False\n"
     "        print(f\"[SimV7] init complete on {ctx.device_name} \"\n"
     "              f\"(own={case.capacities.own_pool_size}, \"\n"
     "              f\"ghost L={case.capacities.leading_ghost_pool_size} \"\n"
     "              f\"T={case.capacities.trailing_ghost_pool_size}, wall_boundary={self.wall_boundary})\")\n",
     "        self._destroyed = False\n"
     "        self._defrag_prefix_records: list[dict] = []        # E9: one record per host scan (utils/defrag_prefix.py)\n"
     "        print(f\"[SimV7] init complete on {ctx.device_name} \"\n"
     "              f\"(own={case.capacities.own_pool_size}, \"\n"
     "              f\"ghost L={case.capacities.leading_ghost_pool_size} \"\n"
     "              f\"T={case.capacities.trailing_ghost_pool_size}, wall_boundary={self.wall_boundary}, \"\n"
     "              f\"defrag_order={'prefix' if case.numerics.use_prefix_sum_defrag else 'atomic'})\")\n"),
])

if not SINGLE:
    # ------------------------------------------------------------ harness configuration line
    edit("docs/cluster_v6/scripts/run_chain_v6.py", [(
        "          f\"defrag_cadence={numerics.get('defrag_cadence')} pool_size={capacities.get('pool_size')} \"\n",
        "          f\"defrag_cadence={numerics.get('defrag_cadence')} defrag_order={numerics.get('defrag_order', 'prefix(default)')} \"\n"
        "          f\"pool_size={capacities.get('pool_size')} \"\n")])
    # ------------------------------------------------------------ the case generators
    for generator, count in (("utils/geometry/_demo_cavity_case.py", 1), ("utils/geometry/_demo_cavity_case_3d.py", 1),
                             ("utils/geometry/_demo_cavity_case_aligned.py", 2)):
        path = ROOT / generator
        text = path.read_text(encoding="utf-8")
        old = "  defrag_cadence: 1000\n  use_prefix_sum_defrag: false\n"
        new = "  defrag_cadence: 1000\n  defrag_order: prefix              # defrag pid order: prefix (voxel-id order, default since 2026-10-27) | atomic (arrival order)\n"
        assert text.count(old) == count, (generator, text.count(old))
        path.write_text(text.replace(old, new), encoding="utf-8", newline="\n")
        print(f"patched {generator} ({count} edits)")
else:
    # ------------------------------------------------------------ v7_single: the same scan around its defrag
    edit("experiment/v7_single/utils/simulator_single.py", [
        ("from experiment.v7.utils.case_v7 import CaseV7, KIND_FLUID\n",
         "from experiment.v7.utils.case_v7 import CaseV7, KIND_FLUID\n"
         "from experiment.v7.utils.defrag_prefix import upload_defrag_prefix\n"),
        ("        self.defrag_cmd: Any = None\n",
         "        self.defrag_cmd: Any = None\n"
         "        self._defrag_prefix_records: list = []             # E9: one record per host scan (utils/defrag_prefix.py)\n"),
        ("        self._record_compute_barrier(cmd)\n"
         "        counter = self.buffers[\"defrag_scratch_counter\"]\n"
         "        vkCmdFillBuffer(cmd, counter.handle, 0, counter.size, 0)\n",
         "        self._record_compute_barrier(cmd)\n"
         "        counter = self.buffers[\"defrag_scratch_counter\"]\n"
         "        if not self.case.numerics.use_prefix_sum_defrag:     # prefix order: the host wrote the total (E9)\n"
         "            vkCmdFillBuffer(cmd, counter.handle, 0, counter.size, 0)\n"),
        ("        command_buffers = [self.step_cmd] + ([self.defrag_cmd] if with_defrag else [])\n",
         "        if with_defrag and self.case.numerics.use_prefix_sum_defrag:\n"
         "            # E9 prefix order: the scan needs the voxel counts of the step before the defrag, so the undrained\n"
         "            # step + defrag submission becomes the step alone followed by the drained defrag\n"
         "            self.submit_step(with_defrag=False)\n"
         "            self.submit_defrag_and_wait()\n"
         "            return\n"
         "        command_buffers = [self.step_cmd] + ([self.defrag_cmd] if with_defrag else [])\n"),
        ("            self.record_defrag(self.defrag_cmd)\n"
         "            vkEndCommandBuffer(self.defrag_cmd)\n"
         "        self.drain()\n"
         "        self.ctx.submit_and_wait(self.defrag_cmd)\n",
         "            self.record_defrag(self.defrag_cmd)\n"
         "            vkEndCommandBuffer(self.defrag_cmd)\n"
         "        self.drain()\n"
         "        if self.case.numerics.use_prefix_sum_defrag:        # E9: the same host scan as SphSimulatorV7\n"
         "            self._defrag_prefix_records.append(upload_defrag_prefix(self, verbose=len(self._defrag_prefix_records) < 2))\n"
         "        self.ctx.submit_and_wait(self.defrag_cmd)\n"),
    ])
print("done")
