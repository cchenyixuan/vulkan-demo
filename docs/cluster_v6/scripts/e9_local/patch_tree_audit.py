"""E9: the V7_DEFRAG_AUDIT=1 diagnostic in SphSimulatorV7.submit_defrag_and_wait (pid order check after every defrag,
utils/defrag_prefix.audit_pid_order). argv[1] = worktree root."""
import pathlib
import sys

ROOT = pathlib.Path(sys.argv[1])
path = ROOT / "experiment/v7/utils/simulator_v7.py"
text = path.read_text(encoding="utf-8")
old_import = "from experiment.v7.utils.defrag_prefix import upload_defrag_prefix\n"
new_import = "from experiment.v7.utils.defrag_prefix import audit_pid_order, upload_defrag_prefix\n"
assert text.count(old_import) == 1
text = text.replace(old_import, new_import)
old = ("        if self.case.numerics.use_prefix_sum_defrag:\n"
       "            self._defrag_prefix_records.append(upload_defrag_prefix(self, verbose=len(self._defrag_prefix_records) < 2))\n"
       "        self.ctx.submit_and_wait(self.defrag_cmd)\n")
new = ("        if self.case.numerics.use_prefix_sum_defrag:\n"
       "            self._defrag_prefix_records.append(upload_defrag_prefix(self, verbose=len(self._defrag_prefix_records) < 2))\n"
       "        self.ctx.submit_and_wait(self.defrag_cmd)\n"
       "        if os.environ.get(\"V7_DEFRAG_AUDIT\") == \"1\":          # diagnostic: the pid order after this defrag\n"
       "            audit_pid_order(self, own_total=self._defrag_prefix_records[-1][\"total\"] if self._defrag_prefix_records else None)\n")
assert text.count(old) == 1
text = text.replace(old, new)
if "\nimport os\n" not in text:
    assert text.count("\nimport struct\n") == 1
    text = text.replace("\nimport struct\n", "\nimport os\nimport struct\n")
    print("import os added")
path.write_text(text, encoding="utf-8", newline="\n")
print(f"patched {path}")
