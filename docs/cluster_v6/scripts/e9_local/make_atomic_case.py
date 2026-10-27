"""Write an 'atomic defrag order' variant of a case.yaml outside the tree (scratchpad), with the geometry / material
paths made absolute so the loader finds the case's particle files: the A/B control of E9 item 6.
Usage: python make_atomic_case.py <case.yaml> <out.yaml>"""
import pathlib
import re
import sys

source = pathlib.Path(sys.argv[1]).resolve()
target = pathlib.Path(sys.argv[2])
case_dir = source.parent
text = source.read_text(encoding="utf-8")
assert "defrag_order" not in text
count = text.count("use_prefix_sum_defrag: false")
assert count == 1, count
text = text.replace("use_prefix_sum_defrag: false", "defrag_order: atomic")
text = re.sub(r"(material_library:\s*)(\S+)", lambda m: m.group(1) + (case_dir / m.group(2)).resolve().as_posix(), text)
text = re.sub(r"(frame:\s*)(\S+\.obj)", lambda m: m.group(1) + (case_dir / m.group(2)).as_posix(), text)
text = re.sub(r"(file:\s*)(\S+\.obj)", lambda m: m.group(1) + (case_dir / m.group(2)).as_posix(), text)
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(text, encoding="utf-8", newline="\n")
print(f"{target}: defrag_order atomic, paths absolute to {case_dir}")
