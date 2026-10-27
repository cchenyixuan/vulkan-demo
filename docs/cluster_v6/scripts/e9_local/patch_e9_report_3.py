"""E9 report chapter 3 (item 7: Section 3 / tab:kernelcost re-measurement + fig:baseline) replacing the placeholder, and the
data copies into docs/cluster_v6/data/e9/section3/."""
import json
import pathlib
import re
import shutil

E8 = pathlib.Path("C:/Users/cchen/PycharmProjects/vulkan-demo-e8")
V7S = pathlib.Path("C:/Users/cchen/PycharmProjects/vulkan-demo-v7-single")
SPE = pathlib.Path("C:/Users/cchen/AppData/Local/Temp/claude/C--Users-cchen-PycharmProjects-vulkan-demo/fd68689d-e4b4-4b3f-8636-fbe3d1d866de/scratchpad/e9")
DATA = E8 / "docs/cluster_v6/data/e9/section3"
DATA.mkdir(parents=True, exist_ok=True)
for src, dst in (("section3_compare.md", "section3_compare.md"), ("section3_compare.json", "section3_compare.json")):
    shutil.copy(SPE / src, DATA / dst)
for name in ("section3_rc4_table2.md", "section3_rc4_sizes.md", "section3_rc4_summary.json", "throughput_rc4_summary.md", "throughput_rc4_summary.json"):
    shutil.copy(V7S / "docs/seam_audit/v7_single" / name, DATA / name)
for name in ("single_gpu_throughput.png", "single_gpu_throughput.pdf", "single_gpu_throughput_e8_atomic.png"):
    shutil.copy(V7S / "manuscripts/fig" / name, DATA / name)

cmp = json.load(open(SPE / "section3_compare.json", encoding="utf-8"))
cmp_md = (SPE / "section3_compare.md").read_text(encoding="utf-8")
cmp_md = re.sub(r"^### ", "#### ", cmp_md, flags=re.M)
table2 = (V7S / "docs/seam_audit/v7_single/section3_rc4_table2.md").read_text(encoding="utf-8")
latex = table2[table2.index("```latex"):table2.index("```", table2.index("```latex") + 8) + 3]
thr = (V7S / "docs/seam_audit/v7_single/throughput_rc4_summary.md").read_text(encoding="utf-8")
thr_table = thr[thr.index("## Throughput (fps)"):thr.index("## Particle throughput")].replace("## Throughput (fps)", "**吞吐（fps，均值 ± 标准差，3 次）**").strip()
old_thr = json.load(open(V7S / "docs/seam_audit/v7_single/throughput_summary.json", encoding="utf-8"))
new_thr = json.load(open(V7S / "docs/seam_audit/v7_single/throughput_rc4_summary.json", encoding="utf-8"))


def sd2(summary, size):
    for key in ("sizes", "rows", "table"):
        node = summary.get(key)
        if isinstance(node, dict) and size in node:
            entry = node[size]
            for k in ("v7_single_d2", "S d2", "S_d2"):
                if k in entry:
                    return entry[k]
    return None


t2 = cmp["table2"]
sz = cmp["sizes"]
sec = f"""## 3. 第 7 项：Section 3 基线重测（工作站 600 W，v7_single，1 在飞，逐核时间戳；`tab:kernelcost` 的原协议）

**协议**：`experiment/v7_single/_run_section3_campaign.py run`（= 2026-10-10 做表 2 时的同一脚本：`_run_kernel_breakdown.py --solver v7_single --interleave 0`，预热 1000 步后第 1001–3000 步全部计时，每步一次提交 + fence，生产录制里的 tick 写 BOTTOM_OF_PIPE 时间戳，defrag 帧剔除；2-D 8M `lid_driven_cavity_2d_8m`（8,128,201 粒子）与 3-D 9.1M `cavity3d_8m_b4`（9,129,329）各 3 次独立运行，其余尺寸各 1 次；CPU 端候选数 / 邻居数统计同样重做）。GPU 1 功率上限恢复为 600 W（你点的 UAC），v7-single 提交 ffd1a2f（= rc4.1 的 solver，外加 kernel-breakdown 驱动的 scan 调用——它自己提交 defrag 命令，第一次跑在 voxel 列表一致性检查上失败，修后重跑）。结果 `docs/cluster_v6/data/e9/section3/`（v7-single 分支镜像 `docs/seam_audit/v7_single/section3_rc4_*`）。

{cmp_md}

**判读**：
- 两个 sweep 核各快 4–6 %，predict / update_voxel / density_copy 的绝对值不变（3-D 的 density_copy +15 µs、predict +4 µs 是 1 µs 量级的散布放大成的百分比，按 0.5 % 规则列为"更新"但不改变任何结论）。2-D 8M 的整步 −5.1 %，与本机第 6 项的 −5.5 %（575 W，链路代码）一致；3-D 9.1M −4.3 %。
- 每候选对的皮秒数随之下降（2-D 5.76 → 5.47，3-D 4.06 → 3.88）：候选数、邻居数、占用体素完全不变（同一物理），少掉的是 atomicAdd 排列带来的访存代价。
- 1M 不动（+0.3 %）：42k 个 voxel 只有 331 个 defrag 工作组，第一波乱序的代价在 1M 上本来就小（本机第 6 项 −0.8 %）。
- 4M 凹陷仍在：4M gen 的每候选皮秒数对 2M / 8M 均值 +5.2 %（旧 +6.1 %），prefix 序不是它的原因。

**tab:kernelcost 的新行**（表 2 全表见 `section3_rc4_table2.md`；LaTeX 行如下，与旧表相差 > 0.5 % 的项都在其中）：

{latex}

### 3.1 fig:baseline 受影响的点

各尺寸 GPU 步长超过 0.5 % 的是 2M、4M、8M、16M、32M 与 3-D 8M；1M 不受影响。fig:baseline 是 2-D 曲线（v7_single 的 S 系列），所以按测量 A 的协议重测了 2M、4M gen、8M、16M、32M（`_run_single_baseline_campaign.py --sizes 2m,4m_gen,8m,16m,32m --configs v7_single:1,v7_single:2,v7_chain:2 --trials 3 --warmup 1000`，窗口 6000 / 4000 / 2000 / 2000 / 2000 步，600 W，45 次运行全部 drift 0），并入 r3 为 `throughput_rc4_merged`（1M、旧 4M、3-D 8M 与 V / Vx 列保留旧值——V / Vx 是 atomic 序的旧运行，它们与新 S 的比值已无意义，图不用它们）。图 `manuscripts/fig/single_gpu_throughput.{{pdf,png}}` 已重画（v7-single 工作树与 vulkan-demo 主库各一份），旧图改名 `single_gpu_throughput_e8_atomic.*`。

{thr_table}

S d2 的变化：2M 371.9 → 385.7（+3.7 %），4M gen 175.1 → 183.8（+5.0 %），8M 94.4 → 99.2（+5.0 %），16M 47.5 → 48.7（+2.5 %），32M 24.5 → 24.8（+1.2 %）；C/S（链路 K = 1 对单卡）在重测点上仍是 0.99–1.00。3-D 8M 的吞吐行没有重测（图里没有它；表 2 的 3-D 值已重测）。

![single_gpu_throughput](../docs/cluster_v6/data/e9/section3/single_gpu_throughput.png)

**协变量**：这次 nvidia-smi 报 GPU 1 的 PCIe 为 Gen 4 x4（10-10 为 Gen 5 x8）——你热插拔换 7900 XTX 再换回来后链路重新协商的结果。单卡核时间不依赖它（1M 点与旧值一致），本节的数字不受影响；本机的传输类测量在重启前不要做。
"""
rep = E8 / "claude/e9_report.md"
text = rep.read_text(encoding="utf-8")
start = text.index("## 3. 第 7 项")
end = text.index("## 4. 集群")
text = text[:start] + sec + "\n" + text[end:]
text = text.replace("- 第 7 项（Section 3 基线重测，600 W）、第 8–11 项（集群）：见第 3、4 节，随进度补。",
                    f"- 第 7 项（第 3 节）：表 2 重测完成——2-D 8M 步长 10502 → {t2['8m']['step total']['new_us']:.0f} µs（{t2['8m']['step total']['change_pct']:+.1f} %），3-D 9.1M 61060 → {t2['3d_8m']['step total']['new_us']:.0f} µs（{t2['3d_8m']['step total']['change_pct']:+.1f} %），两个 sweep 核 −4 ~ −6 %，其余核不变；fig:baseline 的 2M–32M 点已按测量 A 重测并重画（S d2 +1.2 ~ +5.0 %），1M 不受影响。\n- 第 8–11 项（集群）：见第 4 节，随进度补。")
rep.write_text(text, encoding="utf-8", newline="\n")
print("report chapter 3 written")
