"""Report section 7.18 (prefix-sum defrag vs default defrag, 2026-10-27 10:50-11:25) + erratum to 7.17 + memory.
Embeds prefix_defrag.md (per-run table, cost table, layout audits, piece tables) from the scratchpad prefix dir."""
import json
import pathlib
import re
import statistics

E8 = pathlib.Path("C:/Users/cchen/PycharmProjects/vulkan-demo-e8")
MEM = pathlib.Path("C:/Users/cchen/.claude/projects/C--Users-cchen-PycharmProjects-vulkan-demo/memory")
PRE = pathlib.Path("C:/Users/cchen/AppData/Local/Temp/claude/C--Users-cchen-PycharmProjects-vulkan-demo/fd68689d-e4b4-4b3f-8636-fbe3d1d866de/scratchpad/e8/prefix")

res = json.load(open(PRE / "prefix_defrag.json", encoding="utf-8"))
md = (PRE / "prefix_defrag.md").read_text(encoding="utf-8")
md = re.sub(r"^### ", "##### ", md, flags=re.M)
cost = res["cost"]
c8, c64 = cost["8m"], cost["64m"]
a8_off, a64_off = res["pieces"]["8m"]["audit_off"], res["pieces"]["64m"]["audit_off"]
N = {"8m": 8_000_000, "64m": 64_000_000}
W = {"8m": 7, "64m": 13}            # columns of an exact piece (prefix order): 50 / 100 non-empty columns over 8 pieces


def pct(x):
    return f"{x:+.1f} %"


def default_mhz(c):
    return statistics.mean(v["sm_MHz"] for k, v in c["groups"].items() if k.startswith("default"))


# scramble table: per pid piece under the default defrag, how many x columns it holds and how concentrated it is,
# next to the default - prefix piece cost of both kernels
scramble = []
for cube, label in (("8m", "8M"), ("64m", "64M")):
    audit = res["pieces"][cube]["audit_off"]["pieces"]
    pc = res["pieces"][cube]
    scramble += [f"**{label}**（默认 defrag 的审计 `{cube}_off_m1_ctl`；精确片 = {W[cube]} 列）", "",
                 "| pid 片 | 默认 defrag：覆盖的 x 列数 | 最多的 {w} 列占该片 pid 的比例 | corr+dens 默认 − prefix (ms) | force 默认 − prefix (ms) |".replace("{w}", str(W[cube])),
                 "|---|---|---|---|---|"]
    for p in audit:
        cols = p["columns"]
        n = sum(cols.values())
        top = sum(sorted(cols.values(), reverse=True)[:W[cube]])
        k = p["piece"]
        dcd = pc["cd"]["default_ms"][k] - pc["cd"]["prefix_ms"][k]
        dfo = pc["fo"]["default_ms"][k] - pc["fo"]["prefix_ms"][k]
        scramble.append(f"| {k} | {len(cols)}（x {p['x_min']}..{p['x_max']}） | {top / n * 100:.1f} % | {dcd:+.3f} | {dfo:+.3f} |")
    scramble.append("")

piece_notes = []
for cube, label in (("8m", "8M"), ("64m", "64M")):
    for short, kname in (("cd", "corr+dens"), ("fo", "force")):
        rec = res["pieces"][cube].get(short)
        if not rec:
            continue
        p, d = rec["prefix_ms"], rec["default_ms"]
        cols = rec["columns"]
        inner = sorted(p[1:7])
        med_inner = (inner[2] + inner[3]) / 2
        piece_notes.append(
            f"{label} {kname}：prefix 八片 {', '.join(f'{v:.2f}' for v in p)} ms（和 {sum(p):.2f}，默认 {sum(d):.2f}，差 {sum(p) - sum(d):+.2f}）；"
            f"中间六片 {min(inner):.3f}–{max(inner):.3f}（中位 {med_inner:.3f}，极差 {(max(inner) - min(inner)) / med_inner * 100:.1f} %），"
            f"两端片 {p[0]:.3f} / {p[7]:.3f} = 中位的 {p[0] / med_inner:.3f} / {p[7] / med_inner:.3f} 倍（x 列 {cols[0][0]}..{cols[0][1]} / {cols[7][0]}..{cols[7][1]}，各含一根端壁列）")

sec = f"""

### 7.18 defrag 的 pid 排列：`use_prefix_sum_defrag: true` 对默认 atomicAdd 排列的每粒子成本与八片剖面（本机，2026-10-27 10:50–11:25）

**开关在当前树上的状态：不能单独运行。** `defrag.comp` 的 `USE_PREFIX_SUM_DEFRAG` 分支（spec constant 46）把 `voxel_base_offset[voxel_id]` 当作目标基址，但 `experiment/v7` 里没有任何东西写这个缓冲：没有 `prefix_sum.comp`（着色器目录里没有这个文件），主机端也没有扫描，模拟器只分配它（set 1 binding 4）；`docs/paper_material_map.md` 早已记为"prefix-sum defrag 未实现"，v7_single 的 seam audit 记为"只有 USE_PREFIX_SUM_DEFRAG（关）读"。另外 defrag cmd 的第 6 步把 `defrag_scratch_counter` 拷进 `alive_particle_count`，而 prefix 分支从不递增这个计数器（DCE 掉了），cmd 自己先把它填 0。实测（`8m_raw_check`，开关置 true、其余照旧，40 步）：启动 defrag 之后 alive = 0，8M 个 pid 里 7,999,936 个的 voxel id = 0（所有 voxel 都写到 base 0 上互相覆盖），"稳态" 2337 fps（核没有粒子可算），final drift = −8,000,000——与 `logs/dip_diag_2d/case_4m_prefix` 里 V0 的记录（4M 2524 fps、correction 33 µs）是同一种坏法。这就是正确性套件记的问题：开关本身没坏，坏在没有生产者。

**本次的做法**（scratch monkeypatch `local_size_cost/prefix/prefix_defrag_run.py`，经 `run_chain_v6.py` 跑链路代码；`experiment/v7` 代码与 SPIR-V 不动，运行前后 `HEAD:experiment/v7` = 10781f23、工作区 0 改动，批处理日志有核对行）：开关置 true（日志核对行 `spec constant 46 entry=[(46, 'B', 1)]`），缺失的生产者放在主机——每次 defrag 提交前读回 `inside_particle_count`，按 voxel id 顺序对 own voxel 做 exclusive scan，上传到 `voxel_base_offset`（8M 4–6 ms、64M 26–28 ms，启动一次 + 每 defrag_cadence = 1000 帧一次，不在 100–299 的计时窗口内），提交后把 `alive_particle_count` 写回扫描总数。散射本身由树里的 `defrag.comp` prefix 分支完成，其它核不知道差别。协议同 7.12 / 7.17：GPU 1，575 W，K = 1，300 步取 100–299，逐核 trace；prefix 各两次，默认 defrag 本批各一次对照（加上 7.17 的两次 m1_plain 共三次）。

**验证**：drift 0；末态聚合量与 7.17 的原版运行相同（8M：ρ 分位差 ≤ 6e-5、速度分位差 ≤ 4e-8、p_mean 相对差 1.2e-4，原版两次之间 8e-4）。**布局审计**（每次 defrag 后读回 own 粒子的 `position_voxel_id` 与 `material`，按 pid 顺序看 voxel id）：prefix 下 8M 的 125,000 个、64M 的 1,000,000 个 voxel 块严格按 voxel id 升序（0 次块间倒序）；默认 defrag 下块数相同、块内连续，但块的先后是原子到达序：8M 有 {a8_off['descents']:,} 次块间倒序，64M {a64_off['descents']:,} 次。**乱序的分布不均匀**：8M 的八个 pid 片每一片都覆盖全部 46–50 根 x 列（最多的 7 列只占该片 23–35 %），64M 只有前两片（前 16M 个 pid）是乱的（覆盖 x 1..26 的 25–26 列，最多的 13 列占 64–66 %），第 2 片起就是精确的 x 片（13–14 列，98–100 %）。原因是 defrag 按 voxel 派发（128 voxel 一个工作组）：第一波工作组同时填满所有 SM，它们的 atomicAdd 以随机次序完成——8M 的 977 个工作组整个落在第一波里，64M 的 7,813 个只有开头约 2,000 个（≈ 16M 粒子）在第一波，之后工作组按派发序逐个补位、到达序近似派发序。

**每粒子成本**（`data/e8/local_size_cost/prefix_defrag.md`）：

| 立方体 | defrag | 步 (ms) | ns / 粒子 | corr+dens (ms) | force (ms) | SM MHz |
|---|---|---|---|---|---|---|
| 8M | 默认（7.17 两次 + 本批对照） | {c8['default_mean']['step_ms']:.3f} | {c8['default_mean']['step_ms'] / 8e6 * 1e6:.3f} | {c8['default_mean']['cd']:.3f} | {c8['default_mean']['fo']:.3f} | {default_mhz(c8):.0f} |
| 8M | prefix（两次） | {c8['groups']['prefix (host scan)']['step_ms']:.3f} | {c8['groups']['prefix (host scan)']['step_ms'] / 8e6 * 1e6:.3f} | {c8['groups']['prefix (host scan)']['cd']:.3f} | {c8['groups']['prefix (host scan)']['fo']:.3f} | {c8['groups']['prefix (host scan)']['sm_MHz']:.0f} |
| 8M | **差** | **{c8['delta']['step_ms']:+.3f}（{pct(c8['delta_pct']['step_ms'])}）** | {c8['delta']['step_ms'] / 8e6 * 1e6:+.3f} | {c8['delta']['cd']:+.3f}（{pct(c8['delta_pct']['cd'])}） | {c8['delta']['fo']:+.3f}（{pct(c8['delta_pct']['fo'])}） | |
| 64M | 默认（7.17 两次 + 本批对照） | {c64['default_mean']['step_ms']:.3f} | {c64['default_mean']['step_ms'] / 64e6 * 1e6:.3f} | {c64['default_mean']['cd']:.3f} | {c64['default_mean']['fo']:.3f} | {default_mhz(c64):.0f} |
| 64M | prefix（两次） | {c64['groups']['prefix (host scan)']['step_ms']:.3f} | {c64['groups']['prefix (host scan)']['step_ms'] / 64e6 * 1e6:.3f} | {c64['groups']['prefix (host scan)']['cd']:.3f} | {c64['groups']['prefix (host scan)']['fo']:.3f} | {c64['groups']['prefix (host scan)']['sm_MHz']:.0f} |
| 64M | **差** | **{c64['delta']['step_ms']:+.3f}（{pct(c64['delta_pct']['step_ms'])}）** | {c64['delta']['step_ms'] / 64e6 * 1e6:+.3f} | {c64['delta']['cd']:+.3f}（{pct(c64['delta_pct']['cd'])}） | {c64['delta']['fo']:+.3f}（{pct(c64['delta_pct']['fo'])}） | |

**乱序片与成本的对应**（默认 defrag 的每片列覆盖对 m = 8 两组运行的默认 − prefix 片时长）：

""" + "\n".join(scramble) + f"""
**判读**：
- 默认 defrag 的 pid 排列是一个真实的成本，而且是局部的：乱序的 pid 区域每粒子贵 8–12 %，有序的区域与 prefix 相同（64M 第 2–7 片默认 − prefix 在 ±0.15 ms 内，第 0–1 片 +2.3 / +1.8 ms（corr+dens）和 +2.3 / +1.5 ms（force））。8M 整个网格都乱，所以整步省 {-c8['delta']['step_ms']:.1f} ms（{pct(c8['delta_pct']['step_ms'])}）；64M 只有前四分之一乱，省 {-c64['delta']['step_ms']:.1f} ms（{pct(c64['delta_pct']['step_ms'])}）。两个 sweep 核各占一半。
- 乱序区域 ≈ defrag 第一波工作组覆盖的粒子数，是卡的常数（约 16M 粒子）而不是 N 的函数，所以它的代价在 N ≥ 16M 后是一个常数：这正是 7.14–7.17 一直在找的那几毫秒"固定项"（7.16：8M {c8['fixed_terms_716']['cd']:+.2f} + {c8['fixed_terms_716']['fo']:+.2f} = {c8['fixed_terms_716']['cd'] + c8['fixed_terms_716']['fo']:.2f} ms，64M {c64['fixed_terms_716']['cd']:+.2f} + {c64['fixed_terms_716']['fo']:+.2f} = {c64['fixed_terms_716']['cd'] + c64['fixed_terms_716']['fo']:.2f} ms；本节实测的布局代价 8M {-c8['delta']['step_ms']:.1f}、64M {-c64['delta']['step_ms']:.1f} ms）。它不是 dispatch 尾部（7.17 已排除）、不是截面项（7.16 已排除），是 atomicAdd defrag 的布局代价。7.13 / 7.14 的拉长盒子（2×8M = 16M）落在这个常数的过渡区，解释了它们相对立方体 +5 ms 的"固定项"不同——这一点未单独验证。
- 7.12 的规模效应随之消失：默认 defrag 下 8M / 64M 的每粒子成本比 {c8['default_mean']['step_ms'] / 8e6 / (c64['default_mean']['step_ms'] / 64e6):.3f}，prefix 下 {c8['groups']['prefix (host scan)']['step_ms'] / 8e6 / (c64['groups']['prefix (host scan)']['step_ms'] / 64e6):.3f}。
- prefix 运行的 SM 时钟略低（8M {c8['groups']['prefix (host scan)']['sm_MHz']:.0f} 对 {default_mhz(c8):.0f} MHz，575 W 封顶下访存活动更高），等时钟的差会再大一点。
- 对集群数字的含义：E8 全部作业用默认 defrag。K = 1 参考和 K = 8 的每个 slab 都带着这项布局代价，而且它按 slab 粒子数变（slab ≤ 16M 时整个 slab 乱序，+12 %；大 slab 只有前 16M 乱），所以 K = 8 的 slab（64M / 8 = 8M）比 K = 1 的 64M 多付 10 %——这项进入 η 的分母，是 7.12 "8M / 64M 1.09" 的来源之一。要不要把 prefix defrag 做成正式路径（GPU 端 scan 核，或主机 scan——defrag 只在启动和每 1000 帧跑一次，主机 scan 64M 也只要 27 ms）由你定；本节没有改树。

**布局审计与 prefix 下八片的精确列归属**（material 计数直接从 `material` 缓冲读回；默认 defrag 的表说明 pid 片不对应 x 列）：

{md[md.index('##### layout audit'):md.index('##### m = 8 piece durations')].strip()}

**m = 8 八片时长：prefix 对默认（7.17）**（`prefix_defrag_run.py` 导入 7.17 的 `split_dispatch_run_ticks.py` 不改：同样的 SEGMENT 流水线、同样的片界和 tick；默认列 = 7.17 的 cd8t / fo8t 运行；prefix 下第 k 片的 x 列范围来自审计，是精确的）：

{md[md.index('##### m = 8 piece durations'):].strip()}

**各片判读（prefix 下才是真正的 x 剖面）**：
""" + "\n".join(f"- {note}" for note in piece_notes) + f"""
- x 剖面是平的、对称的：中间六片的极差 ≤ 1.5 %，两端片（含端壁列、壁占 21 % / 10.7 %）比中间便宜（8M corr+dens −4 / −6 %、force −15 %；64M corr+dens −2 %、force −7.5 %）——壁粒子在 sweep 核里的工作量小于流体。没有哪一片"明显偏贵"。

**7.17 追加项的勘误**：7.17 把 m = 8 的八片按"pid 序 = voxel 序 = x 从 −x 端到 +x 端"映射到列区间并给出壁 / 盖 / 流体计数。本节的审计表明默认 defrag 下这个前提不成立（8M 每片覆盖 46–50 列；64M 前两片覆盖 x 1..26、互相混合），7.17 的片→列归属、"64M −x 四分之一偏贵 8 %"和"+x 端片最便宜（含壁更便宜）"的 x 判读作废：64M 第 0–1 片贵 8 % 是因为它们是乱序区，不是因为在 −x 端；8M 各片都是整网格的随机混合，片间差异无 x 意义。prefix 下的八片才是真正的 x 剖面（上表）。7.17 的主结论（段边界成本 0.24–0.39 ms，固定项不是 dispatch 尾部）不受影响。

复现：`prefix/run_prefix.sh`、`prefix/run_batch.sh`（13 次运行约 28 min，另加手动的 `8m_raw_check`、`8m_host_m1_r1`、`8m_off_m1_ctl`）、`prefix/prefix_analysis.py`；日志镜像 `logs/e8_local/prefix_20261027/runs/`（含每次运行的 `layout_audit.jsonl`）。
"""
rep = E8 / "claude/e8_3d_report.md"
text = rep.read_text(encoding="utf-8")
assert "### 7.18" not in text
rep.write_text(text.rstrip("\n") + "\n" + sec, encoding="utf-8", newline="\n")
print("report 7.18 appended")

anchor = "docs/cluster_v6/scripts/local_size_cost/split/, logs logs/e8_local/split_20261027/.**"
block = ("\n\n**2026-10-27 10:50-11:25, prefix-sum defrag (report 7.18; user asked for use_prefix_sum_defrag=true at 8M/64M K=1 x2 + m=8 pieces vs default): "
         "the switch ALONE IS BROKEN on the tree (defrag.comp's prefix branch reads voxel_base_offset which nothing writes: no prefix_sum.comp, no host scan; "
         "alive count is copied from the untouched scratch counter = 0) -> 8M raw check: alive 0, 7,999,936 pids with voxel id 0, drift -8M, '2337 fps' "
         "(same as the dip-diag case_4m_prefix garbage). Scratch monkeypatch prefix/prefix_defrag_run.py (tree untouched, 10781f23 before/after): switch on + "
         "host-side exclusive scan of inside_particle_count uploaded to voxel_base_offset before every defrag (8M 5 ms, 64M 27 ms) + alive restored; validated "
         "(drift 0, aggregates = plain, layout audit: exact voxel order vs default's 41,582 (8M) / 56,409 (64M) block descents). RESULT: prefix defrag saves "
         f"{-c8['delta']['step_ms']:.1f} ms/step at 8M ({c8['delta_pct']['step_ms']:+.1f} %) and {-c64['delta']['step_ms']:.1f} ms at 64M ({c64['delta_pct']['step_ms']:+.1f} %), "
         "half in each sweep kernel. MECHANISM (audit + pieces): the atomicAdd order is scrambled only over the defrag dispatch's FIRST WAVE (~2000 workgroups "
         "~ 16M particles): 8M is scrambled everywhere (every pid piece spans all 50 x columns, +12 %/particle), 64M only in pid pieces 0-1 (x 1..26 mixed; "
         "+2.3/+1.8 ms cd, +2.3/+1.5 ms fo) while pieces 2-7 are exact x slabs and cost the same as prefix -> a ~constant penalty for N >= 16M = the 7.16 "
         f"'fixed terms' (5.0 / 5.7 ms); 7.12's size effect 8M/64M {c8['default_mean']['step_ms'] / 8e6 / (c64['default_mean']['step_ms'] / 64e6):.2f} -> "
         f"{c8['groups']['prefix (host scan)']['step_ms'] / 8e6 / (c64['groups']['prefix (host scan)']['step_ms'] / 64e6):.2f} under prefix. Prefix x profile: flat (middle six "
         "within 1.5 %), end pieces (end-wall columns) cheaper. ERRATUM to 7.17's addendum: its piece -> column mapping and the '-x quarter +8 %' reading are void "
         "(pieces 0-1 at 64M are the scrambled region, not the -x end). All E8 cluster numbers carry the default-defrag layout penalty (per slab: whole slab "
         "scrambled when <= 16M, so K=8 slabs of 8M pay ~12 % vs K=1 64M ~2 % -> enters eta). Whether to make prefix defrag a real path is the user's call.**")
p = MEM / "project-e8-3d-scaling.md"
t = p.read_text(encoding="utf-8")
assert t.count(anchor) == 1, t.count(anchor)
p.write_text(t.replace(anchor, anchor + block), encoding="utf-8", newline="\n")
m = MEM / "MEMORY.md"
t = m.read_text(encoding="utf-8")
old = "per-piece timestamps + column mapping in piece_times.md;"
assert t.count(old) == 1, t.count(old)
t = t.replace(old, "per-piece timestamps in piece_times.md (its column mapping is VOID, see 7.18); 7.18 prefix-sum defrag: the switch alone is broken on the tree "
                   "(nothing writes voxel_base_offset), host-scan monkeypatch validated; exact voxel order saves "
                   f"{-c8['delta']['step_ms']:.1f} ms/step at 8M ({c8['delta_pct']['step_ms']:+.1f} %) and {-c64['delta']['step_ms']:.1f} ms at 64M ({c64['delta_pct']['step_ms']:+.1f} %) "
                   "= the 7.16 fixed terms; the atomicAdd defrag scrambles only the first ~16M particles (defrag's first wave) -> 8M fully scrambled, 64M pieces 0-1 only; "
                   "size effect gone under prefix; K=8 slabs of 8M pay ~12 % vs K=1 64M ~2 % (enters eta); prefix x profile flat, end-wall pieces cheaper;")
m.write_text(t, encoding="utf-8", newline="\n")
print("memory patched")
