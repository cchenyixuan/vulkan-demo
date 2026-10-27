"""E9 report chapter 4, part 1: deployment + the two-card short job e9_p1 (1741843)."""
import pathlib
import re

E8 = pathlib.Path("C:/Users/cchen/PycharmProjects/vulkan-demo-e8")
SPE = pathlib.Path("C:/Users/cchen/AppData/Local/Temp/claude/C--Users-cchen-PycharmProjects-vulkan-demo/fd68689d-e4b4-4b3f-8636-fbe3d1d866de/scratchpad/e9")
summary = (SPE / "pull/summary_p1.md").read_text(encoding="utf-8")
runs_table = summary[summary.index("## 2. 运行（RESULT 行）"):summary.index("## 3. 显存")].replace("## 2. 运行（RESULT 行）", "**运行（RESULT 行）**").strip()
vram_table = summary[summary.index("## 3. 显存"):summary.index("## 4. 池")].replace("## 3. 显存：实测峰值对模型", "**显存：实测峰值对模型**").strip()
compare = (SPE / "pull/e7_p1_1741843/compare/compare_8m.md").read_text(encoding="utf-8").strip()

sec = f"""## 4. 集群（第 8–11 项）

### 4.1 部署

- 包：`vd_e9_cb3c9f5c….tar.gz`（342 文件，`git archive v7-rc4`，内容 = E8 的清单 + `docs/cluster_v6/e9_jobs`）与 `vd_v7single_….tar.gz`（v7-single 的 `experiment/{{v7,v7_single}}` + materials）。
- `~/run/vulkan-demo-e9`（11:53–13:31 CST，5921 s：JuiceFS 今天建文件慢，280 个文件拷了 1002 s）：MANIFEST / SCRIPTS 校验、experiment/v7 树 37f99f2b、defrag 默认 prefix、五个 SPIR-V 与 E8 部署逐字节相同、py_compile、`_test_chain_cuts` ALL PASS、11 个算例生成器核对、五个作业脚本的语法与空跑（e9_A 46 次链路运行 / e9_D 30 次 / e9_p1 3 次完整空跑通过；e9_B、e9_C 的空跑被部署脚本 15 分钟上限截断——登录节点上仅启动检查就 14 分钟——不是失败，链结束后用 90 分钟上限重跑中），`sbatch --test-only` 接受。
- `~/run/vulkan-demo-v7single-rc4`：第一次部署的是 969823a（v7-single-rc4，有第 1 节说的死锁），在 e9_p1 结束、队列为空后用 **786e2d4（v7-single-rc4.1）`UPDATE=1` 重新部署**（树 4a8fc71f，SPIR-V 与 e9 逐字节相同）。正式批次在它 OK 之后才提交。

### 4.2 第 8 项：两卡短作业 e9_p1 = SLURM 1741843（13:33:58–13:40:58 CST，wqd10nbj02g1，GPU 16:00.0 / 27:00.0，两卡 575 W，驱动 580.82.07，7 min）

镜像 `C:/Users/cchen/PycharmProjects/vulkan-demo/logs/n56/2026-10-27/01_e9_p1_1741843/`（README / jobs.csv 已加行），数据 `docs/cluster_v6/data/e9/p1/`。PROVENANCE：harness cb3c9f5、experiment/v7 树 37f99f2b，作业末尾 PROVENANCE_END unchanged。bring-up（2-D 1M K1 675 fps、K2 898 fps）通过。

{runs_table}

- 每粒子成本 8M K1：**5.56 ns**（E8 阶段 1 同样两卡协议 6.28 ns，−11.5 %；本机 575 W 5.78 ns）；8M K2 trace 42.8 fps（E8 38.99，+9.8 %）；64M K2 5.5 fps（E8 5.29，+4 %）。三次运行的 defrag 审计（8M K1、8M K2 两 slab、启动与结束各一次）全部 EXACT voxel order，`defrag_order=prefix` 出现在每个 sim 的 init 行。
- seam：K2 的 seam 0（col 26 / col 51）dup=0 OK；drift / overflow / far_migration / stamps / clamp 全 0。

{vram_table}

**场比较 K = 2 对 K = 1（第 300 步，dump_state 按全局粒子 id 对齐，格式同阶段 1）**：

{compare}

（E8 阶段 1 两卡：max |dv|/U 5.7e-5、|dρ|/ρ0 1.8e-7。）**短作业全部检查通过，提交正式批次。**

### 4.3 正式批次（待提交后补）
"""
rep = E8 / "claude/e9_report.md"
text = rep.read_text(encoding="utf-8")
start = text.index("## 4. 集群")
end = text.index("## 5. 数据与复现")
text = text[:start] + sec + "\n" + text[end:]
rep.write_text(text, encoding="utf-8", newline="\n")
print("report chapter 4.1-4.2 written")
