# v7 性能优化(E39,2026-10-08/09):B9、B6、B4、B1、B3

接 [`v6_opt.md`](v6_opt.md)(v6 的发布组合)和 v6-rc2 审计的 B 部分(性能项)。v7 = v6-rc2 加上五项性能优化。每项一个 `V7_*` 开关、一个单独的提交;开关为 0 时就是前一个构建,录制逐条相同,SPIR-V 逐字节相同。`experiment/v6` 保持 v6-rc2 原样,没有改动。代码在分支 `v7-perf`;`v7-wall-bc` 分支上那个旧的 `experiment/v7`(E36 壁面研究)与这里无关,两者不合并、不混用。看完报告后的决定(2026-10-09):B3 默认关(`V7_BAND_OVERLAP=0`),其余默认不变;附注标签 `v7-rc1` 打在本文档所在的提交上。

完整报告(所有表格、方法、日志位置):https://claude.ai/artifact/BSS2eB9iVoiHaGc2H4UHB6(私有,需要分享才能被他人打开)。报告写的是 E39 时的状态(B3 默认 auto、没有标签),页首有 v7-rc1 的说明;v7-rc1 的默认与相应数字以本文为准。

## 结论(TL;DR)

- **速度。** v7-rc1 默认对 v6-rc2(chain bench,两张 RTX 5090,3 次试验;每次试验里一个配置的各构建与 v6-rc2 紧挨着各跑一次,顺序每次轮换,按试验配对。测量时 B3 的默认是 auto,它只在 2-D 1M K = 2 上开启;所以除这一格外,v7-rc1 默认与测量的构建相同,这一格用 B3 关的 B1 构建):
  - 2-D 62k(n250):K = 1 +42.2 %,K = 2 +42.0 %。
  - 2-D 1M:K = 1 +33.4 %,K = 2 +38.6 %(开 B3 为 +47.4 %)。
  - 2-D 16M:K = 1 +32.5 %,K = 2 +32.2 %。
  - 3-D 8M(4 层壁):K = 1 +29.0 %,K = 2 +31.3 %,K = 4(0,1,0,1)+36.2 %。
  - 3-D 1M(9 层壁):K = 1 +36.8 %,K = 2 +43.0 %。
  - adami K = 1:250² +31.4 %,1000² +30.0 %。
- **各项贡献。**
  - B1:+28.1 到 +41.7 %,是大头。
  - B9:2-D +0.8 到 +1.5 %;62k K = 1 与 adami 250² 的 +0.4 % 和 3-D 的 −0.2 到 +0.4 % 在噪声内。
  - B6:只在 K ≥ 2 起作用,+0.6 到 +5.6 %;2-D 16M +0.1 ± 0.6 %,在噪声内。
  - B4:只在 3-D 1M 9 层壁上开启,K = 1 +5.4 %,K = 2 +3.3 %。
  - B3:v7-rc1 默认关。设 `V7_BAND_OVERLAP=auto` 时,本次性能活动的 13 个配置里只有 2-D 1M K = 2 开启(auto 的窗口见"与规格不同的地方"),+6.3 %。
- **η**(K = 2 对两张卡同时跑的 K = 1):
  - 2-D 1M:73.8 → 76.7 %(开 B3 为 81.6 %)。
  - 3-D 8M:95.5 → 97.1 %。
  - 3-D 1M:71.6 → 74.8 %。
  - K = 4 3-D 8M:77.7 → 82.0 %。
  - 2-D 62k(28.7 → 28.6 %)与 2-D 16M(95.7 → 95.5 %)不变。
- **逐位。** 在 5 个标准算例上,下面每个构建都与前一个构建逐位相同:v6-rc2 → v7 全关、B9、B6、B4(屏蔽被跳过的深壁行)、B3。B1 预期不逐位,因为 ρ 的求和顺序变了;单步检验中 L、kernel sum、∇ρ 0 ULP;不开 δρ 时 ρ 每步有 0–32 个粒子差 1 ULP。
- **精度。** 集合检验(E33 的方法):
  - 3-D 1M 的两个时刻、2-D 1M 的 2000 步不可区分。
  - 2-D 1M 300 步:原组检出一个很小的偏移(约为一次 v6 运行起伏的 0.25 倍),独立复核组(另 6 + 6)没有复现。
  - 2-D 1M 2000 步:两组各自按判定规则不可区分。合并 12 + 12 后,v7 的运行更常走到两个可重复的另外轨道分支(12 次中 5 次,v6 1 次;事后检验单侧 p ≈ 0.03)。只关于这一个快照,不说明误差更大。
  - 方腔 Re = 1000(250² 与 500²,adami K = 1 与 simple K = 2,各对 v6-rc2 一次):v7 − v6 与同一算例两次不含 E39 改动的运行之差(噪声对)同量级。
    - 6 个量的最大 |差| / σ:v7 − v6 为 3.9 / 1.5 / 2.7 / 1.2,噪声对 2.6 / 1.5 / 4.9 / 1.0。
    - 最大的 3.9 σ(250² adami 的 L2(u),−0.025 pp)主要来自参照运行偏高:v7 对另一次参照(E36 的 adami_rho0)只差 0.5 σ。
    - 极值与 L2 的差逐量都比同一个量与 Marchi 2021 的偏差小 100 倍以上。
    - 每个比较只有一对运行,与运行间差异同量级的变化分辨不了。
- **不变量与验证层。** 本次所有运行的 drift、overflow 计数器、远迁移、帧戳都是 0。验证层在 K = 2 2-D 1M 与 K = 4 3-D 8M 上各跑 2000 步(E39 时的默认与 v7-rc1 默认各一次),都是 0 条消息。
- **与规格不同的地方。**
  - B4 默认 auto:2-D 关;3-D 只在深壁候选 ≥ 1 % 的 slab 上开。
  - B3 默认关(看完报告后的决定,v7-rc1)。设 `V7_BAND_OVERLAP=auto` 时,只开启 2-D、恰好 2 个 slab、每个 slab 40 万–150 万粒子的链。
  - B1 的 compact band 派发总是回退到分开的 kernel。

## v7 相对 v6 的改动

**改名(0fc247a)。** `experiment/v7` 是 `experiment/v6`(v6-rc2,c3514e5)的机械复制:

- 文件 `*_v6*` → `*_v7*`,模块 `experiment.v7.*`,类 `*V7`。
- 环境变量 `V6_*` → `V7_*`,日志标签 `[xxx_v7]`。

SPIR-V 14/14 与 v6-rc2 逐字节相同。门 0(canonical lists)上 v7 与 `experiment/v6` 逐位相同,覆盖:K = 1 250² simple / adami;K = 2 200 步与 999 步(3 个粒子越过切口);K = 1 3-D 1M。

**不变的接口。** 命令行参数、case.yaml 格式、各 runner 的接口与 v6 相同(chain / dual / single bench、`cavity_runner`、`canonical_dump`、`dump_state`);v7 只多认 `V7_*` 开关。开关表由 `simulator_v7.configured_v7_switches()` 给出,以下地方都打印它:chain bench 的表头、step trace 的 run_meta、`canonical_dump` 的 meta、集群脚本。

**集群脚本(3b756d1)。** 一个开关 `E30_SOLVER=v6|v7`,默认 v6,v6 的输出不变。设为 v7 时,`e30_lib.sh`:

- 清掉继承的 `V5_*` / `V6_*` / `V7_*`;
- 导出 `V7_WORKER_AFFINITY`(同样的 cpulist 和 discrete-first 顺序);
- 把 `--solver v7` 传给配置打印、每次运行和解析。

`run_chain_v6.py`、`parse_run_v6.py`、`bringup_check_v6.py`(SPIR-V 清单按求解器区分,v7 是 22 个文件)、`report_tables_v6.py`、sbatch 文件与 `deploy_v6.sh` 跟随同一开关。

**工具。**

- `canonical_dump --solver v7`(c02cc4c):meta 记录开关表、B4 的逐 slab 跳过记录、B1 的逐 slab 融合判定。
- `cavity_runner` / `cavity_campaign --solver v7` 与 `e39_compare`(51faaf8)。
- `dump_state --version v7` 与 `e39_ensemble`(3b756d1)。
- `fused_single_step`(B1)。
- `e39_perf_campaign`(性能活动,8f8cbd9;v7-rc1 的提交里 b3 构建改为 b1 加 `V7_BAND_OVERLAP=auto`,并拒绝等于 b1 的 b3)。

## 开关与默认值

| 开关 | 项 | 取值 | 默认 | = 0 时 | 生效范围与回退 |
|---|---|---|---|---|---|
| `V7_DENSITY_COPY_COMPUTE` | B9 | 0, 1 | 1 | v6 的 `vkCmdCopyBuffer` 录制 | 所有 K |
| `V7_GHOST_SEND_LANES` | B6 | 0, 4, 8, 16, 32 | 32 | `ghost_send.comp` 及其派发 | 有 peer 的 slab(K = 1 没有 ghost_send) |
| `V7_DEEP_WALL_SKIP` | B4 | 0, 1, auto | auto | B6 构建 | auto:3-D 且初始状态深壁候选 ≥ 1 % 的 slab;2-D 关 |
| `V7_DEEP_WALL_CHECK` | B4 调试 | 0, 1 | 0 | — | 被跳过的墙仍做一次邻居测试,计入 `overflow_deep_wall_skip_count` |
| `V7_FUSED_CORRECTION_DENSITY` | B1 | 0, 1 | 1 | B4 构建 | 以下情况整个 slab 回退到分开的 kernel:correction 与 density 的 band 宽不同(`V7_BAND_WIDTHS` c ≠ d;`V7_BAND_COMPACT_DISPATCH` 只支持 2,3,4,所以总是回退);`V7_DIAG_GHOST_SELF` 只点名其中一个 kernel |
| `V7_BAND_OVERLAP` | B3 | 0, 1, auto | 0(v7-rc1;E39 时为 auto) | B1 构建 | 默认关。auto:每条链判定一次,全开或全关,2-D、恰好 2 个 slab、两个 slab 的初始自有粒子数都在 [400 000, 1 500 000] 内时开。1:每个合法 slab 都开(合法 = 2-D、有 peer、B1 融合生效、`V7_CASCADE_FORCE=1`、`V7_BAND_VOXEL_DISPATCH=1`、`V7_DENSITY_COPY_COMPUTE=1`、simple 壁)。K = 1 与 3-D 在任何取值下都关 |

每个开关在 import 时读一次,解析器严格,不认识的值直接报错。B4、B1、B3 的开关不为 0 时,每个 slab 在构造时打印自己的判定:`[SimV7] V7_X=值: on/off(原因)`;为 0 时不打印。v7-rc1 起 B3 默认 0,所以默认运行里没有 `V7_BAND_OVERLAP` 这一行;B3 的状态看运行表头的开关表(`V7_BAND_OVERLAP=0`)或 `band_overlap_record`(`canonical_dump` 的 meta、step trace 的 run_meta)。

## 逐项

### B9 density 拷贝改为 compute pass(d6a9ecb)

`density_scratch_copy.comp` 按 32 位字原样拷贝:每个 invocation 一个 (ρ, P) 槽 = 一个 uvec2,SPIR-V 中没有浮点类型。区域与 v6 的 `vkCmdCopyBuffer` 完全相同:

- own 范围;
- 每个 peer 方向的 G1 内层 replica 区;
- departed 池。

区域由 `_density_copy_buffer_regions` 给出(两种录制共用),以 spec 101–108 固定到 slab。拷贝前后各一道 compute → compute 屏障。顺带去掉了 v6 录制在 K = 1 下 syncval 报告的 `density_pressure` WRITE_AFTER_WRITE。

**开关解析(ee736f0)。** 报告复核发现,`V7_DENSITY_COPY_COMPUTE` 原来按 `== "1"` 读:任何别的值(如 `true`、`on`)都静默选 v6 的拷贝。现在与其他 E39 开关一样严格,只认 0 / 1,其他值报错。0 与 1 的行为不变;本文的验证运行都没有设这个变量,或设为 0 / 1。

### B6 ghost_send 按 (面体素, 层) 分 lane 组(3d6e1c9)

`ghost_send_lanes.comp`:L 个 lane 一组,负责一个 (面体素, 层) 对。

1. 组长执行 `ghost_send.comp` 原有的分配:同一计数器上的一次 atomicAdd,同样的溢出判断与写 / 不写。
2. 组长经 shared memory 广播 (count, base),在 kernel 唯一一道顶层 barrier 之后读取。
3. lane l 用原来的表达式拷贝槽 l、l + L、…。
4. 组长再串行发送该体素的 migrant:远迁移、migrant 区、departed 拷贝、源端删除、incoming 复位。这保持了每个体素内的 migrant 顺序。

bootstrap 的 ghost 轮、单层 ghost 池、K ≥ 3 中间 slab 的两个方向都用这个 kernel。工作组大小 64(spec 110),lane 数是 spec 109。

**这道 barrier 只靠 `_test_seam_layout` 的文本检查守护。** 每组 ≤ 32 lane 时整组在一个 warp 内;复核的阳性对照中,去掉 barrier 的构建在 RTX 5090 上能通过所有 GPU 门。

### B4 保守的深壁跳过(bf91694)

`deep_wall_marker.comp` 每步从当步的列表重建两个体素标志。运行时机:K = 1 在 correction 之前,K ≥ 2 在 phase B 开头;单命令步与 K = 1 bootstrap 也跑。

- presence:体素里列出了非 BOUNDARY 粒子(墙的 density 要计入的所有种类)。
- deep:邻居循环访问的 3^d 个在网格内的体素都属于本 slab,且都没有 presence(ghost 体素按"有"计)。

位于 density band 之外、在 deep 体素中的 BOUNDARY 粒子被跳过:

- correction 直接返回;
- density 跳过邻居循环,照常对零累加器做收尾,scratch 值不变。

体素边长等于支撑半径,所以判据对循环实际读的东西是精确的。只有 `correction_interior` / `density_deep_interior`(以及无 peer slab 的全域 pipeline)用跳过变体;band、compact、K ≥ 2 bootstrap 的 pipeline 从不跳过。

**`auto` 的规则:** 3-D 且初始状态深壁候选 ≥ 1 % 的 slab 开。2-D 强制开启反而慢 0.4–3.3 %:marker 本身 13.5 µs,而判据只覆盖 11 层壁中外侧 2 层,所以 2-D 关。4 层壁的 3-D 8M 没有候选,也关。**这偏离了"默认全开":B4 的默认是 auto 而不是 1,理由是上面的测量。**

**被跳过的墙保留上一次的 L / kernel sum。** K = 1 时这个旧值是 0,bootstrap 也跳过。后果:

- 不加掩码就显示或比较这两个字段的诊断会看到深壁的旧值(渲染器的 kernel-sum 着色、`single_step`、`opt_validate`、`ab_restart`);
- 被跳过的墙不再计入 `correction_fallback_count`。

### B1 correction 与 density 合并为一次邻居遍历(f646bf6)

`correction_density.comp` 逐语句照搬 correction.comp 的循环:相同的邻居顺序、正则化、行列式阈值、Frobenius 上限、fallback 计数。同一遍里累加 density 的成对项:

- 一对的 drift + diffusion = q_jᵀ L_i ∇W_ij;
- 所以循环里累加 S = Σ q_j ⊗ ∇W_ij,只要对称部分(3-D 6 个累加器,2-D 3 个);
- 求逆之后,再与写进 `correction_inverse` 的那个 L_i 收缩。

density 的粒子对集合与收尾不变:INLET、adami 墙、墙–墙跳过、`V7_DELTA_DENSITY`、EOS、墙存 ρ0。B4 保留在融合核内:interior 与无 peer 全域 pipeline 用 DEEP_WALL_SKIP 变体。

覆盖的录制点:

- phase B interior;
- phase C band:band-voxel 派发与逐粒子 boundary 派发,含两层 ghost 时把内层 ghost 列当 self;
- bootstrap;
- K = 1 单命令步;
- K ≥ 3 中间 slab;
- 单层 ghost;
- cascade force 关;
- adami。

回退见开关表;回退的 slab 录制与开关 0 逐条相同。

**数值。** 不再与 B4 构建逐位相同,因为 ρ / P 的求和顺序变了。单步检验(`experiment/seam_audit/fused_single_step.py`)把同一已发展状态恢复到每个 buffer,分开的 kernel 与融合核各跑一步。结果:

- L、kernel sum、∇ρ 在所有写过的行上 0 ULP。
- 不开 δρ:ρ 每步 0–32 个流体粒子差 1 ULP(绝对存储;2-D 单命令步与 adami 为 0)。P 只在这些行上差,差值是一个 ρ ULP 经 EOS 的量(≤ 1.36 Pa)。
- 开 δρ(存储 ρ − ρ0):差的行更多,max |Δρ| 2.5–2.6 × 10⁻⁷ kg/m³;p99 在 2-D 1M 是 6 ULP,在 adami 是 47 ULP(δρ 本身更小,同样的绝对差对应更多 ULP)。
- 配置共 12 种:K = 1 2-D / 3-D(链式与单命令步)、K = 2 2-D / 3-D、K = 3、单层 ghost、cascade 关、δρ、adami、adami + δρ;评审另跑 7 种。

**0 ULP 依赖驱动对相同表达式做相同的 FMA 合并**,因为 SPIR-V 中没有 NoContraction。换驱动(集群、A100、3090)之后,先重跑 `fused_single_step.py` 再引用这一结论。

另外,v7 的 loader 拒绝有 z 方向运动的 2-D 算例(重力 z 分量或材料初速 z 分量不为 0)。2-D 累加器省略了 z 项,只在 z = 0 平面上精确;现有 38 个 2-D 算例全部满足。

**寄存器。** 原 correction / density 各 40 个;融合核 2-D 56 个、3-D 64 个(3-D 占用率 67 %)。试过的少寄存器写法慢 1–3 %,未采用。

### B3 phase C 的 band kernel 与 cascade force 并发(单队列重排,22701d4)

只有一条 compute 队列,不用第二条队列,也不用 event。所有 compute 屏障都是全局的,所以一个 dispatch 只能与录制时紧挨着、中间没有屏障的 dispatch 重叠。band 链又必须排在两件事之后:phase C 对 upload_done 的等待,以及 install / append_departed 之后的屏障。所以移动的是配对的另一方:`force_deep_interior_scratch` 从 phase B 移到 phase C,按 workgroup 拆成两段,各自紧跟一个延迟受限的 band kernel 录制。新顺序:

- phase B:`correction_density_interior`(3-D 或强制 B4 时前面还有 marker)。
- phase C:expand → install → append_departed → {`correction_density_boundary_band`,段 1} → 屏障 → density 拷贝 → 屏障 → {`force_boundary_band`,段 2} → phase C 结束(frame_done 信号、A(n+1) 开头的全局屏障)。

band kernel 先录。测过的其他布局都更慢:

- force_deep 先录:先录的段占满 SM,band kernel 只能在它的尾部开始;
- 一部分 force_deep 留在 phase B;
- 把拷贝作为第三个伙伴。

**依赖分析。** 逐元素分析,完整表在 `_resolve_band_overlap` 的文档字符串里。带宽为 c / d / f(f ≥ d + 1 ≥ c + 1),列从 peer 一侧数起。

- force_deep 读列 ≥ f − 1 ≥ d 的 scratch ρ / P,以及本粒子列 ≥ f 的 L / kernel sum,这些都由 phase B 写。
- force_deep 写列 ≥ f 的 acceleration / shift。
- band kernel 写列 < c、G1、departed 副本的 L / kernel sum / scratch。
- force band 写列 < f 的 acceleration / shift。

结论:写集不相交,双方不读对方写的元素。install / expand / append_departed 从不参与配对。新迁入粒子的尾槽在 phase C 中属于第 0 / 1 列(< f),在 B1 的 phase B 里是死槽,两种情况下都不被写。三位评审各自重新推导,结论一致。

**段的实现。** 每段是 `force.comp` 的 FORCE_SEGMENT 变体的 base 0 派发(`spv/force_segment.comp.spv`,spec 115 = 段的首线程,同样的粒子、同样的代码);`force.comp.spv` 和其他 SPIR-V 逐字节不变。

先试过 `vkCmdDispatchBase`。在本构建和驱动 576.88 上,force_deep 的 base 派发 (146, 205) 被观察到只执行了 workgroup [146, 205)。这与有没有屏障无关,而且可复现;用简单 kernel 写的孤立探针复现不了,原因未定位(`logs/e39/b3/diag`)。

**录制不变式。**

- 每次录 phase C 都检查 phase B + C 恰好把 force_deep 的全部 workgroup 派发了一次,否则抛错。
- slab 只在录制时 correction + density 融合生效才用 B3 布局。所以 `fused_single_step` 的分开一步录的是 B1 的分开路径。这是第一轮评审抓到的:旧版本会静默丢掉半个 force_deep。

**选择规则(`auto`;v7-rc1 起默认是 0,即关)。** 一条链判定一次,全开或全关:2-D、恰好 2 个 slab、两个 slab 的初始自有粒子数都在 [400 000, 1 500 000] 内时开,否则全关。

- `compute_chain_partition` 把整条链的粒子数交给每个 slab,新字段是 `CaseV7.chain_own_particle_counts`;链外的 case 这个字段为空,auto 下为关。
- 每个 slab 打印判定;`band_overlap_record` 写进 step trace 的 run_meta、`canonical_dump` 的 meta 和 `fused_single_step` 的 JSON。

依据(chain bench,2-D,两张 5090,交错测量,开 − 关):

| K = 2,每 slab 自有粒子 | 实现者(均值 ± 标准差) | 评审复测(3 轮,± SE) |
|---|---|---|
| 62k(37k) | −3.8 % ± 1.5 % | |
| 250k(137k) | −4.7 % ± 0.4 % | |
| 1M(523k) | +5.7 % ± 0.3 % | +6.34 % ± 0.09 %;两个 slab 同在 GPU 1:+5.56 % ± 0.65 % |
| 2M(1.03M) | +2.5 % ± 0.06 % | +2.74 % ± 0.11 % |
| 4M(2.09M) | +0.3 % ± 0.4 %(噪声) | 强制开 +0.01 % ± 0.12 % |
| 16M(8.1M) | −1.4 % ± 0.3 % | |

- 窗口下方:phase B 去掉 force_deep 后盖不住传输链。例:250k K = 2 的 phase B 从 237 µs 降到 114–126 µs,而链长 185–192 µs,接收方每步都要等 upload。
- 窗口上方:配对藏不住多少。例:4M K = 2 配对 743 / 768 µs,串行 671 + 80 µs。

**为什么只开两个 slab 的链。** 本机上 K ≥ 3 只能让 slab 共用 GPU(device map 0,1,0 / 0,1,0,1)。在那里收益的正负随规模和权重变化,只看粒子数分不开(下面都是每个 slab 都开):

- K = 3 1M:−5.6 %;−4.3 %(1,1,1);−4.17 % ± 0.74 %(0.75,1,0.75);+2.59 % ± 0.14 %(0.5,1,0.5)。
- K = 3 2M:+3.7 %、+4.38 % ± 0.13 %、+4.80 % ± 0.24 %。
- K = 3 4M:三次分别 +1.3 %、+0.56 % ± 0.62 %、−0.56 % ± 0.08 %。
- K = 4 62k:+17.5 %。
- K = 4 2M:−7.32 % ± 0.28 %。phase B 从 595–634 µs 降到 290–308 µs,两端 slab 的 B → C 等待从 5–9 µs 升到约 1050 µs。
- K = 4 4M:−1.10 % ± 0.11 %。

`V7_BAND_OVERLAP=1` 可在 K ≥ 3 上强制开启。

**为什么全开或全关。** 第二轮评审发现,逐 slab 判定会让同一条链里 B3 与 B1 的 slab 混用。1M K = 3 用 GPU 均衡权重时,这比 B1 慢 2.1 %:B3 slab 等邻居的 upload 时,失去了 phase B 中原本藏住这段等待的 force_deep。两个 slab 跨在阈值两侧的链按 B1 跑(+0.04 %、−0.07 %);强制全开在那里虽有 +2.76 %、+0.99 %,但规则只用测过的窗口。

**数值。** 与 B1 构建逐位相同:没有任何 kernel 的输入、spec constant 或求和顺序改变。

**验证的边界。**

- 评审做了正对照:去掉 density 拷贝与 force 配对之间那道必需的屏障。结果 GPU 逐位门(n250 K = 2 999 步含跨切面、2-D 1M K = 2 200 步)仍然逐位相同;按归一化签名比较的 syncval 也看不出新签名;只有按 buffer 名称映射的 syncval 报出了真正的 RAW。所以"B3 没有竞争"靠的是逐元素依赖分析,逐位门证明不了。
- B3 开启时 syncval 新出现两类告警:段 1 对 L / kernel sum / scratch 的 RAW,段 2 对 acceleration / shift 的 WAW。两类都是整段 buffer 跟踪造成的、元素不相交的假阳性。

## tick 标签(step trace)

**B1 起。** 融合 slab 上,下面三个标签取代原来的 correction / density 标签:`b_correction_density_interior_end`、`c_correction_density_boundary_end`、`correction_density(_interior)_end`。以下工具已跟随:`bench_v7`、`phase_trace_v7`、`step_trace_model`、`e31_analysis`、`weight_calibration`、各 bench 的键表、opt / perf campaign 与 `opt_tables`。

**B3。** 标签不变,含义变了。B3 开启的 slab 上:

- phase B 在 `b_correction_density_interior_end` 结束(B3 的默认布局把 force_deep 全部移到 phase C,所以没有 `b_force_deep_interior_end`;B3 关的 slab,包括 v7-rc1 的默认运行,录的是 B1,仍有这个标签);
- `c_correction_density_boundary` 与 `c_force` 两段各含 force_deep 的一段。

所以 phase B / phase C 的时长不能与 B1 逐 kernel 对比,只能比 phase 与周期。run_meta 的 `band_overlap` 记录哪些 slab 跑了 B3。

## 验证

日志都在 worktree 的 `logs/e39/` 下(被 gitignore);v7 方腔运行在 `logs/validation/cavity_re1000_v7/`。本节所有运行都在本机两张 RTX 5090 上,驱动 576.88,验证层关(验证层与 syncval 两项除外)。

### 逐位门(E39 时的代码 22701d4,`logs/e39/final/bitwise`)

**构建的设置。** 每个构建 = 22701d4 的代码 + 开关。该构建新增的开关不设(取当时的代码默认),在前一个构建里设为 0,所以 `canonical_dump --compare` 不会看到同一开关被设成两个值。B3 例外:五个标准算例上 B1 不设 `V7_BAND_OVERLAP`(当时为 auto,这些算例上关),B3 设 1;2-D 1M K = 2 一对是 B1 设 0、B3 不设(当时为 auto)。v7-rc1 起 B3 的代码默认是 0,用 v7-rc1 的代码复现时,B3 构建要显式设 `V7_BAND_OVERLAP=1` 或 `=auto`,否则比的是 B1 对 B1。

**算例。** 全部用 canonical lists:

- n250(62k)K = 1 与 K = 2,200 步;
- n250 K = 2,999 步,`--transport-extension`,有 3 个粒子越过切口;
- 3-D 1M K = 1 与 K = 2,200 步。

| 一对构建 | K = 1 n250 | K = 2 n250 | K = 2 n250 999 步 | K = 1 3-D 1M | K = 2 3-D 1M |
|---|---|---|---|---|---|
| v6-rc2 → v7 全关 | 逐位 | 逐位 | 逐位 | 逐位 | 逐位 |
| 全关 → B9 | 逐位 | 逐位 | 逐位 | 逐位 | 逐位 |
| B9 → B6 | 逐位 | 逐位 | 逐位 | 逐位 | 逐位 |
| B6 → B4 | 逐位(2-D auto 关) | 逐位 | 逐位 | 逐位,屏蔽 280 231 行深壁 | 逐位,屏蔽 254 359 行深壁 |
| B4 → B1 | 不逐位(预期) | 不逐位 | 不逐位 | 不逐位 | 不逐位 |
| B1 → B3 | 逐位 | 逐位(强制开) | 逐位(强制开) | 逐位 | 逐位 |

- "屏蔽"指 `canonical_dump --compare` 的默认行为:B4 跳过的深壁行不比较 L 与 kernel sum,其他字段和其他行都比较。屏蔽只在一处起作用。用 `--no-mask` 重新比较:
  - B6 → B4 的 K = 1 3-D 有 280 231 行 L / kernel sum 不同,正好是被屏蔽的深壁行(K = 1 时被跳过的墙保留 0)。
  - 其余三对带屏蔽的比较(B6 → B4 K = 2 3-D,B1 → B3 两个 3-D)不屏蔽也逐位相同。K = 2 时深壁的 L 在 bootstrap 算过,墙不动,所以与每步重算的值相同。
- 3-D 的 200 步里没有粒子越过切口(K = 2 的 migration_install_count 为 0)。3-D 的跨切口门只在各项提交时做过,用 444 / 445 步(B4、B1 的门与 B6 评审)。
- 最终表没有 adami 算例;各项提交时的门含 adami K = 1 200 步。
- B3 在这五个算例上 auto 都关(n250 每个 slab 3.7 万粒子;K = 1;3-D),所以这一行的 B3 用 `V7_BAND_OVERLAP=1` 强制。强制只在 n250 K = 2 上真正开启;K = 1 与 3-D 在任何取值下都关,那三格比较的是同一种录制。
- auto 真正开启的情形另比一对:2-D 1M K = 2 200 步,B1(`V7_BAND_OVERLAP=0`)对当时的默认 auto(两个 slab 都开),逐位相同。改成 v7-rc1 的默认后又比一次:代码默认(两个 slab 都关)对 `V7_BAND_OVERLAP=auto`(都开),逐位相同(`logs/e39/rc1`)。
- B1 的单步检验见 B1 一节:12 种配置,评审另跑 7 种。

### 自复现(既有限制,`logs/e39/final/selfrepro`、`adami_determinism`)

用 canonical lists,999 步,GPU 1:

| 路径 | 运行次数 | 不同结果数 | 各结果的运行数 |
|---|---|---|---|
| v6-rc2 adami K = 1(n250) | 8 | 4 | 4 / 2 / 1 / 1 |
| v7 adami K = 1(n250) | 8 | 5 | 3 / 2 / 1 / 1 / 1 |
| v6-rc2 simple K = 1(n250) | 2 | 1 | 2 |
| v7 simple K = 1(n250) | 4 | 1 | 4 |
| v6-rc2 3-D 1M K = 1 | 2 | 2 | 1 / 1 |
| v7 3-D 1M K = 1 | 2 | 2 | 1 / 1 |

- simple K = 1 两行包括下面 K 不变性比较里同样设置的那次 K = 1 运行。
- adami K = 1 即使用 canonical lists 也不能自复现。这在 v6-rc2 已经如此,两者的频率相近:8 次中 v6-rc2 4 种结果,v7 5 种。
- 同步验证(shader 访问启发式)在 adami K = 1 的 300 步运行里,v6-rc2 与 v7 都没有报告;`wall_extrapolate` 只遍历规范化的列表,没有原子操作。原因没有定位。各项提交时的 adami 门因此用 200 步。
- 3-D 1M K = 1 在 999 步上任何构建都不能自复现。B6 评审中 3-D 1M K = 2 跑两次,到 445 步逐位相同,446 步起不同(`logs/e39/b6_review`),所以 3-D 的跨切口门用 444 / 445 步。
- K = 1 对 K = 2(n250,999 步,有粒子跨切口):v6-rc2 不逐位(加速度 11 行 ≤ 8 ULP,速度 4 行 1 ULP),v7 所有场逐位相同。B1 的融合核在这个算例上对 K 不变,v6-rc2 的分开 kernel 不是。

### 精度:集合检验(`logs/e39/ensemble`)

**设计。** 用 E33 的方法:

- 算例 2-D 1M 与 3-D 1M,K = 1,从 single_step 的 N = 2000 快照重启,开 δρ。
- v6-rc2 与 v7 各 6 次运行,GPU 交替。
- 重启后 300 步与 2000 步对每个粒子 dump。
- 统计量:66 对的逐粒子 rms 差;6 个场 × 2 个箱(全部流体、近壁)= 12 个检验。
- 检验:交叉、含 v7、组间项三种统计量,各自对 924 种重标做联合置换检验。

| 算例 | 重启后步数 | 交叉 p<0.05 / P(≥) | 含 v7 p<0.05 / P(≥) | 组间项 p<0.05 / P(≥) | 跨 / v6 内 | v7 内 / v6 内 | 组间项 D²/s₁² | 判定 |
|---|---|---|---|---|---|---|---|---|
| 2-D 1M | 300 | 6/12 / 0.028 | 0/12 / 1.000 | 6/12 / 0.026 | 1.000–1.022 | 0.976–1.042 | −0.034 … +0.064 | 系统偏移 |
| 2-D 1M | 2000 | 0/12 / 1.000 | 2/12 / 0.079 | 6/12 / 0.037 | 1.005–2.650 | 1.009–3.912 | −0.021 … +1.217 | 不可区分 |
| 3-D 1M | 300 | 0/12 / 1.000 | 0/12 / 1.000 | 0/12 / 1.000 | 0.969–1.000 | 0.934–1.000 | −0.002 … +0.031 | 不可区分 |
| 3-D 1M | 2000 | 0/12 / 1.000 | 0/12 / 1.000 | 0/12 / 1.000 | 1.015–1.227 | 1.092–1.298 | −0.202 … −0.084 | 不可区分 |

**2-D 1M,300 步:系统偏移。**

- 12 个检验的跨版本 / v6 内比值为 1.000–1.022;显著的 6 个(加速度、density、pressure,各两个箱)大 0.4–2.2 %。
- 组间项 D²/s₁² ≤ 0.064,即偏移约为一次 v6 运行起伏的 0.25 倍。
- 最强单检验是近壁加速度:p 0.004,族校正 0.028。
- 原来的解释:B1 每步确定性地改变 ρ 的舍入,运行还没有发散开时,这种确定性的差会被这个方法当作偏移检出;E33 用纯 K = 1 的打乱对照也见过这种短时刻偏移。但下面的独立复核组没有复现这个偏移,所以它更可能是偶然结果。

**2-D 1M,2000 步:按 E33 的判定规则不可区分,但有两个数要说明。**

1. 组间项 6/12 显著(P(≥) 0.037)。规则规定,交叉与含 v7 都不显著时不看组间项。
2. v7 内部的差是 v6 内部的 1.0–3.9 倍。density 与 pressure 的 3.90–3.91 刚好超过各自重标范围的 97.5 % 点(3.86–3.88)。"含 v7"统计量显著的 2 个检验是速度的两个箱(p 0.045),density 与 pressure 的 p 是 0.061。

逐对距离显示,运行在 2000 步时分成几个轨道族,对间速度 rms 从 9.5 × 10⁻⁶ 到 9.1 × 10⁻⁴,跨两个数量级:

- 主族:v6 的 t1、t2、t4、t5、t6 与 v7 的 t1、t2、t5,互相 1 × 10⁻⁵ 到 3 × 10⁻⁴。v7 的 t5 离 v6 的 t2 只有 2.0 × 10⁻⁵,与最近的一对 v6 运行(2.1 × 10⁻⁵)一样近。
- v6 的 t3 与 v7 的 t4、t6:t4 与 t6 互相 2.4 × 10⁻⁵;它们离 v6 的 t3 约 2.5 × 10⁻⁴,离主族约 5 × 10⁻⁴。
- v7 的 t3 单独一族:离所有运行 7.7–9.1 × 10⁻⁴。

v7 内部中位数大,是因为 6 次 v7 运行里有 3 次不在主族,v6 只有 1 次。这个计数本身不显著(Fisher 单侧 p ≈ 0.27),但 6 + 6 次分辨不了 v7 是否更常离开主族。

**复核:独立的第二组 6 + 6(`logs/e39/ensemble_rep`)。** 同一工具、同一快照、同样的代码摘要,12 次运行全部有效,不变量为 0。然后把两组合并成 12 + 12,用 20 000 次随机重标做置换检验(`experiment/seam_audit/e39_pooled_ensemble.py`;原工具逐一枚举重标,24 次运行有 270 万种,不可行)。

| 组 | 重启后步数 | 交叉 p<0.05 / P(≥) | 含 v7 p<0.05 / P(≥) | 组间项 p<0.05 / P(≥) | 跨 / v6 内 | v7 内 / v6 内 | 判定 |
|---|---|---|---|---|---|---|---|
| 复核组 | 300 | 0/12 / 1.000 | 0/12 / 1.000 | 0/12 / 1.000 | 0.993–1.006 | 0.986–1.058 | 不可区分 |
| 复核组 | 2000 | 0/12 / 1.000 | 0/12 / 1.000 | 2/12 / 0.132 | 0.959–1.786 | 0.987–4.583 | 不可区分 |

- **300 步的偏移没有复现。**
  - 复核组 12 个检验都不显著。
  - 两组合并的 24 次运行中,速度与 density(全部流体)的交叉检验 p 为 0.92 与 0.44,v7 内 / v6 内为 1.06 与 1.01。
  - 原组的"系统偏移"(P(≥) 0.028)应看作偶然结果,或分辨极限上的效应。
- **2000 步时 v7 的运行更常走到另外的轨道分支。**
  - 复核组单独看仍不显著:v7 内 / v6 内最大 4.6,在它自己的重标范围 0.18–5.4 内。
  - 两组合并后,速度与 density 的 v7 内 / v6 内为 3.7 与 4.5,单侧置换 p 为 0.036 与 0.028。
  - 这是看过原组之后才定的检验,p 值偏乐观。
- **分支是可重复的离散结果,不是随机散布。** 按对间速度距离(对数,平均连接)聚类,24 次运行分成三组:
  - 主分支:18 次(v6 11、v7 7),组内最大 3.0 × 10⁻⁴。
  - 分支 Y:v7 的 3 次(原组 t4、t6,复核组 t1),互相 ≤ 7.6 × 10⁻⁵,离主分支 4.1 × 10⁻⁴ 以上。v6 原组 t3 在 Y 与主分支之间:离 Y 2.1–2.6 × 10⁻⁴,离主分支至少 2.6 × 10⁻⁴。
  - 分支 X:v7 的 2 次(原组 t3,复核组 t2),是两组各自独立的运行,互相 3.1 × 10⁻⁵,离其他所有运行 ≥ 7.7 × 10⁻⁴。

  不在主分支的:v7 12 次中 5 次,v6 12 次中 1 次(Fisher 单侧 p ≈ 0.08)。
- **一个与数据相符、但没有直接验证的解释。**
  - 300 步时两个构建的运行间散布相同,说明非确定性的"种子"一样大。
  - B1 确定性地改变 ρ 的舍入,也就改变了基准轨道。从这个快照出发,v7 的基准轨道可能更接近某个离散事件(例如一次邻居或体素归属的判定)的分界,于是非确定性的扰动更常把 v7 的运行推到另一个分支。
  - 这只说明这一个快照;换一个快照,更接近分界的也可能是 v6。它也不说明 v7 的误差更大。物理量是否受影响,看下面的方腔对照(长时间平均)。

3-D 1M 两个时刻都不可区分,也没有单个检验突出。族校正的最小 p(交叉 / 含 v7):3-D 300 步 0.413 / 0.940,2000 步 0.606 / 0.245;2-D 2000 步 0.253 / 0.174。

**分辨力。** 合成偏差加在一个检验(速度 / 全部流体)上:在每个 v7 运行上加 rms 为 f × v6 内中位数的偏差。

- 按判定用的联合个数规则,f = 1 的系统偏差只在 2-D 300 步检出(那里真实的偏移已经显著)。其余三个情况检不出:一个场的偏差只改变 12 个检验中的 1 个。
- 按族校正的最小 p(最强单检验),f = 1 在四个情况都检出(交叉或含 v7 的 p ≤ 0.035)。f = 0.5 在 2-D 2000 步(0.162 / 0.141)与 3-D 2000 步(0.104 / 0.141)检不出。

### 精度:方腔 Re = 1000(`logs/e39/cavity`)

**运行。**

- 规格的 4 次 v7 运行(`logs/validation/cavity_re1000_v7`):250² 与 500² 各一次 adami K = 1、一次 simple K = 2,ξ = 0.001,ε² = 0.0025 h²,float32 ρ。都跑到稳态判据之后再 20 个时间单位(t ≥ 100)。
- 求解器代码:
  - 四次运行记录的 physics 哈希(`experiment/v7/utils/*.py`、SPIR-V、算例、材料库)与 22701d4 的文件内容重算一致(工作区里的 `simulator_v7.py` 是 LF 行尾),与 f646bf6、ee736f0 都不一致。
  - runner 与采样代码的哈希与三次新跑的 v6-rc2 运行相同。
  - 有的 v7 运行标 dirty(runner 检查 `experiment/v7` 与 `experiment/validation` 两个目录)。dirty 来自这些哈希以外的文件,日志里无法确定是哪一个。
- v6 参照:
  - adami 250²:E37 的 v6-rc2 运行(d62a1c0;从 d62a1c0 到 v6-rc2,`experiment/v6` 只改了 MANIFEST.txt)。
  - adami 500² 与 simple K = 2 的两个尺寸:本次新跑(`logs/e39/cavity_v6rc2`)。三次运行记录的 physics 哈希与 v6-rc2(c3514e5)的 `experiment/v6` 重算一致;比较输出的构建栏显示 worktree 当时的提交(22701d4 或 ee736f0)。
  - simple K = 2 另与 2026-10-04 的两次 v6 运行比较(`vulkan-demo/logs/validation/cavity_re1000`):
    - 记录为 HEAD 9cde618、dirty。9cde618 的 case loader 把 ε² 固定为 0.01 h²,从 case.yaml 读 ε² 是后来在 16c62a5(10-05)才提交的改动。
    - 两次运行的结果属于 ε² = 0.0025 h²:250² 的 u_min 是 −0.354702,本次 v6-rc2 是 −0.354687;同一 ξ、ε² = 0.01 h² 的运行是 −0.35667(`docs/validation/data/extrema.csv`)。所以当时带着这个未提交的改动;确切代码无法从记录的哈希复原。
    - 切口在第 27 / 52 列;v6-rc2 与 v7 都是 28 / 53。
- 比较工具 `experiment.validation.e39_compare`:
  - 每次运行用 v6 报告自己的分析代码:各自最后 20 个时间单位的平均(工具的默认;表中只有一行用共同窗口,已注明)、线性 MLS、对 Marchi 2021 T_c。
  - 差值为 v7 − v6;"噪声"行为两次参照运行之差。
  - σ = 两次运行各自的时间平均标准误差 std·√(τ_int/n) 的平方和合成。ψ_min 用窗口内每个快照的 ψ_min 序列。
  - 剖面距离 = 两条中线共 56 点的 ‖P₇ − P₆‖ / ‖P₆‖,噪声底按同样的标准误差合成。同一流动的两次独立运行,比值约为 1。

**σ 不一定包含全部运行间散布。**

- 生产运行的体素列表由原子追加建立,求和顺序可能每次不同,所以两次长跑是同一流动的两个独立实现。
- σ 只由单次运行窗口内的时间相关估计(τ_int 来自约 100 个样本或 20 个快照)。如果它完整,两次运行之差大多应在 2 σ 以内;噪声对里个别量的差却达到 4.9 σ(250² simple 的 ψ_min)与 2.6 σ(250² adami 的 L2(u)),逐点最大差 6.5。56 点剖面距离的比值是 0.78–1.14,所以低估只出现在个别量上。
- 表里的"噪声"行就是运行间差异的实测,两次运行都不含 E39 的改动:
  - adami:E36 的 adami_rho0 运行(`v7-wall-bc` 分支,WALL_BC=3)对 v6-rc2 的 adami。E37 把它移植为 adami 选项,在 n250 K = 1、200 步、canonical lists 上 11 个场逐位相同(`logs/e37/A/equivalence`;逐位只在 250² 上验证过)。
  - simple K = 2:9cde618 对 v6-rc2。中间的提交包括 ε² / ξ 的读取与默认值(算例显式给出 ξ、ε²,不改变这两次运行)、ghost 记录打包、band 宽度、切口位置、phase A 不等待等;两次运行不逐位相同。这与 v7 − v6-rc2 是同一类比较。

每格:差值(|差| / σ)。ΔL2 按相对误差的百分点计。

| 算例 | 比较 | Δψ_min(σ) | Δu_min(σ) | Δv_max(σ) | Δv_min(σ) | ΔL2(u)(σ) | ΔL2(v)(σ) | 剖面距离 / 噪声底(比) | 不变量 |
|---|---|---|---|---|---|---|---|---|---|
| 250² adami K = 1 | v7 − v6-rc2(E37 的运行) | +1.9e-05(1.9) | +6.8e-06(0.1) | -1.2e-05(0.2) | +8.4e-07(0.0) | -0.025 pp(3.9) | -0.009 pp(1.4) | 2.5e-04 / 2.3e-04(1.09) | 0 |
|  | 噪声:v6-rc2(E37)− E36 adami_rho0 | -7.1e-06(0.7) | -2.2e-05(0.3) | -3.1e-05(0.6) | +9.6e-06(0.1) | +0.021 pp(2.6) | +0.013 pp(1.9) | 2.9e-04 / 2.6e-04(1.14) | 0 |
|  | 另:v7 − E36 adami_rho0 | +1.2e-05(1.3) | -1.5e-05(0.3) | -4.3e-05(0.8) | +1.0e-05(0.1) | -0.004 pp(0.5) | +0.004 pp(0.7) | 1.8e-04 / 2.6e-04(0.68) | 0 |
| 500² adami K = 1 | v7 − v6-rc2(本次新跑) | +7.9e-06(1.0) | -5.7e-06(0.2) | -9.4e-06(0.4) | +2.4e-06(0.1) | +0.001 pp(0.2) | -0.007 pp(1.5) | 1.0e-04 / 1.1e-04(0.98) | 0 |
|  | 噪声:v6-rc2(本次新跑)− E36 adami_rho0 | -2.5e-06(0.3) | +1.5e-05(0.5) | -8.0e-06(0.3) | +5.6e-05(1.5) | -0.004 pp(1.0) | -0.005 pp(1.5) | 8.7e-05 / 1.1e-04(0.78) | 0 |
|  | 另:v7 − E36 adami_rho0 | +5.4e-06(0.7) | +8.9e-06(0.3) | -1.7e-05(0.7) | +5.9e-05(1.7) | -0.003 pp(0.9) | -0.012 pp(2.6) | 1.2e-04 / 1.1e-04(1.16) | 0 |
| 250² simple K = 2 | v7 − v6-rc2(本次新跑) | -1.5e-05(2.7) | +5.7e-05(0.9) | -3.6e-05(0.8) | +2.4e-06(0.0) | -0.001 pp(0.2) | -0.002 pp(0.2) | 3.3e-04 / 2.7e-04(1.19) | 0 |
|  | 同上,共同窗口(截止 t = 99.96) | -1.4e-05(2.3) | +5.2e-05(0.8) | -3.0e-05(0.7) | -2.0e-05(0.3) | -0.000 pp(0.0) | -0.001 pp(0.2) | 2.8e-04 / 2.7e-04(1.04) | 0 |
|  | 噪声:v6-rc2(本次新跑)− v6 9cde618 | +2.1e-05(4.9) | +1.5e-05(0.2) | -7.0e-05(1.6) | +4.8e-05(0.7) | +0.010 pp(2.3) | +0.007 pp(1.0) | 2.4e-04 / 2.6e-04(0.93) | 0 |
|  | 另:v7 − v6 9cde618 | +6.4e-06(1.2) | +7.2e-05(1.1) | -1.1e-04(2.6) | +5.0e-05(0.6) | +0.009 pp(1.9) | +0.005 pp(0.8) | 3.6e-04 / 2.7e-04(1.37) | 0 |
| 500² simple K = 2 | v7 − v6-rc2(本次新跑) | -2.5e-06(0.5) | +1.9e-05(0.7) | -1.6e-05(0.7) | -1.7e-05(0.5) | -0.002 pp(1.2) | -0.002 pp(0.6) | 9.3e-05 / 1.1e-04(0.82) | 0 |
|  | 噪声:v6-rc2(本次新跑)− v6 9cde618 | -3.4e-07(0.1) | -2.1e-05(0.7) | +2.3e-05(0.9) | +1.5e-05(0.5) | -0.002 pp(1.0) | +0.001 pp(0.2) | 1.1e-04 / 1.2e-04(0.92) | 0 |
|  | 另:v7 − v6 9cde618 | -2.8e-06(0.6) | -2.3e-06(0.1) | +7.6e-06(0.3) | -2.0e-06(0.1) | -0.005 pp(2.1) | -0.001 pp(0.4) | 1.1e-04 / 1.2e-04(0.90) | 0 |

**结论。**

- 规格要求的 4 个比较(v7 − v6-rc2)与同一算例的噪声对同量级。6 个量里最大的 |差| / σ:
  - 250² adami K = 1:3.9(L2(u),−0.025 pp);噪声对 2.6(同一个量,+0.021 pp)。
  - 500² adami K = 1:1.5(L2(v),1.53);噪声对 1.5。
  - 250² simple K = 2:2.7(ψ_min,−1.5 × 10⁻⁵;按共同窗口 2.3);噪声对 4.9(ψ_min,+2.1 × 10⁻⁵)。
  - 500² simple K = 2:1.2(L2(u));噪声对 1.0。
- 在 6 个量、剖面距离、动能和位置上,v7 − v6 与噪声对互有大小;56 点逐点最大差在四个算例里 v7 − v6 都比噪声对小:
  - 剖面距离 / 噪声底:v7 − v6 0.82–1.19,噪声对 0.78–1.14;
  - 56 点逐点最大差 / 合成标准误差:5.1 / 1.7 / 3.6 / 1.8,噪声对 6.5 / 2.0 / 3.8 / 2.0;
  - 流体动能:0.1–2.1 σ,噪声对 1.0–2.6 σ;
  - 极值位置差 ≤ 0.043 Δx,噪声对 ≤ 0.032 Δx。
- 250² adami 的 3.9 σ 是四个 v7 − v6 比较里最大的(本节最大的是噪声对 250² simple 的 ψ_min,4.9 σ)。它主要来自参照 E37 这次运行:
  - 同一窗口(80.64–100.59)上三次运行的 L2(u) 是 E36 3.454 %、v7 3.450 %、E37 3.474 %。
  - v7 − E36 的 L2(u) 只有 −0.004 pp(0.5 σ),6 个量最大 1.3 σ(ψ_min)。
  - 这也说明对 L2(u),σ 低估了运行间差异:不含 v7 的 E37 − E36 是 2.6 σ。
- 250² simple K = 2 的 v7 运行晚一个检查点判为稳态(t = 80.64,v6-rc2 是 70.56),平均窗口因此晚 0.63 个时间单位;其余三个比较的窗口相同。按共同窗口(80.01–99.96)重算:ψ_min 2.3 σ,其余 5 个量 ≤ 0.8 σ,剖面距离比 1.04。
- 绝对大小(按相对误差计):v7 − v6 在极值上最大 0.016 pp,在 L2 上最大 0.025 pp。逐量看,都比同一个量与 Marchi 2021 的偏差小 100 倍以上(最小是 250² adami 的 L2(u):0.025 pp 对 3.45 %,约 140 倍)。位置差不在此列。
- 另一参照的比较在同一量级:250² adami 对 E36 最大 1.3 σ(ψ_min);500² adami 对 E36 最大 2.6 σ(L2(v));simple K = 2 对 9cde618 最大 2.6 σ(250² v_max)与 2.1 σ(500² L2(u))。
- **局限。** 每个比较只有一对运行,噪声对也只有一对,而且噪声对与它对照的 v7 − v6 共用一次参照运行。能排除明显大于运行间差异的系统变化。与运行间差异同量级(250² 上 L2 约 0.01–0.02 pp、ψ_min 约 2 × 10⁻⁵)或更小的变化,分辨不了。
- **与集合检验的关系(旁证)。** 集合检验在 2-D 1M 的一个快照上看到,v7 更常走到另外的轨道分支。方腔长时间平均的量上,没有看到明显超出(每例一对)噪声运行之差的偏差;唯一比它的噪声对略大的一项(250² adami 的 L2(u)),在 v7 对另一次参照 E36 时只有 0.5 σ。两者的算例与规模不同,所以只是旁证。

### 性能(`logs/e39/perf_campaign`、`logs/e39/perf_trace`)

**测量设置。**

- chain bench,3 次试验。每次试验里,一个配置的各构建(含 v6-rc2)紧挨着各跑一次,顺序每次试验轮换一位,按试验配对。
- 这不是严格的 v6 / v7 一对一交替。一个构建与它配对的 v6-rc2 运行在顺序上最多相差 4 个位置,开始时间最多相差 0.3–0.9 分钟(2-D 62k、1M、adami)或 2.2–4.2 分钟(16M、3-D)。
- K = 1 总是两个进程同时跑(GPU 0 与 GPU 1),它们的均值就是 η 的同时参照。
- 每次运行前检查两张卡空闲(GPU 0 带桌面),运行中每秒记录 SM 时钟、功耗和温度。
- 开关不起作用的构建不跑,沿用前一个构建的结果:K = 1 的 B6、B3;2-D 的 B4;没有深壁候选的 3-D 8M 的 B4;auto 关的 B3。
- 144 个单元全部有效,没有不变量或来源违规。
- 测量时 B3 的代码默认是 auto(22701d4)。下面标"v7 默认"的列都是那时的默认。v7-rc1 改为 0,只影响 auto 开启的配置,这里只有 2-D 1M K = 2;它的 v7-rc1 默认就是 B1 构建:1 115.5 ± 9.2 fps,对 v6-rc2 +38.6 % ± 0.3,η 76.7 % ± 0.4(η_min 76.9 %)。

**v7 默认(测量时,B3 auto)对 v6-rc2**(fps,3 次试验的均值 ± 标准差;变化是逐次配对比值的均值 ± 标准差):

| 算例 | K | v6-rc2 | v7 默认 | 变化 |
|---|---|---|---|---|
| 2-D 62k(n250) | 1 | 2 997.5 ± 17.7 | 4 262.7 ± 19.8 | +42.2 % ± 0.8 |
| 2-D 62k(n250) | 2 | 1 718.2 ± 32.4 | 2 439.5 ± 35.0 | +42.0 % ± 0.7 |
| 2-D 1M | 1 | 545.1 ± 1.9 | 727.2 ± 2.7 | +33.4 % ± 0.1 |
| 2-D 1M | 2 | 804.7 ± 5.8 | 1 186.3 ± 6.5 | +47.4 % ± 0.5 |
| 2-D 1M,v7-rc1 默认(B3 关 = B1 构建) | 2 | 804.7 ± 5.8 | 1 115.5 ± 9.2 | +38.6 % ± 0.3 |
| 2-D 16M | 1 | 35.8 ± 0.1 | 47.4 ± 0.1 | +32.5 % ± 0.6 |
| 2-D 16M | 2 | 68.5 ± 0.1 | 90.5 ± 0.0 | +32.2 % ± 0.1 |
| 3-D 8M(4 层壁) | 1 | 13.1 ± 0.0 | 16.9 ± 0.1 | +29.0 % ± 0.8 |
| 3-D 8M(4 层壁) | 2 | 25.1 ± 0.1 | 32.9 ± 0.1 | +31.3 % ± 0.6 |
| 3-D 8M(4 层壁) | 4(0,1,0,1) | 20.4 ± 0.0 | 27.8 ± 0.1 | +36.2 % ± 0.6 |
| 3-D 1M(9 层壁) | 1 | 77.4 ± 0.1 | 105.9 ± 0.1 | +36.8 % ± 0.2 |
| 3-D 1M(9 层壁) | 2 | 110.8 ± 0.1 | 158.4 ± 0.6 | +43.0 % ± 0.4 |
| 2-D adami 250² | 1 | 2 505.3 ± 10.2 | 3 291.8 ± 11.0 | +31.4 % ± 0.1 |
| 2-D adami 1000² | 1 | 506.4 ± 1.1 | 658.4 ± 0.7 | +30.0 % ± 0.4 |

**逐级**(每个构建对前一个,逐次配对;"=" 表示开关在该配置不起作用,沿用前一个构建):

| 算例 | K | B9 | B6 | B4 | B1 | B3 |
|---|---|---|---|---|---|---|
| 2-D 62k | 1 | +0.4 % ± 0.5 | = | = | +41.7 % ± 0.4 | = |
| 2-D 62k | 2 | +0.8 % ± 0.3 | +5.6 % ± 0.2 | = | +33.4 % ± 0.2 | =(auto 关) |
| 2-D 1M | 1 | +1.5 % ± 0.2 | = | = | +31.5 % ± 0.2 | = |
| 2-D 1M | 2 | +0.8 % ± 0.4 | +3.9 % ± 0.5 | = | +32.4 % ± 0.2 | +6.3 % ± 0.3 |
| 2-D 16M | 1 | +1.2 % ± 0.5 | = | = | +30.9 % ± 0.7 | = |
| 2-D 16M | 2 | +0.8 % ± 0.1 | +0.1 % ± 0.6 | = | +31.1 % ± 0.8 | =(auto 关) |
| 3-D 8M | 1 | −0.2 % ± 0.6 | = | = | +29.2 % ± 0.1 | = |
| 3-D 8M | 2 | +0.3 % ± 0.4 | +0.6 % ± 0.1 | = | +30.1 % ± 0.4 | = |
| 3-D 8M | 4 | +0.4 % ± 0.3 | +1.5 % ± 0.3 | = | +33.6 % ± 0.7 | = |
| 3-D 1M | 1 | +0.3 % ± 0.2 | = | +5.4 % ± 0.3 | +29.4 % ± 0.3 | = |
| 3-D 1M | 2 | +0.4 % ± 0.7 | +1.6 % ± 0.6 | +3.3 % ± 0.3 | +35.7 % ± 0.8 | = |
| adami 250² | 1 | +0.4 % ± 0.4 | = | = | +30.9 % ± 0.4 | = |
| adami 1000² | 1 | +1.5 % ± 0.3 | = | = | +28.1 % ± 0.2 | = |

**本地效率。** η = fps_K / (G · 两张卡同时跑 K = 1 的平均 fps),G = 用到的 GPU 数 = 2。η_min 用较慢的那张卡;± 是逐次 η 的标准差;3 次试验。

| 算例 | K | η v6-rc2 | η v7 默认 | η_min v6-rc2 → v7 |
|---|---|---|---|---|
| 2-D 62k | 2 | 28.7 % ± 0.4 | 28.6 % ± 0.3 | 28.8 → 28.7 % |
| 2-D 1M | 2 | 73.8 % ± 0.3 | 81.6 % ± 0.2 | 74.2 → 81.7 % |
| 2-D 1M,v7-rc1 默认(B3 关) | 2 | 73.8 % ± 0.3 | 76.7 % ± 0.4 | 74.2 → 76.9 % |
| 2-D 16M | 2 | 95.7 % ± 0.2 | 95.5 % ± 0.2 | 95.9 → 95.8 % |
| 3-D 8M | 2 | 95.5 % ± 0.4 | 97.1 % ± 0.6 | 95.9 → 97.5 % |
| 3-D 8M | 4(0,1,0,1) | 77.7 % ± 0.1 | 82.0 % ± 0.6 | 78.1 → 82.3 % |
| 3-D 1M | 2 | 71.6 % ± 0.1 | 74.8 % ± 0.4 | 71.6 → 75.0 % |

η 逐构建(百分点,括号里是相对前一个构建的变化;"=" 是开关不起作用、沿用前一个构建):

| 算例 | K | v6-rc2 | +B9 | +B6 | +B4 | +B1 | +B3 |
|---|---|---|---|---|---|---|---|
| 2-D 62k | 2 | 28.7 | 28.8(+0.1) | 30.4(+1.6) | = | 28.6(−1.8) | = |
| 2-D 1M | 2 | 73.8 | 73.3(−0.5) | 76.2(+2.9) | = | 76.7(+0.5) | 81.6(+4.9) |
| 2-D 16M | 2 | 95.7 | 95.3(−0.4) | 95.4(+0.1) | = | 95.5(+0.1) | = |
| 3-D 8M | 2 | 95.5 | 95.9(+0.5) | 96.5(+0.6) | = | 97.1(+0.6) | = |
| 3-D 8M | 4 | 77.7 | 78.1(+0.4) | 79.3(+1.2) | = | 82.0(+2.7) | = |
| 3-D 1M | 2 | 71.6 | 71.7(+0.1) | 72.8(+1.1) | 71.4(−1.4) | 74.8(+3.4) | = |

- B6 在 K ≥ 2 的配置上提高 η +0.6 到 +2.9 个百分点,2-D 16M 除外(+0.1)。B3 开启时在 2-D 1M 上 +4.9 个百分点(v7-rc1 默认关)。
- B1 对 K ≥ 2 与 K = 1 的加速不同,所以它也改变 η:3-D 1M +3.4、K = 4 +2.7、3-D 8M K = 2 +0.6、2-D 1M +0.5、2-D 62k −1.8 个百分点。
- B4 只在 3-D 1M 起作用。它对 K = 1 的加速(+5.4 %)大于对 K = 2 的(+3.3 %),所以 η 降 1.4 个百分点。
- B9 在 −0.5 到 +0.5 个百分点之间。

**时钟**(协变量,逐次记录在 `summary.md`):

- **采样假象。** 2-D 62k 的 K = 1 运行只有 3–4 s,1 Hz 遥测在稳态窗口里只有 2 个样本。B1 的这组运行里有三个中位数含启动中的样本,是采样造成的,不是降频:
  - GPU 1 试验 1、2:1 578 / 1 624 MHz(例:225 与 2 932 MHz 的均值);
  - GPU 0 试验 1:2 846 MHz(2 752 与 2 940)。

  这几次的 fps 与其他试验相同(GPU 1 4 268 ± 7)。其余运行的时钟都在 2 692–2 962 MHz。
- **K 运行与参照的差。** K 运行的时钟一般与它的同时 K = 1 参照相同或更高。不计上面的假象:
  - K = 2:−7 到 +68 MHz(≤ 2.5 %)。
  - K = 4 共用:v6-rc2 +83 到 +128 MHz(3–5 %),v7 +30 到 +67 MHz(1–2.4 %)。

  原因没有逐一确认:2-D 16M 与 3-D 8M 的 K = 2 和它们的 K = 1 参照都在 600 W 上限,时钟仍高 0–67 MHz。
- **对 η 的影响。** η 没有按时钟校正。K = 4 的 η 两边都可能偏高,v6-rc2 偏得更多,所以 77.7 → 82.0 % 的提升可能被低估。v7 的 K = 1 参照本身时钟就更高。

**step trace**(各一条,v6-rc2 对测量时的 v7 默认;2-D 1M K = 2 的 v7 一列是 B3 开。v7-rc1 默认(= B1 构建)在这次 trace 里没有另测;B3 开发时有一条同样录制的 B1 trace(`logs/e39/b3/traces/b1_1m`,f646bf6,1 108 fps,周期约 865 / 872 µs),不与这里的 v6-rc2 配对;µs,稳态完整步的中位数,s0 / s1):

| 量 | 2-D 1M K = 2:v6-rc2(803.6 fps) | v7 默认(1 181.8 fps,B3 开) | 3-D 8M K = 2:v6-rc2(25.1 fps) | v7 默认(32.9 fps) |
|---|---|---|---|---|
| 周期 p50 | 1 216.6 / 1 216.6 | 810.1 / 811.7 | 39 400 / 39 431 | 30 107 / 30 163 |
| phase A | 65.2 / 73.4 | 28.4 / 30.0 | 536 / 571 | 403 / 409 |
| 其中 ghost_send | 43.0 / 51.2 | 6.2 / 6.2 | 145 / 176 | 12.3 / 12.3 |
| phase B | 892.3 / 902.2 | 320.5 / 324.9 | 33 940 / 35 514 | 26 139 / 27 389 |
| 其中 correction + density | 552.3 / 560.3 | 320.5 / 324.9 | 21 091 / 22 146 | 13 355 / 13 974 |
| 其中 force_deep | 340.0 / 342.0 | 0(移到 phase C) | 12 819 / 13 370 | 12 762 / 13 412 |
| B → C 等待 | 4.6 / 4.6 | 4.7 / 4.7 | 1 595 / 4.7 | 1 014 / 5.2 |
| phase C | 232.8 / 224.4 | 435.8 / 441.3 | 3 301 / 3 324 | 2 347 / 2 345 |
| t_chain p50(E29) | 195.2 / 195.9 | 193.3 / 193.6 | 2 060 / 2 048 | 2 043 / 2 032 |
| 传输暴露的步数比例 | 0.027 / 0.074 | 0.044 / 0.101 | 0.000 / 0.795 | 0.000 / 0.726 |

- 2-D 1M:传输链本身不变。B3 把 phase B 缩到约 320 µs,离约 194 µs 的链更近,所以暴露的步数比例略升;净效果仍是周期 −33 %。
- 3-D 8M:s0 的 B → C 等待是两个 slab 的不均衡(s1 的 phase B 长 1.3–1.6 ms),不是传输,v7 缩短了它。

### 不变量与验证层

- 不变量(drift、每个 overflow 计数器、远迁移、GPU / 主机帧戳)在本次每一次运行里都是 0。覆盖:逐位门 37 个 dump 与 v7-rc1 复核的 2 个 dump(`logs/e39/rc1`)、自复现 28 个 dump 与 2 次 syncval 运行、集合检验 24 次与复核 12 次、性能活动 144 个单元(210 个进程)、方腔运行、验证层 4 次运行(见下)。
- Khronos 验证层(默认设置),2000 步:
  - E39 时的代码默认(`logs/e39/final/validation_layer`):K = 2 2-D 1M(0,1;B3 两个 slab 都开)0 条消息;K = 4 3-D 8M(0,1,0,1)0 条消息。
  - v7-rc1 的代码默认(`logs/e39/rc1/validation_layer`):K = 2 2-D 1M(B3 关)0 条消息;K = 4 3-D 8M 0 条消息。
  - 四次都是 drift 0、无溢出。
- 同步验证(shader 访问启发式)在每项的评审里都跑过;B3 的结果见 B3 一节的"验证的边界"。

## 已知限制与后续

- **B1:** compact band 派发总是回退,因为 `band_compact` 只支持 2,3,4。若要融合:只在开关为 1 时允许 compact + 2,2,3;新建 `correction_density_boundary_compact`(mode 2,dispatch 2);phase C 用 meta[0] 间接派发;force 继续用 meta[2](按 4 列算,对 f = 3 只是多派发)。
- **B1:** 单步 0 ULP 依赖驱动的 FMA 合并,见 B1 一节。
- **B6:** barrier 只有文本检查守护,见 B6 一节。
- **B4:**
  - 深壁的 L / kernel sum 是旧值,见 B4 一节;
  - `auto` 的 1 % 阈值只基于 3 个算例;
  - restart 路径只有代码层面的核对;
  - 调试计数器是 uint32,长跑可能回绕。
- **B3:**
  - v7-rc1 默认关;下面几条在设 `V7_BAND_OVERLAP=auto` 或 1 时才相关。
  - 没有竞争只由逐元素依赖分析保证:逐位门抓不到漏掉的屏障,见 B3 一节。以后改动 phase C 的录制,要重做依赖表,并用按 buffer 名映射的 syncval 比较。
  - `vkCmdDispatchBase` 的异常未定位。
  - auto 窗口只在本机标定(两张 5090,K = 2,等权重)。跨阈值的链和 K ≥ 3 按 B1 跑;集群上 K ≥ 3 用不同的 GPU,可能有收益,需要实测。
  - `--weights auto` 的 pilot 用等权重的粒子数做链判定,正式链用标定后的权重。阈值附近两者可能判定不同,这时 pilot 测到的 T_B / T_C 来自另一种录制。
  - 2-D 1M K = 2 上 phase B 变短后,传输暴露的步数比例从 0.03 / 0.07 升到 0.04 / 0.10。传输更慢的机器(集群的主机拷贝)上,窗口可能要重新标定。
- **精度:**
  - 2-D 1M 2000 步,从同一个快照出发的 v7 运行更常走到两个可重复的另外轨道分支(合并 12 + 12 的事后检验,单侧 p ≈ 0.03)。只有一个快照,不能推广;要做一般的结论,需要从几个快照出发重复。
  - 方腔每个比较只有一对 v7 / v6 运行和一对噪声运行,与运行间差异同量级的系统变化分辨不了。要分辨,每个算例需要几次运行(生产运行本身不逐位复现,重跑即是独立样本)。
- **既有问题,v7 没有改变:**
  - 求解器没有任何派发检查 `maxComputeWorkGroupCount`(64M 时逐粒子派发已超过 65 535 个工作组);
  - adami K = 1 与 cavity3d_1m K = 1 在 999 步上不能自复现(见"自复现")。
- **集群:** v7 没有在集群上跑过。`E30_SOLVER=v7` 只做过 CPU 侧的检查;v6 的 `bringup_check` 清单不含 E37 的 `wall_extrapolate`,这一点保持不变。

## 复现

在 worktree 根目录执行;`python` = `vulkan-demo/.venv/Scripts/python.exe`。每次运行前先确认两张卡空闲。

```bash
# 逐位门:一对构建(例:B9 → B6,n250 K = 2 200 步);各构建的开关组合见"逐位门"一节
python experiment/seam_audit/canonical_dump.py --solver v7 --case cases/lid_driven_cavity_2d_n250_xi0p001_eps0p0025/case.yaml \
    --device-map 0,1 --steps 200 --canonical-lists \
    --env V7_GHOST_SEND_LANES=0 --env V7_DEEP_WALL_SKIP=0 --env V7_FUSED_CORRECTION_DENSITY=0 --env V7_BAND_OVERLAP=0 --out b9.npz
python experiment/seam_audit/canonical_dump.py --solver v7 --case cases/lid_driven_cavity_2d_n250_xi0p001_eps0p0025/case.yaml \
    --device-map 0,1 --steps 200 --canonical-lists \
    --env V7_DEEP_WALL_SKIP=0 --env V7_FUSED_CORRECTION_DENSITY=0 --env V7_BAND_OVERLAP=0 --out b6.npz
python -m experiment.seam_audit.canonical_dump --compare b9.npz b6.npz
# 跨切口:--steps 999 --transport-extension;3-D 1M K = 2 另加
#   --env V7_GHOST_POOL_FACTOR=0.5 --env V7_MIGRANT_POOL_FACTOR=0.02 --env V7_DEPARTED_FACE_FRACTION=0.64

# B1 单步检验(K = 1 2-D 1M;K = 2 用 --device-map 0,1,其余配置见脚本的文档字符串)
python experiment/seam_audit/fused_single_step.py --case cases/lid_driven_cavity_2d_gen/case.yaml \
    --state ../vulkan-demo/logs/seam_audit/single_step/cavity2d_1m/snapshot_N2000.npz --device-map 1 --out k1_2d1m.json

# 集合检验(24 次运行,约 10 min)
python -m experiment.seam_audit.e39_ensemble run --out logs/e39/ensemble \
    --snapshot-root ../vulkan-demo/logs/seam_audit/single_step
python -m experiment.seam_audit.e39_ensemble analyze --out logs/e39/ensemble
# 复核组(只有 2-D 1M,12 次运行,约 3 min)与两组合并的检验
python -m experiment.seam_audit.e39_ensemble run --out logs/e39/ensemble_rep --cases cavity2d_1m     --snapshot-root ../vulkan-demo/logs/seam_audit/single_step
python -m experiment.seam_audit.e39_ensemble analyze --out logs/e39/ensemble_rep --cases cavity2d_1m
python -m experiment.seam_audit.e39_pooled_ensemble --horizon 2000   # 以及 --horizon 300

# 方腔:v7 运行(adami K = 1 用 cavity_runner 并加 --slabs 1;simple K = 2 用 campaign),再对 v6 参照比较
python -m experiment.validation.cavity_campaign run --solver v7 \
    --runs n250_k2_float32_xi0p001_eps0p0025,n500_k2_float32_xi0p001_eps0p0025
python -m experiment.validation.e39_compare --v7 <v7 运行目录> --v6 <v6 运行目录> --out logs/e39/cavity/<名字>
#   共同窗口:加 --window-end common;各次调用见 logs/e39/final/run_compare*.sh

# 性能活动(144 个单元,约 1.8 h)、汇总、step trace(约 2 min)
python -m experiment.seam_audit.e39_perf_campaign run --out logs/e39/perf_campaign   # b3 构建显式设 V7_BAND_OVERLAP=auto(v7-rc1 起代码默认为 0)
python -m experiment.seam_audit.e39_perf_campaign summarize --out logs/e39/perf_campaign
python -m experiment.seam_audit.e39_perf_campaign trace --out logs/e39/perf_trace

# 验证层(不设 VK_LOADER_LAYERS_DISABLE)。下面两条是 v7-rc1 的默认(logs/e39/rc1/validation_layer);
# 复现 E39 时的默认(logs/e39/final/validation_layer,2-D 1M K = 2 上 B3 两个 slab 都开):第一条前加 V7_BAND_OVERLAP=auto
python experiment/v7/_run_v7_chain_bench.py --validation --case cases/lid_driven_cavity_2d_gen/case.yaml \
    --weights 1,1 --device-map 0,1 --max-steps 2000 --warmup 200
python experiment/v7/_run_v7_chain_bench.py --validation --case cases/cavity3d_weak4_k2_8m_b4/case.yaml \
    --weights 1,1,1,1 --device-map 0,1,0,1 --max-steps 2000 --warmup 200

# SPIR-V:重编并检查与跟踪的文件逐字节相同
python experiment/v7/compile_shaders_v7.py
python experiment/v7/_test_seam_layout.py
```

## 提交(分支 `v7-perf`,从 v6-rc2 = c3514e5 开出;附注标签 `v7-rc1` 在最后一个提交上)

| 提交 | 内容 |
|---|---|
| 0fc247a | 复制 experiment/v6(v6-rc2)→ experiment/v7 |
| c02cc4c | seam_audit 驱动 v7(solver_adapter、`canonical_dump --solver`) |
| d6a9ecb | B9 |
| 51faaf8 | 工具:`cavity_runner` / `cavity_campaign --solver v7`、`e39_compare` |
| 3b756d1 | 集群脚本的 `E30_SOLVER=v7`、`dump_state` v7、`e39_ensemble` |
| 3d6e1c9 | B6 |
| bf91694 | B4 |
| f646bf6 | B1 |
| 22701d4 | B3 |
| ee736f0 | `V7_DENSITY_COPY_COMPUTE` 严格解析(只认 0 / 1,其他值报错;报告复核发现)|
| 8f8cbd9 | 本文档、SPIR-V MANIFEST(22 个文件)、`e39_perf_campaign.py`、`e39_pooled_ensemble.py`、`bringup_check_v6.py` 的按求解器清单 |
| 本提交(`v7-rc1`) | B3 默认关:`simulator_v7` 的 `V7_BAND_OVERLAP` 默认 0、`_test_seam_layout` 的默认检查、`e39_perf_campaign` 的 b3 构建显式设 auto(规划探针按 auto 规则判定);本文档 |
