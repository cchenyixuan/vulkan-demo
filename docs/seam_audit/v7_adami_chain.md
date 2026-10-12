# E40:Adami 壁面支持多卡(v7-rc3,2026-10-13)

基于 v7-rc2(c56d078,分支 `v7-adami-multi`,worktree `vulkan-demo-v7-rc3`)。此前 `numerics.wall_boundary: adami` 在 slab 数大于 1 时拒绝运行(E37,`v6_opt.md`"壁面选项");本次让它在 K > 1 下运行,并满足:接缝两侧每个壁面粒子的外推输入与时间层与单卡完全相同、默认壁面(simple)在 K = 1 与 K > 1 下与 rc2 逐位相同、adami 的 K = 1 路径与 rc2 逐位相同。每步交换次数不变(仍是一次)。

## 结论

1. **做法:** adami 链的 ghost 加厚到 3 列(G1、G2、G3 = 邻居的 column 0、1、2),correction_density 的 band pass 把 G1 **和 G2** 当作 self 重算(spec 85 `GHOST_SELF_LAYER` = 2),壁面 pass 像 density 一样拆成 phase B 的 interior 变体(wall band 之外的 own 壁粒子,读 scratch)和 phase C 的 band 变体(wall band 内的 own 壁粒子 + G1 当作 self,读拷贝后的 primary);force 的 band 从 3 列加到 4 列。于是 own column 0 的 force 读到的 G1 壁粒子 (ρ0, p_w, u_dummy) 由本卡从本步的 G2、G1、own 0 流体 ρⁿ⁺¹、Pⁿ⁺¹、v^{n+½}、xⁿ⁺¹ 算出,与单卡的输入逐项相同;G2 的 ρⁿ⁺¹ 来自 G3、G2、G1 的 xⁿ⁺¹、v^{n+½}、ρⁿ,也与单卡相同。simple 链与 K = 1 的 adami 不走这些路径,布局与 rc2 相同。
2. **逐位:** 1000² 主序列算例(simple 兄弟算例与 adami 算例),canonical-list 规程,rc2 checkout 对本 checkout:simple K = 1(200 步)、K = 2(200、500 步)、adami K = 1(200、300、400、500 步)全部 BIT-IDENTICAL;adami K = 2 同一构建两次(200 步)BIT-IDENTICAL。K = 1 的 adami 管线用的是 **rc2 的 `wall_extrapolate.comp` 原文件**(SPIR-V 逐字节相同),所以 K = 1 的逐位相同不依赖规程的窗口。规程自身的确定性窗口在 400–500 步之间结束(999 步时同一构建的两次运行也不同;见"逐位门禁")。
3. **接缝(逐粒子):** adami 1000²,K = 2(切口在第 103 列,竖直中线附近)对 K = 1,N = 300 与 900(第一次 defrag 之前),每种两次运行,`dump_state` + `analyze`:按到接缝距离分列的 test / noise 比值在 0.8–1.2 之间,与 K = 1 自身的逐次噪声同量级;N = 300 的第一对在 d = 2–4 列有 1.3–1.7 的比值,但同一处 K = 2 两次运行之间的噪声也是 1.1–1.4,第二对在同一列为 1.0–1.2,N = 900 两对都在 1.0–1.2。壁面压力 p_w(含壁粒子的比较)同样在噪声内。
4. **代价(1000²,chain bench,三次交替):** K = 1 每步时间 adami / simple = 1.106(+10.6 %),K = 2 = 1.191(+19.1 %);K = 2 下 adami 每步多 ≈ 155 µs(每卡,step trace 中位数):phase B 多 88 µs(interior 壁面 pass 57 µs + force_deep 多 39 µs),phase C 多 68 µs(新增的 wall band pass 51 µs + force band 多 18 µs)。phase A(ghost_send 多发一列)与 correction_density 的 band pass(多算一列 self)没有可测的增量。
5. **包:** 每方向 ghost 池 [G1 R | G2 R | G3 R | migrants M](rc2:两层),R、M 不变;每条链每帧 DMA 428.5 → 639.2 KiB(+210.7 KiB = 第三层 replica 的打包记录 6,691 × 32 B + 该列的 voxel 表 + 一个计数字),主机拷贝 322.6 → 484.5 KiB。
6. **band 宽度:** simple 不变(2/2/3);adami 且有邻居的 slab:correction / density 2、**wall 3**(新)、**force 4**(原 3)。

## 设计

记号沿用 `v6_design.md`:column 0 = 紧邻 seam 的 own 列,G1 / G2 / G3 = 邻居的 column 0 / 1 / 2,band c / d / f = correction / density / force 的 band 宽度(own 列数)。

**为什么要第三层。** 壁面 pass 读壁粒子流体邻居的 ρⁿ⁺¹、Pⁿ⁺¹(本步 density 的输出)。own column 0 的 force 要读 G1 壁粒子的 (p_w, u_dummy);它们必须用 G1 壁粒子的流体邻居(G2、G1、own 0)的 ρⁿ⁺¹ 算——G1 的由 rc2 的 band pass(G1 as self)给出,own 0 的由 band density 给出,G2 的在 rc2 里没有(G2 只作为 G1 的邻居被读,保留上传来的 ρⁿ)。要在本卡算出 G2 的 ρⁿ⁺¹,就要 G2 的邻居 G3 的 xⁿ⁺¹、v^{n+½}、ρⁿ、material——第三层 replica。一次交换就够:三层都是 phase A 打包的本步 predict 之后的状态。另一种做法(邻居在它的 phase C 算完 column 0 的壁面再传一次)要在每步加一次交换并把两卡的 phase C 串起来,没有采用。

**时间层。** 与 density 在接缝处的处理同一原则:接收方只用 xⁿ⁺¹、v^{n+½}、ρⁿ(传来的)和本卡刚算出的 ρⁿ⁺¹、Pⁿ⁺¹ 重算,每一项与单卡算同一个壁粒子时读到的相同,唯一差别是求和顺序。

**一步内的顺序(有邻居的 adami slab):**

| 阶段 | kernel | 改动 |
|---|---|---|
| A | predict → update_voxel → ghost_send(三层 replica + migrant) | ghost_send 多发 column 2;三个 replica 计数 |
| B | correction_density_interior(band 2)→ **wall_extrapolate_interior**(wall band 3 之外,读 scratch)→ force_deep_interior_scratch(band 4) | 壁面 pass 的 interior 变体;force band 4 |
| C | expand → install → append_departed → correction_density_boundary_band(own 2 列 + G1、G2 as self)→ scratch→primary 拷贝(own 区间 + 每方向 [G1 R | G2 R] + departed)→ **wall_extrapolate_boundary_band**(own 3 列 + G1 as self,读 primary)→ force_boundary_band(own 4 列) | self 层 2;拷贝多一个区段;新增 wall band pass;force band 4 |
| bootstrap | 两次壁面 pass 各加一次 band 变体(G1 as self) | |

band 宽度的依赖:column c 的壁粒子读 column c ± 1 的流体 ρⁿ⁺¹;phase B 只有 column ≥ d 的 ρⁿ⁺¹,所以 wall band w = d + 1 = 3;column c 的流体 force 读 column c ± 1 壁粒子的 p_w,phase B 只有 column ≥ w 的,所以 f = w + 1 = 4。`partition_v7.configured_band_widths(wall_pass=True)` 的默认是 2,2,4,并要求 f ≥ d + 2。

**不变的部分。** migrant 仍只来自 G1 列的 incoming 表(far_migration 对 G2、G3 列计数,必须为 0);departed 池、install、defrag、G1 as self、拷贝的语义都不变;`V7_BAND_COMPACT_DISPATCH`(它的 band 表只有一层 inner ghost)与 `V7_BAND_OVERLAP`(adami 本来就关)不支持 adami 链,启动时拒绝;`V7_DIAG_GHOST_SELF` 必须同时包含 correction 与 density。

## 改动的文件

| 文件 | 改动 |
|---|---|
| `shaders/common.glsl` | `GHOST_LAYERS`、`GHOST_SELF_LAYER`、`REPLICA_REGION_SIZE` 的注释;GlobalStatus 的 4 个保留字段改为第三层 replica 的发送 / 接收计数(字段 35–38,布局与大小不变) |
| `shaders/helpers.glsl` | `packed_layer_base` = (direction · 8 · GHOST_LAYERS + layer · 8) · R;`migrant_region_offset` = GHOST_LAYERS · R(两层时与原式相同) |
| `shaders/ghost_send.comp`、`ghost_send_lanes.comp` | replica 与 migrant 循环到 GHOST_LAYERS;outbox 列 boundary ± (GHOST_LAYERS − layer);第三层计数器;far_migration 对 layer ≥ 1 计数 |
| `shaders/expand_ghost_lists.comp` | leading 列的 layer = GHOST_LAYERS − 1 − column |
| `shaders/wall_extrapolate_band.comp`(新) | 壁面 pass 的拆分变体:correction 的 spec 常量作模式(47)、wall band(82)、密度来源(56)、band 派发(57/58/59)、self 层(85);`wall_extrapolate.comp` **原样保留**给无邻居的 slab |
| `shaders/wall_boundary.glsl` | 注释 |
| `shaders/spv/` | 改变的 SPIR-V:expand_ghost_lists 4,984 → 5,068 B,ghost_send 18,744 → 19,140 B,ghost_send_lanes 21,272 → 21,668 B,install_migrations 7,600 → 7,600 B(内容变,经 helpers 的 `migrant_region_offset`);新增 wall_extrapolate_band 24,704 B;`wall_extrapolate.comp.spv`(8,216 B)与 rc2 逐字节相同;其余 15 个模块逐字节相同 |
| `utils/partition_v7.py` | `chain_ghost_layers`(adami 且 K > 1 → 3,否则配置值;强制 `V7_GHOST_LAYERS=1` 时报错)、`ADAMI_CHAIN_GHOST_LAYERS`、`ADAMI_CHAIN_BAND_WIDTHS`、`_ghost_pool_layout` 的 GHOST_LAYERS · R + M、`configured_band_widths(wall_pass)`;去掉 K > 1 的拒绝 |
| `utils/simulator_v7.py` | 第三层计数的偏移与字段名;三区段传输段表与计数字;`_ghost_self_layer` = 层数 − 1;拷贝区段 [G1 \| G2];install 线程数与 lane 组数按层数;phase A 清零三个计数;`wall_extrapolate_interior` / `_boundary_band` 管线(`_wall_mode_entries`)与 `_record_wall_band_pass`;phase B / C / bootstrap 的记录;构造时的检查(层数 3、band-voxel 派发、无 compact、self kernels) |
| `utils/bench_v7.py` | phase C 的壁面 tick 在 phase B 已有 `wall_extrapolate_us` 时记为 `wall_extrapolate_band_us` |
| `_run_v7_chain_bench.py`、`_run_v7_soak.py`、`experiment/validation/cavity_runner.py` | 去掉 K > 1 的拒绝(runner 对 v6 求解器保留) |
| `_test_seam_layout.py` | 新检查 `check_adami_chain`(见"门禁");旧检查里 ghost_send 的文本模式与 SPIR-V 钉子按改动更新 |
| `experiment/seam_audit/adami_chain_gates.py`(新) | 本文的 GPU 门禁驱动:bitexact / seam / cost / trace |
| `experiment/validation/cavity_campaign.py` | 两个 adami 1000² 运行(K = 2、K = 1)与它们的校准项 |
| `shaders/README.md` | 两个壁面 kernel 的条目 |

**为什么壁面 kernel 是两个文件。** 第一版把拆分写进 `wall_extrapolate.comp` 本身(模式 0 = 原 kernel 的路径,其余折叠掉)。它对 rc2 的逐位比较在 200、300、400 步相同,500 步时一对运行不同(每个流体粒子的 ρ 差 1 ULP,位置相同),把 rc2 的 SPIR-V 换进本构建时又相同;但随后发现 500 步已在规程的确定性窗口之外(同一构建两次 500 步在三对里有一对不同),这个差别无法归因。K = 1 的逐位相同不该依赖规程的窗口,所以 K = 1 的管线保留 rc2 的源文件与 SPIR-V(逐字节相同,由构造保证),拆分变体另放一个文件。

## 门禁

### CPU

- `_test_seam_layout.py`:ALL PASS。新检查 `check_adami_chain`:分区(adami 链 ghost 层 3,simple 与 K = 1 的 adami 为 2;池 = 3 R + M 且 R、M、own / departed 池、切口与 simple 链相同;每个有邻居的一侧 3 列 ghost;每条链路的 voxel id 偏移把发送方边界列映到接收方最内层 ghost 列;`V7_GHOST_LAYERS=1` 下拒绝;band 默认 2,2,4、拒绝 2,2,3)、每个 slab 的假模拟器(2-D K = 2 / 3,3-D K = 2:band 宽度、两个拆分管线的 spec 条目与模块、fused band 的 self 层 2、force band 4、传输段表(inner / outer / third / migrant 区段铺满池、打包块、四个计数字与最后的帧戳、staging 连续)、install 线程数 = migrant 区段、lane 组 = 3 × face、拷贝区段、phase A / B / C / bootstrap 的记录流;simple K = 2 的 slab 不记录壁面 pass 且保持 2,2,3 / self 层 1 / 2 × face)、三层下的 lane 组映射、shader 文本(各处 GHOST_LAYERS 形式、第三层计数在 GlobalStatus 第 35–38 字段 = 模拟器的偏移、`wall_extrapolate.comp` 与 v7-rc2 的相同)。E39 各项的"开关 0 = 旧构建 SPIR-V"钉子对 E40 改变的 4 个模块跳过(它们的逐位证据是下面的 GPU 门禁),`wall_extrapolate.comp.spv` 照旧钉住。
- `_test_chain_cuts.py` 6,772 项、`_test_partition_chain.py` 2,950 项通过。

### 逐位(`adami_chain_gates.py bitexact`,`canonical_dump --solver v7 --canonical-lists`,rc2 checkout = `vulkan-demo-v7-wake` c56d078,本 checkout;K = 1 在 GPU 1,K = 2 在 GPU 0、1;K = 2 两边都开 `--transport-extension`)

| 比较 | 算例 | K | 步数 | 结果 |
|---|---|---|---|---|
| rc3 对 rc2 | simple(`..._eps0p0025`) | 1 | 200 | BIT-IDENTICAL(10 字段 × 1,046,529) |
| rc3 对 rc2 | simple | 2 | 200 | BIT-IDENTICAL |
| rc3 对 rc2 | simple | 2 | 500 | BIT-IDENTICAL |
| rc3 对 rc2 | adami | 1 | 200 / 300 / 400 / 500 | BIT-IDENTICAL(11 字段,含 wall_dummy_velocity) |
| rc3 对 rc3(自检) | adami | 2 | 200 | BIT-IDENTICAL |
| rc3 对 rc2 | simple / adami | 2 / 1 | 999 | 不同(见下) |
| rc3 对 rc3、rc2 对 rc2(自检) | simple / adami | 2 / 1 | 999 | 不同 |
| rc3 对 rc3(自检) | adami | 1 | 500 | 三对里一对不同 |

- 999 步时同一构建的两次运行也不同(每个字段约 98 % 的粒子,ρ 差 1 ULP 量级、位置差 1–4 ULP):规程只把初始列表排序,之后粒子换 voxel 的原子追加在同一步两个粒子进同一 voxel 时仍可能对调,差别随后经邻居求和传遍全场。1000² 算例的确定性窗口在 400–500 步之间结束(E37 用 250² 算例做到 999 步)。规程内的比较全部相同;K = 1 的 adami 还由 SPIR-V 逐字节相同保证。
- 3-D 没有做:`cavity3d_1m` 在 canonical 规程下(跳过 bootstrap 的 defrag)K = 2 在 rc2 与本构建上同样溢出(200 步 177k 次迁移、ghost 池与 install 尾部溢出),无法比较;3-D 的布局由 CPU 检查覆盖,kernel 改动与维度无关。
- 200 步内没有粒子越过切口;500 与 999 步的 K = 2 运行有 8 个粒子越过(两边都带全局 id)。

### 接缝(`adami_chain_gates.py seam`:`dump_state --version v7`,adami 1000²,K = 1 两次(A1、A2,GPU 1)与 K = 2 两次(B1、B2,GPU 0、1),horizon 300 与 900,`analyze` 的 KD-tree 匹配,全部 id 一致率 1.0,不变量全 0)

比值 = rms|B − A| / rms|A1 − A2|,同一批粒子;d = 到接缝的列数(d = 0 为两侧的 column 0),interior = d ≥ 16。流体粒子(加速度、shift、密度)与含壁粒子的压力(壁粒子的压力即 p_w):

| N | 对 | 字段 | d=0 | 1 | 2 | 3 | 4 | 5 | 8 | interior |
|---|---|---|---|---|---|---|---|---|---|---|
| 300 | B1 对 A1 | 加速度 | 0.944 | 1.02 | 1.32 | 1.55 | 1.36 | 1.01 | 0.919 | 0.99 |
| 300 | B1 对 A1 | shift | 1.30 | 1.07 | 0.872 | 0.679 | 1.22 | 1.05 | 0.677 | 0.993 |
| 300 | B1 对 A1 | 密度 | 0.856 | 1.03 | 1.44 | 1.66 | 1.54 | 1.10 | 0.882 | 0.919 |
| 300 | B1 对 A1 | 压力(含壁) | 0.885 | 1.05 | 1.41 | 1.59 | 1.41 | 1.08 | 0.948 | 0.866 |
| 300 | B2 对 A2 | 加速度 | 0.829 | 0.817 | 0.912 | 1.03 | 1.08 | 1.13 | 1.02 | 1.01 |
| 300 | B2 对 A2 | 密度 | 0.903 | 0.884 | 1.01 | 1.01 | 1.15 | 1.16 | 1.04 | 0.96 |
| 300 | B2 对 A2 | 压力(含壁) | 0.962 | 0.947 | 1.10 | 1.10 | 1.18 | 1.17 | 1.05 | 0.90 |
| 300 | \|B1 − B2\| 对 \|A1 − A2\| | 加速度 | 0.846 | 0.871 | 1.13 | 1.31 | 1.23 | 0.966 | 0.855 | 0.985 |
| 300 | \|B1 − B2\| 对 \|A1 − A2\| | 密度 | 0.94 | 1.01 | 1.29 | 1.38 | 1.41 | 1.08 | 0.945 | 0.964 |
| 900 | B1 对 A1 | 加速度 | 0.968 | 0.997 | 0.964 | 1.04 | 1.10 | 1.06 | 1.07 | 0.973 |
| 900 | B1 对 A1 | shift | 0.855 | 0.987 | 0.968 | 1.05 | 0.985 | 1.08 | 0.983 | 0.98 |
| 900 | B1 对 A1 | 密度 | 1.17 | 1.17 | 1.08 | 1.09 | 1.02 | 1.02 | 1.08 | 1.07 |
| 900 | B1 对 A1 | 压力(含壁) | 1.00 | 1.01 | 1.00 | 1.06 | 1.03 | 1.00 | 1.04 | 1.19 |
| 900 | B2 对 A2 | 加速度 | 0.971 | 0.931 | 0.999 | 1.07 | 1.11 | 1.12 | 0.923 | 1.06 |
| 900 | B2 对 A2 | 密度 | 1.06 | 1.10 | 1.14 | 1.24 | 1.24 | 1.14 | 0.944 | 0.974 |
| 900 | B2 对 A2 | 压力(含壁) | 0.997 | 0.999 | 1.04 | 1.08 | 1.08 | 1.03 | 0.907 | 1.10 |
| 900 | \|B1 − B2\| 对 \|A1 − A2\| | 加速度 | 0.907 | 0.867 | 0.905 | 1.04 | 1.16 | 1.18 | 1.04 | 1.02 |
| 900 | \|B1 − B2\| 对 \|A1 − A2\| | 密度 | 0.779 | 0.786 | 0.793 | 0.877 | 0.96 | 1.01 | 0.925 | 0.951 |

- column 0 的 seam_excess(加速度 rms 比值):N = 300 0.944,N = 900 0.968。
- N = 300 第一对的 d = 2–4(1.3–1.7)在同一列对应 K = 2 自身两次运行的噪声 1.1–1.4,第二对在这些列为 1.0–1.2,N = 900 两对都在 1.0–1.2:不是系统性的。各列的噪声底(rms|A1 − A2|)本身在 0.05–0.07 之间起伏。
- 完整报告(全部 16 列、p99 / max、column 0 的越界分组、kernel_sum 分布、shift 剖面、图):`logs/e40/seam/analysis_N{300,900}_{fluid,all}/report.md`。

### 代价(`adami_chain_gates.py cost`:`_run_v7_chain_bench.py`,1000² simple 兄弟算例对 adami 算例,预热 1,000 步后计时 4,000 步,三次交替(每轮 simple / adami 顺序对调),K = 1 在 GPU 1,K = 2 在 GPU 0、1,drift / overflow / far_migration / 帧戳全 0)

| K | simple fps(三次) | adami fps(三次) | 每步时间 adami / simple |
|---|---|---|---|
| 1 | 728.3 / 730.2 / 730.1 | 659.5 / 660.6 / 658.0 | 1.1064(+10.64 %) |
| 2 | 1103.4 / 1104.2 / 1098.7 | 925.1 / 930.9 / 920.9 | 1.1906(+19.06 %) |

K = 2 对 K = 1 的加速:simple 1.51×,adami 1.40×。

每步分解(`adami_chain_gates.py trace`:`--step-trace --step-trace-detail full`,预热 500 步后 2,000 步,每卡的中位数,µs;step trace 自身有开销,看相对值):

| 区间(中位数 µs) | adami K=1 | adami K=2 s0 | adami K=2 s1 | simple K=1 | simple K=2 s0 | simple K=2 s1 |
|---|---|---|---|---|---|---|
| 步周期 | 1482.9 | 1009.5 | 1021.2 | 1345.1 | 854.3 | 863.1 |
| phase A | 51.0 | 28.4 | 30.1 | 52.7 | 28.3 | 28.4 |
| phase B | 1408.2 | 740.5 | 751.6 | 1270.6 | 652.5 | 664.6 |
| · correction_density_interior | 589.8 | 310.3 | 314.6 | 607.2 | 318.4 | 326.4 |
| · 壁面 pass(interior) | 63.4 | 57.2 | 57.4 | – | – | – |
| · force_deep_interior | 753.8 | 372.7 | 378.9 | 662.5 | 334.0 | 338.0 |
| phase C | 8.4 | 222.4 | 223.7 | 8.4 | 154.8 | 153.9 |
| · install + append + expand | – | 8.8 | 8.7 | – | 10.9 | 8.6 |
| · correction_density band | 0.2 | 69.5 | 69.5 | 0.2 | 69.5 | 69.5 |
| · 密度拷贝 | 8.1 | 4.2 | 4.2 | 8.1 | 4.2 | 4.2 |
| · 壁面 band pass | – | 51.1 | 51.1 | – | – | – |
| · force band | 0.2 | 90.1 | 90.1 | 0.2 | 71.6 | 71.6 |
| B → C 间隙 | 4.1 | 4.7 | 4.5 | 4.1 | 7.2 | 4.7 |

- K = 2 下 adami 每卡每步多 ≈ 155 µs:phase B 多 88(interior 壁面 pass 57 + force_deep 多 39,后者是 adami 的 force 每对邻居多读材料与虚拟速度,K = 1 时为 +91),phase C 多 68(wall band pass 51 + force band 多 18:4 列对 3 列,且是 adami 的 force)。interior 壁面 pass 从 K = 1 的 63 µs 只降到 57 µs(壁粒子少,延迟为主);correction_density band 多算一列 self 没有可测的增量(69.5 对 69.5);phase A 不变。
- 两张卡的分解相同(切口在中线,等权重)。

### 包的字节(K = 2,1000²,chain bench 的 link 计数,每条链路每帧)

| | simple(rc2 布局) | adami(三层) | 增量 |
|---|---|---|---|
| ghost 池 / 方向(槽) | 2 × 6,691 + 165 = 13,547 | 3 × 6,691 + 165 = 20,238 | +6,691 |
| DMA(readback 与 upload 各一次) | 428.5 KiB | 639.2 KiB | +210.7 KiB |
| 主机拷贝(count-aware) | 322.6 KiB | 484.5 KiB | +161.9 KiB |
| bootstrap 的 ghost 同步 | 857.0 KB | 1,278.4 KB | |

增量 = 第三层的打包 replica(每槽 32 B:x y z ρ \| vx vy vz material,与 G1、G2 同一记录,不带压力)6,691 × 32 B = 209.1 KiB + 第三列 ghost voxel 的 `inside_particle_count` 与 `ghost_voxel_first_particle_id`(206 × 4 B × 2 = 1.6 KiB)+ 计数字 4 B。传输 **不**带 `wall_dummy_velocity` 或壁面压力:接收方自己算。

## 复现

```
# CPU
.venv/Scripts/python.exe experiment/v7/_test_seam_layout.py
.venv/Scripts/python.exe experiment/v7/_test_chain_cuts.py
.venv/Scripts/python.exe experiment/v7/_test_partition_chain.py
# GPU(两张 5090;rc2 的 checkout 在 ../vulkan-demo-v7-wake)
.venv/Scripts/python.exe -m experiment.seam_audit.adami_chain_gates bitexact --rc2 ../vulkan-demo-v7-wake
.venv/Scripts/python.exe -m experiment.seam_audit.adami_chain_gates seam
.venv/Scripts/python.exe -m experiment.seam_audit.adami_chain_gates cost
# step trace(四个配置)然后汇总
.venv/Scripts/python.exe experiment/v7/_run_v7_chain_bench.py --case cases/lid_driven_cavity_2d_n1000_xi0p001_eps0p0025_adami/case.yaml \
    --weights 1.0,1.0 --device-map 0,1 --warmup 500 --max-steps 2500 --step-trace logs/e40/trace/adami_k2 --step-trace-detail full
.venv/Scripts/python.exe -m experiment.seam_audit.adami_chain_gates trace --out logs/e40/trace
```

日志与 dump 在 `logs/e40/`(不入库):`bitexact/`(每次 dump 的 npz 与日志、`*.compare.log`、`verdicts.json`)、`seam/`(dump 与四份分析)、`cost/`(每次 bench 的日志、`results.json`、`summary.md`)、`trace/`(四个 step trace、`summary.md`)、`test_seam_layout_*.log`。
