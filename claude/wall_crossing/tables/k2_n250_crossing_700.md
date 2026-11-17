# K = 1 对 K = 2：n250_crossing_700

算例 `cases/lid_driven_cavity_2d_n250_xi0p001_eps0p0025_crossing/case.yaml`，700 步，canonical 体素表；单卡设备 1，链设备 0,1。

- 单卡两次运行逐位相同：是
- 链对单卡逐位相同（体素号除外）：否；切口列 [28]，过切口 {'migration_install_count': [0, 2], 'peak_migration_count': [0, 2], 'peak_departed_count': [1, 0]}

## 第一处差异：第 520 步（第 519 步仍逐位相同）

- 不同的行：66，位置 x ∈ [0.4277, 0.4596]，y ∈ [0.4678, 0.5000]；correction_inverse 44 行（最大相对差 3.0e-07）；density_gradient_kernel_sum 26 行（最大相对差 1.2e-07）；density_pressure_scratch 1 行（最大相对差 6.1e-08）；density_pressure 1 行（最大相对差 6.1e-08）；acceleration 50 行（最大相对差 2.8e-07）；shift 58 行（最大相对差 7.0e-04）
- 该步链上过切口计数：{'migration_install_count': [0, 1], 'peak_migration_count': [0, 1], 'peak_departed_count': [1, 0]}
- 体素表（3 × 3 邻域内各体素，单卡 对 链）：(49,52) 成员不同 {"only_single": [], "only_chain": [59482]}；(50,52) 成员不同 {"only_single": [59482], "only_chain": []}
- 粒子 59482（x = 0.443599969, y = 0.48386851，两种布局位置逐位相同）：单卡的体素表把它列在体素 (50,52)，链的体素表列在 (49,52)。它的 (x − origin_x) / h 的精确值距体素列边界 -3.725e-07 个体素，而 x 的 1 ulp 是 1.490e-06 个体素：落在舍入范围内，两种布局各自从自己的 float32 网格原点装箱（numpy 的 float32 复算：单卡商 50，链商 24 + 列偏移；GPU 的除法舍入不必与 numpy 相同）

记录：wc-20261117-181109-d83d … wc-20261117-181213-3eb7（共 24 条）
