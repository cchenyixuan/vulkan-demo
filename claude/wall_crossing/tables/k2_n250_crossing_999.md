# K = 1 对 K = 2：n250_crossing

算例 `cases/lid_driven_cavity_2d_n250_xi0p001_eps0p0025_crossing/case.yaml`，999 步，canonical 体素表；单卡设备 1，链设备 0,1。

- 单卡两次运行逐位相同：否
- 链对单卡逐位相同（体素号除外）：否；切口列 [28]，过切口 {'migration_install_count': [0, 5], 'peak_migration_count': [0, 5], 'peak_departed_count': [1, 0]}

## 第一处差异：第 520 步（第 519 步仍逐位相同）

- 不同的行：66，位置 x ∈ [0.4277, 0.4596]，y ∈ [0.4678, 0.5000]；correction_inverse 44 行（最大相对差 3.0e-07）；density_gradient_kernel_sum 26 行（最大相对差 1.2e-07）；density_pressure_scratch 1 行（最大相对差 6.1e-08）；density_pressure 1 行（最大相对差 6.1e-08）；acceleration 50 行（最大相对差 2.8e-07）；shift 58 行（最大相对差 7.0e-04）
- 该步链上过切口计数：{'migration_install_count': [0, 1], 'peak_migration_count': [0, 1], 'peak_departed_count': [1, 0]}
- 体素表（3 × 3 邻域内各体素，单卡 对 链）：(49,52) 成员不同 {"only_single": [], "only_chain": [59482]}；(50,52) 成员不同 {"only_single": [59482], "only_chain": []}
- 粒子 59482（x = 0.443599969, y = 0.48386851，两种布局位置逐位相同）：单卡装入体素列 50（float32 商 50），链上装入列 50（商 24）；精确商距列边界 -3.725e-07 个体素，x 的 1 ulp = 1.490e-06 个体素

记录：wc-20261117-180826-8e55 … wc-20261117-181033-9abc（共 26 条）
