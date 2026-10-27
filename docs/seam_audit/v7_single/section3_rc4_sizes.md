# §3 补测 第 3 项:各尺寸的候选数、邻居数、网格与 GPU 步长

初始态:加载器的初始粒子按网格原点与尺寸在 CPU 上分箱(float32,与 initialize_voxelization 相同);窗末:第 3000 步结束时的 GPU 读回。邻居 = |{j ≠ i : r_ij < h}|(float32,与 kernel 的判定相同;double 区间见 JSON)。GPU 步长:1 在飞、时间戳、预热 1000 + 计时 2000(8M 与 3-D 8M 为表 2 三次运行的均值,其余 1 次);摊销步长 = 含 defrag 帧与 defrag 命令;GPU 空闲比例 = 测量 A 的墙钟步长 ÷ 摊销 GPU 步长 − 1。

| 算例 | 粒子 | h/Δx | 网格 Nx × Ny × Nz | 占用体素 初始 / 窗末 | 每体素 均值 / 最大 初始 | 窗末 | 候选/粒子 初始 | 窗末 | 邻居/粒子 初始(全部 / 流体) | 窗末(全部 / 流体) | 候选/邻居 窗末 | GPU 步长 µs(摊销) | ps/候选 步 | 融合核 | force | A 墙钟 d1 / d2 µs | 空闲 d1 | d2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1M | 1,046,529 | 5 | 206 × 206 × 1 | 42,436 / 42,436 | 24.7 / 25 | 24.7 / 31 | 223.2 | 223.3 | 73.52 / 73.82 | 73.86 / 74.18 | 3.02 | 1318 (1322) | 5.64 | 2.57 | 2.83 | 1418 / 1361 | +7.29% | +3.00% |
| 2M | 2,064,969 | 4.99 | 288 × 288 × 1 | 82,944 / 82,944 | 24.9 / 25 | 24.9 / 30 | 223.8 | 223.8 | 72.65 / 72.83 | 71.44 / 71.59 | 3.13 | 2493 (2503) | 5.39 | 2.40 | 2.67 | 2751 / 2689 | +9.90% | +7.45% |
| 4M(旧手工) | 4,182,025 | 5 | 410 × 410 × 1 | 168,100 / 168,100 | 24.9 / 25 | 24.9 / 30 | 224.1 | 224.1 | 73.90 / 74.01 | 74.27 / 74.40 | 3.02 | 5364 (5382) | 5.72 | 2.52 | 2.77 | 5795 / 5715 | +7.68% | +6.20% |
| 4M gen | 4,182,025 | 5.01 | 410 × 410 × 1 | 168,100 / 168,100 | 24.9 / 36 | 24.9 / 36 | 224.3 | 224.3 | 77.44 / 77.60 | 78.48 / 78.67 | 2.86 | 5360 (5378) | 5.71 | 2.49 | 2.79 | 5806 / 5710 | +7.96% | +6.19% |
| 8M | 8,128,201 | 4.99 | 571 × 571 × 1 | 326,041 / 326,041 | 24.9 / 25 | 24.9 / 30 | 224.4 | 224.4 | 74.88 / 74.98 | 70.99 / 71.04 | 3.16 | 9969 (10004) | 5.47 | 2.34 | 2.63 | 10685 / 10589 | +6.81% | +5.85% |
| 16M | 16,184,529 | 5 | 806 × 806 × 1 | 649,636 / 649,636 | 24.9 / 25 | 24.9 / 30 | 224.6 | 224.6 | 75.16 / 75.23 | 75.12 / 75.19 | 2.99 | 20379 (20453) | 5.61 | 2.41 | 2.71 | 21175 / 21065 | +3.53% | +2.99% |
| 32M | 32,251,041 | 5.02 | 1137 × 1137 × 1 | 1,292,769 / 1,292,769 | 24.9 / 25 | 24.9 / 30 | 224.7 | 224.7 | 74.46 / 74.52 | 73.19 / 73.24 | 3.07 | 40062 (40203) | 5.53 | 2.37 | 2.67 | 40952 / 40815 | +1.86% | +1.52% |
| 3-D 8M | 9,129,329 | 4 | 53 × 53 × 53 | 148,877 / 148,877 | 61.3 / 64 | 61.3 / 78 | 1648.6 | 1648.8 | 247.33 / 252.51 | 259.43 / 265.89 | 6.36 | 58414 (58451) | 3.88 | 1.92 | 1.90 | 61344 / 61210 | +4.95% | +4.72% |

## 4M 凹陷

- criterion: the 4M point's ps/candidate (whole step, fused kernel, force) against the mean of 2M and 8M; the run-to-run spread of the step is the yardstick
- step_spread_pct_8m_runs: 0.2304
- ps_per_candidate_step_2m_4m_8m: [5.392600630798017, 5.72344403814845, 5.466264947399583]
- ps_per_candidate_step_4m_vs_neighbours_pct: 5.415
- ps_per_candidate_correction_density_2m_4m_8m: [2.3984301689426744, 2.524569606589429, 2.3395352239668106]
- ps_per_candidate_correction_density_4m_vs_neighbours_pct: 6.568
- ps_per_candidate_force_2m_4m_8m: [2.6683046755446727, 2.7742074047658662, 2.6275354405485025]
- ps_per_candidate_force_4m_vs_neighbours_pct: 4.769
- candidates_per_particle_2m_4m_8m: [223.83620916343054, 224.09186004387826, 224.3701257305686]
- candidates_4m_vs_neighbours_pct: -0.005046
- gpu_throughput_mps_2m_4m_8m: [828.4597238158666, 779.6801503420169, 815.3504040873511]
- gpu_throughput_4m_vs_neighbours_pct: -5.137
- a_throughput_d2_2m_4m_8m: [767.8657435445999, 731.7250110266666, 767.5896417753667]
- a_throughput_d2_4m_vs_neighbours_pct: -4.69
- ps_per_candidate_step_2m_4m_gen_8m: [5.392600630798017, 5.713425872868682, 5.466264947399583]
- ps_per_candidate_step_4m_gen_vs_neighbours_pct: 5.231
- ps_per_candidate_correction_density_2m_4m_gen_8m: [2.3984301689426744, 2.492084351247028, 2.3395352239668106]
- ps_per_candidate_correction_density_4m_gen_vs_neighbours_pct: 5.196
- ps_per_candidate_force_2m_4m_gen_8m: [2.6683046755446727, 2.79027874343047, 2.6275354405485025]
- ps_per_candidate_force_4m_gen_vs_neighbours_pct: 5.376
- candidates_per_particle_2m_4m_gen_8m: [223.83620916343054, 224.32826561295067, 224.3701257305686]
- candidates_4m_gen_vs_neighbours_pct: 0.1004
- gpu_throughput_mps_2m_4m_gen_8m: [828.4597238158666, 780.2241778497508, 815.3504040873511]
- gpu_throughput_4m_gen_vs_neighbours_pct: -5.071
- a_throughput_d2_2m_4m_gen_8m: [767.8657435445999, 732.3590060166667, 767.5896417753667]
- a_throughput_d2_4m_gen_vs_neighbours_pct: -4.607
- grid_2m: {'dimensions': [288, 288, 1], 'total_voxels': 82944, 'end_occupied_voxels': 82944.0, 'end_mean_per_voxel': 24.89594184027778, 'end_max_per_voxel': 30, 'fluid_particles': 2002225, 'wall_particles': 62744, 'h_over_dx': 4.994350282485876}
- grid_4m: {'dimensions': [410, 410, 1], 'total_voxels': 168100, 'end_occupied_voxels': 168100.0, 'end_mean_per_voxel': 24.87819750148721, 'end_max_per_voxel': 30, 'fluid_particles': 4004001, 'wall_particles': 178024, 'h_over_dx': 5.0}
- grid_4m_gen: {'dimensions': [410, 410, 1], 'total_voxels': 168100, 'end_occupied_voxels': 168100.0, 'end_mean_per_voxel': 24.87819750148721, 'end_max_per_voxel': 36, 'fluid_particles': 4092529, 'wall_particles': 89496, 'h_over_dx': 5.006072874493928}
- grid_8m: {'dimensions': [571, 571, 1], 'total_voxels': 326041, 'end_occupied_voxels': 326041.0, 'end_mean_per_voxel': 24.929996534178215, 'end_max_per_voxel': 30, 'fluid_particles': 8003241, 'wall_particles': 124960, 'h_over_dx': 4.994350282485876}
