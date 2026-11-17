| 运行 | 到达的 t | 状态 | 平均密度漂移率（后半段，kg/m³ 每时间单位） | 最大密度偏差（kg/m³） | φ_min / dx | 低于下限 / 越出盒（最大粒子数） | A_f 回退 | 体素表溢出 | 步 / 秒（与别的作业共用 GPU） | 提交 | 记录 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| diag_n250_adami | 6.00 | ok | -0.00029 | 0.482 | 0.244 | 0 / 0 | 0 | 0 | 1043 | 5ee25b1+dirty | wc-20261117-175719-b080 |
| diag_n250_crossing | 6.00 | ok | -0.01579 | 0.169 | 0.281 | 0 / 0 | 0 | 0 | 1205 | 5ee25b1+dirty | wc-20261117-174907-ff46 |
| diag_n250_crossing_floor0p1 | 6.00 | ok | -0.01495 | 0.164 | 0.310 | 0 / 0 | 0 | 0 | 1064 | 5ee25b1+dirty | wc-20261117-180331-abf6 |
| diag_n250_crossing_floor0p5 | 6.00 | ok | -0.01486 | 0.185 | 0.304 | 725 / 0 | 0 | 0 | 929 | a2149ed | wc-20261117-180641-16e7 |
| diag_n250_crossing_no_diffusion | 6.00 | ok | -0.03738 | 21.413 | 0.310 | 0 / 0 | 0 | 0 | 1221 | 5ee25b1+dirty | wc-20261117-175202-5ece |
| diag_n250_crossing_no_pst | — | failed: RuntimeError: step 7000: alive 74523 of 74529, or a non-finite velocity | — | — | — | — | — | — | — | 5ee25b1+dirty | wc-20261117-175155-aeea |
| diag_n250_crossing_wall_velocity | 6.00 | ok | -1.86539 | 11.783 | 0.145 | 1 / 0 | 0 | 0 | 1224 | 5ee25b1+dirty | wc-20261117-180032-8e37 |
| diag_n250_crossing_wall_velocity_no_pst | — | failed: RuntimeError: step 7000: alive 11631 of 74529, or a non-finite velocity | — | — | — | — | — | — | — | 5ee25b1+dirty | wc-20261117-180317-9781 |
| diag_n250_simple | 6.00 | ok | -0.00018 | 1.343 | 0.314 | 0 / 0 | 0 | 0 | 1340 | 5ee25b1+dirty | wc-20261117-175448-8989 |
| diag_n250_simple_no_pst | — | failed: RuntimeError: step 7000: alive 74512 of 74529, or a non-finite velocity | — | — | — | — | — | — | — | 5ee25b1+dirty | wc-20261117-180324-4898 |
