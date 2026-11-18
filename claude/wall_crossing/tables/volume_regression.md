| 检查 | 参照 | 结果 | 结论 | 记录 |
|---|---|---|---|---|
| simple、adami 用的 SPIR-V | 基点（eda5b90）清单里的 23 个文件 | 23 个逐字节相同 | 通过 | MANIFEST.txt |
| 相对 2186032 变了的 SPIR-V | 全部 27 个文件 | correction_density_wall_crossing、correction_density_wall_crossing_budget、force_wall_crossing | — | MANIFEST.txt |
| 完整方案，约束关闭，250²，700 步 | 2186032 的求解器（冻结检出 3176d57，求解器文件相同）的 24 个转储 | 2 个结果类。类 0：参照 18 次，本次 9 次；类 1：参照 6 次，本次 3 次 | 逐位相同 | wc-20261118-183647-e3b7 |
| `_test_seam_layout.py` 的失败清单 | 基点的清单（858 条，基点自身就有） | 858 条，与基点的清单逐行相同 | 通过 | wc-20261118-184813-0dfc |
