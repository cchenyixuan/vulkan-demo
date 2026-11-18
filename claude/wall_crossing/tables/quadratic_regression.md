| 检查 | 参照 | 结果 | 结论 | 记录 |
|---|---|---|---|---|
| simple、adami 用的 SPIR-V | 基点（eda5b90）清单里的 23 个文件 | 23 个逐字节相同 | 通过 | MANIFEST.txt |
| 相对 398f70a 变了的 SPIR-V | 全部 27 个文件 | correction_density_budget、correction_density_wall_crossing、correction_density_wall_crossing_budget、force_wall_crossing | — | MANIFEST.txt |
| simple，250²，999 步 | 基点构建的 2 个转储（2026-11-17） | 1 个结果类。类 0：参照 2 次，本次 4 次 | 逐位相同 | wc-20261118-135159-d59f |
| adami，250²，200 步 | 基点构建的 2 个转储（2026-11-17） | 1 个结果类。类 0：参照 2 次，本次 4 次 | 逐位相同 | wc-20261118-135204-01db |
| crossing，两个开关都是 linear（缺省），250²，700 步 | 398f70a 的求解器的 8 个转储（6c8a55a 时产生；之后到 398f70a 求解器没有改动） | 1 个结果类。类 0：参照 8 次，本次 6 次 | 逐位相同 | wc-20261118-132452-1301 |
| `_test_seam_layout.py` 的失败清单 | 基点的清单（858 条，基点自身就有） | 858 条，与基点的清单逐行相同 | 通过 | wc-20261118-133142-21d5 |
