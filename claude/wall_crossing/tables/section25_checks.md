| 检查 | 算例 | 卡数 | 步数 | 新代码 / 0088ce0 的算例 | 结果类（新代码 / 0088ce0 的运行次数） | 结论 |
|---|---|---|---|---|---|---|
| 1 | simple，2-D 250² | 1 | 200 | 同一算例 | `2e8ecd11ddf6`：新 2、旧 2 | 通过 |
| 1 | adami，2-D 250² | 1 | 200 | 同一算例 | `e3ce9f0f22e0`：新 2、旧 2 | 通过 |
| 2 | crossing、关修正，2-D 250² | 1 | 700 | 新：`…_crossing_no_volume_constraint`；旧：`…_explicit_defaults` | `b65a33c0554d`：新 3、旧 2；`1d3f1d0702b5`：新 1、旧 2 | 通过 |
| 2 | crossing、关修正，3-D 1M | 1 | 200 | 新：`cavity3d_1m_crossing_no_volume_constraint`；旧：`…_explicit_defaults`；3-D 缓冲区系数 | `92faeceaffb0`：新 2、旧 2 | 通过 |
| 2 | crossing、关修正，3-D 1M | 2 | 200 | 同上 | `623e53413adc`：新 2、旧 2 | 通过 |
| 3 | crossing、开修正，2-D 250² | 1 | 700 | 新：`…_crossing`；旧：`…_crossing_explicit_defaults` | `1d3f1d0702b5`：新 3、旧 1；`b65a33c0554d`：新 1、旧 3 | 通过 |
| 3 | crossing、开修正，2-D 250² | 2 | 700 | 同上 | `40240f4c3f14`：新 4、旧 3；`fbaa61fc0140`：新 0、旧 1 | 通过 |
