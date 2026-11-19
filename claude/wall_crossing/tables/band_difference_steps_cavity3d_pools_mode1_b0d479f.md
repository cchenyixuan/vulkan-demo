| 量 | 字段 | 第一个不同的步 | 不同的步数 |
|---|---|---|---|
| 加速度 | `acceleration` | 40 | 27 |
| 修正矩阵（L） | `correction_inverse` | 60 | 25 |
| 密度梯度与核和 | `density_gradient_kernel_sum` | 50 | 26 |
| 密度与压力 | `density_pressure` | 60 | 25 |
| 密度与压力（暂存） | `density_pressure_scratch` | 60 | 25 |
| 扩展字段（全局编号） | `extension_fields` | 无（到第 300 步） | 0 |
| 材料 | `material` | 无（到第 300 步） | 0 |
| 位置 | `position_voxel_id` | 40 | 27 |
| 位移修正 | `shift` | 60 | 25 |
| 速度与质量 | `velocity_mass` | 40 | 27 |

比较的步：0 – 300（31 个）；活着的粒子数不同的步：0。
