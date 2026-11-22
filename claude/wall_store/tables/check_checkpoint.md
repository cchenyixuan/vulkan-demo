| 2-D 250² 方腔 | 运行 | 末时刻 | 最后一次体积修正的时刻 | θ | I | 平均密度 − ρ0 | 动能 |
|---|---|---|---|---|---|---|---|
| 壁面独立存储 | 连续运行到 t = 4 | 4.02 | 3.96 | +2.0828e-07 | +1.4784e-06 | -0.001790 | 19.45255 |
| 壁面独立存储 | t = 2.01 的检查点续跑到 t = 4 | 4.02 | 3.99 | +1.0824e-06 | +1.3499e-06 | -0.000613 | 19.44420 |
| 壁面独立存储 | 差（续跑 − 连续） |  |  | +8.74e-07 | -1.28e-07 | +0.001176 | -0.043 % |
| 标签 crossing-pool | 连续运行到 t = 4 | 4.02 | 3.96 | +2.6993e-06 | +5.5782e-06 | -0.001454 | 19.44812 |
| 标签 crossing-pool | t = 2.01 的检查点续跑到 t = 4 | 4.02 | 3.99 | -3.1351e-06 | +3.7919e-06 | +0.003002 | 19.45040 |
| 标签 crossing-pool | 差（续跑 − 连续） |  |  | -5.83e-06 | -1.79e-06 | +0.004456 | +0.012 % |

旧检查点的报错（标签 crossing-pool 存的检查点；上：box_run --resume，下：直接交给 SphSimulatorV7.restart_init）：

```
[box_run] checkpoint C:\Users\cchen\PycharmProjects\vulkan-demo-wall-crossing\logs\wall_store\new\box_run\ws_old_checkpoint_resume\checkpoints\c0000001000.npz is a wall_boundary crossing checkpoint of the particle-pool version (git tag crossing-pool): its walls are in the particle pool. The wall-store version cannot resume it; start the run anew or resume it with the crossing-pool tag.

ValueError: restart: the state holds 11,528 wall particles: it is a wall_boundary crossing checkpoint of the particle-pool version (git tag crossing-pool), whose walls lived in the particle pool. The wall-store version keeps its walls out of the pool and cannot resume it; restart the run from its initial condition, or resume it with the crossing-pool tag.
```
