# §3 补测 表 2:逐 kernel 成本(v7_single,融合核,第 1001–3000 步连续计时)

每个算例 3 次独立运行;1 在飞(每步一次提交 + fence),生产录制里的 tick 回调写 BOTTOM_OF_PIPE 时间戳(屏障结构与生产步相同),defrag 帧(第 2000、3000 步)剔除;值 = 3 次运行各自均值的平均 ± 3 次的标准差;ns/粒子按全部粒子(含壁);ps/候选 = µs/步 × 10⁶ / 候选对数(第 3000 步结束时的体素列表)。

| kernel | 派发 | 8M µs/步 | ns/粒子 | % | ps/候选 | 3-D 8M µs/步 | ns/粒子 | % | ps/候选 |
|---|---|---|---|---|---|---|---|---|---|
| predict | per particle | 497 ± 0 | 0.06 | 5.0 | – | 504 ± 0 | 0.06 | 0.9 | – |
| update_voxel | per voxel | 331 ± 0 | 0.04 | 3.3 | – | 313 ± 0 | 0.03 | 0.5 | – |
| correction_density | per particle | 4267 ± 12 | 0.52 | 42.8 | 2.34 | 28934 ± 20 | 3.17 | 49.5 | 1.92 |
| density_copy | buffer copy | 82 ± 0 | 0.01 | 0.8 | – | 93 ± 0 | 0.01 | 0.2 | – |
| force | per particle | 4792 ± 11 | 0.59 | 48.1 | 2.63 | 28569 ± 25 | 3.13 | 48.9 | 1.90 |
| **步长** |  | 9969 ± 23 | 1.23 | 100.0 | – | 58414 ± 44 | 6.40 | 100.0 | – |

## 第 3000 步结束时的状态

| 项 | 8M | 3-D 8M |
|---|---|---|
| 粒子数(流体 + 壁) | 8,128,201 | 9,129,329 |
| 流体粒子 | 8,003,241 | 8,120,601 |
| h / Δx | 4.994 | 4.000 |
| 网格 Nx × Ny × Nz | 571 × 571 × 1 | 53 × 53 × 53 |
| 每粒子候选数(全部) | 224.4 | 1648.8 |
| 每粒子候选数(流体) | 225.0 | 1713.2 |
| 候选对数 / 步 | 1,823,725,480 | 15,052,147,316 |
| 每粒子邻居数 r < h,float32(全部) | 70.99 | 259.43 |
| 每粒子邻居数 r < h,float32(流体) | 71.04 | 265.89 |
| 邻居数 double 区间 [h(1−1e−5), h(1+1e−5)](全部) | 69.42 – 72.73 | 259.24 – 259.69 |
| 候选 / 邻居(全部) | 3.16 | 6.36 |
| 候选 / 邻居(流体) | 3.17 | 6.44 |
| 每体素粒子数 均值 / 最大(占用体素) | 24.9 / 30 | 61.3 / 78 |
| 占用体素 / 总体素 | 326,041 / 326,041 | 148,877 / 148,877 |
| 步长 3 次运行的散布(标准差 / 均值) | 0.23 % | 0.08 % |
| 含 defrag 帧与 defrag 命令摊到每步的 GPU 步长 µs | 10004 | 58451 |
| defrag 命令 µs(每次) | 35329 | 35107 |
| 计时帧墙钟 − GPU 帧(host 往返)µs | 260 | 329 |

## 与测量 B(交错,第 1001–5000 步)的差

- 8M:步长 9969 对 B 的 10572 µs(-5.70 %);逐 kernel:predict +0.03 %, update_voxel +0.02 %, correction_density -6.53 %, density_copy +0.03 %, force -5.99 %, gpu_frame -5.70 %;候选数 / 粒子 224.4 对 224.4。
- 3-D 8M:步长 58414 对 B 的 63319 µs(-7.75 %);逐 kernel:predict +1.20 %, update_voxel -1.60 %, correction_density -7.88 %, density_copy +19.44 %, force -7.88 %, gpu_frame -7.75 %;候选数 / 粒子 1648.8 对 1649.1。

## LaTeX 行

```latex
% columns: kernel & dispatch & 8M $\mu$s/step & ns/particle & \% & 3-D 8M $\mu$s/step & ns/particle & \% \\
% 8M: 8,128,201 particles (8,003,241 fluid), h/dx = 4.99, 224 candidates and 71.0 neighbours per particle; 3-D 8M: 9,129,329 particles (8,120,601 fluid), h/dx = 4, 1649 candidates and 259.4 neighbours per particle
% 8M: 3 runs, step spread 0.23 %; 3-D 8M: 3 runs, step spread 0.08 %
predict & per particle & 497 & 0.06 & 5.0 & 504 & 0.06 & 0.9 \\
update\_voxel & per voxel & 331 & 0.04 & 3.3 & 313 & 0.03 & 0.5 \\
correction + density (fused) & per particle & 4267 & 0.52 & 42.8 & 28934 & 3.17 & 49.5 \\
density (scratch$\to$primary copy) & buffer copy & 82 & 0.01 & 0.8 & 93 & 0.01 & 0.2 \\
force & per particle & 4792 & 0.59 & 48.1 & 28569 & 3.13 & 48.9 \\
\midrule
\textbf{step total} &  & 9969 & 1.23 & 100.0 & 58414 & 6.40 & 100.0 \\
% ps per candidate: 8M correction_density 2.34, force 2.63; 3-D 8M correction_density 1.92, force 1.90
```

## 协变量(每次运行窗口内的 nvidia-smi 1 Hz 样本)

| 算例 | 运行 | 样本 | SM MHz | W | 最高 °C | PCIe |
|---|---|---|---|---|---|---|
| 8M | 1 | 20 | 2795 | 600 | 68 | Gen 4 x4 |
| 8M | 2 | 20 | 2793 | 599 | 69 | Gen 4 x4 |
| 8M | 3 | 20 | 2782 | 600 | 69 | Gen 4 x4 |
| 3-D 8M | 1 | 117 | 2808 | 600 | 70 | Gen 4 x4 |
| 3-D 8M | 2 | 117 | 2806 | 600 | 70 | Gen 4 x4 |
| 3-D 8M | 3 | 116 | 2805 | 600 | 70 | Gen 4 x4 |
