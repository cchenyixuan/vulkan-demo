# §3 补测 表 2:逐 kernel 成本(v7_single,融合核,第 1001–3000 步连续计时)

每个算例 3 次独立运行;1 在飞(每步一次提交 + fence),生产录制里的 tick 回调写 BOTTOM_OF_PIPE 时间戳(屏障结构与生产步相同),defrag 帧(第 2000、3000 步)剔除;值 = 3 次运行各自均值的平均 ± 3 次的标准差;ns/粒子按全部粒子(含壁);ps/候选 = µs/步 × 10⁶ / 候选对数(第 3000 步结束时的体素列表)。

| kernel | 派发 | 8M µs/步 | ns/粒子 | % | ps/候选 | 3-D 8M µs/步 | ns/粒子 | % | ps/候选 |
|---|---|---|---|---|---|---|---|---|---|
| predict | per particle | 497 ± 0 | 0.06 | 4.7 | – | 500 ± 0 | 0.05 | 0.8 | – |
| update_voxel | per voxel | 331 ± 0 | 0.04 | 3.2 | – | 316 ± 0 | 0.03 | 0.5 | – |
| correction_density | per particle | 4530 ± 9 | 0.56 | 43.1 | 2.48 | 30354 ± 20 | 3.32 | 49.7 | 2.02 |
| density_copy | buffer copy | 82 ± 0 | 0.01 | 0.8 | – | 78 ± 0 | 0.01 | 0.1 | – |
| force | per particle | 5062 ± 6 | 0.62 | 48.2 | 2.78 | 29813 ± 23 | 3.27 | 48.8 | 1.98 |
| **步长** |  | 10502 ± 15 | 1.29 | 100.0 | – | 61060 ± 43 | 6.69 | 100.0 | – |

## 第 3000 步结束时的状态

| 项 | 8M | 3-D 8M |
|---|---|---|
| 粒子数(流体 + 壁) | 8,128,201 | 9,129,329 |
| 流体粒子 | 8,003,241 | 8,120,601 |
| h / Δx | 4.994 | 4.000 |
| 网格 Nx × Ny × Nz | 571 × 571 × 1 | 53 × 53 × 53 |
| 每粒子候选数(全部) | 224.4 | 1648.8 |
| 每粒子候选数(流体) | 225.0 | 1713.2 |
| 候选对数 / 步 | 1,823,725,626 | 15,052,132,531 |
| 每粒子邻居数 r < h,float32(全部) | 70.99 | 259.45 |
| 每粒子邻居数 r < h,float32(流体) | 71.04 | 265.90 |
| 邻居数 double 区间 [h(1−1e−5), h(1+1e−5)](全部) | 69.42 – 72.73 | 259.25 – 259.71 |
| 候选 / 邻居(全部) | 3.16 | 6.35 |
| 候选 / 邻居(流体) | 3.17 | 6.44 |
| 每体素粒子数 均值 / 最大(占用体素) | 24.9 / 30 | 61.3 / 78 |
| 占用体素 / 总体素 | 326,041 / 326,041 | 148,877 / 148,877 |
| 步长 3 次运行的散布(标准差 / 均值) | 0.14 % | 0.07 % |
| 含 defrag 帧与 defrag 命令摊到每步的 GPU 步长 µs | 10526 | 61090 |
| defrag 命令 µs(每次) | 23786 | 28240 |
| 计时帧墙钟 − GPU 帧(host 往返)µs | 247 | 289 |

## 与测量 B(交错,第 1001–5000 步)的差

- 8M:步长 10502 对 B 的 10572 µs(-0.66 %);逐 kernel:predict +0.02 %, update_voxel +0.01 %, correction_density -0.76 %, density_copy -0.03 %, force -0.69 %, gpu_frame -0.66 %;候选数 / 粒子 224.4 对 224.4。
- 3-D 8M:步长 61060 对 B 的 63319 µs(-3.57 %);逐 kernel:predict +0.22 %, update_voxel -0.77 %, correction_density -3.36 %, density_copy +0.37 %, force -3.87 %, gpu_frame -3.57 %;候选数 / 粒子 1648.8 对 1649.1。

## LaTeX 行

```latex
% columns: kernel & dispatch & 8M $\mu$s/step & ns/particle & \% & 3-D 8M $\mu$s/step & ns/particle & \% \\
% 8M: 8,128,201 particles (8,003,241 fluid), h/dx = 4.99, 224 candidates and 71.0 neighbours per particle; 3-D 8M: 9,129,329 particles (8,120,601 fluid), h/dx = 4, 1649 candidates and 259.4 neighbours per particle
% 8M: 3 runs, step spread 0.14 %; 3-D 8M: 3 runs, step spread 0.07 %
predict & per particle & 497 & 0.06 & 4.7 & 500 & 0.05 & 0.8 \\
update\_voxel & per voxel & 331 & 0.04 & 3.2 & 316 & 0.03 & 0.5 \\
correction + density (fused) & per particle & 4530 & 0.56 & 43.1 & 30354 & 3.32 & 49.7 \\
density (scratch$\to$primary copy) & buffer copy & 82 & 0.01 & 0.8 & 78 & 0.01 & 0.1 \\
force & per particle & 5062 & 0.62 & 48.2 & 29813 & 3.27 & 48.8 \\
\midrule
\textbf{step total} &  & 10502 & 1.29 & 100.0 & 61060 & 6.69 & 100.0 \\
% ps per candidate: 8M correction_density 2.48, force 2.78; 3-D 8M correction_density 2.02, force 1.98
```

## 协变量(每次运行窗口内的 nvidia-smi 1 Hz 样本)

| 算例 | 运行 | 样本 | SM MHz | W | 最高 °C | PCIe |
|---|---|---|---|---|---|---|
| 8M | 1 | 21 | 2788 | 600 | 68 | Gen 5 x8 |
| 8M | 2 | 22 | 2780 | 600 | 68 | Gen 5 x8 |
| 8M | 3 | 21 | 2784 | 600 | 69 | Gen 5 x8 |
| 3-D 8M | 1 | 122 | 2798 | 600 | 70 | Gen 5 x8 |
| 3-D 8M | 2 | 122 | 2797 | 600 | 70 | Gen 5 x8 |
| 3-D 8M | 3 | 122 | 2796 | 600 | 70 | Gen 5 x8 |
