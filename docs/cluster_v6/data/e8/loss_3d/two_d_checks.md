## 4. 2-D checks

### G1 redundancy share, 2-D 64M K8 (x_2d_n8000_K8, job 1694328, wqd10nbj04g2)

Chain trace detail 'phases' (phase ticks only), so the boundary correction + density kernel comes from the anatomy run's sampled frames (3 per sim); period = the chain trace's per-sim mean over the kept steady steps; grid 1606 x 1606 x 1, 16 Ny Nz + 16 = 25,712 B, host copy bytes p50 2,592,752 B -> n0 = n1 = 40,110 particles (one voxel column = 5 layers x 8022); band per side = 3 n0; share = sides n0 c_b / period = boundary us / (3 period). Particle share of the redundant work on an interior card: 2 n0 / (200 columns x n0) = 1.00 %.

| sim | sides | boundary kernel (us, mean of frames) | boundary share of the step | c_b (ns / band particle) | redundant (us) | period (us) | G1 share |
|---|---|---|---|---|---|---|---|
| 0 | 1 | 175.3 (3) | 1.58 % | 1.457 | 58.4 | 11,103 | 0.53 % |
| 1 | 2 | 251.3 (3) | 2.26 % | 1.044 | 83.8 | 11,106 | 0.75 % |
| 2 | 2 | 256.7 (3) | 2.31 % | 1.067 | 85.6 | 11,110 | 0.77 % |
| 3 | 2 | 253.3 (3) | 2.28 % | 1.053 | 84.4 | 11,106 | 0.76 % |
| 4 | 2 | 250.7 (3) | 2.26 % | 1.042 | 83.6 | 11,104 | 0.75 % |
| 5 | 2 | 253.3 (3) | 2.28 % | 1.053 | 84.4 | 11,101 | 0.76 % |
| 6 | 2 | 257.3 (3) | 2.32 % | 1.069 | 85.8 | 11,099 | 0.77 % |
| 7 | 1 | 177.3 (3) | 1.60 % | 1.474 | 59.1 | 11,095 | 0.53 % |

**G1 share, mean over the 8 sims: 0.70 % (0.53–0.77 %)**; the paper's 'about 1 %' is the particle share (1.0 %); in time it is 0.7 %.

### SM clock ratio chain / reference, the nine strong K = 8 2-D points (tables_b2.json: per trial the chain's loop-window SM clock mean over its 8 cards / the mean of the 8 reference runs' (4 pairs' for 11320) SM clock; mean over the trials)

| case | per device | job (node) | trials | ratio mean (min–max) | ratio − 1 | chain SM MHz | reference SM MHz |
|---|---|---|---|---|---|---|---|
| cavity2d_n520 | 0.04M | e7_b2B_1694328 (wqd10nbj04g2) | 3 | 1.1600 (1.0695–1.3097) | +16.00 % | 2718 | 2361 |
| cavity2d_n1000 | 0.13M | e7_b2B_1694328 (wqd10nbj04g2) | 3 | 0.9559 (0.9310–0.9775) | -4.41 % | 2663 | 2786 |
| cavity2d_n1440 | 0.27M | e7_b2S_1695411 (wqd10nbj04g2) | 3 | 0.9849 (0.9784–0.9888) | -1.51 % | 2739 | 2781 |
| cavity2d_n2000 | 0.51M | e7_b2S_1695411 (wqd10nbj04g2) | 3 | 1.0367 (1.0359–1.0377) | +3.67 % | 2866 | 2765 |
| cavity2d_n2840 | 1.02M | e7_b2S_1695411 (wqd10nbj04g2) | 3 | 1.0330 (1.0302–1.0347) | +3.30 % | 2848 | 2757 |
| cavity2d_n4000 | 2.02M | e7_b2S_1695411 (wqd10nbj04g2) | 3 | 1.0240 (1.0183–1.0278) | +2.40 % | 2813 | 2747 |
| cavity2d_n5640 | 4.01M | e7_b2V_1703604 (wqd10nbj04g2) | 3 | 1.0304 (1.0261–1.0329) | +3.04 % | 2801 | 2718 |
| cavity2d_n8000 | 8.04M | e7_b2B_1694328 (wqd10nbj04g2) | 3 | 1.0275 (1.0270–1.0281) | +2.75 % | 2779 | 2705 |
| cavity2d_n11320 | 16.08M | e7_b2B_1694328 (wqd10nbj04g2) | 3 | 1.0187 (1.0184–1.0190) | +1.87 % | 2764 | 2713 |

**Range of the point means: -4.41 % .. +16.00 %** over the nine sizes; -1.51 % .. +3.67 % without the two smallest cases (520², 1000²: 0.04M / 0.13M per device, where the references' own clocks are off the 575 W regime).

### The same ratio along the 64M K = 2 .. 8 series (2d_n8000, wqd10nbj04g2; the E7 report's A line)

| K | point | job | trials | ratio mean (min–max) | ratio − 1 | chain SM MHz | reference SM MHz |
|---|---|---|---|---|---|---|---|
| 2 | 2d_n8000_K2 | e7_b2A_1694327 | 3 | 1.0093 (1.0070–1.0111) | +0.93 % | 2738 | 2713 |
| 3 | 2d_n8000_K3 | e7_b2V_1703604 | 3 | 1.0160 (1.0148–1.0176) | +1.60 % | 2737 | 2694 |
| 4 | 2d_n8000_K4 | e7_b2A_1694327 | 3 | 1.0199 (1.0188–1.0215) | +1.99 % | 2755 | 2701 |
| 5 | 2d_n8000_K5 | e7_b2V_1703604 | 3 | 1.0207 (1.0200–1.0216) | +2.07 % | 2758 | 2702 |
| 6 | 2d_n8000_K6 | e7_b2V_1703604 | 3 | 1.0216 (1.0209–1.0223) | +2.16 % | 2758 | 2700 |
| 7 | 2d_n8000_K7 | e7_b2V_1703604 | 3 | 1.0252 (1.0250–1.0254) | +2.52 % | 2766 | 2698 |
| 8 | x_2d_n8000_K8 | e7_b2B_1694328 | 3 | 1.0275 (1.0270–1.0281) | +2.75 % | 2779 | 2705 |

**K-series range: +0.93 % .. +2.75 %** — this is the paper's 0.9–2.7 %.
