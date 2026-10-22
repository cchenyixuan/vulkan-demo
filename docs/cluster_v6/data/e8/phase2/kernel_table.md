## Per-kernel time per step (E8 phase 2; medians over the steady steps, mean over the sims)

**3d_n192_K8** — 8M (200^3) K = 8, detail full, step period mean of sims 9.980 ms, G1 redundancy share 8.82 % (5.91–10.19 % over the sims)

| phase | kernel | us / step (mean of sims) | ns / particle | share of the step | sims |
|---|---|---|---|---|---|
| A | predict | 25.5 | 0.025 | 0.3 % | 8 |
| A | voxel | 33.1 | 0.033 | 0.3 % | 8 |
| A | ghost_trailing | 15.7 | 0.016 | 0.2 % | 7 |
| B | correction_density_interior | 1,677.6 | 1.652 | 16.8 % | 8 |
| B | force_deep_interior | 461.5 | 0.413 | 4.6 % | 8 |
| C | expand | 19.2 | 0.020 | 0.2 % | 8 |
| C | install_trailing | 2.6 | 0.003 | 0.0 % | 7 |
| C | append_departed | 3.8 | 0.004 | 0.0 % | 8 |
| C | correction_density_boundary | 2,640.5 | 2.684 | 26.5 % | 8 |
| C | density | 9.9 | 0.010 | 0.1 % | 8 |
| C | force | 2,693.2 | 2.740 | 27.0 % | 8 |
| A | ghost_leading | 14.3 | 0.015 | 0.1 % | 7 |
| C | install_leading | 4.1 | 0.004 | 0.0 % | 7 |

per sim (us / step): sim 0: predict 32, voxel 37, ghost_trailing 14, correction_density_interior 2636, force_deep_interior 1890, expand 12, install_trailing 4, append_departed 2, correction_density_boundary 1760, density 8, force 1622; sim 1: predict 23, voxel 31, ghost_leading 14, ghost_trailing 16, correction_density_interior 1328, force_deep_interior 9, expand 22, install_leading 4, install_trailing 2, append_departed 4, correction_density_boundary 2868, density 11, force 2978; sim 2: predict 23, voxel 31, ghost_leading 14, ghost_trailing 16, correction_density_interior 1410, force_deep_interior 10, expand 22, install_leading 4, install_trailing 2, append_departed 4, correction_density_boundary 2845, density 10, force 2966; sim 3: predict 23, voxel 31, ghost_leading 14, ghost_trailing 14, correction_density_interior 1401, force_deep_interior 6, expand 22, install_leading 4, install_trailing 2, append_departed 4, correction_density_boundary 2859, density 10, force 2993; sim 4: predict 24, voxel 31, ghost_leading 14, ghost_trailing 16, correction_density_interior 1374, force_deep_interior 7, expand 22, install_leading 4, install_trailing 2, append_departed 4, correction_density_boundary 2845, density 10, force 2975; sim 5: predict 24, voxel 32, ghost_leading 14, ghost_trailing 16, correction_density_interior 1399, force_deep_interior 7, expand 23, install_leading 4, install_trailing 2, append_departed 4, correction_density_boundary 3054, density 10, force 2973; sim 6: predict 23, voxel 31, ghost_leading 14, ghost_trailing 16, correction_density_interior 1388, force_deep_interior 6, expand 22, install_leading 4, install_trailing 4, append_departed 4, correction_density_boundary 3054, density 10, force 3132; sim 7: predict 32, voxel 41, ghost_leading 14, correction_density_interior 2485, force_deep_interior 1757, expand 11, install_leading 4, append_departed 4, correction_density_boundary 1840, density 8, force 1906

**3d_n392_K8** — 64M (400^3) K = 8, detail full, step period mean of sims 57.474 ms, G1 redundancy share 5.69 % (3.38–6.59 % over the sims)

| phase | kernel | us / step (mean of sims) | ns / particle | share of the step | sims |
|---|---|---|---|---|---|
| A | predict | 470.0 | 0.059 | 0.8 % | 8 |
| A | voxel | 278.0 | 0.035 | 0.5 % | 8 |
| A | ghost_trailing | 66.4 | 0.008 | 0.1 % | 7 |
| B | correction_density_interior | 18,062.0 | 2.252 | 31.4 % | 8 |
| B | force_deep_interior | 14,338.7 | 1.785 | 25.0 % | 8 |
| C | expand | 89.7 | 0.011 | 0.2 % | 8 |
| C | install_trailing | 2.3 | 0.000 | 0.0 % | 7 |
| C | append_departed | 4.1 | 0.001 | 0.0 % | 8 |
| C | correction_density_boundary | 9,815.9 | 1.235 | 17.1 % | 8 |
| C | density | 94.0 | 0.012 | 0.2 % | 8 |
| C | force | 10,090.0 | 1.270 | 17.5 % | 8 |
| A | ghost_leading | 47.0 | 0.006 | 0.1 % | 7 |
| C | install_leading | 4.1 | 0.001 | 0.0 % | 7 |

per sim (us / step): sim 0: predict 464, voxel 293, ghost_trailing 47, correction_density_interior 21560, force_deep_interior 18366, expand 36, install_trailing 4, append_departed 4, correction_density_boundary 5854, density 91, force 5902; sim 1: predict 499, voxel 289, ghost_leading 48, ghost_trailing 70, correction_density_interior 18505, force_deep_interior 14593, expand 108, install_leading 4, install_trailing 2, append_departed 4, correction_density_boundary 11388, density 102, force 11562; sim 2: predict 460, voxel 266, ghost_leading 47, ghost_trailing 70, correction_density_interior 16050, force_deep_interior 12165, expand 107, install_leading 4, install_trailing 2, append_departed 4, correction_density_boundary 11003, density 91, force 11353; sim 3: predict 462, voxel 266, ghost_leading 47, ghost_trailing 70, correction_density_interior 16449, force_deep_interior 12538, expand 107, install_leading 4, install_trailing 2, append_departed 4, correction_density_boundary 11268, density 92, force 11461; sim 4: predict 459, voxel 266, ghost_leading 46, ghost_trailing 70, correction_density_interior 16063, force_deep_interior 12150, expand 107, install_leading 4, install_trailing 2, append_departed 4, correction_density_boundary 11069, density 92, force 11314; sim 5: predict 498, voxel 289, ghost_leading 47, ghost_trailing 70, correction_density_interior 18267, force_deep_interior 14447, expand 108, install_leading 4, install_trailing 2, append_departed 4, correction_density_boundary 11143, density 102, force 11626; sim 6: predict 459, voxel 266, ghost_leading 47, ghost_trailing 70, correction_density_interior 16287, force_deep_interior 12306, expand 107, install_leading 4, install_trailing 2, append_departed 4, correction_density_boundary 11013, density 91, force 11475; sim 7: predict 460, voxel 289, ghost_leading 47, correction_density_interior 21315, force_deep_interior 18145, expand 38, install_leading 4, append_departed 4, correction_density_boundary 5789, density 90, force 6027

**trace_3d_n192_K1** — 8M (200^3) K = 1, detail full, step period mean of sims 51.902 ms, G1 share: n/a (no pool demand)

| phase | kernel | us / step (mean of sims) | ns / particle | share of the step | sims |
|---|---|---|---|---|---|
| A | predict | 462.9 | 0.058 | 0.9 % | 1 |
| A | voxel | 287.1 | 0.036 | 0.6 % | 1 |
| B | correction_density_interior | 25,720.5 | 3.215 | 49.6 % | 1 |
| B | force_deep_interior | 24,934.0 | 3.117 | 48.0 % | 1 |
| C | correction_density_boundary | 0.2 | 0.000 | 0.0 % | 1 |
| C | density | 61.0 | 0.008 | 0.1 % | 1 |
| C | force | 0.2 | 0.000 | 0.0 % | 1 |

per sim (us / step): sim 0: predict 463, voxel 287, correction_density_interior 25721, force_deep_interior 24934, correction_density_boundary 0, density 61, force 0

**trace_3d_n392_K1** — 64M (400^3) K = 1, detail full, step period mean of sims 396.443 ms, G1 share: n/a (no pool demand)

| phase | kernel | us / step (mean of sims) | ns / particle | share of the step | sims |
|---|---|---|---|---|---|
| A | predict | 3,899.5 | 0.061 | 1.0 % | 1 |
| A | voxel | 2,144.8 | 0.034 | 0.5 % | 1 |
| B | correction_density_interior | 186,150.9 | 2.909 | 47.0 % | 1 |
| B | force_deep_interior | 185,186.7 | 2.894 | 46.7 % | 1 |
| C | correction_density_boundary | 0.2 | 0.000 | 0.0 % | 1 |
| C | density | 721.3 | 0.011 | 0.2 % | 1 |
| C | force | 0.2 | 0.000 | 0.0 % | 1 |

per sim (us / step): sim 0: predict 3899, voxel 2145, correction_density_interior 186151, force_deep_interior 185187, correction_density_boundary 0, density 721, force 0
