## Per-kernel time per step (E8 phase 2; medians over the steady steps, mean over the sims)

**3d_n192_K8** — 8M (200^3) K = 8, detail full, step period mean of sims 8.785 ms, G1 redundancy share 9.93 % (6.57–11.82 % over the sims)

| phase | kernel | us / step (mean of sims) | ns / particle | share of the step | sims |
|---|---|---|---|---|---|
| A | predict | 27.2 | 0.027 | 0.3 % | 8 |
| A | voxel | 30.0 | 0.030 | 0.3 % | 8 |
| A | ghost_trailing | 16.1 | 0.016 | 0.2 % | 7 |
| B | correction_density_interior | 1,382.2 | 1.355 | 15.7 % | 8 |
| B | force_deep_interior | 394.7 | 0.353 | 4.5 % | 8 |
| C | expand | 19.3 | 0.020 | 0.2 % | 8 |
| C | install_trailing | 3.2 | 0.003 | 0.0 % | 7 |
| C | append_departed | 3.8 | 0.004 | 0.0 % | 8 |
| C | correction_density_boundary | 2,617.4 | 2.661 | 29.8 % | 8 |
| C | density | 10.4 | 0.010 | 0.1 % | 8 |
| C | force | 2,721.7 | 2.767 | 31.0 % | 8 |
| A | ghost_leading | 14.3 | 0.015 | 0.2 % | 7 |
| C | install_leading | 4.1 | 0.004 | 0.0 % | 7 |

per sim (us / step): sim 0: predict 33, voxel 33, ghost_trailing 14, correction_density_interior 2294, force_deep_interior 1554, expand 12, install_trailing 4, append_departed 2, correction_density_boundary 1725, density 11, force 1734; sim 1: predict 25, voxel 29, ghost_leading 14, ghost_trailing 16, correction_density_interior 1103, force_deep_interior 10, expand 22, install_leading 4, install_trailing 2, append_departed 4, correction_density_boundary 2883, density 10, force 3012; sim 2: predict 25, voxel 29, ghost_leading 14, ghost_trailing 16, correction_density_interior 1077, force_deep_interior 7, expand 22, install_leading 4, install_trailing 2, append_departed 4, correction_density_boundary 2813, density 10, force 2936; sim 3: predict 25, voxel 29, ghost_leading 14, ghost_trailing 16, correction_density_interior 1083, force_deep_interior 7, expand 22, install_leading 4, install_trailing 4, append_departed 4, correction_density_boundary 2869, density 10, force 3004; sim 4: predict 25, voxel 29, ghost_leading 14, ghost_trailing 16, correction_density_interior 1067, force_deep_interior 7, expand 22, install_leading 4, install_trailing 2, append_departed 4, correction_density_boundary 2837, density 10, force 2971; sim 5: predict 25, voxel 29, ghost_leading 14, ghost_trailing 16, correction_density_interior 1091, force_deep_interior 6, expand 23, install_leading 4, install_trailing 4, append_departed 4, correction_density_boundary 2905, density 10, force 3034; sim 6: predict 25, voxel 29, ghost_leading 14, ghost_trailing 16, correction_density_interior 1099, force_deep_interior 6, expand 22, install_leading 4, install_trailing 4, append_departed 4, correction_density_boundary 3112, density 10, force 3171; sim 7: predict 35, voxel 35, ghost_leading 14, correction_density_interior 2245, force_deep_interior 1561, expand 11, install_leading 4, append_departed 4, correction_density_boundary 1795, density 10, force 1911

**3d_n392_K8** — 64M (400^3) K = 8, detail full, step period mean of sims 53.609 ms, G1 redundancy share 6.05 % (3.60–7.00 % over the sims)

| phase | kernel | us / step (mean of sims) | ns / particle | share of the step | sims |
|---|---|---|---|---|---|
| A | predict | 470.8 | 0.059 | 0.9 % | 8 |
| A | voxel | 276.3 | 0.035 | 0.5 % | 8 |
| A | ghost_trailing | 66.3 | 0.008 | 0.1 % | 7 |
| B | correction_density_interior | 16,017.4 | 1.997 | 29.9 % | 8 |
| B | force_deep_interior | 12,866.7 | 1.601 | 24.0 % | 8 |
| C | expand | 89.0 | 0.011 | 0.2 % | 8 |
| C | install_trailing | 2.6 | 0.000 | 0.0 % | 7 |
| C | append_departed | 4.1 | 0.001 | 0.0 % | 8 |
| C | correction_density_boundary | 9,745.9 | 1.227 | 18.2 % | 8 |
| C | density | 93.9 | 0.012 | 0.2 % | 8 |
| C | force | 9,971.2 | 1.255 | 18.6 % | 8 |
| A | ghost_leading | 46.0 | 0.006 | 0.1 % | 7 |
| C | install_leading | 4.1 | 0.001 | 0.0 % | 7 |

per sim (us / step): sim 0: predict 463, voxel 291, ghost_trailing 47, correction_density_interior 19044, force_deep_interior 16270, expand 38, install_trailing 4, append_departed 4, correction_density_boundary 5743, density 91, force 5775; sim 1: predict 461, voxel 264, ghost_leading 45, ghost_trailing 70, correction_density_interior 14480, force_deep_interior 11069, expand 105, install_leading 4, install_trailing 2, append_departed 4, correction_density_boundary 11057, density 92, force 11457; sim 2: predict 500, voxel 287, ghost_leading 46, ghost_trailing 69, correction_density_interior 16240, force_deep_interior 12917, expand 107, install_leading 4, install_trailing 2, append_departed 4, correction_density_boundary 11132, density 102, force 11402; sim 3: predict 462, voxel 264, ghost_leading 47, ghost_trailing 70, correction_density_interior 14162, force_deep_interior 10896, expand 106, install_leading 4, install_trailing 2, append_departed 4, correction_density_boundary 10931, density 92, force 11128; sim 4: predict 460, voxel 264, ghost_leading 45, ghost_trailing 70, correction_density_interior 14205, force_deep_interior 10934, expand 105, install_leading 4, install_trailing 2, append_departed 4, correction_density_boundary 10998, density 92, force 11212; sim 5: predict 502, voxel 287, ghost_leading 47, ghost_trailing 69, correction_density_interior 16350, force_deep_interior 13016, expand 107, install_leading 4, install_trailing 4, append_departed 4, correction_density_boundary 11205, density 102, force 11513; sim 6: predict 461, voxel 264, ghost_leading 47, ghost_trailing 69, correction_density_interior 14317, force_deep_interior 11036, expand 106, install_leading 4, install_trailing 2, append_departed 4, correction_density_boundary 11073, density 92, force 11226; sim 7: predict 458, voxel 289, ghost_leading 45, correction_density_interior 19341, force_deep_interior 16796, expand 38, install_leading 4, append_departed 4, correction_density_boundary 5828, density 88, force 6056

**trace_3d_n192_K1** — 8M (200^3) K = 1, detail full, step period mean of sims 45.870 ms, G1 share: n/a (no pool demand)

| phase | kernel | us / step (mean of sims) | ns / particle | share of the step | sims |
|---|---|---|---|---|---|
| A | predict | 464.1 | 0.058 | 1.0 % | 1 |
| A | voxel | 278.9 | 0.035 | 0.6 % | 1 |
| B | correction_density_interior | 22,660.4 | 2.833 | 49.4 % | 1 |
| B | force_deep_interior | 21,837.3 | 2.730 | 47.6 % | 1 |
| C | correction_density_boundary | 0.2 | 0.000 | 0.0 % | 1 |
| C | density | 67.3 | 0.008 | 0.1 % | 1 |
| C | force | 0.2 | 0.000 | 0.0 % | 1 |

per sim (us / step): sim 0: predict 464, voxel 279, correction_density_interior 22660, force_deep_interior 21837, correction_density_boundary 0, density 67, force 0

**trace_3d_n392_K1** — 64M (400^3) K = 1, detail full, step period mean of sims 381.840 ms, G1 share: n/a (no pool demand)

| phase | kernel | us / step (mean of sims) | ns / particle | share of the step | sims |
|---|---|---|---|---|---|
| A | predict | 3,899.8 | 0.061 | 1.0 % | 1 |
| A | voxel | 2,136.5 | 0.033 | 0.6 % | 1 |
| B | correction_density_interior | 179,043.9 | 2.798 | 46.9 % | 1 |
| B | force_deep_interior | 178,712.5 | 2.792 | 46.8 % | 1 |
| C | correction_density_boundary | 0.2 | 0.000 | 0.0 % | 1 |
| C | density | 720.7 | 0.011 | 0.2 % | 1 |
| C | force | 0.2 | 0.000 | 0.0 % | 1 |

per sim (us / step): sim 0: predict 3900, voxel 2136, correction_density_interior 179044, force_deep_interior 178713, correction_density_boundary 0, density 721, force 0
