### prefix-sum defrag series: per run (kernel times = step-trace medians over steps 100-299, ms)

| run | defrag | spec 46 | split cd/fo | step (ms) | corr+dens (ms) | force (ms) | period (ms) | SM MHz | W | drift | bootstrap defrag (ms) | host scan (ms) | pid order after defrag |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 64m_host_cd8t_r1 | host | 1 | 8/0 | 376.550 | 184.985 | 182.961 | 374.714 | 2767 | 575 | 0 | 232.3 | 26.3 | exact voxel order |
| 64m_host_cd8t_r2 | host | 1 | 8/0 | 376.700 | 185.062 | 183.043 | 374.860 | 2760 | 575 | 0 | 231.6 | 25.8 | exact voxel order |
| 64m_host_fo8t_r1 | host | 1 | 0/8 | 377.150 | 183.704 | 184.901 | 375.398 | 2767 | 575 | 0 | 232.8 | 26.9 | exact voxel order |
| 64m_host_fo8t_r2 | host | 1 | 0/8 | 377.050 | 183.666 | 184.829 | 375.274 | 2767 | 575 | 0 | 232.7 | 27.0 | exact voxel order |
| 64m_host_m1_r1 | host | 1 | 0/0 | 375.200 | 183.608 | 182.969 | 373.381 | 2760 | 575 | 0 | 233.7 | 27.7 | exact voxel order |
| 64m_host_m1_r2 | host | 1 | 0/0 | 375.800 | 183.935 | 183.316 | 374.034 | 2760 | 575 | 0 | 232.3 | 26.2 | exact voxel order |
| 64m_off_m1_ctl | off | 0 | 0/0 | 383.000 | 187.860 | 186.548 | 381.195 | 2760 | 575 | 0 | 205.6 | - | NOT voxel order: 56,409 descents, 57,004,032 pids behind, 0 voxel-id-0 pids |
| 8m_host_cd8t_r1 | host | 1 | 8/0 | 47.700 | 24.597 | 22.024 | 47.449 | 2812 | 575 | 0 | 27.5 | 2.9 | exact voxel order |
| 8m_host_cd8t_r2 | host | 1 | 8/0 | 47.750 | 24.644 | 22.054 | 47.537 | 2812 | 575 | 0 | 28.9 | 4.3 | exact voxel order |
| 8m_host_fo8t_r1 | host | 1 | 0/8 | 47.950 | 22.918 | 23.945 | 47.695 | 2816 | 575 | 0 | 30.6 | 6.0 | exact voxel order |
| 8m_host_fo8t_r2 | host | 1 | 0/8 | 47.900 | 22.889 | 23.937 | 47.655 | 2812 | 575 | 0 | 53.1 | 4.4 | exact voxel order |
| 8m_host_m1_r1 | host | 1 | 0/0 | 46.300 | 23.074 | 22.204 | 46.109 | 2790 | 575 | 0 | 30.0 | 5.5 | exact voxel order |
| 8m_host_m1_r2 | host | 1 | 0/0 | 46.350 | 23.086 | 22.206 | 46.127 | 2797 | 575 | 0 | 35.6 | 3.8 | exact voxel order |
| 8m_off_m1_ctl | off | 0 | 0/0 | 52.550 | 26.113 | 25.344 | 52.288 | 2812 | 575 | 0 | 79.0 | - | NOT voxel order: 41,582 descents, 7,981,056 pids behind, 0 voxel-id-0 pids |

### per-particle cost: prefix defrag vs default defrag (K = 1, 300 steps window 100-299, 575 W)

| cube | defrag | runs | step (ms) | ns / particle | corr+dens (ms) | ns / particle | force (ms) | ns / particle | SM MHz |
|---|---|---|---|---|---|---|---|---|---|
| cube 8M (200^3) | default (7.17 m1_plain) | 2 (52.100, 52.350) | 52.225 | 6.528 | 25.967 | 3.246 | 25.192 | 3.149 | 2807 |
| cube 8M (200^3) | default (this batch, control) | 1 (52.550) | 52.550 | 6.569 | 26.113 | 3.264 | 25.344 | 3.168 | 2812 |
| cube 8M (200^3) | prefix (host scan) | 2 (46.300, 46.350) | 46.325 | 5.791 | 23.080 | 2.885 | 22.205 | 2.776 | 2794 |
| cube 8M (200^3) | **prefix - default (all default runs)** | 3 default | **-6.008** (-11.5 %) | -0.751 | **-2.936** (-11.3 %; 7.16 fixed term +2.63) | -0.367 | **-3.038** (-12.0 %; 7.16 fixed term +2.34) | -0.380 | |
| cube 64M (400^3) | default (7.17 m1_plain) | 2 (383.350, 383.250) | 383.300 | 5.989 | 187.929 | 2.936 | 186.653 | 2.916 | 2760 |
| cube 64M (400^3) | default (this batch, control) | 1 (383.000) | 383.000 | 5.984 | 187.860 | 2.935 | 186.548 | 2.915 | 2760 |
| cube 64M (400^3) | prefix (host scan) | 2 (375.200, 375.800) | 375.500 | 5.867 | 183.772 | 2.871 | 183.143 | 2.862 | 2760 |
| cube 64M (400^3) | **prefix - default (all default runs)** | 3 default | **-7.700** (-2.0 %) | -0.120 | **-4.135** (-2.2 %; 7.16 fixed term +1.64) | -0.065 | **-3.475** (-1.9 %; 7.16 fixed term +3.88) | -0.054 | |

### layout audit after the bootstrap defrag (position_voxel_id read back; own pids in order; pieces = 8 equal pid ranges)

**cube 8M (200^3), default defrag (atomicAdd order)** (8m_off, 1 audited runs; first run shown): 8,000,000 live pids, 125,000 contiguous voxel blocks for 125,000 non-empty voxels, 41,582 block transitions go to a lower voxel id, 7,981,056 pids (99.8 %) sit behind an earlier pid's voxel -> **not voxel order**; material totals fluid 7,077,888 wall 774,656 lid 147,456

| piece | pids | x columns covered (first..last in pid order; min..max) | distinct columns | fluid | wall | lid | wall share |
|---|---|---|---|---|---|---|---|
| 0 | 0..1,000,000 | 39..1 (1..50) | 49 | 886,592 | 94,720 | 18,688 | 9.5 % |
| 1 | 1,000,000..2,000,000 | 21..37 (1..50) | 50 | 892,416 | 87,936 | 19,648 | 8.8 % |
| 2 | 2,000,000..3,000,000 | 37..37 (1..50) | 50 | 902,400 | 79,936 | 17,664 | 8.0 % |
| 3 | 3,000,000..4,000,000 | 41..8 (1..50) | 50 | 893,440 | 87,744 | 18,816 | 8.8 % |
| 4 | 4,000,000..5,000,000 | 31..25 (1..50) | 50 | 871,296 | 110,272 | 18,432 | 11.0 % |
| 5 | 5,000,000..6,000,000 | 1..21 (1..50) | 50 | 884,288 | 96,768 | 18,944 | 9.7 % |
| 6 | 6,000,000..7,000,000 | 21..28 (1..50) | 50 | 882,752 | 97,920 | 19,328 | 9.8 % |
| 7 | 7,000,000..8,000,000 | 28..42 (1..50) | 46 | 864,704 | 119,360 | 15,936 | 11.9 % |

**cube 8M (200^3), prefix defrag (voxel-id order)** (8m_host, 6 audited runs; first run shown): 8,000,000 live pids, 125,000 contiguous voxel blocks for 125,000 non-empty voxels, 0 block transitions go to a lower voxel id, 0 pids (0.0 %) sit behind an earlier pid's voxel -> **exact voxel order**; material totals fluid 7,077,888 wall 774,656 lid 147,456

| piece | pids | x columns covered (first..last in pid order; min..max) | distinct columns | fluid | wall | lid | wall share |
|---|---|---|---|---|---|---|---|
| 0 | 0..1,000,000 | 1..7 (1..7) | 7 | 772,608 | 211,328 | 16,064 | 21.1 % |
| 1 | 1,000,000..2,000,000 | 7..13 (7..13) | 7 | 923,136 | 57,600 | 19,264 | 5.8 % |
| 2 | 2,000,000..3,000,000 | 13..19 (13..19) | 7 | 923,136 | 57,664 | 19,200 | 5.8 % |
| 3 | 3,000,000..4,000,000 | 19..25 (19..25) | 7 | 920,064 | 60,736 | 19,200 | 6.1 % |
| 4 | 4,000,000..5,000,000 | 26..32 (26..32) | 7 | 920,064 | 60,800 | 19,136 | 6.1 % |
| 5 | 5,000,000..6,000,000 | 32..38 (32..38) | 7 | 923,136 | 57,600 | 19,264 | 5.8 % |
| 6 | 6,000,000..7,000,000 | 38..44 (38..44) | 7 | 923,136 | 57,664 | 19,200 | 5.8 % |
| 7 | 7,000,000..8,000,000 | 44..50 (44..50) | 7 | 772,608 | 211,264 | 16,128 | 21.1 % |

**cube 64M (400^3), default defrag (atomicAdd order)** (64m_off, 1 audited runs; first run shown): 64,000,000 live pids, 1,000,000 contiguous voxel blocks for 1,000,000 non-empty voxels, 56,409 block transitions go to a lower voxel id, 57,004,032 pids (89.1 %) sit behind an earlier pid's voxel -> **not voxel order**; material totals fluid 60,236,288 wall 3,149,056 lid 614,656

| piece | pids | x columns covered (first..last in pid order; min..max) | distinct columns | fluid | wall | lid | wall share |
|---|---|---|---|---|---|---|---|
| 0 | 0..8,000,000 | 2..23 (1..25) | 25 | 7,033,664 | 842,048 | 124,288 | 10.5 % |
| 1 | 8,000,000..16,000,000 | 23..20 (1..26) | 26 | 7,717,440 | 255,168 | 27,392 | 3.2 % |
| 2 | 16,000,000..24,000,000 | 20..38 (3..38) | 28 | 7,682,048 | 239,872 | 78,080 | 3.0 % |
| 3 | 24,000,000..32,000,000 | 38..51 (38..51) | 14 | 7,683,712 | 238,272 | 78,016 | 3.0 % |
| 4 | 32,000,000..40,000,000 | 51..63 (50..63) | 14 | 7,682,880 | 238,400 | 78,720 | 3.0 % |
| 5 | 40,000,000..48,000,000 | 63..76 (63..76) | 14 | 7,685,696 | 236,288 | 78,016 | 3.0 % |
| 6 | 48,000,000..56,000,000 | 76..88 (75..88) | 14 | 7,680,832 | 240,448 | 78,720 | 3.0 % |
| 7 | 56,000,000..64,000,000 | 88..100 (88..100) | 13 | 7,070,016 | 858,560 | 71,424 | 10.7 % |

**cube 64M (400^3), prefix defrag (voxel-id order)** (64m_host, 6 audited runs; first run shown): 64,000,000 live pids, 1,000,000 contiguous voxel blocks for 1,000,000 non-empty voxels, 0 block transitions go to a lower voxel id, 0 pids (0.0 %) sit behind an earlier pid's voxel -> **exact voxel order**; material totals fluid 60,236,288 wall 3,149,056 lid 614,656

| piece | pids | x columns covered (first..last in pid order; min..max) | distinct columns | fluid | wall | lid | wall share |
|---|---|---|---|---|---|---|---|
| 0 | 0..8,000,000 | 1..13 (1..13) | 13 | 7,068,544 | 859,328 | 72,128 | 10.7 % |
| 1 | 8,000,000..16,000,000 | 13..25 (13..25) | 13 | 7,683,200 | 238,400 | 78,400 | 3.0 % |
| 2 | 16,000,000..24,000,000 | 26..38 (26..38) | 13 | 7,683,200 | 238,400 | 78,400 | 3.0 % |
| 3 | 24,000,000..32,000,000 | 38..50 (38..50) | 13 | 7,683,200 | 238,400 | 78,400 | 3.0 % |
| 4 | 32,000,000..40,000,000 | 51..63 (51..63) | 13 | 7,683,200 | 238,400 | 78,400 | 3.0 % |
| 5 | 40,000,000..48,000,000 | 63..75 (63..75) | 13 | 7,683,200 | 238,400 | 78,400 | 3.0 % |
| 6 | 48,000,000..56,000,000 | 76..88 (76..88) | 13 | 7,683,200 | 238,400 | 78,400 | 3.0 % |
| 7 | 56,000,000..64,000,000 | 88..100 (88..100) | 13 | 7,068,544 | 859,328 | 72,128 | 10.7 % |

### m = 8 piece durations: prefix defrag vs default defrag (ms; mean over two runs of the per-step medians over steps 100-299; piece 0 = lowest pids)

**cube 8M (200^3), correction_density_interior** (prefix: 8m_host_cd8t_r1, 8m_host_cd8t_r2; default: 7.17 8m_cd8t_r1, 8m_cd8t_r2)

| piece | x columns (prefix, exact) | pids | prefix (ms) | prefix per 1M (ms) | default (ms) | default per 1M (ms) | prefix - default |
|---|---|---|---|---|---|---|---|
| 0 | 1..7 | 1,000,000 | 2.987 | 2.987 | 3.526 | 3.526 | -0.539 |
| 1 | 7..13 | 1,000,000 | 3.119 | 3.119 | 3.556 | 3.556 | -0.438 |
| 2 | 13..19 | 1,000,000 | 3.122 | 3.122 | 3.558 | 3.558 | -0.436 |
| 3 | 19..25 | 1,000,000 | 3.087 | 3.087 | 3.597 | 3.597 | -0.510 |
| 4 | 26..32 | 1,000,000 | 3.134 | 3.134 | 3.612 | 3.612 | -0.478 |
| 5 | 32..38 | 1,000,000 | 3.121 | 3.121 | 3.504 | 3.504 | -0.383 |
| 6 | 38..44 | 1,000,000 | 3.121 | 3.121 | 3.482 | 3.482 | -0.360 |
| 7 | 44..50 | 1,000,000 | 2.924 | 2.924 | 3.388 | 3.388 | -0.464 |
| sum / max-median | | | 24.615 / 1.005 | | 28.223 / 1.020 | | -3.608 |
- prefix: max piece 4 (3.134 ms), min piece 7 (2.924 ms), middle-six median 3.121; per-run medians agree within 0.016 ms; the unsplit kernel in these runs: 22.039 ms

**cube 8M (200^3), force_deep_interior_scratch** (prefix: 8m_host_fo8t_r1, 8m_host_fo8t_r2; default: 7.17 8m_fo8t_r1, 8m_fo8t_r2)

| piece | x columns (prefix, exact) | pids | prefix (ms) | prefix per 1M (ms) | default (ms) | default per 1M (ms) | prefix - default |
|---|---|---|---|---|---|---|---|
| 0 | 1..7 | 1,000,000 | 2.654 | 2.654 | 3.500 | 3.500 | -0.846 |
| 1 | 7..13 | 1,000,000 | 3.115 | 3.115 | 3.552 | 3.552 | -0.437 |
| 2 | 13..19 | 1,000,000 | 3.101 | 3.101 | 3.535 | 3.535 | -0.434 |
| 3 | 19..25 | 1,000,000 | 3.101 | 3.101 | 3.601 | 3.601 | -0.500 |
| 4 | 26..32 | 1,000,000 | 3.112 | 3.112 | 3.558 | 3.558 | -0.446 |
| 5 | 32..38 | 1,000,000 | 3.121 | 3.121 | 3.488 | 3.488 | -0.367 |
| 6 | 38..44 | 1,000,000 | 3.097 | 3.097 | 3.507 | 3.507 | -0.410 |
| 7 | 44..50 | 1,000,000 | 2.640 | 2.640 | 3.259 | 3.259 | -0.619 |
| sum / max-median | | | 23.942 / 1.007 | | 28.001 / 1.023 | | -4.059 |
- prefix: max piece 5 (3.121 ms), min piece 7 (2.640 ms), middle-six median 3.107; per-run medians agree within 0.016 ms; the unsplit kernel in these runs: 22.903 ms

**cube 64M (400^3), correction_density_interior** (prefix: 64m_host_cd8t_r1, 64m_host_cd8t_r2; default: 7.17 64m_cd8t_r1, 64m_cd8t_r2)

| piece | x columns (prefix, exact) | pids | prefix (ms) | prefix per 1M (ms) | default (ms) | default per 1M (ms) | prefix - default |
|---|---|---|---|---|---|---|---|
| 0 | 1..13 | 8,000,000 | 22.782 | 2.848 | 25.130 | 3.141 | -2.348 |
| 1 | 13..25 | 8,000,000 | 23.244 | 2.906 | 25.073 | 3.134 | -1.829 |
| 2 | 26..38 | 8,000,000 | 23.235 | 2.904 | 23.296 | 2.912 | -0.061 |
| 3 | 38..50 | 8,000,000 | 23.092 | 2.887 | 23.206 | 2.901 | -0.113 |
| 4 | 51..63 | 8,000,000 | 23.154 | 2.894 | 23.238 | 2.905 | -0.084 |
| 5 | 63..75 | 8,000,000 | 23.277 | 2.910 | 23.423 | 2.928 | -0.146 |
| 6 | 76..88 | 8,000,000 | 23.420 | 2.927 | 23.468 | 2.933 | -0.048 |
| 7 | 88..100 | 8,000,000 | 22.828 | 2.854 | 22.818 | 2.852 | +0.010 |
| sum / max-median | | | 185.033 / 1.010 | | 189.652 / 1.076 | | -4.619 |
- prefix: max piece 6 (23.420 ms), min piece 0 (22.782 ms), middle-six median 23.239; per-run medians agree within 0.017 ms; the unsplit kernel in these runs: 183.002 ms

**cube 64M (400^3), force_deep_interior_scratch** (prefix: 64m_host_fo8t_r1, 64m_host_fo8t_r2; default: 7.17 64m_fo8t_r1, 64m_fo8t_r2)

| piece | x columns (prefix, exact) | pids | prefix (ms) | prefix per 1M (ms) | default (ms) | default per 1M (ms) | prefix - default |
|---|---|---|---|---|---|---|---|
| 0 | 1..13 | 8,000,000 | 21.795 | 2.724 | 24.105 | 3.013 | -2.310 |
| 1 | 13..25 | 8,000,000 | 23.648 | 2.956 | 25.114 | 3.139 | -1.466 |
| 2 | 26..38 | 8,000,000 | 23.551 | 2.944 | 23.522 | 2.940 | +0.030 |
| 3 | 38..50 | 8,000,000 | 23.436 | 2.929 | 23.428 | 2.928 | +0.008 |
| 4 | 51..63 | 8,000,000 | 23.421 | 2.928 | 23.380 | 2.922 | +0.041 |
| 5 | 63..75 | 8,000,000 | 23.558 | 2.945 | 23.575 | 2.947 | -0.017 |
| 6 | 76..88 | 8,000,000 | 23.645 | 2.956 | 23.651 | 2.956 | -0.006 |
| 7 | 88..100 | 8,000,000 | 21.797 | 2.725 | 21.729 | 2.716 | +0.068 |
| sum / max-median | | | 184.850 / 1.007 | | 188.503 / 1.066 | | -3.652 |
- prefix: max piece 1 (23.648 ms), min piece 0 (21.795 ms), middle-six median 23.555; per-run medians agree within 0.015 ms; the unsplit kernel in these runs: 183.685 ms
