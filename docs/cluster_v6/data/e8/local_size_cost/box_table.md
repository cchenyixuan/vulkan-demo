interface model (cluster fit): c = 5.61 + 8.64 f_w ns; local cubes sit at 0.9720 x that model on average (platform offset); local-anchored interface model from the local cubes: c = 5.344 + 9.69 f_w ns

| case | runs | N_total | f_w (walls incl. lid) | ns/particle (total) | ns/particle (fluid) | corr+dens (ns/total) | force (ns/total) | SM p50 (MHz) | W | pred. interface 5.61 + 8.64 f_w | x local scale | pred. cube interp. (same N) | measured - interface | measured - cube interp. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cube 8M (200^3) | 2 | 8,000,000 | 0.1153 | 6.4562 (±0.0125) | 7.2974 | 3.2147 | 3.1138 | 2838 | 575.0 | 6.6059 | 6.4211 | 6.4562 (anchor) | +0.54 % (scaled) | +0.00 % |
| cube 16M (252^3) | 2 | 16,003,008 | 0.0922 | 6.2535 (±0.0016) | 6.8890 | 3.1044 | 3.0221 | 2812 | 575.0 | 6.4070 | 6.2278 | 6.2535 (anchor) | +0.41 % (scaled) | +0.00 % |
| cube 32M (320^3) | 2 | 32,768,000 | 0.0731 | 6.0364 (±0.0031) | 6.5127 | 2.9843 | 2.9283 | 2801 | 575.0 | 6.2419 | 6.0674 | 6.0364 (anchor) | -0.51 % (scaled) | +0.00 % |
| cube 64M (400^3) | 2 | 64,000,000 | 0.0588 | 5.9203 (±0.0008) | 6.2902 | 2.9154 | 2.8817 | 2794 | 574.9 | 6.1181 | 5.9470 | 5.9203 (anchor) | -0.45 % (scaled) | +0.00 % |
| box 2 x 8M (192^2 x 384) | 2 | 15,680,000 | 0.0972 | 6.5625 (±0.0032) | 7.2691 | 3.2486 | 3.1856 | 2833 | 574.9 | 6.4499 | 6.2695 | 6.2595 | +4.46 % (scaled) | +4.62 % |
| box 4 x 8M (192^2 x 768) | 2 | 31,040,000 | 0.0879 | 6.1993 (±0.0008) | 6.7967 | 3.0705 | 3.0048 | 2820 | 575.0 | 6.3695 | 6.1913 | 6.0528 | +0.13 % (scaled) | +2.36 % |
| box 8 x 8M (192^2 x 1536) | 2 | 61,760,000 | 0.0832 | 6.0168 (±0.0049) | 6.5627 | 2.9812 | 2.9135 | 2812 | 575.0 | 6.3286 | 6.1516 | 5.9265 | -2.24 % (scaled) | +1.50 % |

box vs cube of the same N (local): box 2 x 8M (192^2 x 384): measured 6.5625, cube interp. 6.2595, interface (scaled) 6.2695; box 4 x 8M (192^2 x 768): measured 6.1993, cube interp. 6.0528, interface (scaled) 6.1913; box 8 x 8M (192^2 x 1536): measured 6.0168, cube interp. 5.9265, interface (scaled) 6.1516