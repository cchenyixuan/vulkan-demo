### cube 8M (200^3): pieces (m = 8, by particle order = x from -x to +x; live N = 8,000,000; 50 x-columns of 4 cells: -x end wall, 48 fluid columns, +x end wall; fluid column = 160,000 particles = fluid 147,456 + lid 3,072 + side walls 9,472; end wall column = 160,000)

| piece | particles | columns covered (x order) | fluid | lid | wall | wall share | correction_density_interior (ms) | per 1M (ms) | force_deep_interior_scratch (ms) | per 1M (ms) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1,000,000 | -x end wall (100 %) .. fluid column 6 (25 %) | 774,144 | 16,128 | 209,728 | 21.0 % | 3.526 | 3.526 | 3.500 | 3.500 |
| 1 | 1,000,000 | fluid column 6 (75 %) .. fluid column 12 (50 %) | 921,600 | 19,200 | 59,200 | 5.9 % | 3.556 | 3.556 | 3.552 | 3.552 |
| 2 | 1,000,000 | fluid column 12 (50 %) .. fluid column 18 (75 %) | 921,600 | 19,200 | 59,200 | 5.9 % | 3.558 | 3.558 | 3.535 | 3.535 |
| 3 | 1,000,000 | fluid column 18 (25 %) .. fluid column 24 (100 %) | 921,600 | 19,200 | 59,200 | 5.9 % | 3.597 | 3.597 | 3.601 | 3.601 |
| 4 | 1,000,000 | fluid column 25 (100 %) .. fluid column 31 (25 %) | 921,600 | 19,200 | 59,200 | 5.9 % | 3.612 | 3.612 | 3.558 | 3.558 |
| 5 | 1,000,000 | fluid column 31 (75 %) .. fluid column 37 (50 %) | 921,600 | 19,200 | 59,200 | 5.9 % | 3.504 | 3.504 | 3.488 | 3.488 |
| 6 | 1,000,000 | fluid column 37 (50 %) .. fluid column 43 (75 %) | 921,600 | 19,200 | 59,200 | 5.9 % | 3.482 | 3.482 | 3.507 | 3.507 |
| 7 | 1,000,000 | fluid column 43 (25 %) .. +x end wall (100 %) | 774,144 | 16,128 | 209,728 | 21.0 % | 3.388 | 3.388 | 3.259 | 3.259 |
- correction_density_interior: pieces sum 28.223 ms; min 3.388 (piece 7), max 3.612 (piece 4), max / median 1.020; per-run medians agree within 0.075 ms
- force_deep_interior_scratch: pieces sum 28.001 ms; min 3.259 (piece 7), max 3.601 (piece 3), max / median 1.023; per-run medians agree within 0.093 ms

### cube 64M (400^3): pieces (m = 8, by particle order = x from -x to +x; live N = 64,000,000; 100 x-columns of 4 cells: -x end wall, 98 fluid columns, +x end wall; fluid column = 640,000 particles = fluid 614,656 + lid 6,272 + side walls 19,072; end wall column = 640,000)

| piece | particles | columns covered (x order) | fluid | lid | wall | wall share | correction_density_interior (ms) | per 1M (ms) | force_deep_interior_scratch (ms) | per 1M (ms) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 8,000,000 | -x end wall (100 %) .. fluid column 12 (50 %) | 7,068,544 | 72,128 | 859,328 | 10.7 % | 25.130 | 3.141 | 24.105 | 3.013 |
| 1 | 8,000,000 | fluid column 12 (50 %) .. fluid column 24 (100 %) | 7,683,200 | 78,400 | 238,400 | 3.0 % | 25.073 | 3.134 | 25.114 | 3.139 |
| 2 | 8,000,000 | fluid column 25 (100 %) .. fluid column 37 (50 %) | 7,683,200 | 78,400 | 238,400 | 3.0 % | 23.296 | 2.912 | 23.522 | 2.940 |
| 3 | 8,000,000 | fluid column 37 (50 %) .. fluid column 49 (100 %) | 7,683,200 | 78,400 | 238,400 | 3.0 % | 23.206 | 2.901 | 23.428 | 2.928 |
| 4 | 8,000,000 | fluid column 50 (100 %) .. fluid column 62 (50 %) | 7,683,200 | 78,400 | 238,400 | 3.0 % | 23.238 | 2.905 | 23.380 | 2.922 |
| 5 | 8,000,000 | fluid column 62 (50 %) .. fluid column 74 (100 %) | 7,683,200 | 78,400 | 238,400 | 3.0 % | 23.423 | 2.928 | 23.575 | 2.947 |
| 6 | 8,000,000 | fluid column 75 (100 %) .. fluid column 87 (50 %) | 7,683,200 | 78,400 | 238,400 | 3.0 % | 23.468 | 2.933 | 23.651 | 2.956 |
| 7 | 8,000,000 | fluid column 87 (50 %) .. +x end wall (100 %) | 7,068,544 | 72,128 | 859,328 | 10.7 % | 22.818 | 2.852 | 21.729 | 2.716 |
- correction_density_interior: pieces sum 189.652 ms; min 22.818 (piece 7), max 25.130 (piece 0), max / median 1.076; per-run medians agree within 0.127 ms
- force_deep_interior_scratch: pieces sum 188.503 ms; min 21.729 (piece 7), max 25.114 (piece 1), max / median 1.066; per-run medians agree within 0.202 ms
