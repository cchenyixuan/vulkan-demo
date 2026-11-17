| 比较（算例、步数窗口、K、两侧） | 结论 | 比较时本分支的提交 | 记录 |
|---|---|---|---|
| adami_n250_canonical K=1 base_self | BIT-IDENTICAL | 029313a | wc-20261117-171905-2971 |
| adami_n250_canonical K=1 base_vs_this | BIT-IDENTICAL | 029313a | wc-20261117-171905-88f1 |
| adami_n250_canonical K=1 this_self | BIT-IDENTICAL | 029313a | wc-20261117-171905-aed3 |
| adami_n250_canonical K=2 base_self | BIT-IDENTICAL | a2149ed+dirty | wc-20261117-182512-b196 |
| adami_n250_canonical K=2 base_vs_this | BIT-IDENTICAL | a2149ed+dirty | wc-20261117-182512-6148 |
| adami_n250_canonical K=2 this_self | BIT-IDENTICAL | a2149ed+dirty | wc-20261117-182512-aa71 |
| adami_n250_nodefrag_3000 K=1 base_self | NOT bit-identical | 029313a | wc-20261117-172007-7816 |
| adami_n250_nodefrag_3000 K=1 base_vs_this | NOT bit-identical | 029313a | wc-20261117-172007-4fb0 |
| adami_n250_nodefrag_3000 K=1 this_self | NOT bit-identical | 029313a | wc-20261117-172007-1fb8 |
| adami_n250_plain K=1 base_self | NOT bit-identical | 029313a | wc-20261117-171921-d234 |
| adami_n250_plain K=1 base_vs_this | NOT bit-identical | 029313a | wc-20261117-171921-e91f |
| adami_n250_plain K=1 this_self | NOT bit-identical | 029313a | wc-20261117-171921-a87f |
| crossing 700 steps: V7_DENSITY_BUDGET=1 vs 0 | ok | 7b443b7+dirty | wc-20261118-025100-3e6e |
| crossing 999 steps: V7_DENSITY_BUDGET=1 vs 0 | failed: exit code 1 | 7b443b7+dirty | wc-20261118-025101-2207 |
| crossing 999 steps: no budget, two runs | ok | 7b443b7+dirty | wc-20261118-025101-62c7 |
| crossing default: build before the diagnostic switch (a2149ed) vs the build with body_force, K=1 700 steps | ok | d10e144+dirty | wc-20261118-013443-f377 |
| crossing default: build before the diagnostic switch (a2149ed) vs with it, K=1 700 steps | ok | 6a1bada+dirty | wc-20261117-224106-e5d5 |
| crossing default: build before the diagnostic switches (a2149ed) vs the build with the budget variants, K=1 700 steps | ok | e643008+dirty | wc-20261118-021600-3d5c |
| crossing: V7_DENSITY_BUDGET=1 (final build) vs 0, K=1 700 steps | ok | 1932d2b+dirty | wc-20261118-023325-37e2 |
| crossing: V7_DENSITY_BUDGET=1 (twelve regions) vs 0, K=1 700 steps | failed: exit code 1 | 7b443b7+dirty | wc-20261118-024510-f644 |
| crossing: V7_DENSITY_BUDGET=1 vs 0, K=1 700 steps | ok | e643008+dirty | wc-20261118-021637-39a4 |
| simple 700 steps: V7_DENSITY_BUDGET=1 vs 0 | ok | 7b443b7+dirty | wc-20261118-025100-0dc7 |
| simple 999 steps: V7_DENSITY_BUDGET=1 vs 0 | ok | 7b443b7+dirty | wc-20261118-025101-70a2 |
| simple: V7_DENSITY_BUDGET=1 (final build) vs 0, K=1 700 steps | ok | 1932d2b+dirty | wc-20261118-023326-206f |
| simple: V7_DENSITY_BUDGET=1 (own traversal) vs 0, K=1 700 steps | ok | e643008+dirty | wc-20261118-022244-28f8 |
| simple: V7_DENSITY_BUDGET=1 (twelve regions) vs 0, K=1 700 steps | ok | 7b443b7+dirty | wc-20261118-024510-e1ca |
| simple: V7_DENSITY_BUDGET=1 vs 0, K=1 700 steps | failed: exit code 1 | e643008+dirty | wc-20261118-021638-da5d |
| simple_n250_canonical K=1 base_self | BIT-IDENTICAL | 029313a | wc-20261117-171901-3467 |
| simple_n250_canonical K=1 base_vs_this | BIT-IDENTICAL | 029313a | wc-20261117-171901-11cd |
| simple_n250_canonical K=1 this_self | BIT-IDENTICAL | 029313a | wc-20261117-171901-d36c |
| simple_n250_canonical K=2 base_self | BIT-IDENTICAL | a2149ed+dirty | wc-20261117-182501-3690 |
| simple_n250_canonical K=2 base_vs_this | BIT-IDENTICAL | a2149ed+dirty | wc-20261117-182501-0600 |
| simple_n250_canonical K=2 this_self | BIT-IDENTICAL | a2149ed+dirty | wc-20261117-182501-75b2 |
| simple_n250_nodefrag_3000 K=1 base_self | BIT-IDENTICAL | 029313a | wc-20261117-171958-091b |
| simple_n250_nodefrag_3000 K=1 base_vs_this | NOT bit-identical | 029313a | wc-20261117-171959-e3c8 |
| simple_n250_nodefrag_3000 K=1 this_self | NOT bit-identical | 029313a | wc-20261117-171959-0f65 |
| simple_n250_nodefrag_8000 K=1 base_self | NOT bit-identical | 029313a | wc-20261117-172019-d629 |
| simple_n250_nodefrag_8000 K=1 base_vs_this | NOT bit-identical | 029313a | wc-20261117-172020-54c2 |
| simple_n250_nodefrag_8000 K=1 this_self | NOT bit-identical | 029313a | wc-20261117-172019-e756 |
| simple_n250_plain K=1 base_self | NOT bit-identical | 029313a | wc-20261117-171913-01a2 |
| simple_n250_plain K=1 base_vs_this | NOT bit-identical | 029313a | wc-20261117-171913-ad2d |
| simple_n250_plain K=1 this_self | NOT bit-identical | 029313a | wc-20261117-171913-57f0 |
