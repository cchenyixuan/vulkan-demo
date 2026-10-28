## v7_single vs chain K = 1 (E8 phase 2; same card, 2 in flight)

Protocol off = --no-defrag on both halves (the chain's fps without defrag events); cadence = Sec. 5.1's, both at the case's defrag cadence 1000 with the window from step warmup + 1 (the chain's fps with its defrag events). A pair whose v7_single half never defragmented (protocol off without the bench's bootstrap defrag) is INVALID and excluded from the means.

| job | case | protocol | pair | steps (warmup + measured) | v7_single fps (bootstrap defrag) | chain K = 1 fps used (with / without defrag) | ratio chain / single | valid | single status / drift / overflow | chain status |
|---|---|---|---|---|---|---|---|---|---|---|
| 02_e9_e9B_1741917 | 8M (200^3) | cadence | 1 | 1000 + 1000 | 20.711 (yes) | 20.679 (20.679 / 20.716) | 0.9985 | yes | ok / 0 / {} | pass |
| 02_e9_e9B_1741917 | 8M (200^3) | cadence | 2 | 1000 + 1000 | 20.756 (yes) | 20.718 (20.718 / 20.758) | 0.9982 | yes | ok / 0 / {} | pass |
| 02_e9_e9B_1741917 | 8M (200^3) | cadence | 3 | 1000 + 1000 | 20.736 (yes) | 20.645 (20.645 / 20.680) | 0.9956 | yes | ok / 0 / {} | pass |
| 02_e9_e9B_1741917 | 8M (200^3) | cadence | **mean / mean** | | 20.734 | 20.680 | **0.9974** (pairs 0.9956–0.9985) | 3 valid pairs | | |
