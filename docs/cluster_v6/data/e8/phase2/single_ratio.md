## v7_single vs chain K = 1 (E8 phase 2; same card, 2 in flight)

Protocol off = --no-defrag on both halves (the chain's fps without defrag events); cadence = Sec. 5.1's, both at the case's defrag cadence 1000 with the window from step warmup + 1 (the chain's fps with its defrag events). A pair whose v7_single half never defragmented (protocol off without the bench's bootstrap defrag) is INVALID and excluded from the means.

| job | case | protocol | pair | steps (warmup + measured) | v7_single fps (bootstrap defrag) | chain K = 1 fps used (with / without defrag) | ratio chain / single | valid | single status / drift / overflow | chain status |
|---|---|---|---|---|---|---|---|---|---|---|
| 02_e8_e8B_1721839 | 8M (200^3) | - | 1 | - | - (NO) | 19.373 (19.372 / 19.373) | - | failed (no RESULT) | None / None / None | pass |
| 02_e8_e8B_1721839 | 8M (200^3) | - | 2 | - | - (NO) | 19.384 (19.383 / 19.384) | - | failed (no RESULT) | None / None / None | pass |
| 02_e8_e8B_1721839 | 8M (200^3) | - | 3 | - | - (NO) | 19.381 (19.379 / 19.381) | - | failed (no RESULT) | None / None / None | pass |
| 02_e8_e8B_1721839 | 8M (200^3) | - | **mean / mean** | | - | - | **none: 0 invalid, 3 failed** | | | |
| 05_e8_e8S_1722130 | 8M (200^3) | off | 1 | 500 + 1000 | 16.480 (NO) | 19.069 (19.069 / 19.069) | 1.1571 | **INVALID** | ok / 0 / {} | pass |
| 05_e8_e8S_1722130 | 8M (200^3) | off | 2 | 500 + 1000 | 16.451 (NO) | 19.130 (19.131 / 19.130) | 1.1629 | **INVALID** | ok / 0 / {} | pass |
| 05_e8_e8S_1722130 | 8M (200^3) | off | 3 | 500 + 1000 | 16.387 (NO) | 19.305 (19.304 / 19.305) | 1.1781 | **INVALID** | ok / 0 / {} | pass |
| 05_e8_e8S_1722130 | 8M (200^3) | off | **mean / mean** | | - | - | **none: 3 invalid, 0 failed** | | | |
| 06_e8_e8S2_1723164 | 8M (200^3) | cadence | 1 | 1000 + 1000 | 18.413 (yes) | 18.267 (18.267 / 18.294) | 0.9920 | yes | ok / 0 / {} | pass |
| 06_e8_e8S2_1723164 | 8M (200^3) | cadence | 2 | 1000 + 1000 | 18.292 (yes) | 18.297 (18.297 / 18.327) | 1.0003 | yes | ok / 0 / {} | pass |
| 06_e8_e8S2_1723164 | 8M (200^3) | cadence | 3 | 1000 + 1000 | 18.326 (yes) | 18.377 (18.377 / 18.409) | 1.0028 | yes | ok / 0 / {} | pass |
| 06_e8_e8S2_1723164 | 8M (200^3) | cadence | **mean / mean** | | 18.344 | 18.314 | **0.9984** (pairs 0.9920–1.0028) | 3 valid pairs | | |

Invalid pairs: 05_e8_e8S_1722130 pair 1 (chain / single 1.1571); 05_e8_e8S_1722130 pair 2 (chain / single 1.1629); 05_e8_e8S_1722130 pair 3 (chain / single 1.1781). Reason: v7_single never defragmented: the bench before 2026-10-22 skipped the bootstrap defrag under --no-defrag while the chain's bootstrap_all always runs one (claude/e8_single_vs_chain_k1.md).

### Bootstrap-defrag control (v7_single alone, --no-defrag, job 1722130's steps; e8_single_control)

| job | case | steps (warmup + measured) | with bootstrap defrag: fps (defrag ms) | without: fps | ratio with / without |
|---|---|---|---|---|---|
| 06_e8_e8S2_1723164 | 8M (200^3) | 500 + 1000 | 19.580 (25.5) | 16.720 | **1.1711** |
