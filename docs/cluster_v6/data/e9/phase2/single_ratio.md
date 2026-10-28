## v7_single vs chain K = 1 (E8 phase 2; same card, 2 in flight)

Protocol off = --no-defrag on both halves (the chain's fps without defrag events); cadence = Sec. 5.1's, both at the case's defrag cadence 1000 with the window from step warmup + 1 (the chain's fps with its defrag events). A pair whose v7_single half never defragmented (protocol off without the bench's bootstrap defrag) is INVALID and excluded from the means.

| job | case | protocol | pair | steps (warmup + measured) | v7_single fps (bootstrap defrag) | chain K = 1 fps used (with / without defrag) | ratio chain / single | valid | single status / drift / overflow | chain status |
|---|---|---|---|---|---|---|---|---|---|---|
