# E7 rc2 candidate: local validation (item 6 of the batch-1 review)

Candidate: branch `v7-wake-fix`, commit `c56d078` (tree of `experiment/v7` `a4b3a831`), the B2 dest guard (relay of
the reverse worker's source wait, `V7_DEST_GUARD=relay`, `V7_DEST_GUARD_PRECHECK=counter` by default). Reference:
`v7-rc1` (`d0c8dcb`) exported with `git archive`. Rig: 2x RTX 5090, Windows, driver 576.88, 2026-10-10.

| Check | Result | Files |
|---|---|---|
| CPU tests | `_test_transport_relay.py` (K=2 x 10,000 frames per pre-check and rc1 mode, K=3 every pre-check, drain / release / abort, negative controls), `_test_chain_cuts`, `_test_weight_calibration`, `_test_partition_chain`, `_test_seam_layout`: ALL PASS | `cpu_tests/` |
| Bitwise gate (canonical lists, rc1 vs rc2) | BIT-IDENTICAL: K=2 n250 200 steps under every switch setting (counter, zero_wait, none, wait; and rc2 vs rc2), K=2 n250 999 steps with `--transport-extension`, K=2 3-D 1M, K=3 and K=4 n250 300 steps, K=4 2-D 1M 300 steps | `gates.out`, `gates/bitwise/` |
| Validation layer | K=2 and K=4 2-D 1M, 2000 steps: 0 messages, drift 0; every dest guard completed on the counter or the relay (fallback 0, blocking 0) | `gates/validation/` |
| Step trace, 2-D 2M K=2, 3000 steps (warmup 1000) | late wakes 0 in rc1 and rc2 (this rig had none in rc1 either); blocked dest guards complete p50 44 / 53 us after the signal (rc1 26 / 28 us), p99 121 / 199 us (rc1 86 / 112 us); run_meta carries `dest_guard` | `trace/trace_rc*`, `trace/late_wake.txt` |
| Interleaved timing, 3 runs per solver (order rc1 rc2, rc2 rc1, rc1 rc2) | 2-D 2M: rc2 vs rc1 +0.37 % (SE 0.39); 2-D 1M (transport exposed, ~1090 fps): +0.12 % (SE 0.23) | `trace/timing_table.txt`, `trace/time*.log` |

Scripts: `run_gates.sh`, `run_trace.sh`, `timing_table.py` (paths point at the session scratchpad where they ran).
