## 1. 作业

    === E30 e7_p1 job 1721673 node wqd10nbl09g4 Wed Oct 21 11:54:26 PM CST 2026 local=0 dry=0 solver=v7 ===
    job cpus: pid 3035228's current affinity list: 38-45,102-109
    0, NVIDIA GeForce RTX 5090, 00000000:C8:00.0, GPU-5cbf6f3f-3c29-95bc-c30f-647639435c8d, 580.82.07, 32607 MiB, 575.00 W
    1, NVIDIA GeForce RTX 5090, 00000000:D8:00.0, GPU-43694a93-e305-dee6-d0c6-b5b01d88accd, 580.82.07, 32607 MiB, 575.00 W
    PROVENANCE host=wqd10nbl09g4 slurm_job=1721673 job=p1 line=S harness=0ad381481ca9201382b8504eb1767d5563ead87b experiment_v7_tree=10781f23663c3c3b8c95cbe7235c3fa7653b07aa (v7-e8p1 10781f23663c3c3b8c95cbe7235c3fa7653b07aa) driver=58
    GPU health: 2 GPUs listed without an error on wqd10nbl09g4
      [1] NVIDIA GeForce RTX 5090  type=DISCRETE  api=1.4.312
    [VulkanContextV7] selected: [1] NVIDIA GeForce RTX 5090 on queue_family[0]
    VulkanContextV7[1]: OK — NVIDIA GeForce RTX 5090 (compute qf 0, transfer qf 1, transfer queues 2 (split))
    [caps] v6[0] smi[0] NVIDIA GeForce RTX 5090 bus=0000:c8:00.0 uuid=5cbf6f3f driver=580.82.07 api=1.4.312 vram=31.84GiB period=1.0ns compute_qf=0 transfer_qf=1x2 split=1 domains_KHR=['DEVICE', 'CLOCK_MONOTONIC', 'CLOCK_MONOTONIC_RAW
    [caps] v6[1] smi[1] NVIDIA GeForce RTX 5090 bus=0000:d8:00.0 uuid=43694a93 driver=580.82.07 api=1.4.312 vram=31.84GiB period=1.0ns compute_qf=0 transfer_qf=1x2 split=1 domains_KHR=['DEVICE', 'CLOCK_MONOTONIC', 'CLOCK_MONOTONIC_RAW
    === E7 job p1 (line S) ready at 23:55:41: 5 cases staged
      [0] NVIDIA GeForce RTX 5090  type=DISCRETE  api=1.4.312
      [1] NVIDIA GeForce RTX 5090  type=DISCRETE  api=1.4.312

## 2. 运行（RESULT 行）

| label | status | K | steps | steady fps | wall s | build s | alive start→end | drift / overflow / far / stamps / clamp | seam / NaN | own columns | own particles | device-local + scratch MiB | VRAM peak MiB per card | SM MHz (median / min) | W median | °C max | ns / particle-step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 8m_K1 | pass (rc 0) | 1 | 300 | 19.9 | 24.3 | 8.75 | 8000000→8000000 | 0 / 0 / 0 / 0+0 / 0 | - / None | [52] | [8000000] | 2,641 | 0:2,698 | 2865/180 / 2662/180 | 575/23 | 51/24 | 6.28 |
| 8m_K2_trace | pass (rc 0) | 2 | 300 | 39.0 | 23.2 | 8.69 | 8000000→8000000 | 0 / 0 / 0 / 0+0 / 0 | {'seam': 0, 'column': 26, 'left_overshoot_dx': -0.19, 'right_overshoot_dx': -0.46, 'duplicates': 0, 'ok': True} / False | [26, 26] | [4000000, 4000000] | 1,512/1,512 | 0:1,569/1:1,553 | 2880/2872 / 2857/2872 | 575/563 | 53/49 | 6.41 |
| s48_K1 | pass (rc 0) | 1 | 300 | 819.1 | 2.8 | 1.95 | 175616→175616 | 0 / 0 / 0 / 0+0 / 0 | - / None | [16] | [175616] | 61 | 0:110 | 2400/172 / 2400/172 | 72/23 | 29/26 | 6.95 |
| s48_K2 | pass (rc 0) | 2 | 300 | 526.5 | 4.5 | 3.20 | 175616→175616 | 0 / 0 / 0 / 0+0 / 0 | {'seam': 0, 'column': 8, 'left_overshoot_dx': -0.41, 'right_overshoot_dx': -0.47, 'duplicates': 0, 'ok': True} / False | [8, 8] | [87808, 87808] | 44/44 | 0:101 | 2595/2752 / 2595/2752 | 100/96 | 34/34 | 21.63 |
| weak2_K2 | pass (rc 0) | 2 | 300 | 20.0 | 41.3 | 17.31 | 15680000→15680000 | 0 / 0 / 0 / 0+0 / 0 | {'seam': 0, 'column': 50, 'left_overshoot_dx': -0.19, 'right_overshoot_dx': -0.46, 'duplicates': 0, 'ok': True} / False | [50, 50] | [7840000, 7840000] | 2,818/2,818 | 0:2,875/1:2,859 | 2865/2872 / 2850/2865 | 575/575 | 51/52 | 6.38 |
| 64m_K2 | pass (rc 0) | 2 | 300 | 5.3 | 163.3 | 68.03 | 64000000→64000000 | 0 / 0 / 0 / 0+0 / 0 | {'seam': 0, 'column': 51, 'left_overshoot_dx': -0.02, 'right_overshoot_dx': -0.49, 'duplicates': 0, 'ok': True} / False | [51, 51] | [32000000, 32000000] | 11,454/11,454 | 0:11,511/1:11,495 | 2805/2812 / 2775/2767 | 575/575 | 61/64 | 5.90 |

## 3. 显存：实测峰值对模型

| run | card | slab own columns | model device-local + scratch MiB | measured | model VRAM | measured peak | diff |
|---|---|---|---|---|---|---|---|
| 8m_K1 | 0 | 52 | 2,616 | 2,641 | 2,666 | 2,698 | +1.2 % |
| 8m_K2_trace | 0 | 26 | 1,512 | 1,512 | 1,562 | 1,569 | +0.4 % |
| 8m_K2_trace | 1 | 26 | 1,512 | 1,512 | 1,562 | 1,553 | -0.6 % |
| s48_K1 | 0 | 16 | 58 | 61 | 108 | 110 | +1.8 % |
| s48_K2 | 0 | 8 | 44 | 44 | 94 | 101 | +7.0 % |
| s48_K2 | 1 | 8 | 44 | 44 | 94 | 85 | -10.0 % |
| weak2_K2 | 0 | 50 | 2,819 | 2,818 | 2,869 | 2,875 | +0.2 % |
| weak2_K2 | 1 | 50 | 2,819 | 2,818 | 2,869 | 2,859 | -0.4 % |
| 64m_K2 | 0 | 51 | 11,459 | 11,454 | 11,509 | 11,511 | +0.0 % |
| 64m_K2 | 1 | 51 | 11,459 | 11,454 | 11,509 | 11,495 | -0.1 % |

## 4. 池：区的用量（V7_POOL_PEAKS）与 pool_health

**8m_K1**

    [partition_v7] chain N=1: columns [0,52) of 52; particles 8,000,000
    [e30] sim0 (dev0) pool_health: peak_tail_high_water=0 own_pool_size=9200000 free_margin=9200000 peak_migration_count=0 peak_departed_count=0 departed_pool_size=0

**8m_K2_trace**

    [partition_v7] chain N=2: columns [0,26), [26,52) of 52; particles 4,000,000, 4,000,000
    [partition_v7] seam: ghost_layers=2 keep_departed=1 ghost pool/direction=434,371 (replica regions 2 x 216,320 + migrants 1,731) departed pool/slab=1731/1731
    [e30] sim0 (dev0) pool_health: peak_tail_high_water=0 own_pool_size=4800000 free_margin=4800000 peak_migration_count=0 peak_departed_count=0 departed_pool_size=1731
    [e30] sim1 (dev1) pool_health: peak_tail_high_water=0 own_pool_size=4800000 free_margin=4800000 peak_migration_count=0 peak_departed_count=0 departed_pool_size=1731
    [e30] pool s0_to_s1 inner: capacity=216320 peak=160000 frame_of_peak=0 p999=160000 mean=160000.0 last=160000 frames=300 occupancy=0.7396
    [e30] pool s0_to_s1 outer: capacity=216320 peak=160000 frame_of_peak=0 p999=160000 mean=160000.0 last=160000 frames=300 occupancy=0.7396
    [e30] pool s0_to_s1 migrant: capacity=1731 peak=0 frame_of_peak=0 p999=0 mean=0.0 last=0 frames=300 occupancy=0.0000
    [e30] pool s1_to_s0 inner: capacity=216320 peak=160000 frame_of_peak=0 p999=160000 mean=160000.0 last=160000 frames=300 occupancy=0.7396
    [e30] pool s1_to_s0 outer: capacity=216320 peak=160000 frame_of_peak=0 p999=160000 mean=160000.0 last=160000 frames=300 occupancy=0.7396
    [e30] pool s1_to_s0 migrant: capacity=1731 peak=0 frame_of_peak=0 p999=0 mean=0.0 last=0 frames=300 occupancy=0.0000

**s48_K1**

    [partition_v7] chain N=1: columns [0,16) of 16; particles 175,616
    [e30] sim0 (dev0) pool_health: peak_tail_high_water=0 own_pool_size=201984 free_margin=201984 peak_migration_count=0 peak_departed_count=0 departed_pool_size=0

**s48_K2**

    [partition_v7] chain N=2: columns [0,8), [8,16) of 16; particles 87,808, 87,808
    [partition_v7] seam: ghost_layers=2 keep_departed=1 ghost pool/direction=41,124 (replica regions 2 x 20,480 + migrants 164) departed pool/slab=164/164
    [e30] sim0 (dev0) pool_health: peak_tail_high_water=0 own_pool_size=105472 free_margin=105472 peak_migration_count=0 peak_departed_count=0 departed_pool_size=164
    [e30] sim1 (dev1) pool_health: peak_tail_high_water=0 own_pool_size=105472 free_margin=105472 peak_migration_count=0 peak_departed_count=0 departed_pool_size=164
    [e30] pool s0_to_s1 inner: capacity=20480 peak=12544 frame_of_peak=0 p999=12544 mean=12544.0 last=12544 frames=300 occupancy=0.6125
    [e30] pool s0_to_s1 outer: capacity=20480 peak=12544 frame_of_peak=0 p999=12544 mean=12544.0 last=12544 frames=300 occupancy=0.6125
    [e30] pool s0_to_s1 migrant: capacity=164 peak=0 frame_of_peak=0 p999=0 mean=0.0 last=0 frames=300 occupancy=0.0000
    [e30] pool s1_to_s0 inner: capacity=20480 peak=12544 frame_of_peak=0 p999=12544 mean=12544.0 last=12544 frames=300 occupancy=0.6125
    [e30] pool s1_to_s0 outer: capacity=20480 peak=12544 frame_of_peak=0 p999=12544 mean=12544.0 last=12544 frames=300 occupancy=0.6125
    [e30] pool s1_to_s0 migrant: capacity=164 peak=0 frame_of_peak=0 p999=0 mean=0.0 last=0 frames=300 occupancy=0.0000

**weak2_K2**

    [partition_v7] chain N=2: columns [0,50), [50,100) of 100; particles 7,840,000, 7,840,000
    [partition_v7] seam: ghost_layers=2 keep_departed=1 ghost pool/direction=434,371 (replica regions 2 x 216,320 + migrants 1,731) departed pool/slab=1731/1731
    [e30] sim0 (dev0) pool_health: peak_tail_high_water=0 own_pool_size=9408000 free_margin=9408000 peak_migration_count=0 peak_departed_count=0 departed_pool_size=1731
    [e30] sim1 (dev1) pool_health: peak_tail_high_water=0 own_pool_size=9408000 free_margin=9408000 peak_migration_count=0 peak_departed_count=0 departed_pool_size=1731
    [e30] pool s0_to_s1 inner: capacity=216320 peak=160000 frame_of_peak=0 p999=160000 mean=160000.0 last=160000 frames=300 occupancy=0.7396
    [e30] pool s0_to_s1 outer: capacity=216320 peak=160000 frame_of_peak=0 p999=160000 mean=160000.0 last=160000 frames=300 occupancy=0.7396
    [e30] pool s0_to_s1 migrant: capacity=1731 peak=0 frame_of_peak=0 p999=0 mean=0.0 last=0 frames=300 occupancy=0.0000
    [e30] pool s1_to_s0 inner: capacity=216320 peak=160000 frame_of_peak=0 p999=160000 mean=160000.0 last=160000 frames=300 occupancy=0.7396
    [e30] pool s1_to_s0 outer: capacity=216320 peak=160000 frame_of_peak=0 p999=160000 mean=160000.0 last=160000 frames=300 occupancy=0.7396
    [e30] pool s1_to_s0 migrant: capacity=1731 peak=0 frame_of_peak=0 p999=0 mean=0.0 last=0 frames=300 occupancy=0.0000

**64m_K2**

    [partition_v7] chain N=2: columns [0,51), [51,102) of 102; particles 32,000,000, 32,000,000
    [partition_v7] seam: ghost_layers=2 keep_departed=1 ghost pool/direction=1,671,299 (replica regions 2 x 832,320 + migrants 6,659) departed pool/slab=6659/6659
    [e30] sim0 (dev0) pool_health: peak_tail_high_water=0 own_pool_size=38400000 free_margin=38400000 peak_migration_count=0 peak_departed_count=0 departed_pool_size=6659
    [e30] sim1 (dev1) pool_health: peak_tail_high_water=0 own_pool_size=38400000 free_margin=38400000 peak_migration_count=0 peak_departed_count=0 departed_pool_size=6659
    [e30] pool s0_to_s1 inner: capacity=832320 peak=640000 frame_of_peak=0 p999=640000 mean=640000.0 last=640000 frames=300 occupancy=0.7689
    [e30] pool s0_to_s1 outer: capacity=832320 peak=640000 frame_of_peak=0 p999=640000 mean=640000.0 last=640000 frames=300 occupancy=0.7689
    [e30] pool s0_to_s1 migrant: capacity=6659 peak=0 frame_of_peak=0 p999=0 mean=0.0 last=0 frames=300 occupancy=0.0000
    [e30] pool s1_to_s0 inner: capacity=832320 peak=640000 frame_of_peak=0 p999=640000 mean=640000.0 last=640000 frames=300 occupancy=0.7689
    [e30] pool s1_to_s0 outer: capacity=832320 peak=640000 frame_of_peak=0 p999=640000 mean=640000.0 last=640000 frames=300 occupancy=0.7689
    [e30] pool s1_to_s0 migrant: capacity=6659 peak=0 frame_of_peak=0 p999=0 mean=0.0 last=0 frames=300 occupancy=0.0000

## 5. 链路字节（每条链路每步）

| run | link | host copy KiB | DMA KiB | frames |
|---|---|---|---|---|
| 8m_K2_trace | s0_to_s1 | 10042.3 | 13636.6 | 300 |
| 8m_K2_trace | s1_to_s0 | 10042.3 | 13636.6 | 300 |
| s48_K2 | s0_to_s1 | 788.0 | 1291.1 | 300 |
| s48_K2 | s1_to_s0 | 788.0 | 1291.1 | 300 |
| weak2_K2 | s0_to_s1 | 10042.3 | 13636.6 | 300 |
| weak2_K2 | s1_to_s0 | 10042.3 | 13636.6 | 300 |
| 64m_K2 | s0_to_s1 | 40162.6 | 52468.7 | 300 |
| 64m_K2 | s1_to_s0 | 40162.6 | 52468.7 | 300 |

## 6. trace（8m_K2）

```
[stderr] Traceback (most recent call last):
  File "C:\Users\cchen\PycharmProjects\vulkan-demo-e8\docs\cluster_v6\scripts\e7_trace_phases.py", line 105, in <module>
    sys.exit(main())
             ~~~~^^
  File "C:\Users\cchen\PycharmProjects\vulkan-demo-e8\docs\cluster_v6\scripts\e7_trace_phases.py", line 94, in main
    blocks.append(markdown(name, results[name]))
                  ~~~~~~~~^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\cchen\PycharmProjects\vulkan-demo-e8\docs\cluster_v6\scripts\e7_trace_phases.py", line 78, in markdown
    lines.append(f"| {entry['sim']} | {entry.get('device', '')} | {particles:,} | {entry.get('own_columns', '')} | "
                                                                  ^^^^^^^^^^^^^
TypeError: unsupported format string passed to NoneType.__format__
```

```
pooled over every link and steady step, p50 / p95 (us)

| trace | links x steps | t_chain | readback | host | upload | readback_dma | memcpy | host_rest | readback_to_worker | stamp_check | upload_dma |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 8m_K2 | 2 x 200 | 1,349 / 1,420 | 259 / 264 | 789 / 846 | 299 / 309 | 253 / 254 | 574 / 603 | 219 / 263 | 216 / 260 | 3 / 4 | 263 / 265 |

range of the per-link p50 (us)

| trace | links x steps | t_chain | readback | host | upload | readback_dma | memcpy | host_rest | readback_to_worker | stamp_check | upload_dma |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 8m_K2 | 2 | 1,328-1,369 | 259-259 | 768-808 | 299-299 | 253-253 | 555-584 | 212-223 | 209-220 | 3-3 | 262-264 |

8m_K2: per link, p50 / p95 (us), bytes p50

| link | host copy B | DMA B | t_chain | readback | host | upload | readback_dma | memcpy | host_rest | readback_to_worker | stamp_check | upload_dma |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s0_to_s1 | 10,283,280 | 13,963,924 | 1,369 / 1,428 | 259 / 264 | 808 / 855 | 299 / 306 | 253 / 254 | 584 / 608 | 223 / 265 | 220 / 262 | 3 / 4 | 264 / 265 |
| s1_to_s0 | 10,283,280 | 13,963,924 | 1,328 / 1,393 | 259 / 264 | 768 / 819 | 299 / 310 | 253 / 254 | 555 / 567 | 212 / 261 | 209 / 258 | 3 / 4 | 262 / 263 |
```


## 7. 场比较（第 300 步，K = 2 对 K = 1，流体粒子）

**compare_8m**: cuts [26], 7,077,888 particles, missing 0 / 0; max |dv|/U 5.741e-05, rms 2.624e-06; max |drho|/rho0 1.831e-07, rms 2.116e-08

| distance to the seam (columns) | particles | max dv/U | rms dv/U | max drho/rho0 | rms drho/rho0 | max dx (m) |
|---|---|---|---|---|---|---|
| 0 | 294,912 | 1.804e-05 | 4.314e-06 | 1.221e-07 | 2.724e-08 | 2.950e-06 |
| 1 | 294,912 | 2.105e-05 | 4.174e-06 | 1.221e-07 | 2.755e-08 | 2.653e-06 |
| 2 | 294,912 | 1.817e-05 | 3.480e-06 | 1.221e-07 | 2.484e-08 | 2.563e-06 |
| 3 | 294,912 | 1.618e-05 | 2.720e-06 | 1.831e-07 | 2.213e-08 | 2.535e-06 |
| 4 | 294,912 | 1.273e-05 | 2.253e-06 | 1.221e-07 | 2.062e-08 | 2.743e-06 |
| 5 | 294,912 | 1.492e-05 | 2.108e-06 | 1.221e-07 | 2.020e-08 | 2.355e-06 |
| 6 | 294,912 | 1.366e-05 | 1.995e-06 | 1.221e-07 | 1.972e-08 | 2.178e-06 |
| 7 | 294,912 | 1.592e-05 | 1.911e-06 | 1.221e-07 | 1.943e-08 | 2.593e-06 |
| 8 | 294,912 | 1.592e-05 | 1.920e-06 | 1.221e-07 | 1.951e-08 | 2.533e-06 |
| 9 | 294,912 | 2.088e-05 | 1.860e-06 | 1.221e-07 | 1.937e-08 | 3.130e-06 |
| 10 | 294,912 | 1.520e-05 | 1.800e-06 | 1.221e-07 | 1.944e-08 | 2.861e-06 |
| 11 | 294,912 | 1.401e-05 | 1.737e-06 | 1.221e-07 | 1.902e-08 | 4.325e-06 |
| 12 | 294,912 | 2.477e-05 | 1.806e-06 | 1.221e-07 | 1.928e-08 | 4.650e-06 |
| 13 | 294,912 | 2.438e-05 | 1.896e-06 | 1.221e-07 | 1.946e-08 | 4.113e-06 |
| 14 | 294,912 | 3.022e-05 | 2.071e-06 | 1.221e-07 | 1.949e-08 | 4.697e-06 |
| 15 | 294,912 | 3.181e-05 | 2.293e-06 | 1.221e-07 | 2.011e-08 | 5.413e-06 |
| interior | 2,359,296 | 5.741e-05 | 2.801e-06 | 1.831e-07 | 2.093e-08 | 1.131e-05 |

**compare_s48**: cuts [8], 110,592 particles, missing 0 / 0; max |dv|/U 2.130e-05, rms 4.625e-06; max |drho|/rho0 1.221e-07, rms 2.386e-08

| distance to the seam (columns) | particles | max dv/U | rms dv/U | max drho/rho0 | rms drho/rho0 | max dx (m) |
|---|---|---|---|---|---|---|
| 0 | 18,432 | 2.130e-05 | 6.264e-06 | 1.221e-07 | 2.818e-08 | 7.692e-06 |
| 1 | 18,432 | 1.960e-05 | 5.817e-06 | 6.104e-08 | 2.892e-08 | 9.187e-06 |
| 2 | 18,432 | 1.703e-05 | 4.182e-06 | 1.221e-07 | 2.291e-08 | 5.302e-06 |
| 3 | 18,432 | 1.452e-05 | 3.886e-06 | 6.104e-08 | 2.199e-08 | 6.230e-06 |
| 4 | 18,432 | 1.608e-05 | 3.461e-06 | 6.104e-08 | 2.011e-08 | 6.326e-06 |
| 5 | 18,432 | 1.596e-05 | 3.274e-06 | 6.104e-08 | 1.933e-08 | 5.827e-06 |

dump dump_8m_K1_N300: valid True, drift 0, overflow 0, missing 0, duplicates 0, stamps 0/0, far 0, alive 8000000, frames 15.2 s
dump dump_8m_K2_N300: valid True, drift 0, overflow 0, missing 0, duplicates 0, stamps 0/0, far 0, alive 8000000, frames 7.9 s
dump dump_s48_K1_N300: valid True, drift 0, overflow 0, missing 0, duplicates 0, stamps 0/0, far 0, alive 175616, frames 0.4 s
dump dump_s48_K2_N300: valid True, drift 0, overflow 0, missing 0, duplicates 0, stamps 0/0, far 0, alive 175616, frames 0.5 s

## 8. bring-up 与步骤

**s0_bringup**
    RESULT label=bringup_k1 status=pass rc=0 steady_fps=723.1 total_fps=743.6 drift=0 overflow_total=0 far_migration=0 stamps=0+0 alive=1044484->1044484 clamp=0 nan=False fields=OK columns=[206] tq2=1 pin
    K1 STAGE: PASS (rc=0)
    ===== K2 STAGE (chain bench, weights 1,1, devices 0,1, 300 steps) =====
    RESULT label=bringup_k2 status=pass rc=0 steady_fps=931.5 total_fps=1011.8 drift=0 overflow_total=0 far_migration=0 stamps=0+0 alive=1044484->1044484 clamp=0 nan=False fields=OK columns=[103, 103] tq2
    K2 STAGE: PASS (rc=0)
    BRINGUP: ALL PASS — report: /dev/shm/scxm138/e7_p1_1721673/bringup/bringup_report.txt

**s0_caps**

**s0_obj_cache**
    [obj_cache] check cavity3d_n392/domain.obj: cached domain_0d1b228ecfd03bf27dcb.npy (722,835,584 B) -> staged
    [obj_cache] check cavity3d_n392/wall.obj: cached wall_159d264d77dc44a0544f.npy (37,788,800 B) -> staged
    [obj_cache] check cavity3d_n392/wall_top.obj: cached wall_top_293e9d7099dffc8992a8.npy (7,376,000 B) -> staged
    [obj_cache] check cavity3d_n392/frame.obj: cached frame_c7247c59f35297328c4b.npy (224 B) -> staged
    [obj_cache] staged 1,066,804,240 B to /dev/shm/scxm138/e7_p1_1721673_obj_cache in 15.6 s (68 MB/s)
    [obj_cache] check: 0 missing

