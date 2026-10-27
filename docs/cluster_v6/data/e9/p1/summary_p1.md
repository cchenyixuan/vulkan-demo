## 1. 作业

    === E30 e7_p1 job 1741843 node wqd10nbj02g1 Tue Oct 27 01:34:05 PM CST 2026 local=0 dry=0 solver=v7 ===
    job cpus: pid 2122531's current affinity list: 0-7,64-71
    0, NVIDIA GeForce RTX 5090, 00000000:16:00.0, GPU-e463aac7-fb22-031e-b2e7-4a0aa4049e75, 580.82.07, 32607 MiB, 575.00 W
    1, NVIDIA GeForce RTX 5090, 00000000:27:00.0, GPU-efc47892-6d0c-6541-a589-536bdcda5fa7, 580.82.07, 32607 MiB, 575.00 W
    PROVENANCE host=wqd10nbj02g1 slurm_job=1741843 job=p1 line=S harness=cb3c9f5cab754e59a2776b2414612a47b82dbb47 experiment_v7_tree=37f99f2b25f85f01e31f6915bddf12c43f743993 (v7-e9p1 37f99f2b25f85f01e31f6915bddf12c43f743993) driver=58
    GPU health: 2 GPUs listed without an error on wqd10nbj02g1
    power caps: 0, 575.00 W;1, 575.00 W;
    power caps: all 2 GPUs at 575.00 W on wqd10nbj02g1
    power caps: 0, 575.00 W;1, 575.00 W;
    power caps: all 2 GPUs at 575 W on wqd10nbj02g1 (E7_POWER_CAP_W)
      [1] NVIDIA GeForce RTX 5090  type=DISCRETE  api=1.4.312
    [VulkanContextV7] selected: [1] NVIDIA GeForce RTX 5090 on queue_family[0]
    VulkanContextV7[1]: OK — NVIDIA GeForce RTX 5090 (compute qf 0, transfer qf 1, transfer queues 2 (split))
    [caps] v6[0] smi[0] NVIDIA GeForce RTX 5090 bus=0000:16:00.0 uuid=e463aac7 driver=580.82.07 api=1.4.312 vram=31.84GiB period=1.0ns compute_qf=0 transfer_qf=1x2 split=1 domains_KHR=['DEVICE', 'CLOCK_MONOTONIC', 'CLOCK_MONOTONIC_RAW

## 2. 运行（RESULT 行）

| label | status | K | steps | steady fps | wall s | build s | alive start→end | drift / overflow / far / stamps / clamp | seam / NaN | own columns | own particles | device-local + scratch MiB | VRAM peak MiB per card | SM MHz (median / min) | W median | °C max | ns / particle-step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 8m_K1 | pass (rc 0) | 1 | 300 | 22.5 | 24.2 | 9.10 | 8000000→8000000 | 0 / 0 / 0 / 0+0 / 0 | - / None | [52] | [8000000] | 2,641 | 0:2,698 | 2827/180 / 2820/172 | 575/17 | 51/24 | 5.56 |
| 8m_K2_trace | pass (rc 0) | 2 | 300 | 42.8 | 26.4 | 11.90 | 8000000→8000000 | 0 / 0 / 0 / 0+0 / 0 | {'seam': 0, 'column': 26, 'left_overshoot_dx': -0.19, 'right_overshoot_dx': -0.46, 'duplicates': 0, 'ok': True} / False | [26, 26] | [4000000, 4000000] | 1,512/1,512 | 0:1,569/1:1,553 | 2835/2865 / 2835/2850 | 569/530 | 52/46 | 5.84 |
| 64m_K2 | pass (rc 0) | 2 | 300 | 5.5 | 146.4 | 54.90 | 64000000→64000000 | 0 / 0 / 0 / 0+0 / 0 | {'seam': 0, 'column': 51, 'left_overshoot_dx': -0.02, 'right_overshoot_dx': -0.49, 'duplicates': 0, 'ok': True} / False | [51, 51] | [32000000, 32000000] | 11,454/11,454 | 0:11,511/1:11,495 | 2775/2827 / 2745/2797 | 575/575 | 63/64 | 5.68 |

## 3. 显存：实测峰值对模型

| run | card | slab own columns | model device-local + scratch MiB | measured | model VRAM | measured peak | diff |
|---|---|---|---|---|---|---|---|
| 8m_K1 | 0 | 52 | 2,616 | 2,641 | 2,666 | 2,698 | +1.2 % |
| 8m_K2_trace | 0 | 26 | 1,512 | 1,512 | 1,562 | 1,569 | +0.4 % |
| 8m_K2_trace | 1 | 26 | 1,512 | 1,512 | 1,562 | 1,553 | -0.6 % |
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
| 8m_K2 | 2 x 200 | 1,212 / 1,302 | 283 / 295 | 624 / 658 | 299 / 370 | 277 / 286 | 610 / 645 | 14 / 17 | 13 / 15 | 1 / 2 | 261 / 263 |

range of the per-link p50 (us)

| trace | links x steps | t_chain | readback | host | upload | readback_dma | memcpy | host_rest | readback_to_worker | stamp_check | upload_dma |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 8m_K2 | 2 | 1,201-1,223 | 280-288 | 620-631 | 296-302 | 275-280 | 604-618 | 13-16 | 12-14 | 1-2 | 260-262 |

8m_K2: per link, p50 / p95 (us), bytes p50

| link | host copy B | DMA B | t_chain | readback | host | upload | readback_dma | memcpy | host_rest | readback_to_worker | stamp_check | upload_dma |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s0_to_s1 | 10,283,280 | 13,963,924 | 1,201 / 1,290 | 280 / 287 | 620 / 637 | 302 / 364 | 275 / 281 | 604 / 623 | 16 / 18 | 14 / 16 | 2 / 2 | 262 / 263 |
| s1_to_s0 | 10,283,280 | 13,963,924 | 1,223 / 1,319 | 288 / 295 | 631 / 666 | 296 / 370 | 280 / 286 | 618 / 653 | 13 / 14 | 12 / 13 | 1 / 1 | 260 / 261 |
```


## 7. 场比较（第 300 步，K = 2 对 K = 1，流体粒子）

**compare_8m**: cuts [26], 7,077,888 particles, missing 0 / 0; max |dv|/U 4.517e-05, rms 2.457e-06; max |drho|/rho0 1.831e-07, rms 2.046e-08

| distance to the seam (columns) | particles | max dv/U | rms dv/U | max drho/rho0 | rms drho/rho0 | max dx (m) |
|---|---|---|---|---|---|---|
| 0 | 294,912 | 1.896e-05 | 4.353e-06 | 1.221e-07 | 2.723e-08 | 3.576e-06 |
| 1 | 294,912 | 2.064e-05 | 4.283e-06 | 1.221e-07 | 2.787e-08 | 2.980e-06 |
| 2 | 294,912 | 2.021e-05 | 3.547e-06 | 1.831e-07 | 2.508e-08 | 2.801e-06 |
| 3 | 294,912 | 1.804e-05 | 2.779e-06 | 1.221e-07 | 2.230e-08 | 2.742e-06 |
| 4 | 294,912 | 1.736e-05 | 2.293e-06 | 1.221e-07 | 2.075e-08 | 3.368e-06 |
| 5 | 294,912 | 1.496e-05 | 2.122e-06 | 1.221e-07 | 2.034e-08 | 3.219e-06 |
| 6 | 294,912 | 1.366e-05 | 2.010e-06 | 1.221e-07 | 2.002e-08 | 2.593e-06 |
| 7 | 294,912 | 2.003e-05 | 1.875e-06 | 1.221e-07 | 1.944e-08 | 3.399e-06 |
| 8 | 294,912 | 1.635e-05 | 1.902e-06 | 1.221e-07 | 1.943e-08 | 3.101e-06 |
| 9 | 294,912 | 1.799e-05 | 1.849e-06 | 1.221e-07 | 1.930e-08 | 3.368e-06 |
| 10 | 294,912 | 1.503e-05 | 1.812e-06 | 1.221e-07 | 1.943e-08 | 2.833e-06 |
| 11 | 294,912 | 1.431e-05 | 1.768e-06 | 1.831e-07 | 1.922e-08 | 3.726e-06 |
| 12 | 294,912 | 2.452e-05 | 1.815e-06 | 1.221e-07 | 1.931e-08 | 3.938e-06 |
| 13 | 294,912 | 2.720e-05 | 1.829e-06 | 1.221e-07 | 1.913e-08 | 4.387e-06 |
| 14 | 294,912 | 3.031e-05 | 1.914e-06 | 1.221e-07 | 1.891e-08 | 4.584e-06 |
| 15 | 294,912 | 3.327e-05 | 2.010e-06 | 1.221e-07 | 1.874e-08 | 5.467e-06 |
| interior | 2,359,296 | 4.517e-05 | 2.296e-06 | 1.221e-07 | 1.884e-08 | 1.003e-05 |
| slab | device | particles | max dv/U | rms dv/U | max drho/rho0 | rms drho/rho0 |
|---|---|---|---|---|---|---|
| 0 | 0 | 3,538,944 | 2.064e-05 | 2.151e-06 | 1.831e-07 | 1.845e-08 |
| 1 | 1 | 3,538,944 | 4.517e-05 | 2.729e-06 | 1.831e-07 | 2.230e-08 |

dump dump_8m_K1_N300: valid True, drift 0, overflow 0, missing 0, duplicates 0, stamps 0/0, far 0, alive 8000000, frames 13.4 s
dump dump_8m_K2_N300: valid True, drift 0, overflow 0, missing 0, duplicates 0, stamps 0/0, far 0, alive 8000000, frames 7.0 s

## 8. bring-up 与步骤

**s0_bringup**
    RESULT label=bringup_k1 status=pass rc=0 steady_fps=675.3 total_fps=702.9 drift=0 overflow_total=0 far_migration=0 stamps=0+0 alive=1044484->1044484 clamp=0 nan=False fields=OK columns=[206] tq2=1 pin
    K1 STAGE: PASS (rc=0)
    ===== K2 STAGE (chain bench, weights 1,1, devices 0,1, 300 steps) =====
    RESULT label=bringup_k2 status=pass rc=0 steady_fps=897.7 total_fps=980.2 drift=0 overflow_total=0 far_migration=0 stamps=0+0 alive=1044484->1044484 clamp=0 nan=False fields=OK columns=[103, 103] tq2=
    K2 STAGE: PASS (rc=0)
    BRINGUP: ALL PASS — report: /dev/shm/scxm138/e7_p1_1741843/bringup/bringup_report.txt

**s0_caps**

**s0_obj_cache**
    [obj_cache] check cavity3d_n392/domain.obj: cached domain_0d1b228ecfd03bf27dcb.npy (722,835,584 B) -> staged
    [obj_cache] check cavity3d_n392/wall.obj: cached wall_159d264d77dc44a0544f.npy (37,788,800 B) -> staged
    [obj_cache] check cavity3d_n392/wall_top.obj: cached wall_top_293e9d7099dffc8992a8.npy (7,376,000 B) -> staged
    [obj_cache] check cavity3d_n392/frame.obj: cached frame_c7247c59f35297328c4b.npy (224 B) -> staged
    [obj_cache] staged 876,535,632 B to /dev/shm/scxm138/e7_p1_1741843_obj_cache in 15.0 s (58 MB/s)
    [obj_cache] check: 0 missing

