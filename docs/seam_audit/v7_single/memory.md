# v7_single device-local memory per case (buffer table built on the CPU, no device)

Pool slot = the set-0 SoA fields (position_voxel_id 16, density_pressure 8, density_pressure_scratch 8, velocity_mass 16, acceleration 16, shift 16, material 4, correction_inverse 32, density_gradient_kernel_sum 16, extension_fields 16 B; wall_dummy_velocity 16 B only with adami walls) plus the defrag twins of the nine DEFRAG_SET0_BINDINGS fields (everything but density_pressure_scratch and wall_dummy_velocity). Voxel lists = set 1: inside / incoming counts (4 B each), inside index (4 x max_per_voxel), incoming index (4 x max_incoming), voxel_base_offset (4), per voxel (slot 0 unused). Fixed = global status, pool health, material parameters, counters and the stubs.

| case | particles | pool | pool / N | B / slot (SoA + twin) | SoA B | twin B | voxels | B / voxel (lists) | lists B / particle | fixed B | device-local total | B / particle |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1m | 1,046,529 | 1,203,585 | 1.150 | 288 (148 + 140) | 169.9 MiB | 160.7 MiB | 42,437 | 460 | 18.7 | 368 | 349.2 MiB | 349.9 |
| 8m | 8,128,201 | 9,347,457 | 1.150 | 288 (148 + 140) | 1,319.3 MiB | 1,248.0 MiB | 326,042 | 460 | 18.5 | 368 | 2,710.4 MiB | 349.7 |
| 32m | 32,251,041 | 37,088,769 | 1.150 | 288 (148 + 140) | 5,234.8 MiB | 4,951.9 MiB | 1,292,770 | 460 | 18.4 | 368 | 10,753.9 MiB | 349.6 |
| 3d_8m | 9,129,329 | 10,498,817 | 1.150 | 288 (148 + 140) | 1,481.8 MiB | 1,401.7 MiB | 148,878 | 652 | 10.6 | 368 | 2,976.2 MiB | 341.8 |

Bytes per particle = device-local total / N; the SoA + twin part alone is 1m 331.2, 8m 331.2, 32m 331.2, 3d_8m 331.2 B / particle (slot bytes x pool / N).

## 1m (cases/lid_driven_cavity_2d_gen/case.yaml): 24 buffers + 9 twins

| buffer | set | binding | bytes | twin bytes | per |
|---|---|---|---|---|---|
| position_voxel_id | 0 | 0 | 19,257,360 | 19,257,360 | 16 B/slot + twin 16 B/slot |
| density_pressure | 0 | 1 | 9,628,680 | 9,628,680 | 8 B/slot + twin 8 B/slot |
| density_pressure_scratch | 0 | 2 | 9,628,680 | 0 | 8 B/slot |
| velocity_mass | 0 | 3 | 19,257,360 | 19,257,360 | 16 B/slot + twin 16 B/slot |
| acceleration | 0 | 4 | 19,257,360 | 19,257,360 | 16 B/slot + twin 16 B/slot |
| shift | 0 | 5 | 19,257,360 | 19,257,360 | 16 B/slot + twin 16 B/slot |
| material | 0 | 6 | 4,814,340 | 4,814,340 | 4 B/slot + twin 4 B/slot |
| correction_inverse | 0 | 7 | 38,514,720 | 38,514,720 | 32 B/slot + twin 32 B/slot |
| density_gradient_kernel_sum | 0 | 8 | 19,257,360 | 19,257,360 | 16 B/slot + twin 16 B/slot |
| extension_fields | 0 | 9 | 19,257,360 | 19,257,360 | 16 B/slot + twin 16 B/slot |
| wall_dummy_velocity | 0 | 10 | 16 | 0 | 16 B |
| inside_particle_count | 1 | 0 | 169,748 | 0 | 4 B/voxel |
| incoming_particle_count | 1 | 1 | 169,748 | 0 | 4 B/voxel |
| inside_particle_index | 1 | 2 | 16,295,808 | 0 | 384 B/voxel |
| incoming_particle_index | 1 | 3 | 2,715,968 | 0 | 64 B/voxel |
| voxel_base_offset | 1 | 4 | 169,748 | 0 | 4 B/voxel |
| ghost_voxel_first_particle_id | 1 | 5 | 4 | 0 | 4 B |
| band_compact_list | 1 | 6 | 4 | 0 | 4 B |
| ghost_packed_words | 1 | 7 | 4 | 0 | 4 B |
| global_status | 3 | 0 | 160 | 0 | 160 B |
| pool_health | 3 | 1 | 16 | 0 | 16 B |
| material_parameters | 3 | 7 | 144 | 0 | 144 B |
| defrag_scratch_counter | 3 | 8 | 4 | 0 | 4 B |
| band_compact_meta | 3 | 9 | 16 | 0 | 16 B |

## 8m (cases/lid_driven_cavity_2d_8m/case.yaml): 24 buffers + 9 twins

| buffer | set | binding | bytes | twin bytes | per |
|---|---|---|---|---|---|
| position_voxel_id | 0 | 0 | 149,559,312 | 149,559,312 | 16 B/slot + twin 16 B/slot |
| density_pressure | 0 | 1 | 74,779,656 | 74,779,656 | 8 B/slot + twin 8 B/slot |
| density_pressure_scratch | 0 | 2 | 74,779,656 | 0 | 8 B/slot |
| velocity_mass | 0 | 3 | 149,559,312 | 149,559,312 | 16 B/slot + twin 16 B/slot |
| acceleration | 0 | 4 | 149,559,312 | 149,559,312 | 16 B/slot + twin 16 B/slot |
| shift | 0 | 5 | 149,559,312 | 149,559,312 | 16 B/slot + twin 16 B/slot |
| material | 0 | 6 | 37,389,828 | 37,389,828 | 4 B/slot + twin 4 B/slot |
| correction_inverse | 0 | 7 | 299,118,624 | 299,118,624 | 32 B/slot + twin 32 B/slot |
| density_gradient_kernel_sum | 0 | 8 | 149,559,312 | 149,559,312 | 16 B/slot + twin 16 B/slot |
| extension_fields | 0 | 9 | 149,559,312 | 149,559,312 | 16 B/slot + twin 16 B/slot |
| wall_dummy_velocity | 0 | 10 | 16 | 0 | 16 B |
| inside_particle_count | 1 | 0 | 1,304,168 | 0 | 4 B/voxel |
| incoming_particle_count | 1 | 1 | 1,304,168 | 0 | 4 B/voxel |
| inside_particle_index | 1 | 2 | 125,200,128 | 0 | 384 B/voxel |
| incoming_particle_index | 1 | 3 | 20,866,688 | 0 | 64 B/voxel |
| voxel_base_offset | 1 | 4 | 1,304,168 | 0 | 4 B/voxel |
| ghost_voxel_first_particle_id | 1 | 5 | 4 | 0 | 4 B |
| band_compact_list | 1 | 6 | 4 | 0 | 4 B |
| ghost_packed_words | 1 | 7 | 4 | 0 | 4 B |
| global_status | 3 | 0 | 160 | 0 | 160 B |
| pool_health | 3 | 1 | 16 | 0 | 16 B |
| material_parameters | 3 | 7 | 144 | 0 | 144 B |
| defrag_scratch_counter | 3 | 8 | 4 | 0 | 4 B |
| band_compact_meta | 3 | 9 | 16 | 0 | 16 B |

## 32m (cases/lid_driven_cavity_2d_32m/case.yaml): 24 buffers + 9 twins

| buffer | set | binding | bytes | twin bytes | per |
|---|---|---|---|---|---|
| position_voxel_id | 0 | 0 | 593,420,304 | 593,420,304 | 16 B/slot + twin 16 B/slot |
| density_pressure | 0 | 1 | 296,710,152 | 296,710,152 | 8 B/slot + twin 8 B/slot |
| density_pressure_scratch | 0 | 2 | 296,710,152 | 0 | 8 B/slot |
| velocity_mass | 0 | 3 | 593,420,304 | 593,420,304 | 16 B/slot + twin 16 B/slot |
| acceleration | 0 | 4 | 593,420,304 | 593,420,304 | 16 B/slot + twin 16 B/slot |
| shift | 0 | 5 | 593,420,304 | 593,420,304 | 16 B/slot + twin 16 B/slot |
| material | 0 | 6 | 148,355,076 | 148,355,076 | 4 B/slot + twin 4 B/slot |
| correction_inverse | 0 | 7 | 1,186,840,608 | 1,186,840,608 | 32 B/slot + twin 32 B/slot |
| density_gradient_kernel_sum | 0 | 8 | 593,420,304 | 593,420,304 | 16 B/slot + twin 16 B/slot |
| extension_fields | 0 | 9 | 593,420,304 | 593,420,304 | 16 B/slot + twin 16 B/slot |
| wall_dummy_velocity | 0 | 10 | 16 | 0 | 16 B |
| inside_particle_count | 1 | 0 | 5,171,080 | 0 | 4 B/voxel |
| incoming_particle_count | 1 | 1 | 5,171,080 | 0 | 4 B/voxel |
| inside_particle_index | 1 | 2 | 496,423,680 | 0 | 384 B/voxel |
| incoming_particle_index | 1 | 3 | 82,737,280 | 0 | 64 B/voxel |
| voxel_base_offset | 1 | 4 | 5,171,080 | 0 | 4 B/voxel |
| ghost_voxel_first_particle_id | 1 | 5 | 4 | 0 | 4 B |
| band_compact_list | 1 | 6 | 4 | 0 | 4 B |
| ghost_packed_words | 1 | 7 | 4 | 0 | 4 B |
| global_status | 3 | 0 | 160 | 0 | 160 B |
| pool_health | 3 | 1 | 16 | 0 | 16 B |
| material_parameters | 3 | 7 | 144 | 0 | 144 B |
| defrag_scratch_counter | 3 | 8 | 4 | 0 | 4 B |
| band_compact_meta | 3 | 9 | 16 | 0 | 16 B |

## 3d_8m (cases/cavity3d_8m_b4/case.yaml): 24 buffers + 9 twins

| buffer | set | binding | bytes | twin bytes | per |
|---|---|---|---|---|---|
| position_voxel_id | 0 | 0 | 167,981,072 | 167,981,072 | 16 B/slot + twin 16 B/slot |
| density_pressure | 0 | 1 | 83,990,536 | 83,990,536 | 8 B/slot + twin 8 B/slot |
| density_pressure_scratch | 0 | 2 | 83,990,536 | 0 | 8 B/slot |
| velocity_mass | 0 | 3 | 167,981,072 | 167,981,072 | 16 B/slot + twin 16 B/slot |
| acceleration | 0 | 4 | 167,981,072 | 167,981,072 | 16 B/slot + twin 16 B/slot |
| shift | 0 | 5 | 167,981,072 | 167,981,072 | 16 B/slot + twin 16 B/slot |
| material | 0 | 6 | 41,995,268 | 41,995,268 | 4 B/slot + twin 4 B/slot |
| correction_inverse | 0 | 7 | 335,962,144 | 335,962,144 | 32 B/slot + twin 32 B/slot |
| density_gradient_kernel_sum | 0 | 8 | 167,981,072 | 167,981,072 | 16 B/slot + twin 16 B/slot |
| extension_fields | 0 | 9 | 167,981,072 | 167,981,072 | 16 B/slot + twin 16 B/slot |
| wall_dummy_velocity | 0 | 10 | 16 | 0 | 16 B |
| inside_particle_count | 1 | 0 | 595,512 | 0 | 4 B/voxel |
| incoming_particle_count | 1 | 1 | 595,512 | 0 | 4 B/voxel |
| inside_particle_index | 1 | 2 | 76,225,536 | 0 | 512 B/voxel |
| incoming_particle_index | 1 | 3 | 19,056,384 | 0 | 128 B/voxel |
| voxel_base_offset | 1 | 4 | 595,512 | 0 | 4 B/voxel |
| ghost_voxel_first_particle_id | 1 | 5 | 4 | 0 | 4 B |
| band_compact_list | 1 | 6 | 4 | 0 | 4 B |
| ghost_packed_words | 1 | 7 | 4 | 0 | 4 B |
| global_status | 3 | 0 | 160 | 0 | 160 B |
| pool_health | 3 | 1 | 16 | 0 | 16 B |
| material_parameters | 3 | 7 | 144 | 0 | 144 B |
| defrag_scratch_counter | 3 | 8 | 4 | 0 | 4 B |
| band_compact_meta | 3 | 9 | 16 | 0 | 16 B |
