"""Task B.3 — geometry / transport sizing table for 3-D slab decomposition (cube vs stretched), from the solver's
conventions: h = support radius = 4 dx (voxel edge = h), 1-voxel-thick ghost, bands correction/density/force =
2/3/4 voxel planes, ghost packet = 9 SoA fields = 140 B/particle, ghost pool per side = face_voxels x
(max_per_voxel + max_incoming) x factor, walls border (default 9 layers, or 4 = one support radius).
Measured inputs (fill from local runs): band per-particle cost relative to interior (2-D: 2.0x), fps.
usage: analyze_3d_prep.py > table.md"""
import math

HDX = 4                      # support radius in dx = voxel edge
GHOST_BYTES = 140            # 9 SoA fields per ghost particle
MAX_PER_VOXEL, MAX_INCOMING = 128, 32
BAND_COST = 2.0              # band kernel per-particle cost / interior (2-D measurement)
BAND_PLANES = {"correction": 2, "density": 3, "force": 4}

def tier(name, side, half_x_dx, K, border):
    """side: fluid lattice side in dx (y,z); half_x_dx: fluid x extent in dx (total); K slabs along x."""
    nx = half_x_dx
    slab_dx = nx / K
    slab_vox = slab_dx / HDX
    n_fluid = nx * side * side
    n_wall = 2 * (side + 2 * border) ** 2 * border * 0 + border * (2 * (nx * side) * 2 + 2 * side * side)  # 4 long faces + 2 ends (approx)
    n_wall = border * (4 * nx * side + 2 * side * side)   # shell around the fluid box, 6 faces, no corner double count
    n_total = n_fluid + n_wall
    face_vox = math.ceil((side + 2 * border) / HDX) ** 2            # voxels per y-z plane incl. shell
    halo_particles = HDX * (side + 2 * border) ** 2                  # one voxel plane, incl. wall shell
    halo_frac = HDX / slab_dx                                        # fraction of a slab's x extent per side
    sides = 2 if K > 1 else 0
    link_bytes = halo_particles * GHOST_BYTES                        # per direction per frame
    slab_particles = n_total / K
    force_band_frac = min(1.0, sides * BAND_PLANES["force"] * HDX / slab_dx)
    dens_band_frac = min(1.0, sides * BAND_PLANES["density"] * HDX / slab_dx)
    corr_band_frac = min(1.0, sides * BAND_PLANES["correction"] * HDX / slab_dx)
    # phase B ~ interior work of correction + density + force (weights from 2-D anatomy 3.5 : 3.7 : 5.0)
    w = {"correction": 3.5, "density": 3.7, "force": 5.0}
    B = (w["correction"] * (1 - corr_band_frac) + w["density"] * (1 - dens_band_frac) + w["force"] * (1 - force_band_frac))
    C = BAND_COST * (w["correction"] * corr_band_frac + w["density"] * dens_band_frac + w["force"] * force_band_frac)
    pool_slots_side = face_vox * (MAX_PER_VOXEL + MAX_INCOMING)
    live = halo_particles
    factor_needed = live / pool_slots_side
    return dict(name=name, K=K, nx=nx, side=side, slab_dx=slab_dx, slab_vox=slab_vox, n_fluid=n_fluid, n_wall=n_wall,
                halo_frac=halo_frac, link_MB=link_bytes / 1e6, C_over_B=C / B if B > 0 else float("inf"),
                force_band_frac=force_band_frac, factor_needed=factor_needed,
                factor_reco=min(1.0, math.ceil(factor_needed * 1.5 * 20) / 20), slab_particles=slab_particles)

rows = []
for border in (9, 4):
    # 8M/GPU tier: cube 200^3 per slab -> stretched 1600x200x200 (64M); cube 400^3 (64M) as the 1-D-slab limit
    for K in (1, 2, 4, 8):
        rows.append(tier(f"stretched 8M/GPU  border {border}", 201, 201 * K, K, border))
    rows.append(tier(f"cube 400^3 (64M)   border {border}", 401, 401, 8, border))
    # 4M/GPU tier: 159^3 per slab -> 1272x159x159 (32M)
    for K in (1, 2, 4, 8):
        rows.append(tier(f"stretched 4M/GPU  border {border}", 159, 159 * K, K, border))
    rows.append(tier(f"cube 318^3 (32M)   border {border}", 319, 319, 8, border))

print("| geometry | K | fluid N | wall N (%) | slab width dx (voxels) | halo/side | ghost MB/link/frame | force band frac | est. C/B | ghost pool factor needed (reco) |")
print("|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    print(f"| {r['name']} | {r['K']} | {r['n_fluid'] / 1e6:.1f}M | {r['n_wall'] / 1e6:.1f}M ({100 * r['n_wall'] / r['n_fluid']:.0f}%) | "
          f"{r['slab_dx']:.0f} ({r['slab_vox']:.1f}) | {100 * r['halo_frac']:.1f}% | {r['link_MB']:.1f} | {100 * r['force_band_frac']:.0f}% | "
          f"{r['C_over_B']:.2f} | {r['factor_needed']:.2f} ({r['factor_reco']:.2f}) |")
print()
print("2-D reference (64M K=8, 8M/GPU): slab 205 voxel columns, halo/side 0.5%, force band 3.9% of the slab, measured C/B = 1.16/12.4 = 0.09, ghost pool factor 0.25 used (live/pool 0.22).")
