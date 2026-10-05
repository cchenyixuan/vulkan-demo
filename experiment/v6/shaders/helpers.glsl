// ============================================================================
// helpers.glsl
//
// Shared device-side helper functions for all SPH compute shaders.
// Dependency: MUST be included AFTER common.glsl (reads its spec constants
// and buffer declarations).
//
// Usage:
//     #version 460
//     #extension GL_GOOGLE_include_directive : enable
//     #include "common.glsl"
//     #include "helpers.glsl"
//
// Contents:
//   - Wendland C4 kernel evaluation (value + gradient)
//   - 1-based voxel_id ↔ 3D voxel coord conversions + grid bounds check
//   - Symmetric 3×3 correction_inverse unpacking (2 vec4 → mat3)
// ============================================================================

#ifndef SPH_HELPERS_GLSL_INCLUDED
#define SPH_HELPERS_GLSL_INCLUDED

// ============================================================================
// Wendland C4 kernel and gradient.
//
// Support radius = h, q = r / h ∈ [0, 1). Normalization baked into
// KERNEL_COEFFICIENT / KERNEL_GRADIENT_COEFFICIENT (Python-side precomputed,
// including 1/h^DIM and 1/h^(DIM+1)).
//
//   W(q)   = KERNEL_COEFFICIENT          · (1-q)^6 · (35/3·q² + 6q + 1)
//   dW/dq  = (1-q)^5 · q · (-280/3·q - 56/3)
//   ∇W     = KERNEL_GRADIENT_COEFFICIENT · dW/dq · r̂
// ============================================================================

float evaluate_kernel(float distance) {
    float normalized_distance = distance / SMOOTHING_LENGTH;
    if (normalized_distance >= 1.0) return 0.0;

    float one_minus_q    = 1.0 - normalized_distance;
    float one_minus_q_sq = one_minus_q * one_minus_q;
    float one_minus_q_6  = one_minus_q_sq * one_minus_q_sq * one_minus_q_sq;

    // (35/3) · q² + 6q + 1
    float profile_polynomial =
        normalized_distance * ((35.0 / 3.0) * normalized_distance + 6.0) + 1.0;

    return KERNEL_COEFFICIENT * one_minus_q_6 * profile_polynomial;
}

vec3 evaluate_kernel_gradient(vec3 relative_position, float distance) {
    if (distance < 1e-12) return vec3(0.0);
    float normalized_distance = distance / SMOOTHING_LENGTH;
    if (normalized_distance >= 1.0) return vec3(0.0);

    float one_minus_q    = 1.0 - normalized_distance;
    float one_minus_q_sq = one_minus_q * one_minus_q;
    float one_minus_q_4  = one_minus_q_sq * one_minus_q_sq;
    float one_minus_q_5  = one_minus_q_4 * one_minus_q;

    // q · (-280/3 · q + -56/3)
    float derivative_polynomial =
        normalized_distance * ((-280.0 / 3.0) * normalized_distance + (-56.0 / 3.0));

    float gradient_magnitude_scalar =
        KERNEL_GRADIENT_COEFFICIENT * one_minus_q_5 * derivative_polynomial;

    return gradient_magnitude_scalar * (relative_position / distance);
}

// ============================================================================
// 1-based voxel_id ↔ 3D voxel coord.   (V1: x-slowest encoding)
//
// V1 partitions along X. The voxel_id encoding is "x-slowest" so that
// each x-column of voxels (NY*NZ voxels at a fixed x) occupies a CONTIGUOUS
// block of voxel_id values. This makes:
//   * the leading-ghost and trailing-ghost voxel ranges contiguous segments
//     of the global voxel_id space
//   * "is this voxel my own?" reduce to a single range comparison
//   * predict / update_voxel / defrag dispatch naturally over the own range
//
// Encoding:  voxel_id = y + z*GRID_DIMENSION_Y + x*GRID_DIMENSION_Y*GRID_DIMENSION_Z + 1
// Inverse:   y    = (id-1) % NY
//            z    = ((id-1) / NY) % NZ
//            x    = (id-1) / (NY*NZ)
//
// Slot 0 of every voxel buffer is unused (1-based). Particles store their
// voxel_id in position_voxel_id.w (0 = dead sentinel).
//
// V1 merged-buffer scheme: GRID_DIMENSION_X is the EXTENDED nx — it covers
// own columns plus leading/trailing ghost columns. Ghost particles share
// set 0 / set 1 with own particles; ranges are split by spec const.
// (See LEADING_GHOST_VOXEL_COUNT / TRAILING_GHOST_VOXEL_COUNT in common.glsl.)
// ============================================================================

ivec3 own_coord_of(uint voxel_id) {
    uint zero_based = voxel_id - 1u;
    uint y           = zero_based % GRID_DIMENSION_Y;
    uint after_y     = zero_based / GRID_DIMENSION_Y;
    uint z           = after_y    % GRID_DIMENSION_Z;
    uint x           = after_y    / GRID_DIMENSION_Z;
    return ivec3(x, y, z);
}

uint own_voxel_id_of(ivec3 coord) {
    return uint(coord.y)
         + uint(coord.z) * GRID_DIMENSION_Y
         + uint(coord.x) * GRID_DIMENSION_Y * GRID_DIMENSION_Z
         + 1u;
}

bool in_own_grid(ivec3 coord) {
    return coord.x >= 0 && coord.x < int(GRID_DIMENSION_X)
        && coord.y >= 0 && coord.y < int(GRID_DIMENSION_Y)
        && coord.z >= 0 && coord.z < int(GRID_DIMENSION_Z);
}

// ============================================================================
// V1 own-vs-ghost classification on the merged voxel_id range.
//
// Voxel_id layout in extended grid:
//   [1, M]                          = leading ghost  (peer's data, end GPUs: M=0)
//   [M+1, EXTENDED_TOTAL - N]       = own            (this GPU's particles)
//   [EXTENDED_TOTAL - N + 1, TOTAL] = trailing ghost (peer's data, end GPUs: N=0)
//
// EXTENDED_TOTAL = GRID_DIMENSION_X * GRID_DIMENSION_Y * GRID_DIMENSION_Z.
// M = LEADING_GHOST_VOXEL_COUNT, N = TRAILING_GHOST_VOXEL_COUNT.
// ============================================================================

// ============================================================================
// V6_PACKED_REPLICAS: word offsets in ghost_packed_words (see common.glsl):
// 8 words per record (x y z rho | vx vy vz material-bits) per layer, 2 layers
// per direction -> 16 R words per direction.
// ============================================================================
uint packed_layer_base(uint direction, uint layer) {
    return (direction * 16u + layer * 8u) * REPLICA_REGION_SIZE;
}

void store_packed_vec4(uint word, vec4 value) {
    ghost_packed_words[word + 0u] = floatBitsToUint(value.x);
    ghost_packed_words[word + 1u] = floatBitsToUint(value.y);
    ghost_packed_words[word + 2u] = floatBitsToUint(value.z);
    ghost_packed_words[word + 3u] = floatBitsToUint(value.w);
}

vec4 load_packed_vec4(uint word) {
    return vec4(uintBitsToFloat(ghost_packed_words[word + 0u]),
                uintBitsToFloat(ghost_packed_words[word + 1u]),
                uintBitsToFloat(ghost_packed_words[word + 2u]),
                uintBitsToFloat(ghost_packed_words[word + 3u]));
}

// ============================================================================
// V6_DELTA_DENSITY helpers (identity / the V5 expression when the switch is off)
// ============================================================================
float density_from_stored(float stored_density) {
    return DELTA_DENSITY ? REFERENCE_DENSITY + stored_density : stored_density;
}

float stored_rest_density(float rest_density) {
    return DELTA_DENSITY ? rest_density - REFERENCE_DENSITY : rest_density;
}

// Tait EOS P = B ((rho / rho0)^gamma - 1). Off: the V5 expression on the stored
// (absolute) density. On: x = (rho - rho0) / rho0 from the stored delta and
// (1 + x)^gamma - 1 = sum_k C(gamma, k) x^k, k = 1..8 by Horner (an exact
// polynomial for integer gamma; for |x| < 0.1 the truncation is < 1e-9 relative
// for non-integer gamma <= 8).
float tait_pressure(float stored_density, float rest_density, float eos_constant) {
    if (!DELTA_DENSITY) {
        return eos_constant * (pow(stored_density / rest_density, POWER_PARAMETER) - 1.0);
    }
    float x = (REFERENCE_DENSITY - rest_density + stored_density) / rest_density;
    float coefficient[9];
    coefficient[0] = 1.0;
    for (int order = 1; order <= 8; order++) {
        coefficient[order] = coefficient[order - 1] * (POWER_PARAMETER - float(order - 1)) / float(order);
    }
    float series = coefficient[8];
    for (int order = 7; order >= 1; order--) {
        series = coefficient[order] + x * series;
    }
    return eos_constant * x * series;
}

uint extended_voxel_count() {
    return GRID_DIMENSION_X * GRID_DIMENSION_Y * GRID_DIMENSION_Z;
}

bool is_own_voxel(uint voxel_id) {
    return voxel_id > LEADING_GHOST_VOXEL_COUNT
        && voxel_id <= extended_voxel_count() - TRAILING_GHOST_VOXEL_COUNT;
}

bool is_leading_ghost_voxel(uint voxel_id) {
    return voxel_id >= 1u && voxel_id <= LEADING_GHOST_VOXEL_COUNT;
}

bool is_trailing_ghost_voxel(uint voxel_id) {
    return voxel_id > extended_voxel_count() - TRAILING_GHOST_VOXEL_COUNT
        && voxel_id <= extended_voxel_count();
}

// ============================================================================
// V4 boundary-band classification (own coord → "near a peer interface?").
//
// See docs/sph_v4_design.md §7. A own-particle column is in the boundary band
// if its kernel support radius could either:
//   (a) reach into the ghost zone (column 0, i.e. NEIGHBOR_X_RANGE = 1), or
//   (b) reach into a column that itself reaches into the ghost zone — i.e.
//       a column that will receive new migrants when install_migration runs in
//       Submit 3 (column 1, requiring NEIGHBOR_X_RANGE ≥ 2).
//
// The leading and trailing sides are gated independently on whether a peer
// ghost exists on that side, so end-of-chain GPUs (LEADING_GHOST_VOXEL_COUNT
// or TRAILING_GHOST_VOXEL_COUNT == 0) correctly report only one boundary band.
//
// Ghost x-thickness derivation: V1 voxel encoding is x-slowest, so each
// x-column occupies exactly GRID_DIMENSION_Y * GRID_DIMENSION_Z voxels.
// LEADING_GHOST_VOXEL_COUNT is therefore an integer multiple of that product;
// dividing gives the thickness in columns.
// ============================================================================

uint leading_ghost_x_thickness() {
    return LEADING_GHOST_VOXEL_COUNT / (GRID_DIMENSION_Y * GRID_DIMENSION_Z);
}

uint trailing_ghost_x_thickness() {
    return TRAILING_GHOST_VOXEL_COUNT / (GRID_DIMENSION_Y * GRID_DIMENSION_Z);
}

bool in_boundary_band(ivec3 coord) {
    int leading_x  = int(leading_ghost_x_thickness());
    int trailing_x = int(trailing_ghost_x_thickness());
    int range      = int(NEIGHBOR_X_RANGE);
    if (FAKE_BAND_COLUMN > 0u) {
        // diagnostic band in the interior of a single-GPU domain
        return coord.x >= int(FAKE_BAND_COLUMN) && coord.x < int(FAKE_BAND_COLUMN) + range;
    }

    bool near_leading  = leading_x  > 0 && coord.x <  leading_x + range;
    int  own_last_x    = int(GRID_DIMENSION_X) - 1 - trailing_x;
    bool near_trailing = trailing_x > 0 && coord.x >  own_last_x - range;

    return near_leading || near_trailing;
}

// ============================================================================
// V3.4 band-voxel dispatch (BAND_VOXEL_DISPATCH = 1) for the boundary
// pipelines. The band of width `range` is exactly the set of voxels for
// which in_boundary_band() is true: leading band = own columns
// [leading_x, leading_x + range) when a leading peer exists, trailing band =
// the last `range` own columns when a trailing peer exists. Voxel ids of one
// column are contiguous (vid = 1 + x*NY*NZ + y + z*NY), so band voxel index
// i -> column i / (NY*NZ), yz = i % (NY*NZ). A thread is one (voxel, slot)
// pair; each pair is visited exactly once, so each band particle is
// processed exactly once. The simulator sizes the dispatch with the same
// formula (band_voxel_count(range) * MAX_PARTICLES_PER_VOXEL threads).
// ============================================================================
// V6 (GHOST_SELF_LAYER = 1, V6_GHOST_LAYERS = 2): every side with a peer
// walks ONE more column outward — the inner ghost column (the peer's seam
// column) — so its particles are processed as self. Per side the band is
// then columns [leading_x - 1, leading_x + range) / [own_last_x - range + 1,
// own_last_x + 1]. GHOST_SELF_LAYER = 0 reproduces V5 exactly.
uint band_voxel_count(uint range) {
    uint face = GRID_DIMENSION_Y * GRID_DIMENSION_Z;
    if (FAKE_BAND_COLUMN > 0u) return range * face;
    uint side_columns = range + GHOST_SELF_LAYER;
    uint columns = 0u;
    if (leading_ghost_x_thickness()  > 0u) columns += side_columns;
    if (trailing_ghost_x_thickness() > 0u) columns += side_columns;
    return columns * face;
}

// band voxel index (0 .. band_voxel_count(range)-1) -> voxel id of that band voxel
bool band_thread_voxel(uint voxel_index, uint range, out uint voxel_id) {
    uint face        = GRID_DIMENSION_Y * GRID_DIMENSION_Z;
    if (FAKE_BAND_COLUMN > 0u) {
        if (voxel_index >= range * face) return false;
        voxel_id = 1u + (FAKE_BAND_COLUMN + voxel_index / face) * face + (voxel_index % face);
        return true;
    }
    uint side_columns = range + GHOST_SELF_LAYER;
    uint leading_x   = leading_ghost_x_thickness();
    uint trailing_x  = trailing_ghost_x_thickness();
    uint leading_voxels = (leading_x > 0u) ? side_columns * face : 0u;
    uint column;
    if (voxel_index < leading_voxels) {
        column = leading_x - GHOST_SELF_LAYER + voxel_index / face;
    } else {
        uint trailing_index = voxel_index - leading_voxels;
        if (trailing_x == 0u || trailing_index >= side_columns * face) return false;
        uint own_last_x = GRID_DIMENSION_X - 1u - trailing_x;
        column = own_last_x - range + 1u + trailing_index / face;
    }
    voxel_id = 1u + column * face + (voxel_index % face);
    return true;
}

// V6_BAND_COMPACT_DISPATCH (BAND_VOXEL_DISPATCH = 2): thread -> pid through
// the compacted band list. Per side the list holds 4 own columns + L inner ghost
// columns (L = 1 with V6_GHOST_LAYERS = 2); a kernel of width `range` walking
// GHOST_SELF_LAYER (<= L) ghost columns takes leading list columns
// [L - GHOST_SELF_LAYER, L + range) and trailing list columns
// [block + 4 - range, block + 4 + GHOST_SELF_LAYER) — the same particles as
// band_thread_particle, in the same per-voxel order.
bool compact_thread_particle(uint thread_id, uint range, out uint self_particle_id) {
    uint list_self_layer = (GHOST_LAYERS >= 2u) ? 1u : 0u;
    bool leading  = leading_ghost_x_thickness() > 0u;
    bool trailing = trailing_ghost_x_thickness() > 0u;
    uint trailing_block = leading ? 4u + list_self_layer : 0u;
    uint leading_begin = band_compact_column_start[list_self_layer - GHOST_SELF_LAYER];
    uint leading_count = leading
        ? band_compact_column_start[list_self_layer + range] - leading_begin : 0u;
    if (thread_id < leading_count) {
        self_particle_id = band_compact_list[leading_begin + thread_id];
        return true;
    }
    thread_id -= leading_count;
    if (!trailing) return false;
    uint trailing_begin = band_compact_column_start[trailing_block + 4u - range];
    uint trailing_count = band_compact_column_start[trailing_block + 4u + GHOST_SELF_LAYER]
                        - trailing_begin;
    if (thread_id >= trailing_count) return false;
    self_particle_id = band_compact_list[trailing_begin + thread_id];
    return true;
}

bool band_thread_particle(uint thread_id, uint range, out uint self_particle_id) {
    uint face        = GRID_DIMENSION_Y * GRID_DIMENSION_Z;
    uint voxel_index = thread_id / MAX_PARTICLES_PER_VOXEL;
    uint slot        = thread_id % MAX_PARTICLES_PER_VOXEL;
    uint side_columns = range + GHOST_SELF_LAYER;
    uint leading_x   = leading_ghost_x_thickness();
    uint trailing_x  = trailing_ghost_x_thickness();
    uint leading_voxels = (leading_x > 0u) ? side_columns * face : 0u;
    uint column;
    if (FAKE_BAND_COLUMN > 0u) {
        if (voxel_index >= range * face) return false;
        column = FAKE_BAND_COLUMN + voxel_index / face;
    } else if (voxel_index < leading_voxels) {
        column = leading_x - GHOST_SELF_LAYER + voxel_index / face;
    } else {
        uint trailing_index = voxel_index - leading_voxels;
        if (trailing_x == 0u || trailing_index >= side_columns * face) return false;
        uint own_last_x = GRID_DIMENSION_X - 1u - trailing_x;
        column = own_last_x - range + 1u + trailing_index / face;
    }
    // leading_voxels is a multiple of face, so this holds on both sides.
    uint yz = voxel_index % face;
    uint voxel_id = 1u + column * face + yz;
    if (slot >= inside_particle_count[voxel_id]) return false;
    self_particle_id = inside_particle_index[voxel_id * MAX_PARTICLES_PER_VOXEL + slot];
    return true;
}

// ============================================================================
// V1 own / ghost pid range helpers (mirrors voxel layout).
//
// Pid layout in set 0:
//   [1, LEADING_GHOST_POOL_SIZE]                            = leading ghost
//   [own_first_pid, own_last_pid]                           = own
//   [own_last_pid+1, own_last_pid+TRAILING_GHOST_POOL_SIZE] = trailing ghost
// ============================================================================

uint own_first_pid() {
    return LEADING_GHOST_POOL_SIZE + 1u;
}

uint own_last_pid() {
    return LEADING_GHOST_POOL_SIZE + OWN_POOL_SIZE;
}

uint leading_ghost_first_pid() { return 1u; }
uint leading_ghost_last_pid()  { return LEADING_GHOST_POOL_SIZE; }

uint trailing_ghost_first_pid() {
    return LEADING_GHOST_POOL_SIZE + OWN_POOL_SIZE + 1u;
}
uint trailing_ghost_last_pid() {
    return LEADING_GHOST_POOL_SIZE + OWN_POOL_SIZE + TRAILING_GHOST_POOL_SIZE;
}

// V6 departed pool (V6_KEEP_DEPARTED): right after the trailing ghost range.
uint departed_first_pid() {
    return LEADING_GHOST_POOL_SIZE + OWN_POOL_SIZE + TRAILING_GHOST_POOL_SIZE + 1u;
}

// V6 ghost pool layout (GHOST_LAYERS = 2): [inner replicas | outer replicas |
// migrants]; with the V5 mixed pool (REPLICA_REGION_SIZE = 0) migrants start
// at offset 0 interleaved with the replicas.
uint migrant_region_offset() {
    return (GHOST_LAYERS >= 2u) ? 2u * REPLICA_REGION_SIZE : 0u;
}

// ============================================================================
// Unpack symmetric 3×3 correction_inverse from 2 vec4.
//
//   Storage: [pid*2]   = (M[0][0], M[1][1], M[2][2], M[0][1])
//            [pid*2+1] = (M[0][2], M[1][2], _, _)
// By symmetry M[1][0] = M[0][1], M[2][0] = M[0][2], M[2][1] = M[1][2].
// ============================================================================

mat3 unpack_correction_inverse(uint particle_id) {
    vec4 a = correction_inverse[particle_id * 2u];
    vec4 b = correction_inverse[particle_id * 2u + 1u];
    return mat3(
        vec3(a.x, a.w, b.x),   // column 0
        vec3(a.w, a.y, b.y),   // column 1
        vec3(b.x, b.y, a.z)    // column 2
    );
}

#endif  // SPH_HELPERS_GLSL_INCLUDED
