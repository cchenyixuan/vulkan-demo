// ============================================================================
// common.glsl
//
// Single source of truth for all SPH compute shaders: specialization constants,
// descriptor set bindings, and scalar constants. Every compute shader in
// shaders/sph/ must include this file and use only the fields it needs. SPIR-V
// optimization (compile with -O) strips unused bindings at compile time, so
// declaring everything here does not force every shader to bind every buffer.
//
// Convention:
//   - Each shader's .comp file starts with:
//         #version 460
//         #extension GL_GOOGLE_include_directive : enable
//         #include "common.glsl"
//   - Shader-specific bindings and local helpers go in the .comp file after
//     the include, before main().
//
// Descriptor set layout:
//   set 0 : own particle SoA (per-particle persistent + scratch fields)
//   set 1 : own voxel cell structures
//   set 2 : ghost particle + ghost voxel structures (V1 multi-GPU; V0-a: empty)
//   set 3 : global status, transport, diagnostics, material parameters
// ============================================================================

#ifndef SPH_COMMON_GLSL_INCLUDED
#define SPH_COMMON_GLSL_INCLUDED

// ============================================================================
// Specialization constants (VkSpecializationInfo)
//
// ID range plan:
//   0  - 9   : global physics scalars + own grid origin  (id=3 reserved)
//   10       : multi-GPU bit-exactness toggle
//   11 - 13  : own grid dimensions
//   14 - 16  : correction regularization tunables
//   17 - 19  : gravity
//   20 - 29  : voxel layout / reserved
//   30 - 33  : dimension + kernel coefficients
//   34 - 39  : reserved for future kernel / simulation options
//   40 - 42  : SPH numerical parameters (ε_h², PST main, PST anti)
//   43 - 46  : algorithm ablation toggles (KCG, density diffusion, PST, prefix-sum defrag)
//   47 - 49  : reserved for future ablation toggles
//   50 - 53  : capacities + workgroup size + pool size
//   54 - 79  : reserved
//   80 - 88  : multi-GPU ghost grid parameters
//   89 - 127 : reserved
//
// Per-material parameters (rest_density, viscosity, eos_constant, radius,
// volume, rotor_angular_velocity) are NOT spec constants — they live in
// MaterialParametersBuffer (set 3 binding 7), indexed by group_id.
// ============================================================================

// --- Global physics scalars ---
layout(constant_id = 0) const float SMOOTHING_LENGTH   = 0.009;
layout(constant_id = 1) const float SPEED_OF_SOUND     = 300.0;   // c0
layout(constant_id = 2) const float DELTA_COEFFICIENT  = 0.1;     // δ-plus diffusion coefficient
// constant_id = 3 reserved (was EPSILON_SHIFT, a legacy unused PST placeholder)
layout(constant_id = 4) const float POWER_PARAMETER    = 7.0;     // EOS γ
layout(constant_id = 5) const float CFL_NUMBER         = 0.15;    // informational; dt computed on CPU
layout(constant_id = 6) const float TIMESTEP           = 4.5e-6;  // dt = CFL · H / c0 = 0.15 · 0.009 / 300

// --- Own grid origin (world-space corner of voxel [0,0,0] on this GPU) ---
layout(constant_id = 7) const float OWN_ORIGIN_X = -1.758406;
layout(constant_id = 8) const float OWN_ORIGIN_Y = -1.758406;
layout(constant_id = 9) const float OWN_ORIGIN_Z =  0.0;

// --- Multi-GPU bit-exactness ---
// V0 (single-GPU): default false — no precise qualifiers required.
// V1+ (multi-GPU ghost integration): Python should override to true.
layout(constant_id = 10) const bool STRICT_BIT_EXACT = false;

// --- Own grid dimensions (in voxels) ---
layout(constant_id = 11) const uint GRID_DIMENSION_X = 128u;
layout(constant_id = 12) const uint GRID_DIMENSION_Y = 128u;
layout(constant_id = 13) const uint GRID_DIMENSION_Z = 1u;        // 1 in 2D

// Total number of voxels in the own grid. Spec-constant-derived constant
// (folded at pipeline creation). Used by per-voxel kernels for bounds checks.
const uint TOTAL_VOXEL_COUNT = GRID_DIMENSION_X * GRID_DIMENSION_Y * GRID_DIMENSION_Z;

// --- Correction (KCG) regularization ---
layout(constant_id = 14) const float REGULARIZATION_XI                    = 0.1;
layout(constant_id = 15) const float REGULARIZATION_DETERMINANT_THRESHOLD = 1e-4;
layout(constant_id = 16) const float REGULARIZATION_MAX_FROBENIUS_NORM    = 10.0;

// --- Gravity ---
layout(constant_id = 17) const float GRAVITY_X = 0.0;
layout(constant_id = 18) const float GRAVITY_Y = 0.0;
layout(constant_id = 19) const float GRAVITY_Z = 0.0;

// --- Voxel layout ---
layout(constant_id = 20) const uint VOXEL_ORDER = 0u;             // 0 = linear z-major; 1 = Morton (future)

// --- Micropolar (V0 reserved, not integrated) ---
layout(constant_id = 21) const float MICROPOLAR_THETA = 2.0;

// --- Dimension handling (2D as degenerate 3D) ---
layout(constant_id = 30) const uint  DIMENSION          = 2u;          // 2 or 3
layout(constant_id = 31) const uint  NEIGHBOR_Z_RANGE   = 0u;          // 0 for 2D, 1 for 3D

// --- Kernel normalization. Python-side precomputed including h^DIM factor. ---
// Wendland C4 (support radius = h, NOT 2h):
//     2D coefficient: 9 / (π · H²)
//     3D coefficient: 495 / (32 · π · H³)
// Profile:    W(q) = KERNEL_COEFFICIENT · (1-q)^6 · (35/3·q² + 6q + 1)    for q ∈ [0, 1]
// Gradient:   ∇W = KERNEL_GRADIENT_COEFFICIENT · (1-q)^5 · q · (-280/3·q - 56/3) · r̂
// Note: KERNEL_GRADIENT_COEFFICIENT = KERNEL_COEFFICIENT / H (the 1/h factor from chain rule).
layout(constant_id = 32) const float KERNEL_COEFFICIENT           = 35367.765;  // 9 / (π · 0.009²)
layout(constant_id = 33) const float KERNEL_GRADIENT_COEFFICIENT  = 3929751.7;  // 9 / (π · 0.009³)

// --- Density diffusion / viscosity division-by-zero guard ---
// Used in δ-SPH density diffusion and artificial-viscosity expressions where a
// 1/(r² + ε_h²) term would otherwise blow up when two particles approach each
// other. Typical value: 0.01 · H² (Antuono et al. δ-SPH).
layout(constant_id = 40) const float EPS_H_SQUARED = 8.1e-7;  // 0.01 · 0.009²

// PST main-shift scale coefficient (Sun 2017 δ-plus empirical constant).
// Enters as:  pst_base_factor = CFL · PST_MAIN_SHIFT_COEFFICIENT · 2·h²
// which then premultiplies BOTH the main and anti accumulators inside the loop.
layout(constant_id = 41) const float PST_MAIN_SHIFT_COEFFICIENT = 0.1;

// PST anti-shift (cohesion) magnitude multiplier on top of the main scale.
// Legacy value 0.005.
layout(constant_id = 42) const float PST_ANTI_SHIFT_COEFFICIENT = 0.005;

// ----- Algorithm ablation toggles (id 43-46) -------------------------------
// All bool spec constants; with glslc -O the dead branches are DCE-removed,
// so disabling a feature actually skips its computation (not just zeroes).
//
// Convention: when a toggle is false, the associated coefficients
// (delta_coefficient, pst_main/anti, regularization tunables) are still
// spec-const-supplied but unused — case.py may pass any value.

// KCG correction: when false, density.comp / force.comp use identity for M⁻¹
// and zero for ∇ρ instead of reading correction.comp's outputs. Used for
// comparing against non-KCG SPH codebases. correction.comp still runs
// (kernel_sum drives PST blend), but its M⁻¹ / ∇ρ are ignored downstream.
layout(constant_id = 43) const bool USE_KCG_CORRECTION = true;

// Density diffusion (the δ term in dρ/dt): when false, density.comp skips the
// δ-SPH continuity diffusion term. Result is vanilla WCSPH continuity.
// DELTA_COEFFICIENT is then unused.
layout(constant_id = 44) const bool USE_DENSITY_DIFFUSION = true;

// Particle shifting technique (PST): when false, force.comp skips the shift
// computation and writes shift = 0. Predict's drift then becomes pure
// x_{n+1} = x_n + v_{n+1/2}·dt with no δ-plus correction.
// PST_MAIN_SHIFT_COEFFICIENT / PST_ANTI_SHIFT_COEFFICIENT are unused.
layout(constant_id = 45) const bool USE_PST = true;

// Defrag base-offset source: when true, defrag.comp reads from
// voxel_base_offset[] (deterministic, voxel-id ordered, requires prefix_sum
// pass to populate it first). When false, defrag.comp uses an atomicAdd on
// defrag_scratch_counter — order is non-deterministic but defrag still
// works standalone (no prefix_sum dependency). Default false; flip to true
// after prefix_sum is verified.
layout(constant_id = 46) const bool USE_PREFIX_SUM_DEFRAG = false;
// ----- end ablation toggles ------------------------------------------------

// --- Capacity / dispatch ---
layout(constant_id = 50) const uint MAX_PARTICLES_PER_VOXEL = 96u;
layout(constant_id = 51) const uint WORKGROUP_SIZE          = 128u;
layout(constant_id = 52) const uint MAX_INCOMING_PER_VOXEL  = 16u;
// POOL_SIZE = number of particle slots (1..POOL_SIZE). Buffer is sized
// (POOL_SIZE + 1) under 1-based indexing. Padding threads beyond POOL_SIZE
// must early-return to avoid OOB access.
layout(constant_id = 53) const uint POOL_SIZE               = 1000000u;

// --- Multi-GPU ghost grid (V0-a: all zero / disabled; V1: populated) ---
layout(constant_id = 80) const uint  GHOST_DIMENSION_X = 0u;
layout(constant_id = 81) const uint  GHOST_DIMENSION_Y = 0u;
layout(constant_id = 82) const uint  GHOST_DIMENSION_Z = 0u;
layout(constant_id = 83) const float GHOST_ORIGIN_X    = 0.0;
layout(constant_id = 84) const float GHOST_ORIGIN_Y    = 0.0;
layout(constant_id = 85) const float GHOST_ORIGIN_Z    = 0.0;
// Integer offset such that `ghost_coord = own_coord + OWN_TO_GHOST_OFFSET`
// maps an out-of-own-bounds own coord onto the adjacent ghost region.
layout(constant_id = 86) const int OWN_TO_GHOST_OFFSET_X = 0;
layout(constant_id = 87) const int OWN_TO_GHOST_OFFSET_Y = 0;
layout(constant_id = 88) const int OWN_TO_GHOST_OFFSET_Z = 0;

// ============================================================================
// Scalar constants (compile-time, shared by all shaders)
// ============================================================================

// --- Material kind tags (stored inside MaterialParameters.kind) ---
const uint MATERIAL_FLUID    = 0u;
const uint MATERIAL_BOUNDARY = 1u;
const uint MATERIAL_INLET    = 2u;
const uint MATERIAL_ROTOR    = 3u;

// --- Dead / sentinel values ---
// Convention: EVERY id in the SPH pipeline is 1-based. 0 is reserved as the
// "unallocated / dead / empty" sentinel across all buffers:
//   - particle_id  ∈ [1, pool_size]      (slot 0 of all particle buffers unused)
//   - voxel_id     ∈ [1, voxel_count]    (slot 0 of all voxel buffers unused)
//   - slot entries ∈ [1, pool_size]      (inside_particle_index stores particle_id directly)
// Zero-init of buffers therefore naturally means "all slots empty / all particles dead".
const uint VOXEL_ID_DEAD         = 0u;
const uint INSIDE_SLOT_EMPTY     = 0u;
const uint PARTICLE_ID_NONE      = 0u;

// ============================================================================
// Descriptor set 0 — Own particle SoA
// ----------------------------------------------------------------------------
// Per-particle persistent state. All particles (fluid / boundary / inlet /
// rotor) live in this unified pool; they are distinguished by
// material[pid] (group_id), which indexes into MaterialParametersBuffer
// (set 3 binding 7) to obtain per-material kind + parameters.
//
// INDEXING: particle_id is 1-based, particle_id ∈ [1, pool_size]. Every
// particle buffer below is sized pool_size + 1 (slot 0 is unused; 0 is the
// PARTICLE_ID_NONE sentinel). Shader entry converts gl_GlobalInvocationID.x
// (0-based) to particle_id via `+ 1u`.
// ============================================================================

layout(std430, set = 0, binding = 0) buffer PositionVoxelIdBuffer {
    // (x, y, z, voxel_id_as_float)
    // voxel_id is 1-based; 0 marks dead/uninitialized.
    // Decode: uint voxel_id = uint(round(position_voxel_id[pid].w))
    vec4 position_voxel_id[];
};

layout(std430, set = 0, binding = 1) buffer DensityPressureABuffer {
    // (ρ, P)  ping-pong partner A
    // Density stage writes to "next" buffer; binding is swapped by descriptor update.
    vec2 density_pressure_a[];
};

layout(std430, set = 0, binding = 2) buffer DensityPressureBBuffer {
    // (ρ, P)  ping-pong partner B
    vec2 density_pressure_b[];
};

layout(std430, set = 0, binding = 3) buffer VelocityMassBuffer {
    // (vx, vy, vz, mass)
    // Leapfrog scheme: .xyz holds v_{n+1/2} (half-step velocity) between steps.
    // The predict kernel updates it via a full-step kick: v_{n+1/2} = v_{n-1/2} + a_n * dt.
    vec4 velocity_mass[];
};

layout(std430, set = 0, binding = 4) buffer AccelerationBuffer {
    // (ax, ay, az, reserved)
    // w slot reserved (previously used as temperature — now in ExtensionFieldsBuffer).
    vec4 acceleration[];
};

layout(std430, set = 0, binding = 5) buffer ShiftBuffer {
    // (shift_x, shift_y, shift_z, reserved)
    // w slot reserved.
    vec4 shift[];
};

layout(std430, set = 0, binding = 6) buffer MaterialBuffer {
    // group_id indexing into MaterialParametersBuffer (set 3 binding 7)
    uint material[];
};

layout(std430, set = 0, binding = 7) buffer CorrectionInverseBuffer {
    // Symmetric 3×3 M⁻¹, packed into 2 vec4 per particle:
    //   correction_inverse[pid*2]     = (m00, m11, m22, m01)
    //   correction_inverse[pid*2 + 1] = (m02, m12, _, _)
    vec4 correction_inverse[];
};

layout(std430, set = 0, binding = 8) buffer DensityGradientKernelSumBuffer {
    // (∇ρ.x, ∇ρ.y, ∇ρ.z, kernel_sum)
    // density_gradient:  Σ_j V_j · (ρ_j - ρ_i) · ∇W_ij
    // kernel_sum:        Σ_j V_j · W_ij
    vec4 density_gradient_kernel_sum[];
};

layout(std430, set = 0, binding = 9) buffer ExtensionFieldsBuffer {
    // Per-particle scalar fields not core to V0 SPH physics. Reserved slots
    // for future / debug use. V0 pipeline reads and writes nothing here.
    //   .x : temperature (passive transport or heat diffusion, future)
    //   .y : reserved (future: micropolar angular velocity scalar, or species concentration)
    //   .z : reserved
    //   .w : reserved
    vec4 extension_fields[];
};

// binding 10 reserved for GlobalIdBuffer (FTLE / Lagrangian tracking)

// ============================================================================
// Descriptor set 1 — Own voxel cell structures
// ----------------------------------------------------------------------------
// INDEXING: voxel_id is 1-based, voxel_id ∈ [1, voxel_count]. Every voxel
// buffer below is sized (voxel_count + 1) in the count dimension (slot 0
// unused; 0 is VOXEL_ID_DEAD).
//
// Slot values inside the flat `*_particle_index` buffers store 1-based
// particle_ids directly (no +1 encoding); 0 is the INSIDE_SLOT_EMPTY sentinel.
// ============================================================================

layout(std430, set = 1, binding = 0) buffer InsideParticleCountBuffer {
    // Indexed directly by 1-based voxel_id. Size = voxel_count + 1, slot 0 unused.
    uint inside_particle_count[];
};

layout(std430, set = 1, binding = 1) buffer IncomingParticleCountBuffer {
    // predict writes atomic-append count here; update_voxel consumes and resets to 0.
    //
    // OVERFLOW contract: on atomic-append overflow (slot ≥ MAX_INCOMING_PER_VOXEL),
    // predict still increments this counter but does NOT write into
    // incoming_particle_index (to avoid OOB). So after predict, this count MAY
    // exceed MAX_INCOMING_PER_VOXEL. update_voxel MUST clamp its iteration via
    // `min(count, MAX_INCOMING_PER_VOXEL)` before reading incoming_particle_index.
    uint incoming_particle_count[];
};

layout(std430, set = 1, binding = 2) buffer InsideParticleIndexBuffer {
    // flat [voxel_id * MAX_PARTICLES_PER_VOXEL + slot]; slot stores 1-based particle_id.
    // Total size = (voxel_count + 1) * MAX_PARTICLES_PER_VOXEL.
    //
    // INVARIANT 1 (contiguous packing): for any voxel, valid particle_ids occupy
    //   slots [0, inside_particle_count[voxel_id]); all trailing slots are 0
    //   (INSIDE_SLOT_EMPTY). update_voxel MUST produce this layout; neighbor loops
    //   rely on it for the early `break` on empty slots.
    //
    // INVARIANT 2 (no inlet/dead): inlet particles and dead particles (voxel_id=0)
    //   never appear here. Enforced by predict.comp (never atomic-appends inlet or
    //   particles whose drift lands outside domain) and by initial voxelization
    //   (skip inlet when populating at t=0). Neighbor loops therefore do not need
    //   to filter neighbors by kind.
    uint inside_particle_index[];
};

layout(std430, set = 1, binding = 3) buffer IncomingParticleIndexBuffer {
    // flat [voxel_id * MAX_INCOMING_PER_VOXEL + slot]; slot stores 1-based particle_id.
    // Same inlet/dead exclusion invariant as inside_particle_index.
    uint incoming_particle_index[];
};

layout(std430, set = 1, binding = 4) buffer VoxelBaseOffsetBuffer {
    // Exclusive prefix sum of inside_particle_count[]:
    //   voxel_base_offset[v] = Σ inside_particle_count[0..v-1]
    //
    // Populated by prefix_sum.comp; consumed by defrag.comp as the
    // deterministic destination-SoA base index for voxel v's particles.
    // When defrag's USE_PREFIX_SUM_DEFRAG spec constant is false, this buffer is
    // declared but unread — DCE removes the load from defrag.spv.
    // Other shaders never reference it, so glslc -O strips it from their
    // SPIR-V entirely.
    //
    // Size = (voxel_count + 1); slot 0 unused for symmetry with other
    // voxel buffers.
    uint voxel_base_offset[];
};
// binding 5 reserved

// ============================================================================
// Descriptor set 2 — Ghost particles + ghost voxel structures (V1 multi-GPU)
// ----------------------------------------------------------------------------
// Received from peer GPU each step; compute shaders read-only here. In V0-a
// GHOST_DIMENSION_* == 0, so ghost branches are dead-code-eliminated and this
// set can be bound to a dummy or left unused.
// ============================================================================

layout(std430, set = 2, binding = 0) buffer GhostPositionVoxelIdBuffer {
    vec4 ghost_position_voxel_id[];
};

layout(std430, set = 2, binding = 1) buffer GhostDensityPressureBuffer {
    // Single (not ping-pong); ghost density is received authoritatively from peer.
    vec2 ghost_density_pressure[];
};

layout(std430, set = 2, binding = 3) buffer GhostVelocityMassBuffer {
    vec4 ghost_velocity_mass[];
};

layout(std430, set = 2, binding = 4) buffer GhostAccelerationBuffer {
    // Required so that both GPUs can run bit-exact leapfrog kick on ghost particles.
    vec4 ghost_acceleration[];
};

layout(std430, set = 2, binding = 5) buffer GhostShiftBuffer {
    vec4 ghost_shift[];
};

layout(std430, set = 2, binding = 6) buffer GhostMaterialBuffer {
    uint ghost_material[];
};

// ghost bindings 7-9 reserved (parallels set 0 layout)

layout(std430, set = 2, binding = 10) buffer GhostInsideParticleCountBuffer {
    uint ghost_inside_particle_count[];
};

layout(std430, set = 2, binding = 12) buffer GhostInsideParticleIndexBuffer {
    uint ghost_inside_particle_index[];
};

// ============================================================================
// Descriptor set 3 — Global status, transport, material parameters, diagnostics
// ============================================================================

layout(std430, set = 3, binding = 0) buffer GlobalStatusBuffer {
    uint  alive_particle_count;
    uint  frame_counter;
    float maximum_velocity;
    uint  inlet_spawn_count;
    uint  overflow_inside_count;
    uint  overflow_incoming_count;
    uint  first_overflow_voxel_inside;
    uint  first_overflow_voxel_incoming;
    uint  correction_fallback_count;   // particles whose M_inv fell to identity
    uint  reserved_status_1;
    uint  reserved_status_2;
    uint  reserved_status_3;
    uint  reserved_status_4;
    uint  reserved_status_5;
    uint  reserved_status_6;
    uint  reserved_status_7;            // total 16 uint = 64 B, one cache line
};

layout(std430, set = 3, binding = 1) buffer OverflowLogBuffer {
    uint  log_event_count;              // atomic counter, modulo ring size
    uint  reserved_log_pad_0;
    uint  reserved_log_pad_1;
    uint  reserved_log_pad_2;
    uvec4 log_events[];                 // ring buffer: (voxel_id, step, kind, lost_particle_id)
};

// Inlet template: one full particle state per template slot. The inlet spawn
// kernel copies from a slot into a free main-pool slot when triggered.
struct InletTemplateEntry {
    vec4 position_voxel_id;
    vec4 velocity_mass;
    uint material;
    uint reserved_0;
    uint reserved_1;
    uint reserved_2;
};

layout(std430, set = 3, binding = 2) buffer InletTemplateBuffer {
    InletTemplateEntry inlet_template[];
};

layout(std430, set = 3, binding = 3) buffer DispatchIndirectBuffer {
    // Consumed by vkCmdDispatchIndirect. A small updater kernel sets x when
    // alive_particle_count changes.
    uint dispatch_indirect_x;
    uint dispatch_indirect_y;
    uint dispatch_indirect_z;
};

layout(std430, set = 3, binding = 4) buffer GhostOutPacketBuffer {
    // (V1+) boundary particles packed for send to peer GPU
    vec4 ghost_out_packet[];
};

layout(std430, set = 3, binding = 5) buffer GhostInStagingBuffer {
    // (V1+) receive destination from peer GPU, before unpack into set 2
    vec4 ghost_in_staging[];
};

layout(std430, set = 3, binding = 6) buffer DiagnosticBuffer {
    // (optional, debug builds only) curl, FTLE, vorticity for visualization
    vec4 diagnostic[];
};

// ----------------------------------------------------------------------------
// MaterialParametersBuffer — per-group SPH parameters.
// MaterialBuffer[particle] stores a group_id that indexes here.
// Struct is 48 B (12 × 4 B); for 16 groups the entire buffer is 768 B
// and lives in L1 after first read.
// ----------------------------------------------------------------------------
struct MaterialParameters {
    // --- Core (V0 used) -----------------------------------------------------
    uint  kind;                   // MATERIAL_FLUID / BOUNDARY / INLET / ROTOR
    float rest_density;
    float viscosity;
    float eos_constant;           // c0² · rest_density / power_parameter
    float smoothing_length;       // V0: same as global SMOOTHING_LENGTH; future: per-group for multi-res
    float radius;
    float volume;
    float rotor_angular_velocity; // Used by ROTOR kind only; 0 for others. Assumed about z-axis in V0.

    // --- Reserved for __future__ (V0 unused) --------------------------------
    float viscosity_transfer;     // micropolar mass transfer
    float viscosity_rotation;     // micropolar rotation
    uint  reserved_material_0;
    uint  reserved_material_1;
};  // 48 B total

layout(std430, set = 3, binding = 7) buffer MaterialParametersBuffer {
    MaterialParameters material_parameters[];
};

layout(std430, set = 3, binding = 8) buffer DefragScratchCounterBuffer {
    // Single uint, atomic-incremented by defrag.comp when USE_PREFIX_SUM_DEFRAG=false.
    // CPU resets to 0 before each defrag dispatch (vkCmdFillBuffer).
    // When USE_PREFIX_SUM_DEFRAG=true, declared but unwritten — DCE removes it
    // from defrag.spv. Other shaders never reference it; glslc -O strips
    // it from their SPIR-V entirely.
    uint defrag_scratch_counter;
};

#endif  // SPH_COMMON_GLSL_INCLUDED
