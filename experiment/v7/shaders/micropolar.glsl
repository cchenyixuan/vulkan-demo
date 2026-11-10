// ============================================================================
// micropolar.glsl - the micropolar terms (docs/micropolar_design.md). Included
// only under #ifdef MICROPOLAR, after common.glsl / helpers.glsl (and
// wall_boundary.glsl where the kernel has it), by the MICROPOLAR variants of
// compile_shaders_v7.py SHADER_VARIANTS (<kernel>_micropolar.comp.spv). The plain
// build of every kernel does not see this file, so the Newtonian SPIR-V does
// not change.
//
// Model (Bender et al. 2019 as written in Chen, Shao & Sugiyama 2026, Eqs. 32-33):
//
//   Dv/Dt       = -(1/rho) grad p + nu lap v + c curl w + g
//   Theta Dw/Dt =  xi lap w + c (curl v - 2 w),          c = nu + nu_t
//
//   nu    = MaterialParameters.viscosity            (= (mu + kappa) / rho in Eringen's notation)
//   nu_t  = MaterialParameters.viscosity_transfer   (may be negative; c = kappa / rho >= 0)
//   xi    = MaterialParameters.viscosity_rotation   (= gamma / rho)
//   Theta = MICROPOLAR_THETA, spec constant 21 of common.glsl (= j)
//
// Discrete operators, per pair inside force.comp's neighbour loop, with
// G_ij = kernel_gradient_corrected, r_ij = r_i - r_j, V_j = m / rho_j:
//
//   (curl w)_i = sum_j V_j (w_i - w_j) x G_ij
//   (curl v)_i = sum_j V_j (v_i - v~_j) x G_ij          v~_j = the viscous term's neighbour velocity
//   (lap  w)_i = 2 sum_j V_j (w_i - w_j) (r_ij . G_ij) / (r_ij^2 + EPS_H_SQUARED)
//
// The curls have the physical sign: v = Omega x r gives curl v = +2 Omega.
// Both are difference forms, exact for linear fields with KGC (up to the
// M (M + xi I)^-1 factor of REGULARIZATION_XI); they are not adjoint to each
// other, so the discrete energy balance of the coupling holds to truncation
// order only.
//
// Time integration: w is stored at half steps like v. predict kicks
// w^{n+1/2} = w^{n-1/2} + dt alpha^n; force evaluates alpha from v^{n+1/2},
// w^{n+1/2}, rho^{n+1}; bootstrap_half_kick sets w^{-1/2} = w^0 - dt/2 alpha^0.
//
// Walls: a wall particle's angular_velocity slot holds its prescribed
// micro-rotation w_wall (host upload, material key wall_angular_velocity) and
// is never integrated. wall_boundary simple: neighbours read that slot.
// adami: the wall pass writes the dummy micro-rotation 2 w_wall - w~ (w~ = the
// Shepard average of the fluid neighbours' w, the sums of the dummy velocity)
// into wall_dummy_angular_velocity and fluid particles read that.
//
// Spec constant ids: 21 MICROPOLAR_THETA (common.glsl), 116
// MICROPOLAR_REPLICA_LAYERS (here). Set 0 bindings 12, 13, 14 (binding 11 stays
// reserved for the global id buffer).
// ============================================================================

#ifndef SPH_MICROPOLAR_GLSL_INCLUDED
#define SPH_MICROPOLAR_GLSL_INCLUDED

// The replica layers, counted from the inner one, whose angular_velocity
// travels over a link: the layers whose fluid state feeds the receiver's
// force, directly (the inner layer) or through the adami wall pass on the
// inner ghost column (the outer layer of the three-layer ghost). The host
// passes GHOST_LAYERS - 1 on a slab with a peer, 0 otherwise.
layout(constant_id = 116) const uint MICROPOLAR_REPLICA_LAYERS = 0u;

layout(std430, set = 0, binding = 12) buffer AngularVelocityBuffer {
    // (w_x, w_y, w_z, reserved). Fluid: w^{n+1/2} between steps (predict's
    // kick). Wall: the prescribed w_wall. Permuted by defrag, transported with
    // replicas / migrants / departed copies, part of the restart state.
    vec4 angular_velocity[];
};

layout(std430, set = 0, binding = 13) buffer AngularAccelerationBuffer {
    // (alpha_x, alpha_y, alpha_z, reserved), written by force for every particle
    // it writes acceleration for, read by the next predict. Permuted by defrag,
    // part of the restart state, never transported (a migrant lands inside the
    // force band, which rewrites it before predict reads it).
    vec4 angular_acceleration[];
};

layout(std430, set = 0, binding = 14) buffer WallDummyAngularVelocityBuffer {
    // adami: per wall particle (2 w_wall - w~_w, Sigma_f W_wf), written by the
    // wall pass every step before force reads it. Same coverage, lifetime and
    // transport rules as wall_dummy_velocity: fluid slots never written, not
    // permuted by defrag, not transported, not in the restart state. simple: a
    // 16-byte stub, never read.
    vec4 wall_dummy_angular_velocity[];
};

#endif  // SPH_MICROPOLAR_GLSL_INCLUDED
