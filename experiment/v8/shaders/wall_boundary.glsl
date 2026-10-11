// ============================================================================
// wall_boundary.glsl - the wall boundary option (E37). Included by density.comp,
// force.comp and wall_extrapolate.comp only, after common.glsl / helpers.glsl,
// so the other modules' SPIR-V does not change.
//
// WALL_BOUNDARY (spec const 100) = case.yaml numerics.wall_boundary:
//   0 = simple (default): the v6 walls. A wall particle integrates its density
//       from its fluid neighbours (density.comp skips wall-wall pairs), stores
//       rho0 and the pressure EOS(rho0 + dt * drho/dt); force uses its stored
//       velocity (0, or the lid speed) in the viscous term.
//   1 = adami: the wall pressure and no-slip condition of Adami, Hu & Adams
//       (2012), walls storing rho0 (E36's adami_rho0 = WALL_BC 3 of the
//       v7-wall-bc branch). wall_extrapolate.comp, after density and before
//       every force, sets each wall particle's pressure to the Shepard average
//       of its fluid neighbours' pressure (+ the body-force term), keeps its
//       stored density at rho0 and writes a dummy velocity 2 u_w - (Shepard
//       average of the fluid velocity), which force uses for wall neighbours in
//       the viscous term. Walls do not integrate density. Continuity, delta
//       term, KCG and the wall volume see rho0 and the wall's stored,
//       prescribed velocity, as with simple.
//       One slab only (K = 1): the wall pass reads local neighbours and the
//       ghost transport does not carry wall_dummy_velocity; SphSimulatorV6
//       refuses an adami slab with a peer.
// ============================================================================

layout(constant_id = 100) const uint WALL_BOUNDARY = 0u;
const uint WALL_BOUNDARY_SIMPLE = 0u;
const uint WALL_BOUNDARY_ADAMI  = 1u;
// adami: walls skip the density pass and take (rho0, p_w) from the wall pass; a wall neighbour's viscous
// velocity is its dummy velocity
const bool WALL_ADAMI = (WALL_BOUNDARY == WALL_BOUNDARY_ADAMI);

layout(std430, set = 0, binding = 10) buffer WallDummyVelocityBuffer {
    // adami: per wall particle, written by wall_extrapolate.comp every step before force reads it:
    // (u_dummy = 2 u_w - u~_w, Sigma_f W_wf). Fluid slots are never written. Transient: defrag does not permute
    // it (it is recomputed before every read); not part of the restart state. simple: a 16-byte stub, never read.
    vec4 wall_dummy_velocity[];
};
