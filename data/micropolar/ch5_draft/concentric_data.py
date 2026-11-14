"""
concentric_data.py - the concentric rows of the pack in one form, whichever analysis file they come from
(Stage 4: couette_stage4.json; Stage 6: concentric.json). Adds what the older file lacks: the rule in its slip
form with the hydrodynamic radii of the control.
"""
import style
from experiment.validation import micropolar_couette_reference as reference
from experiment.validation.micropolar_couette import design_parameters
from experiment.validation.micropolar_stage5 import slip_rule_torque_change

INNER_RADIUS, OUTER_RADIUS, GAP, WALL_SPEED, FAR_VISCOSITY, DENSITY, MACH = 0.4, 0.5, 0.1, 1.0, 0.01, 1000.0, 0.02
ROTATION_RATE = WALL_SPEED / INNER_RADIUS
RATIOS = (3.0, 1.0, 0.5)


def load(source: dict) -> dict:
    analysis = style.load_json(source["analysis"])
    runs = analysis["runs"]
    for name, record in runs.items():
        if "relative_torque_change" not in record:
            continue
        control = runs[record["control"]]
        displacement = record["wall_displacement_in_spacings"] * GAP / record["gap_in_spacings"]
        record.setdefault("slip_rule", slip_rule_torque_change(control["hydrodynamic_inner_radius"],
                                                               control["hydrodynamic_outer_radius"], displacement))
        record.setdefault("slip_rule_over_exact", record["slip_rule"] / record["exact_hydrodynamic"])
    return analysis


def exact_change(coupling_ratio: float, gap_over_two_lengths: float) -> float:
    design = design_parameters(FAR_VISCOSITY, coupling_ratio, GAP / (2.0 * gap_over_two_lengths))
    eringen = reference.eringen_parameters_from_solver_parameters(
        DENSITY, design["viscosity"], design["coupling"], design["viscosity_rotation"])
    return reference.relative_torque_change(INNER_RADIUS, OUTER_RADIUS, eringen.shear_viscosity,
                                            eringen.coupling_viscosity, eringen.spin_viscosity, WALL_SPEED, 0.0,
                                            0.0, 0.0)


def slip_rule_change(coupling_ratio: float, gap_over_two_lengths: float) -> float:
    length = GAP / (2.0 * gap_over_two_lengths)
    design = design_parameters(FAR_VISCOSITY, coupling_ratio, length)
    displacement = design["coupling"] * length / (2.0 * design["viscosity"])
    return slip_rule_torque_change(INNER_RADIUS, OUTER_RADIUS, displacement)


def tags(runs: dict, names) -> str:
    """The git describe of the named runs, shortest first."""
    found = sorted({runs[name]["revision"]["describe"] for name in names if runs[name].get("revision")},
                   key=lambda tag: (len(tag), tag))
    return ", ".join(f"`{tag}`" for tag in found)


def parameters(record: dict) -> str:
    """K, l / dx, m, g / dx, Ma, the two Reynolds numbers of one micropolar row, as caption text."""
    return (f"K = {record['coupling_ratio']:g}, l = {record['length_in_spacings']:g} dx, "
            f"m = {record['gap_over_two_lengths']:g}, g = {record['gap_in_spacings']} dx, Ma = {MACH:g}, "
            f"Re = {record['reynolds_far_viscosity']:g} on mu_inf ({record['reynolds_shear_viscosity']:g} on mu)")
