"""
Item 10 - the film function over h^3 against h / l at K = 3: micropolar lubrication theory, the wall rule, its
first-order term alone, and the sharper form of the rule (core with the slip length - delta plus the flux of the
two wall layers). Host only.

Run from the repository root:  python data/micropolar/ch5_draft/fig_10_film_function.py
"""
import matplotlib.pyplot as plt
import numpy as np

import style
from experiment.validation import micropolar_film as film

COUPLING_RATIO = 3.0
RUN_GAPS = (1.6, 2.8, 4.0, 7.0, 8.0, 14.0)          # narrowest gap / l of the eccentric runs


def main() -> int:
    style.setup()
    ratio = film.displacement_over_length(COUPLING_RATIO)
    gap = np.geomspace(0.5, 60.0, 600)
    figure, axis = plt.subplots(figsize=(style.NARROW, 0.92 * style.NARROW))
    cube = gap ** 3
    lubrication = film.film_function("lubrication", gap, 1.0, ratio) / cube
    axis.plot(gap, lubrication, color=style.INK, **style.FILM, label="lubrication theory")
    axis.plot(gap, film.film_function("sharper", gap, 1.0, ratio) / cube, color=style.MUTED, linewidth=0.8,
              linestyle=(0, (8, 2)), label=r"$h^3 - 6\delta h^2 + 12\delta\ell h$")
    axis.plot(gap, film.film_function("rule", gap, 1.0, ratio) / cube, color=style.INK, **style.RULE,
              label=r"rule, $(h - 2\delta)^3$")
    axis.plot(gap, film.film_function("first_order", gap, 1.0, ratio) / cube, color=style.INK, **style.FIRST_ORDER,
              label=r"first order, $h^3 - 6\delta h^2$")
    # the four-constant solution of the model's two equations at the narrowest gaps of the runs
    checked = np.array(RUN_GAPS)
    solved = np.array([12.0 * float(film.plane_film(value, 1.0, ratio, gradient=-1.0)["flux"][0]) / value ** 3
                       for value in checked])
    axis.plot(checked, solved, **{**style.run_markers(0, filled=False), "markeredgecolor": style.INK},
              label="model equations, solved")
    limit = (1.0 + 0.5 * COUPLING_RATIO) / (1.0 + COUPLING_RATIO)
    axis.axhline(limit, color=style.GRID, linewidth=0.5, zorder=0)
    axis.text(58.0, limit + 0.012, r"$\nu_\infty/\nu$", ha="right", va="bottom", fontsize=style.FONT_SIZE - 1.5)
    axis.set_xscale("log")
    axis.set_xlim(0.5, 60.0)
    axis.set_ylim(0.0, 1.02)
    axis.set_xticks([1, 2, 5, 10, 20, 50])
    axis.set_xticklabels(["1", "2", "5", "10", "20", "50"])
    axis.set_xlabel(r"$h\,/\,\ell$")
    axis.set_ylabel(r"$f(h)\,/\,h^3$")
    axis.legend(loc="lower right", handletextpad=0.4, borderaxespad=0.2)
    figure.tight_layout(pad=0.4)
    style.save(figure, "fig_10_film_function")

    def at(kind, value):
        return float(film.film_function(kind, value, 1.0, ratio)) / float(film.film_function("lubrication", value,
                                                                                             1.0, ratio))
    table = "; ".join(f"h / l = {value:g}: rule {at('rule', value):.3f}, first order {at('first_order', value):.3f}, "
                      f"sharper form {at('sharper', value):.4f}" for value in (4.0, 7.0, 10.0, 14.0, 20.0))
    caption = (
        "Pressure-driven flux of a plane film between two walls with w = 0, as f(h) / h^3 against h / l (flux = "
        f"- f(h) (dp/dx) / (12 mu_inf)), K = {COUPLING_RATIO:g} (delta = K l / (2 (1 + K)) = {ratio:g} l). "
        "Dash-dotted: the film function of micropolar lubrication theory, f = h^3 - 6 delta h^2 coth(h / (2 l)) + "
        "12 delta l h (in the notation of that theory h^3 + 12 l'^2 h - 6 N l' h^2 coth(N h / (2 l')) with N^2 = "
        "delta / l and l' = N l; the user quotes it from Prakash and Sinha 1975, neither of us has compared it with "
        "the original). Circles: the two ordinary differential equations of the model solved between the walls at "
        "the narrowest gaps of the eccentric runs; they agree with the formula to 1e-15. Dashed: the wall rule "
        "(both walls displaced by delta). Dotted: its first-order term alone. Long dashes: the sharper form of the "
        "rule, a Newtonian core with the slip length - delta plus delta l times the wall shear rate carried by each "
        "wall layer; it differs from the film function by 6 delta h^2 (coth(h / (2 l)) - 1), about 12 delta h^2 "
        "exp(-h / l). The grey line is the limit h -> 0, nu_inf / nu = 0.625 (a Newtonian fluid of viscosity mu + "
        f"kappa). Ratios to the film function: {table}.")
    style.record("10", "The film function over h^3 against h / l at K = 3", "final (host only)", caption,
                 ["fig_10_film_function.pdf", "fig_10_film_function.png"],
                 ["experiment/validation/micropolar_film.py", "data/micropolar/stage6/film_check.txt"],
                 {"limit": limit}, "host")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
