"""currently 28.11.2022 consider beam problem for ttol=[1e-1, 1e-2, 1e-3]

plot error against number of dofs for each target tol

ttol -> different marker styles
distr --> different color
"""

from src.definitions import ExampleConfig
from multi.postprocessing import read_bam_colors
from multi.plotting_context import PlottingContext
import numpy as np


def plot_error_norm(examples, out_file):
    """plot error norm against number for dofs"""

    BEAM = [examples[0], examples[1], examples[2]]
    HBEAM = [examples[3], examples[4], examples[5]]

    bamcd = read_bam_colors()
    red = bamcd["red"]
    blue = bamcd["blue"]
    green = bamcd["green"]

    # same tolerance, but the material is varied
    marker = {
            "beam_1": "s", "beam_2": "o", "beam_3": "P",
            "hbeam_1": "s", "hbeam_2": "o", "hbeam_3": "P"
            }

    color = {"normal": tuple(red[0]), "multivariate_normal": tuple(blue[0]), "hierarchical": tuple(green[0])}

    dist_to_label = {"normal": "uncorrelated", "multivariate_normal": "correlated", "hierarchical": "hierarchical"}

    # for label
    ratios = [1, 1.5, 2]
    frac = r"\nicefrac{E_{\mathrm{a}}}{E_{\mathrm{m}}}"

    with PlottingContext([__file__, out_file], "pdiercks_multi") as fig:
        # figure.figsize: 4.773, 2.950
        width = 4.773
        height = 2.95
        factor = 1.0
        fig.set_size_inches(factor * width, factor * height)
        ax = fig.subplots()

        for kwargs in BEAM:
            ex = ExampleConfig(**kwargs)

            name = ex.name.replace("_", "")
            mat_index = int(name.strip("hbeam")) - 1
            ratio = ratios[mat_index]

            for distr in ["normal", "multivariate_normal"]:
                mark = marker[ex.name]
                clr = color[distr]
                npz = ex.mean_error(distr)
                data = np.load(npz.as_posix())

                npz_std = ex.std_error(distr)
                data_std = np.load(npz_std.as_posix())
                std = data_std["std"]

                label = r"{} ${}={}$".format(dist_to_label[distr], frac, ratio)

                ax.semilogy(
                        data["num_dofs"], data["err"],
                        color=clr, marker=mark, label=label
                        )
                ax.fill_between(
                        data["num_dofs"], data["err"] - std, data["err"] + std,
                        alpha=0.2, color=clr
                        )

        for kwargs in HBEAM:
            ex = ExampleConfig(**kwargs)

            name = ex.name.replace("_", "")
            mat_index = int(name.strip("hbeam")) - 1
            ratio = ratios[mat_index]

            distr = "hierarchical"
            mark = marker[ex.name]
            clr = color[distr]
            npz = ex.mean_error(distr)
            data = np.load(npz.as_posix())

            label = r"{} ${}={}$".format(dist_to_label[distr], frac, ratio)

            ax.semilogy(
                    data["num_dofs"][::2], data["err"][::2],
                    color=clr, marker=mark, label=label
                    )

        ax.set_xlabel("Number of DOFs.")
        numerator = r"\norm{u_{\mathrm{fom}} - u_{\mathrm{rom}}}"
        denominator = r"\norm{u_{\mathrm{fom}}}"
        ax.set_ylabel(r"$\nicefrac{{{}}}{{{}}}$".format(numerator, denominator))

        ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=9)
        # ax.set_ylim(1e-5, 0.5)
        # ax.set_ylim(5e-8, 0.5)
        # ax.set_xlim(0, 550)
        # ax.yaxis.set_ticks(np.array([0.1, 0.01, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]))
