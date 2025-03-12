from src.definitions import ExampleConfig
from multi.postprocessing import read_bam_colors
from multi.plotting_context import PlottingContext
import numpy as np


def plot_error_norm(examples, out_file):
    """plot error norm against number for dofs"""

    bamcd = read_bam_colors()
    red = bamcd["red"]
    blue = bamcd["blue"]

    marker = {"lpanel_1": "s", "lpanel_2": "o", "lpanel_3": "P"}
    color = {"normal": tuple(red[0]), "multivariate_normal": tuple(blue[0])}

    with PlottingContext([__file__, out_file], "pdiercks_multi") as fig:
        # figure.figsize: 4.773, 2.950
        width = 4.773
        height = 2.95
        factor = 1.0
        fig.set_size_inches(factor * width, factor * height)
        ax = fig.subplots()

        for kwargs in examples:
            ex = ExampleConfig(**kwargs)
            for distr in color.keys():
                mark = marker[ex.name]
                clr = color[distr]
                npz = ex.mean_error(distr)
                data = np.load(npz.as_posix())

                npz_std = ex.std_error(distr)
                data_std = np.load(npz_std.as_posix())
                std = data_std["std"]

                if distr == "multivariate_normal":
                    label = "correlated, "
                if distr == "normal":
                    label = "uncorrelated, "
                label += r"$\texttt{ttol}"+r"={}".format(ex.ttol)+"$"

                ax.semilogy(
                        data["num_dofs"], data["err"],
                        color=clr, marker=mark, label=label
                        )
                ax.fill_between(data["num_dofs"], data["err"]-std, data["err"]+std,
                        alpha=0.2, color=clr)

        ax.set_xlabel("Number of DOFs.")
        numerator = r"\norm{u_{\mathrm{fom}} - u_{\mathrm{rom}}}"
        denominator = r"\norm{u_{\mathrm{fom}}}"
        ax.set_ylabel(r"$\nicefrac{{{}}}{{{}}}$".format(numerator, denominator))
        ax.legend(loc="best", bbox_to_anchor=(0.65, 1.0))
