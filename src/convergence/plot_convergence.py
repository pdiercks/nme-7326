"""
plot results from task convergence

Usage:
    plot_convergence.py [options] A B

Arguments:
    A          The file containing convergence results for RCE type 01.
    B          The file containing convergence results for RCE type 02.

Options:
    -h, --help     Show this message and exit.
    --legend       If True, show legend.
    --pdf=FILE     Save plot to FILE.
"""

import sys
import numpy
from pathlib import Path
from docopt import docopt
from multi.plotting_context import PlottingContext
from multi.postprocessing import read_bam_colors


def parse_args(args):
    args = docopt(__doc__, args)
    args["A"] = Path(args["A"])
    args["B"] = Path(args["B"])
    return args


def main(args):
    args = parse_args(args)

    bamcd = read_bam_colors()
    red = [tuple(r) for r in bamcd["red"]]
    blue = [tuple(b) for b in bamcd["blue"]]
    colors = [blue[0], red[0]]
    txt_files = [args["A"], args["B"]]

    plot_argv = [__file__, args["--pdf"]] if args["--pdf"] else [__file__]
    with PlottingContext(plot_argv, "pdiercks_multi") as fig:
        ax = fig.subplots()
        for i, txtfile in enumerate(txt_files):
            data = numpy.genfromtxt(txtfile, delimiter=",")
            dofs = numpy.sqrt(data[:-1, -1])
            err = data[:-1, 0]
            ax.loglog(
                dofs,
                err,
                color=colors[i],
                marker="o",
                label=f"mesoscale subdomain type {int(txtfile.stem.split('_')[-1]):02d}",
            )
        xlabel = r"\sqrt{N_{\mathrm{DoFs}}}"
        ax.set_xlabel(r"${}$".format(xlabel), fontsize=12)
        ylabel = r"\norm{u_{\mathrm{ref}} - u_{\mathrm{fem}}} / \norm{u_{\mathrm{ref}}}"
        ax.set_ylabel(r"${}$".format(ylabel), fontsize=12)
        if args["--legend"]:
            ax.legend()


if __name__ == "__main__":
    main(sys.argv[1:])
