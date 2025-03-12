import re
from collections import defaultdict
import dolfinx
import ufl
from mpi4py import MPI
import numpy as np

from multi.postprocessing import read_bam_colors
from multi.plotting_context import PlottingContext

from src.definitions import ExampleConfig


def compute_std_error(input_files, out_file):
    """computes the standard deviation of the error

    Each input file contains the error for a single
    realization"""
    nmodes = []
    for npz in input_files:
        data = np.load(npz)
        num_modes = data["num_modes"].size
        nmodes.append(num_modes)

    min_num_modes = np.amin(nmodes)
    errs = []
    for npz in input_files:
        data = np.load(npz)
        errs.append(data["err"][:min_num_modes])

    E = np.vstack(errs)
    std = np.std(E, axis=0)  # shape (min_num_modes,)
    np.savez(out_file, std=std)


def compute_mean_minmax_modes(input_files):
    """for each realization get the minimum over
    the local maxima of number of edge modes.
    Then compute the mean over all realizations.
    """
    modes = []
    for npz in input_files:
        data = np.load(npz)
        num_modes = data["max_modes"]
        minmax = np.amin(np.amax(num_modes, axis=1))
        modes.append(minmax)
    return int(np.mean(modes))


def compute_mean_error(input_files, out_file):
    """computes the mean of the error and associated
    number of DOFs"""
    errs, ndofs, sizes = [], [], []
    for npz in input_files:
        data = np.load(npz)
        errs.append(data["err"])
        ndofs.append(data["num_dofs"])
        sizes.append(data["err"].size)

    # max number of modes may differ per real
    N = np.amin(sizes)
    error = []
    num_dofs = []
    for e, d in zip(errs, ndofs):
        error.append(e[:N])
        num_dofs.append(d[:N])
    E = np.vstack(error)
    D = np.vstack(num_dofs)
    mean = np.mean(E, axis=0)
    mean_dofs = np.mean(D, axis=0)
    np.savez(out_file, err=mean, num_dofs=mean_dofs)


def plot_pod_modes(example=None, distr=None, real=None, cell_index=None, edge=None, num_modes=4, sub=0, title=None, edge_grid_cells=10, out_file=None):

    assert edge in ("bottom", "left", "right", "top")
    assert cell_index in list(range(example.num_cells))

    if sub in ["x", "X"]:
        sub = 0
    if sub in ["y", "Y"]:
        sub = 1

    assert sub in (0, 1)

    domain = dolfinx.mesh.create_unit_interval(MPI.COMM_SELF, edge_grid_cells)
    ve = ufl.VectorElement("P", domain.ufl_cell(), degree=2, dim=2)
    V = dolfinx.fem.FunctionSpace(domain, ve)

    npzfile = example.pod_bases(distr, real, cell_index)
    data = np.load(npzfile.as_posix())
    modes = data[edge]

    xcomp = modes[:, ::2]
    ycomp = modes[:, 1::2]
    xdofs = V.tabulate_dof_coordinates()

    if edge in ("bottom", "top"):
        xx = xdofs[:, 0]
    elif edge in ("right", "left"):
        xx = xdofs[:, 1]
    oo = np.argsort(xx)

    if out_file is not None:
        plot_argv = [__file__, out_file]
    else:
        plot_argv = [__file__]

    if sub == 0:

        with PlottingContext(plot_argv, "halfwidth") as fig:
            ax = fig.subplots()

            if title is not None:
                ax.set_title(title)
            for i, y in enumerate(xcomp[:num_modes]):
                ax.plot(xx[oo], y[oo], marker=".", label=str(i))
            ax.legend()


    elif sub == 1:

        with PlottingContext(plot_argv, "halfwidth") as fig:
            ax = fig.subplots()

            if title is not None:
                ax.set_title(title)
            for i, y in enumerate(ycomp[:num_modes]):
                ax.plot(xx[oo], y[oo], marker=".", label=str(i))

            ax.legend()


def plot_hierarchical_modes(example=None, distr=None, real=None, cell_index=None, edge=None, num_modes=4, sub=0, title=None, edge_grid_cells=10, out_file=None):

    assert edge in ("bottom", "left", "right", "top")
    assert cell_index in list(range(example.num_cells))

    if sub in ["x", "X"]:
        sub = 0
    if sub in ["y", "Y"]:
        sub = 1

    assert sub in (0, 1)

    domain = dolfinx.mesh.create_unit_interval(MPI.COMM_SELF, edge_grid_cells)
    ve = ufl.VectorElement("P", domain.ufl_cell(), degree=2, dim=2)
    V = dolfinx.fem.FunctionSpace(domain, ve)

    npzfile = example.pod_bases(distr, real, cell_index)
    data = np.load(npzfile.as_posix())
    modes = data[edge]

    xcomp = modes[:, ::2]
    ycomp = modes[:, 1::2]
    xdofs = V.tabulate_dof_coordinates()

    if edge in ("bottom", "top"):
        xx = xdofs[:, 0]
    elif edge in ("right", "left"):
        xx = xdofs[:, 1]
    oo = np.argsort(xx)

    if out_file is not None:
        plot_argv = [__file__, out_file]
    else:
        plot_argv = [__file__]

    # mask: take every second non-zero x(or y)-component mode
    mask = list(range(sub, num_modes * 2, 2))

    if sub == 0:

        with PlottingContext(plot_argv, "pdiercks_multi") as fig:
            ax = fig.subplots()

            if title is not None:
                ax.set_title(title)
            for i, y in enumerate(xcomp[mask]):
                ax.plot(xx[oo], y[oo], marker=".", label=str(i))
            ax.legend()


    elif sub == 1:

        with PlottingContext(plot_argv, "pdiercks_multi") as fig:
            ax = fig.subplots()

            if title is not None:
                ax.set_title(title)
            for i, y in enumerate(ycomp[:mask]):
                ax.plot(xx[oo], y[oo], marker=".", label=str(i))

            ax.legend()


def plot_xttol_yerr_yndofs(examples, out_file):
    """

    Parameters
    ----------
    examples 
        The examples to consider.
    """

    distributions = ("normal", "multivariate_normal")
    dlabel = ("uncorrelated", "correlated")

    data = {}
    for d in distributions:
        data[d] = {"tols": [], "err": [], "ndofs": []}

    for kwargs in examples:
        ex = ExampleConfig(**kwargs)
        ttol = ex.ttol
        for d in distributions:
            rom_data = [ex.rom_solution(d, r) for r in range(ex.num_real)]
            minmaxmodes = compute_mean_minmax_modes(rom_data)
            mean_error_npz = ex.mean_error(d)
            mean_error = np.load(mean_error_npz)

            err = mean_error["err"][minmaxmodes]
            ndofs = mean_error["num_dofs"][minmaxmodes]

            data[d]["tols"].append(ttol)
            data[d]["err"].append(err)
            data[d]["ndofs"].append(ndofs)


    bamcd = read_bam_colors()
    red = bamcd["red"]
    blue = bamcd["blue"]
    marker = {"error": "x", "ndofs": "o"}
    colors = {"normal": tuple(red[0]), "multivariate_normal": tuple(blue[0])}

    plot_argv = [__file__, out_file]
    with PlottingContext(plot_argv, "halfwidth") as fig:

        ax = fig.subplots()

        ax.set_xlabel("Target tolerance")
        ax.set_ylabel("Error (marker='x')")
        ax_r = ax.twinx()
        ax_r.set_ylabel("Number of DOFs (marker='o')")

        for d, label in zip(distributions, dlabel):
            ax.loglog(data[d]["tols"], data[d]["err"], color=colors[d],
                      marker=marker["error"], label=label)
        for d, label in zip(distributions, dlabel):
            ax_r.semilogx(
                    data[d]["tols"], data[d]["ndofs"], color=colors[d],
                    marker=marker["ndofs"], label=label)

        ax.legend(markerscale=0., loc="best")


def plot_xerr_yndofs_ynsamples(examples, out_file):
    """

    Parameters
    ----------
    examples 
        The examples to consider.
    """

    distributions = ("normal", "multivariate_normal")
    dlabel = ("uncorrelated", "correlated")

    data = {}
    for d in distributions:
        data[d] = {"nsamp": [], "err": [], "ndofs": []}

    for kwargs in examples:
        ex = ExampleConfig(**kwargs)
        for d in distributions:
            rom_data = [ex.rom_solution(d, r) for r in range(ex.num_real)]
            minmaxmodes = compute_mean_minmax_modes(rom_data)
            mean_error_npz = ex.mean_error(d)
            mean_error = np.load(mean_error_npz)

            err = mean_error["err"][minmaxmodes]
            ndofs = mean_error["num_dofs"][minmaxmodes]

            data[d]["err"].append(err)
            data[d]["ndofs"].append(ndofs)

            # number of samples from log
            log_npz = ex.mean_log_data(d)
            log = np.load(log_npz.as_posix())
            nsamp = log["num_samples"]

            data[d]["nsamp"].append(nsamp)


    bamcd = read_bam_colors()
    red = bamcd["red"]
    blue = bamcd["blue"]
    marker = {"nsamp": "x", "ndofs": "o"}
    colors = {"normal": tuple(red[0]), "multivariate_normal": tuple(blue[0])}

    plot_argv = [__file__, out_file]
    with PlottingContext(plot_argv, "halfwidth") as fig:

        ax = fig.subplots()

        ax.set_xlabel("Error")
        ax.set_ylabel("Number of DOFs (marker='o')")
        ax_r = ax.twinx()
        ax_r.set_ylabel("Number of training samples (marker='x')")

        for d, label in zip(distributions, dlabel):
            ax.loglog(data[d]["err"], data[d]["ndofs"], color=colors[d],
                      marker=marker["ndofs"], label=label)
        for d, label in zip(distributions, dlabel):
            ax_r.semilogx(
                    data[d]["err"], data[d]["nsamp"], color=colors[d],
                    marker=marker["nsamp"], label=label)

        ax.legend(markerscale=0., loc="best")


def parse_log(search, logfile):
    """parse given logfile for keys in search"""

    def int_or_float(numstr):
        try:
            return int(numstr)
        except ValueError:
            return float(numstr)

    R = defaultdict(list)
    with open(logfile, "r") as instream:
        for line in instream.readlines():
            for key, chars in search.items():
                if chars in line:
                    filtered_string = re.sub("[^.0-9e-]", "", line.split(chars)[1])
                    number = int_or_float(filtered_string.strip("."))
                    R[key].append(number)
    # NOTE
    # there might be several lines in the log
    # that match the search key
    # either several runs in the log or in case of the rom
    # we have assembly and solve time for increasing number of modes
    return R
