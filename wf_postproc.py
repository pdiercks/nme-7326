import sys
import pathlib
import numpy as np

from doit import get_var
from doit.tools import create_folder

from src.definitions import ExampleConfig

PYTHON = sys.executable

DOIT_CONFIG = {
    "verbosity": 2,
    "action_string_formatting": "new",
    "backend": "dbm",
    "dep_file": ".postproc.db",
}

CLI = {"mode": get_var("mode", "DEBUG"), "nreal": get_var("nreal", "1")}
NREAL = int(CLI["nreal"])

# dir structure
ROOT = pathlib.Path(__file__).parent
SRC = ROOT / "src"
WORK = ROOT / "work"
if not WORK.exists():
    create_folder(WORK)
    create_folder(WORK / "tex")

# Global parameters
# #################
DEGREE = 2
MATERIAL = SRC / "material.yaml"
DISTRIBUTIONS = ("normal", "multivariate_normal")
# measure functions in source and range space
# of transfer problem in these norms
SOURCENORM = "l2"
RANGENORM = "h1"
# measure global error in this norm
ERRNORM = "h1"
# convergence
PPE_RCE_01 = [3, 5, 11, 21, 41]  # points per edge
PPE_RCE_02 = [6, 11, 21, 31, 41, 51]

BLOCK = []
BEAM = []
HBEAM = []
LPANEL = []
target_tol = [1e-1, 1e-2, 1e-3]

if CLI["mode"] == "DEBUG":
    LOGLVL = "DEBUG"
    for k, ttol in enumerate(target_tol):
        BLOCK.append(
            {
                "degree": DEGREE,
                "material": MATERIAL.as_posix(),
                "name": f"block_{k+1}",
                "num_real": NREAL,
                "num_testvecs": 20,
                "nx": 3,
                "ny": 3,
                "range_product": RANGENORM,
                "rce_type": 1,
                "source_product": SOURCENORM,
                "ttol": ttol,
            }
        )
        BEAM.append(
            {
                "degree": DEGREE,
                "material": (SRC / f"beam/material_{k+1}.yaml").as_posix(),
                "name": f"beam_{k+1}",
                "num_real": NREAL,
                "num_testvecs": 20,
                "nx": 5,
                "ny": 3,
                "range_product": RANGENORM,
                "rce_type": 2,
                "source_product": SOURCENORM,
                "ttol": 0.1,
            }
        )
        HBEAM.append(
            {
                "degree": DEGREE,
                "material": (SRC / f"beam/material_{k+1}.yaml").as_posix(),
                "name": f"hbeam_{k+1}",
                "num_real": 1,
                "num_testvecs": 20,
                "nx": 5,
                "ny": 3,
                "range_product": RANGENORM,
                "rce_type": 2,
                "source_product": SOURCENORM,
                "ttol": 0.1,
            }
        )
        LPANEL.append(
            {
                "degree": DEGREE,
                "material": MATERIAL.as_posix(),
                "name": f"lpanel_{k+1}",
                "num_real": NREAL,
                "num_testvecs": 20,
                "nxy": 6,
                "range_product": RANGENORM,
                "rce_type": 2,
                "source_product": SOURCENORM,
                "ttol": ttol,
            }
        )

elif CLI["mode"] == "PAPER":
    LOGLVL = "DEBUG"
    for k, ttol in enumerate(target_tol):
        BLOCK.append(
            {
                "degree": DEGREE,
                "material": MATERIAL.as_posix(),
                "name": f"block_{k+1}",
                "num_real": NREAL,
                "num_testvecs": 20,
                "nx": 5,
                "ny": 5,
                "range_product": RANGENORM,
                "rce_type": 1,
                "source_product": SOURCENORM,
                "ttol": ttol,
            }
        )
        BEAM.append(
            {
                "degree": DEGREE,
                "material": (SRC / f"beam/material_{k+1}.yaml").as_posix(),
                "name": f"beam_{k+1}",
                "num_real": NREAL,
                "num_testvecs": 20,
                "nx": 50,
                "ny": 5,
                "range_product": RANGENORM,
                "rce_type": 2,
                "source_product": SOURCENORM,
                "ttol": 0.1,
            }
        )
        HBEAM.append(
            {
                "degree": DEGREE,
                "material": (SRC / f"beam/material_{k+1}.yaml").as_posix(),
                "name": f"hbeam_{k+1}",
                "num_real": 1,
                "num_testvecs": 20,
                "nx": 50,
                "ny": 5,
                "range_product": RANGENORM,
                "rce_type": 2,
                "source_product": SOURCENORM,
                "ttol": 0.1,
            }
        )
        LPANEL.append(
            {
                "degree": DEGREE,
                "material": MATERIAL.as_posix(),
                "name": f"lpanel_{k+1}",
                "num_real": NREAL,
                "num_testvecs": 20,
                "nxy": 20,
                "range_product": RANGENORM,
                "rce_type": 2,
                "source_product": SOURCENORM,
                "ttol": ttol,
            }
        )

else:
    raise KeyError(
        "The workflow is run in DEBUG mode (smaller problems) by default."
        "To run the real thing type `doit mode=PAPER`."
    )
print(f"Running in {CLI['mode']} mode")
print(f"Number of realizations: {CLI['nreal']}")


def get_rce_meshfiles(path, points_per_edge):
    """returns list of a rce meshfiles under path"""
    return [(path / f"rce_{ppe:02d}.msh") for ppe in points_per_edge]


def task__get_rce_metadata():
    """task to compute rce metadata for the converged rce meshes

    The metadata includes:
        - filepath
        - unit_length
        - points_per_edge
        - num_max_modes

    """

    def get_max_modes(ppe):
        """max number of fine scale modes depending on discretization"""
        nverts = 2 * ppe - 1
        ndofs = 2 * nverts
        return ndofs - 4

    def get_unit_length(meshfile):
        import meshio

        mesh = meshio.read(meshfile)
        x = mesh.points
        xmax = np.amax(x, axis=0)
        xmin = np.amin(x, axis=0)

        unit_length = float(abs(xmax[0] - xmin[0]))
        return unit_length

    def write(mesh_files, result_file, targets):
        data = np.genfromtxt(result_file, delimiter=",")
        err = data[:, 0]
        converged = np.argmax(err < 1e-2)
        filepath = mesh_files[converged]
        unit_length = get_unit_length(filepath.absolute().as_posix())
        print(f"Converged mesh: {filepath.absolute().as_posix()}")
        points_per_edge = int(filepath.stem.split("_")[1])
        num_cells = points_per_edge - 1
        rce_metadata = {
            "filepath": filepath.absolute().as_posix(),
            "unit_length": unit_length,
            "points_per_edge": points_per_edge,
            "num_cells": num_cells,
            "num_max_modes": get_max_modes(points_per_edge),
        }
        return rce_metadata

    result_files = [
        WORK / "convergence/convergence_01.txt",
        WORK / "convergence/convergence_02.txt",
    ]
    mesh_files_01 = get_rce_meshfiles(WORK / "convergence/rce_01", PPE_RCE_01)
    mesh_files_02 = get_rce_meshfiles(WORK / "convergence/rce_02", PPE_RCE_02)
    mesh_files = [mesh_files_01, mesh_files_02]

    name = 0
    for result, meshes in zip(result_files, mesh_files):
        name += 1
        yield {
            "name": f"{name:02d}",
            "file_dep": meshes + [result],
            "actions": [(write, [meshes, result])],
            "verbosity": 1,
            "clean": True,
        }


def task_mean_error():
    """compute the mean over number of realizations"""

    def mean_error(dependencies, targets):
        from src.postprocessing import compute_mean_error

        compute_mean_error(dependencies, targets[0])

    examples = BLOCK + BEAM + LPANEL
    for kwargs in examples:
        ex = ExampleConfig(**kwargs)
        num_real = ex.num_real
        for distr in DISTRIBUTIONS:

            # collect all result files (error.npz)
            deps = []
            for i in range(num_real):
                deps.append(ex.error(distr, i))

            target = ex.mean_error(distr)

            yield {
                "name": ":".join([ex.name, distr]),
                "file_dep": deps,
                "actions": [mean_error],
                "targets": [target],
                "clean": True,
            }


def task_mean_std_error():
    """compute the std of the error over number of realizations"""

    def compute_std(dependencies, targets):
        from src.postprocessing import compute_std_error
        compute_std_error(dependencies, targets[0])

    examples = BLOCK + BEAM + LPANEL
    for kwargs in examples:
        ex = ExampleConfig(**kwargs)
        num_real = ex.num_real
        for distr in DISTRIBUTIONS:

            # collect all result files (error.npz)
            deps = []
            for i in range(num_real):
                deps.append(ex.error(distr, i))

            target = ex.std_error(distr)

            yield {
                "name": ":".join([ex.name, distr]),
                "file_dep": deps,
                "actions": [compute_std],
                "targets": [target],
                "clean": True,
            }



def task_mean_error_hbeam():
    """compute the mean over number of realizations"""

    def mean_error(dependencies, targets):
        from src.postprocessing import compute_mean_error

        compute_mean_error(dependencies, targets[0])

    examples = HBEAM
    for kwargs in examples:
        ex = ExampleConfig(**kwargs)
        num_real = 1
        distr = "hierarchical"

        # collect all result files (error.npz)
        deps = []
        for i in range(num_real):
            deps.append(ex.error(distr, i))

        target = ex.mean_error(distr)

        yield {
            "name": ":".join([ex.name, distr]),
            "file_dep": deps,
            "actions": [mean_error],
            "targets": [target],
            "clean": True,
        }


def task_plot_rce_grid():
    """plot converged rce meshes"""
    plot_mesh = SRC / "convergence" / "plot_mesh.py"
    rce_types = [1, 2]
    for typ in rce_types:
        png = WORK / f"tex/converged_rce_{typ:02d}.png"
        yield {
            "name": f"{typ:02d}",
            "file_dep": [plot_mesh],
            "actions": [
                (create_folder, [png.parent]),
                f"{PYTHON} {plot_mesh} {{mesh}} --subdomains"
                f" --transparent --png={{targets}} --colormap=bam-RdBu",
            ],
            "getargs": {"mesh": (f"_get_rce_metadata:{typ:02d}", "filepath")},
            "targets": [png],
            "clean": True,
        }


def task_plot_convergence():
    """plot convergence data and save to pdf"""
    script = SRC / "convergence" / "plot_convergence.py"
    result_files = [
        WORK / "convergence/convergence_01.txt",
        WORK / "convergence/convergence_02.txt",
    ]
    pdf = WORK / "tex/convergence_plot.pdf"
    return {
        "file_dep": result_files + [script],
        "actions": [
            (create_folder, [pdf.parent]),
            f"{PYTHON} {script} {result_files[0]} {result_files[1]}"
            f" --legend --pdf={pdf}",
        ],
        "targets": [pdf],
        "clean": True,
    }


def task_get_log_data_basis():
    """average number of samples over realizations"""
    # in the previous step: task_parse_log_basis
    # the data is averaged over all oversapmling problems

    def get_mean(dependencies, targets):
        from collections import defaultdict

        r = defaultdict(list)
        for infile in dependencies:
            data = np.load(infile)
            for k in data.files:
                r[k].append(data[k])

        R = {}
        for k, v in r.items():
            R[k] = np.mean(v)

        np.savez(targets[0], **R)

    examples = BLOCK + BEAM + LPANEL
    for kwargs in examples:
        example = ExampleConfig(**kwargs)
        for distr in DISTRIBUTIONS:
            deps = []
            for real in range(example.num_real):
                deps.append(
                    example.empirical_basis_log(distr, real, 0).parent
                    / "gathered_basis_log_cells.npz"
                )
            target = example.mean_log_data(distr)
            yield {
                "name": ":".join([example.name, distr]),
                "actions": [(get_mean)],
                "targets": [target],
                "file_dep": deps,
            }


def task_parse_log_basis():
    """parse log files for a particular distribution"""

    def parse(dependencies, targets):
        """for a fixed realization
        take the mean over all oversampling problems
        """
        from src.postprocessing import parse_log
        from collections import defaultdict

        search = {
            "rrf_runtime": "Ran range finder algorithm in t=",
            "num_samples": "Number of training samples=",
            "compute_pod_modes_runtime": "Execution of compute_pod_modes took",
            "ext_time": "Extended pod modes in t=",
            "phi_time": "Computed coarse scale basis in",
            "extend_pod_modes_runtime": "Execution of extend_pod_modes took",
            }

        r = defaultdict(list)
        for log in dependencies:
            data = parse_log(search, log)
            for k, v in data.items():
                r[k].append(v[0])

        mean = {}
        for k, v in r.items():
            mean[k] = np.mean(v)
        np.savez(targets[0], **mean)

    examples = BLOCK + BEAM + LPANEL
    for kwargs in examples:
        example = ExampleConfig(**kwargs)
        for distr in DISTRIBUTIONS:
            for real in range(example.num_real):
                deps = []
                for ci in range(example.num_cells):
                    deps.append(example.empirical_basis_log(distr, real, ci))

                target = deps[0].parent / "gathered_basis_log_cells.npz"

                yield {
                    "name": ":".join([example.name, distr, str(real)]),
                    "actions": [parse],
                    "targets": [target],
                    "file_dep": deps,
                    "clean": True,
                }


def task_plot_block_num_edge_modes():
    """plot num edge modes"""

    for kwargs in BLOCK:
        block = ExampleConfig(**kwargs)
        for d in DISTRIBUTIONS:
            # use the first realization
            grid = block.coarse_grid
            sol = block.rom_solution(d, 0)
            target = WORK / "tex" / f"{block.name}_{d}_max_modes.png"

            yield {
                "name": f"{block.name}_{d}",
                "file_dep": [grid, sol],
                "targets": [target],
                "actions": [
                    " ".join(
                        [
                            PYTHON,
                            (SRC / "plot_num_edge_modes.py").as_posix(),
                            grid.as_posix(),
                            sol.as_posix(),
                            "--png",
                            target.as_posix(),
                            "--transparent",
                        ]
                    )
                ],
                "clean": True,
            }


def task_plot_block_error():
    """plot error against Ndofs"""

    def plot(example, targets):
        from src.block.plot_error_norm import plot_error_norm

        plot_error_norm(example, targets[0])

    deps = []
    for kwargs in BLOCK:
        block = ExampleConfig(**kwargs)
        for d in DISTRIBUTIONS:
            deps.append(block.mean_error(d))
            deps.append(block.std_error(d))

    target = WORK / "tex" / "block_error_plot.pdf"
    return {
        "file_dep": deps,
        "actions": [(plot, [BLOCK])],
        "targets": [target],
        "clean": True,
    }


def task_plot_block_xttol_2y():
    """plot error and ndofs against ttol"""

    def plot(examples, targets):
        from src.postprocessing import plot_xttol_yerr_yndofs

        plot_xttol_yerr_yndofs(examples, targets[0])

    deps = []
    for kwargs in BLOCK:
        block = ExampleConfig(**kwargs)
        for d in DISTRIBUTIONS:
            deps.append(block.mean_error(d))
            deps.append(block.mean_log_data(d))

    target = WORK / "tex" / "block_xttol_2y.pdf"
    return {
        "file_dep": deps,
        "actions": [(plot, [BLOCK])],
        "targets": [target],
        "clean": True,
    }


def task_plot_block_xerr_2y():
    """plot ndofs and num samples against err"""

    def plot(examples, targets):
        from src.postprocessing import plot_xerr_yndofs_ynsamples

        plot_xerr_yndofs_ynsamples(examples, targets[0])

    deps = []
    for kwargs in BLOCK:
        block = ExampleConfig(**kwargs)
        for d in DISTRIBUTIONS:
            deps.append(block.mean_error(d))
            deps.append(block.mean_log_data(d))

    target = WORK / "tex" / "block_xerr_2y.pdf"
    return {
        "file_dep": deps,
        "actions": [(plot, [BLOCK])],
        "targets": [target],
        "clean": True,
    }


def task_plot_beam_error():
    """plot error against Ndofs"""

    def plot(examples, targets):
        from src.beam.plot_error_norm import plot_error_norm

        plot_error_norm(examples, targets[0])

    deps = []
    for kwargs in BEAM:
        ex = ExampleConfig(**kwargs)
        for distr in DISTRIBUTIONS:
            deps.append(ex.mean_error(distr))
            deps.append(ex.std_error(distr))
    for kwargs in HBEAM:
        ex = ExampleConfig(**kwargs)
        deps.append(ex.mean_error("hierarchical"))

    target = WORK / "tex" / "beam_error_plot.pdf"
    return {
        "file_dep": deps,
        "actions": [(plot, [BEAM + HBEAM])],
        "targets": [target],
        "clean": True,
    }


def task_plot_lpanel_xttol_2y():
    """plot error and ndofs against ttol"""

    def plot(examples, targets):
        from src.postprocessing import plot_xttol_yerr_yndofs

        plot_xttol_yerr_yndofs(examples, targets[0])

    deps = []
    for kwargs in LPANEL:
        block = ExampleConfig(**kwargs)
        for d in DISTRIBUTIONS:
            deps.append(block.mean_error(d))

    target = WORK / "tex" / "lpanel_xttol_2y.pdf"
    return {
        "file_dep": deps,
        "actions": [(plot, [LPANEL])],
        "targets": [target],
        "clean": True,
    }


def task_plot_lpanel_xerr_2y():
    """plot ndofs and num samples against err"""

    def plot(examples, targets):
        from src.postprocessing import plot_xerr_yndofs_ynsamples

        plot_xerr_yndofs_ynsamples(examples, targets[0])

    deps = []
    for kwargs in LPANEL:
        block = ExampleConfig(**kwargs)
        for d in DISTRIBUTIONS:
            deps.append(block.mean_error(d))
            deps.append(block.mean_log_data(d))

    target = WORK / "tex" / "lpanel_xerr_2y.pdf"
    return {
        "file_dep": deps,
        "actions": [(plot, [LPANEL])],
        "targets": [target],
        "clean": True,
    }


def task_plot_lpanel_error():
    """plot error against Ndofs"""

    def plot(example, targets):
        from src.lpanel.plot_error_norm import plot_error_norm

        plot_error_norm(example, targets[0])

    deps = []
    for kwargs in LPANEL:
        lpanel = ExampleConfig(**kwargs)
        for d in DISTRIBUTIONS:
            deps.append(lpanel.mean_error(d))
            deps.append(lpanel.std_error(d))

    target = WORK / "tex" / "lpanel_error_plot.pdf"
    return {
        "file_dep": deps,
        "actions": [(plot, [LPANEL])],
        "targets": [target],
        "clean": True,
    }


def task_plot_pod_basis():
    """plot the pod basis for some edge"""

    def plot(example, distr, sub, cell_index, num_cells, targets):
        from src.postprocessing import plot_pod_modes

        inputs = {
            "example": example,
            "distr": distr,
            "real": 0,  # just use the first realization
            "cell_index": cell_index,
            "edge": "top",
            "num_modes": 4,
            "sub": sub,  # x- or y-component
            "edge_grid_cells": num_cells,
            "out_file": targets[0],
        }
        plot_pod_modes(**inputs)

    examples = []
    examples.append(BEAM[0])
    examples.append(BEAM[2])
    examples.append(BLOCK[2])

    for kwargs in examples:
        example = ExampleConfig(**kwargs)

        if example.name.startswith("block"):
            cell_index = int((example.num_cells - 1) / 2)
            typ = 1
        elif example.name.startswith("beam"):
            cell_index = 117
            typ = 2

        for distr in DISTRIBUTIONS:
            for sub in ["x", "y"]:
                target = WORK/f"tex/pod_modes_{example.name}_{distr}_{sub}.pdf"
                yield {
                    "name": ":".join([example.name, distr, sub]),
                    "actions": [(plot, [example, distr, sub, cell_index])],
                    "getargs": {
                        "num_cells": (f"_get_rce_metadata:{typ:02d}", "num_cells")
                    },
                    "targets": [target],
                    "clean": True,
                }


def task_plot_hierarchical_basis():
    """plot the hierarchical basis for some edge"""

    def plot(example, distr, sub, cell_index, num_cells, targets):
        from src.postprocessing import plot_hierarchical_modes

        inputs = {
            "example": example,
            "distr": distr,
            "real": 0,  # just use the first realization
            "cell_index": cell_index,
            "edge": "bottom",
            "num_modes": 4,
            "sub": sub,  # x- or y-component
            "edge_grid_cells": num_cells,
            "out_file": targets[0],
        }
        plot_hierarchical_modes(**inputs)

    examples = []
    examples.append(HBEAM[0])

    for kwargs in examples:
        example = ExampleConfig(**kwargs)

        if example.name.startswith("hbeam"):
            cell_index = int((example.num_cells - 1) / 2)
            typ = 2

        distr = "hierarchical"
        sub = "x"
        target = WORK / f"tex/pod_modes_{example.name}_{distr}_{sub}.pdf"
        yield {
            "name": ":".join([example.name, distr, sub]),
            "actions": [(plot, [example, distr, sub, cell_index])],
            "getargs": {"num_cells": (f"_get_rce_metadata:{typ:02d}", "num_cells")},
            "targets": [target],
            "clean": True,
        }


def task_get_fom_timings():
    """parse the log for fom timings"""

    def get_fom_timings(dependencies):
        from src.postprocessing import parse_log
        from collections import defaultdict

        search = {
            "fom_ndofs": "FOM system size N=",
            "fom_assembly": "Assembled FOM in t=",
            "fom_solve": "Computed FOM solution in t=",
            }

        r = defaultdict(list)
        for logfile in dependencies:
            data = parse_log(search, logfile)
            for k, v in data.items():
                r[k].append(max(v))

        # return mean over all realizations
        R = {}
        for k, v in r.items():
            R[k] = np.mean(v)
        return R

    distr = "normal"
    examples = [BLOCK[2], BEAM[2], LPANEL[2]]

    for kwargs in examples:
        ex = ExampleConfig(**kwargs)
        name = ex.name.split("_")[0]
        deps = [ex.error_log(distr, real) for real in range(ex.num_real)]

        yield {
            "name": name,
            "file_dep": deps,
            "actions": [(get_fom_timings)],
            "clean": True,
        }


def task_get_rom_timings():
    """parse the log for rom timings"""

    def get_rom_timings(num_modes, dependencies):
        """
        first for each realization get the runtime
        for assembly, solve etc. for prescribed number
        of modes.
        In a second step, take the mean over all
        realizations
        """
        from src.postprocessing import parse_log
        from collections import defaultdict
        search = {
                "rom_ndofs": "ROM system size N=",
                "rom_assembly": "Assembled ROM in t=",
                "rom_solve": "Computed ROM solution in t=",
                "rom_max_modes": "Global max number of modes per edge NUM_MAX_MODES=",
                }

        r = defaultdict(list)
        for logfile in dependencies:
            data = parse_log(search, logfile)
            for k, v in data.items():
                try:
                    r[k].append(v[num_modes])
                except IndexError:
                    assert k == "rom_max_modes"
                    r[k].append(max(v))

        R = {}
        for k, v in r.items():
            assert len(v) == len(dependencies)
            R[k] = np.mean(v)
        print(R)
        return R

    examples = [BLOCK[2], BEAM[2], LPANEL[2]]
    number_of_modes = [10, 6, 6]
    for kwargs, num_modes in zip(examples, number_of_modes):
        ex = ExampleConfig(**kwargs)
        name = ex.name.split("_")[0]
        for distr in DISTRIBUTIONS:
            deps = [ex.rom_log(distr, real) for real in range(ex.num_real)]

            yield {
                "name": ":".join([name, distr]),
                "file_dep": deps,
                "actions": [(get_rom_timings, [num_modes])],
                "clean": True,
                "verbosity": 2,
            }


# setup task for write macros
def task_get_basis_timings():
    """parse the log for basis construction runtimes"""

    def get_basis_timings(example, distr):
        """get the min, max, avg of rrf runtime etc
        over all oversampling problems
        In a second step, take the mean over all
        realizations
        """
        from src.postprocessing import parse_log
        from collections import defaultdict

        search = {
            "rrf_runtime": "Ran range finder algorithm in t=",
            "num_samples": "Number of training samples=",
            "compute_pod_modes_runtime": "Execution of compute_pod_modes took",
            "ext_time": "Extended pod modes in t=",
            "phi_time": "Computed coarse scale basis in",
            "extend_pod_modes_runtime": "Execution of extend_pod_modes took",
            }

        # results
        AVG = {}  # average over realizations
        mma = defaultdict(list)  # min, max, avg values
        r = defaultdict(list)  # local values

        for real in range(example.num_real):

            for ci in range(example.num_cells):
                log = example.empirical_basis_log(distr, real, ci)
                data = parse_log(search, log)
                for k, v in data.items():
                    # v is a list, with len(v) > 0 if there are
                    # logs from multiple runs in the file
                    # always take the first run
                    r[k].append(v[0])

            # 1. take min, max, avg over cells
            mma["rrf_min"].append(np.amin(r["rrf_runtime"]))
            mma["rrf_max"].append(np.amax(r["rrf_runtime"]))
            mma["rrf_mean"].append(np.mean(r["rrf_runtime"]))
            mma["ext_min"].append(np.amin(r["ext_time"]))
            mma["ext_max"].append(np.amax(r["ext_time"]))
            mma["ext_mean"].append(np.mean(r["ext_time"]))

        # 2. take avg of each of the above over realizations
        for k, v in mma.items():
            AVG[k] = np.mean(v)
        return AVG

    examples = [BLOCK[2], BEAM[2], LPANEL[2]]
    for kwargs in examples:
        ex = ExampleConfig(**kwargs)
        name = ex.name.split("_")[0]
        for distr in DISTRIBUTIONS:
            deps = []
            for real in range(ex.num_real):
                for cell_index in range(ex.num_cells):
                    deps.append(ex.empirical_basis_log(distr, real, cell_index))

            yield {
                "name": ":".join([name, distr]),
                "file_dep": deps,
                "actions": [(get_basis_timings, [ex, distr])],
                "clean": True,
                "verbosity": 2,
            }


def task_write_macros():
    """write macros for paper.tex"""

    def write_macros(example_name, distr, values, dependencies, targets):
        import string
        name = example_name.upper()

        if distr == "normal":
            dist = distr.upper()
        elif distr == "multivariate_normal":
            dist = "MVN"
        elif distr is None:
            dist = ""
        else:
            raise NotImplementedError

        macros = {}
        for k, v in values.items():
            variable = k.replace("_", "").upper()
            value = round(v, 2)
            macros[name+dist+variable] = value

        with open(targets[0], "w") as out_file:
            with open(dependencies[0], "r") as in_file:
                raw = string.Template(in_file.read())
                out_file.write(raw.substitute(macros))

    examples = [BLOCK[2], BEAM[2], LPANEL[2]]
    for kwargs in examples:
        ex = ExampleConfig(**kwargs)
        name = ex.name.split("_")[0]

        # ### fom macros
        yield {
                "name": ":".join([name, "fom"]),
                "file_dep": [SRC / f"tex/{name}_fom_macros.tex.template"],
                "actions": [(write_macros, [name, None])],
                "getargs": {"values": (f"get_fom_timings:{name}", None)},
                "targets": [WORK/f"tex/{name}_fom_macros.tex"],
                }
        for distr in DISTRIBUTIONS:
            for log_type in ["basis", "rom"]:
                yield {
                    "name": ":".join([name, log_type, distr]),
                    "file_dep": [SRC / f"tex/{name}_{log_type}_{distr}_macros.tex.template"],
                    "actions": [(write_macros, [name, distr])],
                    "getargs": {"values": (f"get_{log_type}_timings:{name}:{distr}", None)},
                    "targets": [WORK / f"tex/{name}_{log_type}_{distr}_macros.tex"],
                }
