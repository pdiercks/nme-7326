import sys
import pathlib
import numpy as np

from doit import get_var
from doit.action import CmdAction
from doit.tools import create_folder

from src.definitions import ExampleConfig

PYTHON = sys.executable

DOIT_CONFIG = {
    "verbosity": 2,
    "action_string_formatting": "new",
    "backend": "dbm",
    "dep_file": ".beam.db"
}

# mode in which to run the workflow
# use DEBUG or PAPER
CLI = {"mode": get_var("mode", "DEBUG"), "nreal": get_var("nreal", "1")}
NREAL = int(CLI["nreal"])

assert NREAL <= 20

# dir structure
ROOT = pathlib.Path(__file__).parent
SRC = ROOT / "src"
WORK = ROOT / "work"
if not WORK.exists():
    create_folder(WORK)
    create_folder(WORK / "tex")
    """tex folder to build paper

    write all plots/figures to this path
    parse log and write macros.tex
    copy paper.tex from source directory
    finally latexmk -pdf paper.tex
    """

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

# default configurations based on run mode
BEAM = []
HBEAM = []
target_tol = [1e-1, 1e-2, 1e-3]

if CLI["mode"] == "DEBUG":
    LOGLVL = "DEBUG"
    for k, ttol in enumerate(target_tol):
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

elif CLI["mode"] == "PAPER":
    LOGLVL = "DEBUG"
    for k, ttol in enumerate(target_tol):
        # parse debug log and estimate needed FOM size based on offline runtime?
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

else:
    raise KeyError(
        "The workflow is run in DEBUG mode (smaller problems) by default."
        "To run the real thing type `doit mode=PAPER`."
    )
print(f"Running in {CLI['mode']} mode")
print(f"Realization number: {CLI['nreal']}")


def to_cli(kwargs) -> list[str]:
    """convert dict to CLI options"""
    r = []
    for k, v in kwargs.items():
        r.append(f"--{k}")
        r.append(f"{v}")
    return r


def create_action_coarse_rom(output, loglvl, logfile, **kwargs) -> str:
    name = kwargs.get("name")
    if name.startswith("block"):
        script = SRC / "block/coarse_rom.py"
    elif name.startswith("beam"):
        script = SRC / "beam/coarse_rom.py"
    elif name.startswith("lpanel"):
        script = SRC / "lpanel/coarse_rom.py"

    action = [PYTHON, script.as_posix(), "--output", output.as_posix()]
    action += ["--loglvl", loglvl, "--logfile", logfile.as_posix()]
    action += to_cli(kwargs)
    return " ".join(action)


def create_action_pod_basis(
    cell_index, distribution, real, output, loglvl, logfile, **kwargs
) -> str:
    script = SRC / "compute_pod_basis.py"
    action = [PYTHON, script.as_posix(), str(cell_index), distribution, str(real)]
    action += ["--output", output.as_posix()]
    action += ["--loglvl", loglvl, "--logfile", logfile.as_posix()]
    action += to_cli(kwargs)
    return " ".join(action)


def create_action_extension(
    cell_index, distribution, real, output, loglvl, logfile, **kwargs
) -> str:
    script = (SRC / "extend_edge_modes.py").as_posix()
    action = [PYTHON, script, str(cell_index), distribution, str(real)]
    action += ["--output", output.as_posix()]
    action += ["--loglvl", loglvl, "--logfile", logfile.as_posix()]
    action += to_cli(kwargs)
    return " ".join(action)


def create_action_rom(distr, real, product, loglvl, logfile, **kwargs) -> str:
    name = kwargs.get("name")
    if name.startswith("block"):
        script = SRC / "block/block_rom.py"
    elif name.startswith("beam"):
        script = SRC / "beam/beam_rom.py"
    elif name.startswith("hbeam"):
        script = SRC / "beam/beam_rom.py"
    elif name.startswith("lpanel"):
        script = SRC / "lpanel/lpanel_rom.py"

    action = [PYTHON, script.as_posix(), distr, str(real), "--product", product]
    action += ["--loglvl", loglvl, "--logfile", logfile.as_posix()]
    action += to_cli(kwargs)
    return " ".join(action)


def create_action_rom_error(distr, real, product, loglvl, logfile, **kwargs) -> str:
    name = kwargs.get("name")
    if name.startswith("block"):
        script = SRC / "block/block_error.py"
    elif name.startswith("beam"):
        script = SRC / "beam/beam_error.py"
    elif name.startswith("hbeam"):
        script = SRC / "beam/beam_error.py"
    elif name.startswith("lpanel"):
        script = SRC / "lpanel/lpanel_error.py"

    action = [PYTHON, script.as_posix(), distr, str(real), "--product", product]
    action += ["--loglvl", loglvl, "--logfile", logfile.as_posix()]
    action += to_cli(kwargs)
    return " ".join(action)


def create_action_fields(distr, real, num_modes, loglvl, logfile, **kwargs) -> str:
    name = kwargs.get("name")
    if name.startswith("block"):
        script = SRC / "block/block_fields.py"
    elif name.startswith("beam"):
        script = SRC / "beam/beam_fields.py"
    elif name.startswith("hbeam"):
        script = SRC / "beam/beam_fields.py"
    elif name.startswith("lpanel"):
        script = SRC / "lpanel/lpanel_fields.py"

    action = [PYTHON, script.as_posix(), distr, str(real), str(num_modes)]
    try:
        action += ["--loglvl", loglvl, "--logfile", logfile.as_posix()]
    except AttributeError:
        action += ["--loglvl", loglvl]
    action += to_cli(kwargs)
    return " ".join(action)


def create_action_merge_pvd(pvdfiles, output) -> str:
    script = SRC / "merge_pvtu_datasets.py"
    action = [PYTHON, script.as_posix()]
    for f in pvdfiles:
        action.append(f.as_posix())
    action += ["--output", output.as_posix()]
    return " ".join(action)


def fix_deps(dependencies, fnames):
    """unfortunately the order of the `file_dep` is not guaranteed
    see https://github.com/pydoit/doit/issues/254
    Parameters
    ----------
    fnames : list of str
        The filenames in the correct order.
    """
    deps = [pathlib.Path(d).name for d in dependencies]
    order = [deps.index(fn) for fn in fnames]
    return [dependencies[i] for i in order]


def clean_xdmf(targets):
    for xf in targets:
        xdmf = pathlib.Path(xf)
        h5 = xdmf.with_suffix(".h5")
        xdmf.unlink()
        h5.unlink()


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


def task_coarse_rom_solution():
    """compute the coarse rom solution"""

    examples = BEAM
    for kwargs in examples:
        example = ExampleConfig(**kwargs)
        rom = example.coarse_rom_solution
        logfile = example.coarse_rom_log

        yield {
            "name": example.name,
            "file_dep": [example.coarse_grid, example.fine_grid]
            + [example.subdomain_grid(i) for i in range(example.num_cells)],
            "actions": [
                (create_folder, [rom.parent]),
                CmdAction((create_action_coarse_rom, [rom, LOGLVL, logfile], kwargs)),
            ],
            "targets": [rom, logfile],
            "clean": True,
        }


def task_hierarchical_basis():
    """compute hierarchical basis for each edge"""

    def construct_basis(example, max_degree, num_cells):
        from src.beam.hierarchical import compute_hierarchical_modes

        inputs = {
            "example": example,
            "num_cells": num_cells,
            "max_degree": max_degree,
            "orthonormalize": True,
            "product": example.range_product,
        }
        compute_hierarchical_modes(**inputs)

    distr = "hierarchical"
    real = 0
    degree = 7  # max degree for hierarchical basis
    typ = 2  # rce type
    for kwargs in HBEAM:
        example = ExampleConfig(**kwargs)
        targets = [
            example.pod_bases(distr, real, ci) for ci in range(example.num_cells)
        ]
        yield {
            "name": example.name,
            "actions": [
                (create_folder, [targets[0].parent]),
                (construct_basis, [example, degree]),
            ],
            "getargs": {"num_cells": (f"_get_rce_metadata:{typ:02d}", "num_cells")},
            "targets": targets,
            "clean": True,
        }


def task_pod_basis():
    """compute pod basis for each edge
    for `normal` or `multivariate_normal` distribution"""

    examples = BEAM
    for kwargs in examples:
        example = ExampleConfig(**kwargs)
        deps = [example.coarse_grid, example.fine_grid, example.material]

        for distr in DISTRIBUTIONS:
            if distr == "multivariate_normal":
                coarse_rom = example.coarse_rom_solution
                deps.append(coarse_rom)
            real = example.num_real
            for cell_index in range(example.num_cells):
                deps.append(example.coarse_patch(cell_index))
                deps.append(example.fine_patch(cell_index))
                deps.append(example.subdomain_grid(cell_index))

                pod_bases = example.pod_bases(distr, real, cell_index)
                logfile = example.empirical_basis_log(distr, real, cell_index)

                yield {
                    "name": example.name
                    + ":"
                    + distr
                    + ":"
                    + str(real)
                    + ":"
                    + str(cell_index),
                    "file_dep": deps,
                    "actions": [
                        (create_folder, [pod_bases.parent]),
                        (create_folder, [logfile.parent]),
                        CmdAction(
                            (
                                create_action_pod_basis,
                                [
                                    cell_index,
                                    distr,
                                    real,
                                    pod_bases,
                                    LOGLVL,
                                    logfile,
                                ],
                                kwargs,
                            )
                        ),
                    ],
                    "targets": [pod_bases, logfile],
                    "clean": True,
                }


def task_basis_extension():
    """extend pod edge basis into subdomains"""

    examples = BEAM

    for kwargs in examples:
        example = ExampleConfig(**kwargs)
        for distr in DISTRIBUTIONS:
            real = example.num_real
            # for real in range(example.num_real):
            for ci in range(example.num_cells):
                pod_bases = [
                    example.pod_bases(distr, real, i)
                    for i in range(example.num_cells)
                ]
                basis = example.basis(distr, real, ci)
                logfile = example.empirical_basis_log(distr, real, ci)
                yield {
                    "name": example.name
                    + ":"
                    + distr
                    + ":"
                    + str(real)
                    + ":"
                    + str(ci),
                    "file_dep": pod_bases,
                    "actions": [
                        CmdAction(
                            (
                                create_action_extension,
                                [ci, distr, real, basis, LOGLVL, logfile],
                                kwargs,
                            )
                        )
                    ],
                    "targets": [basis],
                    "clean": True,
                }


def task_basis_extension_hierarchical():
    """extend hierarchical basis into subdomains"""

    examples = HBEAM

    for kwargs in examples:
        example = ExampleConfig(**kwargs)
        distr = "hierarchical"
        real = 0
        for ci in range(example.num_cells):
            pod_bases = [
                example.pod_bases(distr, real, i) for i in range(example.num_cells)
            ]
            basis = example.basis(distr, real, ci)
            logfile = example.empirical_basis_log(distr, real, ci)
            yield {
                "name": example.name + ":" + str(ci),
                "file_dep": pod_bases,
                "actions": [
                    (create_folder, [basis.parent]),
                    (create_folder, [logfile.parent]),
                    CmdAction(
                        (
                            create_action_extension,
                            [ci, distr, real, basis, LOGLVL, logfile],
                            kwargs,
                        )
                    ),
                ],
                "targets": [basis],
                "clean": True,
            }


def task_beam_rom():
    """compute the rom solution for the beam"""

    product = ERRNORM
    for kwargs in BEAM:
        beam = ExampleConfig(**kwargs)
        for distr in DISTRIBUTIONS:
            # for real in range(beam.num_real):
            real = beam.num_real
            # dependencies
            # should be enough to only add the bases
            deps = [beam.basis(distr, real, ci) for ci in range(beam.num_cells)]

            # targets
            rom = beam.rom_solution(distr, real)
            logfile = beam.rom_log(distr, real)

            yield {
                "name": beam.name + ":" + distr + ":" + str(real),
                "file_dep": deps,
                "actions": [
                    (create_folder, [rom.parent]),
                    CmdAction(
                        (
                            create_action_rom,
                            [distr, real, product, LOGLVL, logfile],
                            kwargs,
                        )
                    )
                ],
                "targets": [rom, logfile],
                "clean": True,
            }


def task_beam_error():
    """compute the fom solution and rom error"""

    product = ERRNORM
    for kwargs in BEAM:
        beam = ExampleConfig(**kwargs)
        for distr in DISTRIBUTIONS:
            real = beam.num_real
            # for real in range(beam.num_real):
            # dependencies
            deps = [beam.basis(distr, real, ci) for ci in range(beam.num_cells)]
            rom = beam.rom_solution(distr, real)
            deps.append(rom)

            # targets
            fom = beam.fom_solution(distr, real)
            err = beam.error(distr, real)
            logfile = beam.error_log(distr, real)

            yield {
                "name": beam.name + ":" + distr + ":" + str(real),
                "file_dep": deps,
                "actions": [
                    (create_folder, [fom.parent]),
                    CmdAction(
                        (
                            create_action_rom_error,
                            [distr, real, product, LOGLVL, logfile],
                            kwargs,
                        )
                    ),
                ],
                "targets": [fom, err, logfile],
                "clean": True,
            }


def task_hbeam_rom():
    """compute the rom solution"""

    distr = "hierarchical"
    real = 0
    product = ERRNORM
    for kwargs in HBEAM:
        beam = ExampleConfig(**kwargs)
        # dependencies
        # should be enough to only add the bases
        deps = [beam.basis(distr, real, ci) for ci in range(beam.num_cells)]

        # targets
        rom = beam.rom_solution(distr, real)
        logfile = beam.rom_log(distr, real)

        yield {
            "name": beam.name,
            "file_dep": deps,
            "actions": [
                (create_folder, [rom.parent]),
                CmdAction(
                    (
                        create_action_rom,
                        [distr, real, product, LOGLVL, logfile],
                        kwargs,
                    )
                ),
            ],
            "targets": [rom, logfile],
            "clean": True,
        }


def task_hbeam_error():
    """compute the fom solution and rom error"""

    distr = "hierarchical"
    real = 0
    product = ERRNORM
    for kwargs in HBEAM:
        beam = ExampleConfig(**kwargs)
        # dependencies
        deps = [beam.basis(distr, real, ci) for ci in range(beam.num_cells)]
        rom = beam.rom_solution(distr, real)
        deps.append(rom)

        # targets
        fom = beam.fom_solution(distr, real)
        err = beam.error(distr, real)
        logfile = beam.error_log(distr, real)

        yield {
            "name": beam.name,
            "file_dep": deps,
            "actions": [
                (create_folder, [fom.parent]),
                CmdAction(
                    (
                        create_action_rom_error,
                        [distr, real, product, LOGLVL, logfile],
                        kwargs,
                    )
                ),
            ],
            "targets": [fom, err, logfile],
            "clean": True,
        }
