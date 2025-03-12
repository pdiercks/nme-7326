import sys
import pathlib
import numpy as np

from doit import get_var
from doit.tools import create_folder, config_changed

from src.definitions import ExampleConfig

PYTHON = sys.executable

DOIT_CONFIG = {
    "verbosity": 2,
    "action_string_formatting": "new",
    "backend": "dbm",
    "dep_file": ".preprocessing.db"
}

# mode in which to run the workflow
# use DEBUG or PAPER
CLI = {"mode": get_var("mode", "DEBUG"), "nreal": get_var("nreal", "1")}
NREAL = int(CLI["nreal"])

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
        # parse debug log and estimate needed FOM size based on offline runtime?
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


def clean_xdmf(targets):
    for xf in targets:
        xdmf = pathlib.Path(xf)
        h5 = xdmf.with_suffix(".h5")
        xdmf.unlink()
        h5.unlink()


def task_grid_rce_01():
    """generate grids in .msh format for RCE type 01"""

    def create_grid(num_cells, targets):
        from multi.preprocessing import create_rce_grid_01

        xmin = 0.0
        xmax = 1.0
        ymin = 0.0
        ymax = 1.0
        radius = 0.2

        for target in targets:
            out_file = pathlib.Path(target)
            create_rce_grid_01(
                xmin,
                xmax,
                ymin,
                ymax,
                z=0.0,
                radius=radius,
                num_cells=num_cells,
                out_file=out_file.as_posix(),
            )

    for ppe in PPE_RCE_01:
        n = int(ppe - 1)
        mesh = WORK / "convergence" / "rce_01" / f"rce_{ppe:02d}.msh"
        yield {
            "name": f"{ppe:02d}",
            "actions": [(create_folder, [mesh.parent]), (create_grid, [n])],
            "targets": [mesh],
            "clean": True,
            "uptodate": [True],
        }


def task_grid_rce_02():
    """generate grids in .msh format for RCE type 02"""

    def create_rce_grid(num_cells, targets):
        from multi.preprocessing import create_rce_grid_02

        create_rce_grid_02(
            0.0, 20.0, 0.0, 20.0, num_cells=num_cells, facets=True, out_file=targets[0]
        )

    for num_points in PPE_RCE_02:
        num_cells = int(num_points - 1)
        msh_file = WORK / f"convergence/rce_02/rce_{num_points:02d}.msh"
        yield {
            "name": f"{num_points:02d}",
            "actions": [
                (create_folder, [msh_file.parent]),
                (create_rce_grid, [num_cells]),
            ],
            "clean": True,
            "targets": [msh_file],
            "uptodate": [True],
        }


def get_rce_meshfiles(path, points_per_edge):
    """returns list of a rce meshfiles under path"""
    return [(path / f"rce_{ppe:02d}.msh") for ppe in points_per_edge]


def task_convergence_analysis():
    """run mesh convergence analysis for both rce types"""
    script = SRC / "convergence" / "convergence.py"
    paths = [WORK / "convergence/rce_01", WORK / "convergence/rce_02"]
    points = [PPE_RCE_01, PPE_RCE_02]
    results = [
        WORK / "convergence/convergence_01.txt",
        WORK / "convergence/convergence_02.txt",
    ]
    for path, ppe, target in zip(paths, points, results):
        meshes = get_rce_meshfiles(path, ppe)
        cmd = f"{PYTHON} {script} {path} {DEGREE} {MATERIAL} --output={target} --product={{product}} -l {{loglevel}}"
        yield {
            "name": target.stem.split("_")[-1],
            "actions": [cmd],
            "targets": [target],
            "file_dep": [script, MATERIAL] + meshes,
            "clean": True,
            "params": [
                {
                    "name": "product",
                    "default": "energy",
                    "short": "p",
                    "long": "product",
                },
                {"name": "loglevel", "short": "l", "long": "log", "default": 30},
            ],
            "verbosity": 2,
        }


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


def task_create_global_coarse_grid():
    """create global coarse grid (quadrilateral cells)"""

    def create_grid(example, unit_length, targets):
        from multi.preprocessing import create_rectangle_grid

        xmin = ymin = 0.0
        nx = example.nx
        ny = example.ny
        xmax = nx * unit_length
        ymax = ny * unit_length
        create_rectangle_grid(
            xmin,
            xmax,
            ymin,
            ymax,
            num_cells=(nx, ny),
            facets=True,
            recombine=True,
            out_file=targets[0],
        )

    examples = BLOCK + BEAM + HBEAM
    for kwargs in examples:
        example = ExampleConfig(**kwargs)
        grid = example.coarse_grid
        yield {
            "name": example.name,
            "actions": [(create_folder, [grid.parent]), (create_grid, [example])],
            "uptodate": [True, config_changed({"nx": example.nx, "ny": example.ny})],
            "getargs": {
                "unit_length": (
                    f"_get_rce_metadata:{example.rce_type:02}",
                    "unit_length",
                )
            },
            "targets": [grid],
            "clean": True,
        }


def task_create_global_coarse_grid_lpanel():
    """create global coarse grid for lpanel"""

    def create_grid(example, unit_length, targets):
        from src.lpanel.create_lpanel_grid import create_lpanel_grid

        xmin = ymin = 0.0
        n = example.nxy
        xmax = ymax = n * unit_length
        create_lpanel_grid(
            xmin,
            xmax,
            ymin,
            ymax,
            num_cells=n,
            recombine=True,
            facets=True,
            out_file=targets[0],
        )

    for kwargs in LPANEL:
        lpanel = ExampleConfig(**kwargs)
        coarse_grid = lpanel.coarse_grid
        yield {
            "name": lpanel.name,
            "actions": [(create_folder, [coarse_grid.parent]), (create_grid, [lpanel])],
            "getargs": {
                "unit_length": (
                    f"_get_rce_metadata:{lpanel.rce_type:02}",
                    "unit_length",
                )
            },
            "targets": [coarse_grid],
            "clean": True,
            "uptodate": [config_changed(lpanel.__dict__)],
        }


def task_create_global_fine_grid():
    """create the global fine scale grid"""

    def create_grid(num_cells_per_edge, method, dependencies, targets):
        from src.preprocessing import GridType, DomainType, create_grid

        create_grid(
            dependencies[0],
            GridType.FINE,
            DomainType.GLOBAL,
            targets[0],
            subdomain_grid_method=[method],
            num_cells=num_cells_per_edge,
        )

    examples = BLOCK + BEAM + LPANEL + HBEAM
    for kwargs in examples:
        example = ExampleConfig(**kwargs)
        coarse_grid = example.coarse_grid
        fine_grid = example.fine_grid
        yield {
            "name": example.name,
            "file_dep": [coarse_grid],
            "actions": [(create_grid, [])],
            "targets": [fine_grid],
            "getargs": {
                "num_cells_per_edge": (
                    f"_get_rce_metadata:{example.rce_type:02}",
                    "num_cells",
                ),
                "method": (f"_get_rce_metadata:{example.rce_type:02}", "filepath"),
            },
            "clean": [clean_xdmf],
        }


def task_create_coarse_patches():
    """create coarse scale patch grids"""

    def create_grid(cell_index, dependencies, targets):
        from src.preprocessing import GridType, DomainType, create_grid

        create_grid(
            dependencies[0],
            GridType.COARSE,
            DomainType.PATCH,
            targets[0],
            cell_index=cell_index,
        )

    examples = BLOCK + BEAM + LPANEL + HBEAM
    for kwargs in examples:
        example = ExampleConfig(**kwargs)
        coarse_grid = example.coarse_grid
        for ci in range(example.num_cells):
            patch = example.coarse_patch(ci)
            yield {
                "name": example.name + ":" + str(ci),
                "file_dep": [coarse_grid],
                "actions": [(create_folder, [patch.parent]), (create_grid, [ci])],
                "targets": [patch],
                "clean": [clean_xdmf],
            }


def task_create_fine_patches():
    """create fine scale patch grids"""

    def create_grid(cell_index, num_cells_per_edge, method, dependencies, targets):
        from src.preprocessing import GridType, DomainType, create_grid

        create_grid(
            dependencies[0],
            GridType.FINE,
            DomainType.PATCH,
            targets[0],
            cell_index=cell_index,
            subdomain_grid_method=[method],
            num_cells=num_cells_per_edge,
        )

    examples = BLOCK + BEAM + LPANEL + HBEAM
    for kwargs in examples:
        example = ExampleConfig(**kwargs)
        coarse_grid = example.coarse_grid
        for ci in range(example.num_cells):
            patch = example.fine_patch(ci)
            yield {
                "name": example.name + ":" + str(ci),
                "file_dep": [coarse_grid],
                "actions": [
                    (create_folder, [patch.parent]),
                    (create_grid, [ci, example.rce_type]),
                ],
                "getargs": {
                    "num_cells_per_edge": (
                        f"_get_rce_metadata:{example.rce_type:02}",
                        "num_cells",
                    ),
                    "method": (f"_get_rce_metadata:{example.rce_type:02}", "filepath"),
                },
                "targets": [patch],
                "clean": [clean_xdmf],
            }


def task_create_subdomain_grids():
    """create fine scale subdomain grids"""

    def create_grid(cell_index, num_cells_per_edge, method, dependencies, targets):
        from src.preprocessing import GridType, DomainType, create_grid

        create_grid(
            dependencies[0],
            GridType.FINE,
            DomainType.SUBDOMAIN,
            targets[0],
            cell_index=cell_index,
            subdomain_grid_method=[method],
            num_cells=num_cells_per_edge,
        )

    examples = BLOCK + BEAM + LPANEL + HBEAM

    for kwargs in examples:
        example = ExampleConfig(**kwargs)
        coarse_grid = example.coarse_grid
        for ci in range(example.num_cells):
            subdomain = example.subdomain_grid(ci)
            yield {
                "name": example.name + ":" + str(ci),
                "file_dep": [coarse_grid],
                "actions": [
                    (create_folder, [subdomain.parent]),
                    (create_grid, [ci, example.rce_type]),
                ],
                "getargs": {
                    "num_cells_per_edge": (
                        f"_get_rce_metadata:{example.rce_type:02}",
                        "num_cells",
                    ),
                    "method": (f"_get_rce_metadata:{example.rce_type:02}", "filepath"),
                },
                "targets": [subdomain],
                "clean": [clean_xdmf],
            }


def task_preprocessing():
    """group of all preprocessing tasks"""

    return {
        "actions": None,
        "task_dep": [
            "create_global_coarse_grid",
            "create_global_coarse_grid_lpanel",
            "create_global_fine_grid",
            "create_coarse_patches",
            "create_fine_patches",
            "create_subdomain_grids",
        ],
    }
