import sys
import pathlib

from doit import get_var
from doit.tools import create_folder

from src.definitions import ExampleConfig

PYTHON = sys.executable

DOIT_CONFIG = {
    "verbosity": 2,
    "action_string_formatting": "new",
    "backend": "dbm",
    "dep_file": ".pv.db"
}

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

# mode in which to run the workflow
# use DEBUG or PAPER
CLI = {"mode": get_var("mode", "DEBUG"), "nreal": get_var("nreal", "1")}
NREAL = int(CLI["nreal"])

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

BLOCK = []
LPANEL = []
target_tol = [1e-1, 1e-2, 1e-3]

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


def task_plot_block_error_field():
    """plot the error field"""
    real = 0
    kwargs = BLOCK[2]
    block = ExampleConfig(**kwargs)
    script = SRC / "block/trace_err_warp_by_vector.py"
    for distr in DISTRIBUTIONS:
        pvdfile = block.fields_subdomain(distr, real, 0).parent / "fields.pvd"
        pngfile_untrimmed = WORK / f"tex/block_{distr}_error_field_untrimmed.png"
        pngfile = WORK / f"tex/block_{distr}_error_field.png"
        yield {
                "name": distr,
                "file_dep": [pvdfile],
                "actions": [f"pvbatch {script} {{dependencies}} {pngfile_untrimmed}",
                            "convert -trim {targets}"],
                "targets": [pngfile_untrimmed, pngfile],
                "clean": True,
                }


def task_plot_lpanel_error_field():
    """plot the error field"""
    real = 4
    kwargs = LPANEL[2]
    lpanel = ExampleConfig(**kwargs)
    script = SRC / "lpanel/trace_err_warp_by_vector.py"
    for distr in DISTRIBUTIONS:
        pvdfile = lpanel.fields_subdomain(distr, real, 0).parent / "fields.pvd"
        pngfile_untrimmed = WORK / f"tex/lpanel_{distr}_error_field_untrimmed.png"
        pngfile = WORK / f"tex/lpanel_{distr}_error_field.png"
        yield {
                "name": distr, 
                "file_dep": [pvdfile],
                "actions": [f"pvbatch {script} {{dependencies}} {pngfile_untrimmed}",
                            "convert -trim {targets}"],
                "targets": [pngfile_untrimmed, pngfile],
                "clean": True,
                }
