import sys
import pathlib

from doit import get_var
from doit.action import CmdAction
from doit.tools import create_folder

from src.definitions import ExampleConfig

PYTHON = sys.executable

DOIT_CONFIG = {
    "verbosity": 2,
    "action_string_formatting": "new",
    "backend": "dbm",
    "dep_file": ".lpanel.db"
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
LPANEL = []
target_tol = [1e-1, 1e-2, 1e-3]

if CLI["mode"] == "DEBUG":
    LOGLVL = "DEBUG"
    for k, ttol in enumerate(target_tol):
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


def task_coarse_rom_solution():
    """compute the coarse rom solution"""

    examples = LPANEL
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


def task_pod_basis():
    """compute pod basis for each edge
    for `normal` or `multivariate_normal` distribution"""

    examples = LPANEL
    for kwargs in examples:
        example = ExampleConfig(**kwargs)
        deps = [example.coarse_grid, example.fine_grid, example.material]

        for distr in DISTRIBUTIONS:
            if distr == "multivariate_normal":
                coarse_rom = example.coarse_rom_solution
                deps.append(coarse_rom)
            real = example.num_real
            # for real in range(example.num_real):
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

    examples = LPANEL

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


def task_lpanel_rom():
    """compute rom solution"""

    product = ERRNORM
    for kwargs in LPANEL:
        problem = ExampleConfig(**kwargs)
        for distr in DISTRIBUTIONS:
            real = problem.num_real
            # for real in range(problem.num_real):
            # dependencies
            # should be enough to only add the bases
            deps = [
                problem.basis(distr, real, ci) for ci in range(problem.num_cells)
            ]

            # targets
            rom = problem.rom_solution(distr, real)
            logfile = problem.rom_log(distr, real)

            yield {
                "name": problem.name + ":" + distr + ":" + str(real),
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


def task_lpanel_error():
    """compute the fom and rom solution and the error"""

    product = ERRNORM
    for kwargs in LPANEL:
        problem = ExampleConfig(**kwargs)
        for distr in DISTRIBUTIONS:
            real = problem.num_real
            # for real in range(problem.num_real):
            # dependencies
            deps = [
                problem.basis(distr, real, ci) for ci in range(problem.num_cells)
            ]
            rom = problem.rom_solution(distr, real)
            deps.append(rom)

            # targets
            fom = problem.fom_solution(distr, real)
            err = problem.error(distr, real)
            logfile = problem.error_log(distr, real)

            yield {
                "name": problem.name + ":" + distr + ":" + str(real),
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


def task_lpanel_fields():
    """write solution fields ufom, urom and err"""

    num_modes = 12
    real = 4

    kwargs = LPANEL[2]
    lpanel = ExampleConfig(**kwargs)
    for distr in DISTRIBUTIONS:
        # deps
        bases = [lpanel.basis(distr, real, ci) for ci in range(lpanel.num_cells)]
        rom = lpanel.rom_solution(distr, real)

        # targets
        fields = [lpanel.fields_subdomain(distr, real, ci) for ci in range(lpanel.num_cells)]
        pvd = fields[0].parent / "fields.pvd"

        logfile = None

        yield {
            "name": ":".join([lpanel.name, distr, str(real)]),
            "file_dep": bases + [rom],
            "actions": [
                (create_folder, [rom.parent]),
                CmdAction(
                    (
                        create_action_fields,
                        [distr, real, num_modes, LOGLVL, logfile],
                        kwargs,
                    )
                ),
                CmdAction((create_action_merge_pvd, [fields, pvd], {})),
            ],
            "targets": fields + [pvd],
            "clean": True,
        }
