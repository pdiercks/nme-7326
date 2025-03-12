"""tex related tasks"""

import pathlib

DOIT_CONFIG = {
    "verbosity": 2,
    "action_string_formatting": "new",
    "backend": "dbm",
    "dep_file": ".tex.db"
}

# dir structure
ROOT = pathlib.Path(__file__).parent
SRC = ROOT / "src"
WORK = ROOT / "work"
if not WORK.exists():
    raise ValueError


def task_figures():
    """compile figures"""

    figures = SRC / "tex/figures"
    deps = figures.glob("*.tex")

    for d in deps:
        if d.stem == "bamcolors":
            continue
        pdf = WORK / "tex" / (d.stem + ".pdf")
        yield {
                "name": d.stem,
                "file_dep": [d],
                "actions": ["tectonic {dependencies} -o ./work/tex"],
                "targets": [pdf],
                "clean": True,
                }


def task_copy_paper_sources():
    """copy sources for main document and response to referees"""
    deps = [
            SRC / "tex/NJDnatbib.sty",
            SRC / "tex/WileyNJD-AMA.bst",
            SRC / "tex/WileyNJD-v2.cls",
            SRC / "tex/paper.tex",
            SRC / "tex/references.bib",
            ]
    for fd in deps:
        yield {
                "name": fd.stem,
                "actions": ["cp {dependencies} {targets}"],
                "file_dep": [fd],
                "targets": [WORK / "tex" / fd.name],
                "clean": True,
                }


def task_paper():
    """compile the paper"""

    paper = WORK / "tex/paper.tex"

    deps = [
            WORK / "tex/beam_basis_multivariate_normal_macros.tex",
            WORK / "tex/beam_basis_normal_macros.tex",
            WORK / "tex/beam_coarse_grid.pdf",
            WORK / "tex/beam_config_examples.pdf",
            WORK / "tex/beam_error_plot.pdf",
            WORK / "tex/beam_fom_macros.tex",
            WORK / "tex/beam_rom_multivariate_normal_macros.tex",
            WORK / "tex/beam_rom_normal_macros.tex",
            WORK / "tex/beam_sketch.pdf",
            WORK / "tex/block_1_multivariate_normal_max_modes.png",
            WORK / "tex/block_1_normal_max_modes.png",
            WORK / "tex/block_2_multivariate_normal_max_modes.png",
            WORK / "tex/block_2_normal_max_modes.png",
            WORK / "tex/block_3_multivariate_normal_max_modes.png",
            WORK / "tex/block_3_normal_max_modes.png",
            WORK / "tex/block_basis_multivariate_normal_macros.tex",
            WORK / "tex/block_basis_normal_macros.tex",
            WORK / "tex/block_error_plot.pdf",
            WORK / "tex/block_fom_macros.tex",
            WORK / "tex/block_rom_multivariate_normal_macros.tex",
            WORK / "tex/block_rom_normal_macros.tex",
            WORK / "tex/block_sketch.pdf",
            WORK / "tex/block_xerr_2y.pdf",
            WORK / "tex/block_xttol_2y.pdf",
            WORK / "tex/converged_rce_01.png",
            WORK / "tex/converged_rce_02.png",
            WORK / "tex/convergence_plot.pdf",
            WORK / "tex/domain_partitions.pdf",
            WORK / "tex/lpanel_basis_multivariate_normal_macros.tex",
            WORK / "tex/lpanel_basis_normal_macros.tex",
            WORK / "tex/lpanel_configurations.pdf",
            WORK / "tex/lpanel_error_plot.pdf",
            WORK / "tex/lpanel_fom_macros.tex",
            WORK / "tex/lpanel_multivariate_normal_error_field.png",
            WORK / "tex/lpanel_normal_error_field.png",
            WORK / "tex/lpanel_rom_multivariate_normal_macros.tex",
            WORK / "tex/lpanel_rom_normal_macros.tex",
            WORK / "tex/lpanel_sketch.pdf",
            WORK / "tex/lpanel_specimen_with_rce.pdf",
            WORK / "tex/lpanel_xerr_2y.pdf",
            WORK / "tex/lpanel_xttol_2y.pdf",
            WORK / "tex/NJDnatbib.sty",
            WORK / "tex/oversampling_domain.pdf",
            WORK / "tex/references.bib",
            WORK / "tex/WileyNJD-AMA.bst",
            WORK / "tex/WileyNJD-v2.cls",
            ]

    deps.append(paper)
    return {
            "file_dep": deps,
            "actions": [f"latexmk -cd -pdf {paper}"],
            "targets": [paper.with_suffix(".pdf")],
            "clean": True,
            }
