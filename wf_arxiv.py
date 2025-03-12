"""arxiv submission"""

import pathlib
from doit.tools import create_folder

DOIT_CONFIG = {
    "verbosity": 2,
    "action_string_formatting": "new",
    "backend": "dbm",
    "dep_file": ".arxiv.db"
}

# dir structure
ROOT = pathlib.Path(__file__).parent
SRC = ROOT / "src"
WORK = ROOT / "work"
ARXIV = ROOT / "arxiv"
if not ARXIV.exists():
    create_folder(ARXIV)


macros = [
        WORK / "tex/beam_basis_multivariate_normal_macros.tex",
        WORK / "tex/beam_basis_normal_macros.tex",
        WORK / "tex/beam_fom_macros.tex",
        WORK / "tex/beam_rom_multivariate_normal_macros.tex",
        WORK / "tex/beam_rom_normal_macros.tex",
        WORK / "tex/block_basis_multivariate_normal_macros.tex",
        WORK / "tex/block_basis_normal_macros.tex",
        WORK / "tex/block_fom_macros.tex",
        WORK / "tex/block_rom_multivariate_normal_macros.tex",
        WORK / "tex/block_rom_normal_macros.tex",
        WORK / "tex/lpanel_basis_multivariate_normal_macros.tex",
        WORK / "tex/lpanel_basis_normal_macros.tex",
        WORK / "tex/lpanel_fom_macros.tex",
        WORK / "tex/lpanel_rom_multivariate_normal_macros.tex",
        WORK / "tex/lpanel_rom_normal_macros.tex",
        WORK / "tex/merged_macros.tex",
        ]


paper_deps = [
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
        WORK / "tex/block_normal_error_field.png",
        WORK / "tex/block_multivariate_normal_error_field.png",
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
        WORK / "tex/merged_macros.tex",
        WORK / "tex/pod_modes_hbeam_1_hierarchical_x.pdf",
        WORK / "tex/pod_modes_beam_1_multivariate_normal_x.pdf",
        WORK / "tex/pod_modes_beam_1_multivariate_normal_y.pdf",
        WORK / "tex/pod_modes_beam_3_multivariate_normal_x.pdf",
        WORK / "tex/pod_modes_beam_3_multivariate_normal_y.pdf",
        WORK / "tex/NJDnatbib.sty",
        WORK / "tex/oversampling_domain.pdf",
        WORK / "tex/references.bib",
        WORK / "tex/WileyNJD-AMA.bst",
        WORK / "tex/WileyNJD-v2.cls",
        WORK / "tex/paper.tex",
        ]


def task_copy_paper_sources():
    """copy sources for main document"""
    for fd in paper_deps:
        if fd.stem == "paper":
            continue
        yield {
                "name": fd.stem,
                "actions": ["cp {dependencies} {targets}"],
                "file_dep": [fd],
                "targets": [ARXIV / fd.name],
                "clean": True,
                }


def task_paper_pdfoutput():
    """add \pdfoutput=1 to preample"""

    def add_stuff(dependencies, targets):

        with open(targets[0], "w") as out_fh:
            with open(dependencies[0], "r") as in_fh:
                for k, line in enumerate(in_fh.readlines()):
                    out_fh.write(line)
                    if k == 0:
                        out_fh.write(r"\pdfoutput=1")

    return {
            "actions": [add_stuff],
            "file_dep": [WORK / "tex/paper.tex"],
            "targets": [ARXIV / "paper.tex"],
            }


def task_bbl():
    return {
            "actions": ["tectonic -k {dependencies}"],
            "file_dep": [ARXIV / "paper.tex"],
            "targets": [ARXIV / "paper.bbl"],
            }


def make_arxiv_zip(targets):
    from zipfile import ZipFile

    ancillary = []
    ancillary.append(ROOT / "README.md")
    ancillary.append(ROOT / "LICENSE")
    ancillary.append(ROOT / "envs/paraview_v5.11.0.yaml")
    ancillary.append(ROOT / "envs/tex.yaml")

    # add everything under source
    for root, subdirs, files in pathlib.os.walk(SRC):
        if any([p.startswith((".", "_", "data", "test", "multi.egg-info")) for p in root.split("/")]):
            continue
        else:
            for f in files:
                if not f.startswith((".", "response", "study")):
                    fpath = pathlib.os.path.join(root, f)
                    ancillary.append(fpath)

    for f in ROOT.glob("wf*.py"):
        ancillary.append(f)

    # ### multicode
    multi_files = []
    MULTI = ROOT.parent / "multicode/multi"
    multi_src = MULTI / "src/multi"

    multi_files.append(MULTI / "LICENSE")
    multi_files.append(MULTI / "README.md")
    multi_files.append(MULTI / "pyproject.toml")
    for f in multi_src.glob("*.*"):
        multi_files.append(f)


    with ZipFile(targets[0], "w") as archive:

        for f in ancillary:
            path = pathlib.Path(f)
            rr = path.relative_to(ROOT).as_posix()
            archive.write(filename=rr, arcname="anc/" + rr)

        for suffix in ["tex", "pdf", "png", "bbl", "bst", "cls"]:
            for f in ARXIV.glob(f"*.{suffix}"):
                path = pathlib.Path(f)
                if not path.name == "paper.pdf":
                    rr = path.relative_to(ROOT).as_posix()
                    ra = path.relative_to(ARXIV).as_posix()
                    archive.write(filename=rr, arcname=ra)

        for f in multi_files:
            path = pathlib.Path(f)
            rr = path.relative_to(MULTI).as_posix()
            archive.write(filename=path.as_posix(), arcname="anc/multicode/" + rr)


def task_make_zip():
    return {
            "file_dep": [ARXIV / "paper.tex", ARXIV / "paper.bbl"],
            "actions": [make_arxiv_zip],
            "targets": [ROOT / "arxiv.zip"],
            "clean": True,
            }
