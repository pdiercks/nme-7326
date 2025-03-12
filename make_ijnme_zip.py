"""create a zip file for ijnme submission"""


from zipfile import ZipFile

prefix = "./work/tex/"


style_files = [
        "WileyNJD-AMA.bst",
        "WileyNJD-v2.cls",
        "NJDnatbib.sty",
        ]
latex_files = [
        "paper.tex",
        "references.bib",
        "merged_macros.tex",
        ]
figures = [
        "beam_coarse_grid.pdf",
        "beam_config_examples.pdf",
        "beam_error_plot.pdf",
        "beam_sketch.pdf",
        "block_1_multivariate_normal_max_modes.png",
        "block_1_normal_max_modes.png",
        "block_2_multivariate_normal_max_modes.png",
        "block_2_normal_max_modes.png",
        "block_3_multivariate_normal_max_modes.png",
        "block_3_normal_max_modes.png",
        "block_error_plot.pdf",
        "block_normal_error_field.png",
        "block_multivariate_normal_error_field.png",
        "block_sketch.pdf",
        "block_xerr_2y.pdf",
        "block_xttol_2y.pdf",
        "converged_rce_01.png",
        "converged_rce_02.png",
        "convergence_plot.pdf",
        "domain_partitions.pdf",
        "lpanel_configurations.pdf",
        "lpanel_error_plot.pdf",
        "lpanel_multivariate_normal_error_field.png",
        "lpanel_normal_error_field.png",
        "lpanel_sketch.pdf",
        "lpanel_specimen_with_rce.pdf",
        "lpanel_xerr_2y.pdf",
        "lpanel_xttol_2y.pdf",
        "pod_modes_hbeam_1_hierarchical_x.pdf",
        "pod_modes_beam_1_multivariate_normal_x.pdf",
        "pod_modes_beam_1_multivariate_normal_y.pdf",
        "pod_modes_beam_3_multivariate_normal_x.pdf",
        "pod_modes_beam_3_multivariate_normal_y.pdf",
        "oversampling_domain.pdf",
        ]

with ZipFile("./ijnme.zip", "w") as archive:

    all_files = style_files + latex_files + figures
    for f in all_files:
        archive.write(filename=prefix+f)
