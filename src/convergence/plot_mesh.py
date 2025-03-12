"""
plot the given mesh

Usage:
    plot_mesh.py [options] GRID

Arguments:
    GRID         The mesh data to be plotted.

Options:
    -h, --help         Show this message and exit.
    --subdomains       Plot subdomains as well.
    --transparent      Use a transparent background.
    --colormap=CMAP    Choose the colormap [default: viridis].
    --png=FILE         Write result to PNG.
"""

import sys
from docopt import docopt
from pathlib import Path

from dolfinx import plot
from dolfinx.io import XDMFFile, gmshio
from mpi4py import MPI

import pyvista
from matplotlib.colors import ListedColormap



def parse_arguments(args):
    args = docopt(__doc__, args)
    args["GRID"] = Path(args["GRID"])
    args["--png"] = Path(args["--png"]) if args["--png"] else None
    args["--colormap"] = str(args["--colormap"])
    supported_cmaps = ("viridis", "bam-RdBu", "RdYlBu")
    if args["--colormap"] not in supported_cmaps:
        print(
            f"Colormap not in supported colormaps {supported_cmaps}. Using defaut value 'viridis'."
        )
        args["--colormap"] = "viridis"
    return args


def main(args):
    args = parse_arguments(args)
    meshfile = args["GRID"].as_posix()

    if args["GRID"].suffix == ".xdmf":
        with XDMFFile(MPI.COMM_SELF, meshfile, "r") as xdmf:
            domain = xdmf.read_mesh(name="Grid")
            celltags = xdmf.read_meshtags(domain, name="Grid")
    elif args["GRID"].suffix == ".msh":
        domain, celltags, _ = gmshio.read_from_msh(meshfile, MPI.COMM_SELF, gdim=2)
    else:
        raise NotImplementedError

    if args["--colormap"] == "bam-RdBu":
        from multi.postprocessing import read_bam_colormap
        bam_RdBu = read_bam_colormap()
        cmap = ListedColormap(bam_RdBu, name="bam-RdBu")
    elif args["--colormap"] == "RdYlBu":
        cmap = "RdYlBu"
    else:
        cmap = "viridis"

    pyvista.global_theme.font.size = 48
    pyvista.start_xvfb()

    tdim = domain.topology.dim
    topology, cell_types, geometry = plot.create_vtk_mesh(domain, tdim)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

    if args["--subdomains"]:
        grid.cell_data["Marker"] = celltags.values
        grid.set_active_scalars("Marker")

    plotter = pyvista.Plotter(off_screen=True)
    plotter.add_mesh(grid, show_edges=True, show_scalar_bar=False, cmap=cmap)
    plotter.show_bounds(
            color="black",
            font_size=24,
            font_family="times",
            xlabel="x",
            ylabel="y",
            use_2d=False,
            bold=True,
            )

    plotter.view_xy()
    plotter.screenshot(
            args["--png"].as_posix(),
            transparent_background=args["--transparent"]
            )


if __name__ == "__main__":
    main(sys.argv[1:])
