"""
Plot mesh and annotate edges with max number of modes

Usage:
    plot_num_edge_modes.py [options] GRID NPZ

Arguments:
    GRID           The mesh file.
    NPZ            The npz file containing max num modes.

Options:
    -h, --help         Show this message and exit.
    --transparent      Use a transparent background.
    --png=FILE         Write result to PNG.
"""

import sys
import pathlib
import numpy as np
import pyvista
from docopt import docopt

from dolfinx.io import gmshio, XDMFFile
from dolfinx import plot
from mpi4py import MPI

from multi.domain import StructuredQuadGrid


def parse_args(args):
    args = docopt(__doc__, args)
    args["GRID"] = pathlib.Path(args["GRID"])
    args["NPZ"] = pathlib.Path(args["NPZ"])
    args["--png"] = pathlib.Path(args["--png"]) if args["--png"] else None
    return args


def main(args):
    args = parse_args(args)
    meshfile = args["GRID"].as_posix()

    if args["GRID"].suffix == ".xdmf":
        with XDMFFile(MPI.COMM_SELF, meshfile, "r") as xdmf:
            domain = xdmf.read_mesh(name="Grid")
            celltags = xdmf.read_meshtags(domain, name="Grid")
    elif args["GRID"].suffix == ".msh":
        domain, celltags, _ = gmshio.read_from_msh(meshfile, MPI.COMM_SELF, gdim=2)
    else:
        raise NotImplementedError


    tdim = domain.topology.dim
    topology, cell_types, geometry = plot.create_vtk_mesh(domain, tdim)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

    cgrid = StructuredQuadGrid(domain)
    domain.topology.create_connectivity(1, 1)
    num_edges = len(domain.topology.connectivity(1, 1))
    num_cells = cgrid.num_cells
    edge_coord = cgrid.get_entity_coordinates(1, np.arange(num_edges, dtype=np.int32))

    data = np.load(args["NPZ"].as_posix())
    max_modes = data["max_modes"]
    num_edge_modes = {}
    for cell in range(num_cells):
        edge_tags = cgrid.get_entities(1, cell)
        for loc_edge, gl_edge in enumerate(edge_tags):
            if gl_edge not in num_edge_modes.keys():
                local_num_modes = max_modes[cell]
                num_edge_modes[gl_edge] = local_num_modes[loc_edge]

    edges = np.array(list(num_edge_modes.keys()))
    values = np.array(list(num_edge_modes.values()))
    idx = np.argsort(edges)

    pyvista.start_xvfb()
    plotter = pyvista.Plotter(off_screen=True)
    plotter.add_mesh(grid, show_edges=True)
    plotter.add_point_labels(edge_coord, values[idx])
    plotter.view_xy()
    plotter.screenshot(args["--png"].as_posix(), transparent_background=args["--transparent"])


if __name__ == "__main__":
    main(sys.argv[1:])
