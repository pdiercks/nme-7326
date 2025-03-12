"""offline phase"""

import sys
import argparse
import re
import dolfinx
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from pymor.bindings.fenicsx import FenicsxVectorSpace
from pymor.core.logger import getLogger
from pymor.tools.timing import Timer

from multi.bcs import BoundaryDataFactory
from multi.shapes import NumpyQuad
from multi.dofmap import QuadrilateralDofLayout
from multi.domain import RectangularDomain
from multi.problems import LinearElasticityProblem

from src.definitions import ExampleConfig


@Timer("extend_edge_modes", getLogger("src.empirical_basis.extend_edge_modes", level="INFO"))
def extend_edge_modes(
    cell_index,
    distribution,
    real,
    output=None,
    loglvl="INFO",
    logfile=None,
    **kwargs,
):
    """extend pod edge modes into the subdomains containing the edge

    Parameters
    ----------
    cell_index : int
        The index of the subdomain (coarse grid cell).
    distribution : str
        The distribution that was used to compute the POD basis.
    real : int
        The integer specifying the realization.
    output : optional, str
        The FilePath to write the extended basis.
    loglvl : optional
        The logging level.
    logfile : optional
        The logfile.

    """

    logger = getLogger(
        "src.empirical_basis.extend_pod_modes", level=loglvl, filename=logfile
    )
    timer = Timer("empirical_basis")

    example_config = ExampleConfig(**kwargs)

    # ### Instantiate multiscale problem
    if re.search("block", example_config.name):
        from src.block.block_problem import BlockProblem as ExampleProblem

    elif re.search("beam", example_config.name):
        from src.beam.beam_problem import BeamProblem as ExampleProblem

    elif re.search("lpanel", example_config.name):
        from src.lpanel.lpanel_problem import LpanelProblem as ExampleProblem


    multiscale_problem = ExampleProblem(
        example_config.coarse_grid, example_config.fine_grid
    )
    multiscale_problem.material = example_config.material
    multiscale_problem.degree = example_config.degree
    multiscale_problem.setup_coarse_grid()
    multiscale_problem.setup_coarse_space()

    coarse_grid = multiscale_problem.coarse_grid
    edge_map = multiscale_problem.edge_map
    dof_layout = QuadrilateralDofLayout()
    W_global = multiscale_problem.W
    fe_family = W_global.element.basix_element.family

    material = multiscale_problem.material
    E = material["Material parameters"]["E"]["value"]
    NU = material["Material parameters"]["NU"]["value"]
    plane_stress = material["Constraints"]["plane_stress"]


    subdomain_xdmf = example_config.subdomain_grid(cell_index)
    with dolfinx.io.XDMFFile(MPI.COMM_SELF, subdomain_xdmf.as_posix(), "r") as xdmf:
        subdomain = xdmf.read_mesh(name="Grid")
        cell_tags = xdmf.read_meshtags(subdomain, name="Grid")

    # Might re-evaluate concept for
    # RectangularDomain.create_edge_grids
    # this actually takes quite a bit and might be
    # done more efficiently once in the pre-processing phase

    Ω = RectangularDomain(subdomain, cell_tags)
    Ω.create_edge_grids()

    V = dolfinx.fem.VectorFunctionSpace(subdomain, (fe_family, example_config.degree))
    source = FenicsxVectorSpace(V)

    problem = LinearElasticityProblem(Ω, V, E=E, NU=NU, plane_stress=plane_stress)
    problem.setup_edge_spaces()
    problem.create_map_from_V_to_L()

    # ### create matrix and vector
    a = dolfinx.fem.form(problem.form_lhs)
    A = dolfinx.fem.petsc.create_matrix(a)

    L = dolfinx.fem.form(problem.form_rhs)
    rhs = dolfinx.fem.petsc.create_vector(L)

    # ### setup the PETSc solver
    solver = PETSc.KSP().create(subdomain.comm)
    solver.setOperators(A)

    petsc_options = {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}

    # Not using global PETSc.Options helps?
    solver.setType(petsc_options["ksp_type"])
    solver.getPC().setType(petsc_options["pc_type"])
    solver.getPC().setFactorSolverType(petsc_options["pc_factor_mat_solver_type"])

    data_factory = BoundaryDataFactory(subdomain, V)

    # dummy bc to apply to A
    g = dolfinx.fem.Function(V)  # re-use g to define extension
    g.x.set(0.0)
    bc_zero = data_factory.create_bc(g)

    # ### Assemble the matrix
    A.zeroEntries()
    dolfinx.fem.petsc.assemble_matrix(A, a, bcs=[bc_zero])
    A.assemble()

    # ### Start loop over edges of the cell
    psi = {}  # store extended functions
    ψ = dolfinx.fem.Function(V)  # extended function psi
    timer.start()

    cell_edges = coarse_grid.get_entities(1, cell_index)
    for local_ent, edge in enumerate(cell_edges):

        # ### get pod basis functions
        # modes on global edge `edge` 
        # are stored in file ci (int) under loc_e (str)
        (ci, loc_e) = edge_map[edge]
        npzfile = example_config.pod_bases(distribution, real, ci)
        pod_modes = np.load(npzfile)[loc_e]
        if cell_index == ci:
            # read from ci and do loc_e of not current cell
            boundary = loc_e
        else:
            # read from ci and do local_ent of current cell_index
            boundary = dof_layout.local_edge_index_map[local_ent]

        edge_dofs = problem.V_to_L[boundary]

        extensions = []
        # ### extend each pod mode into the subdomain
        for mode in pod_modes:

            # update boundary data
            g.vector.zeroEntries()
            g.vector.setValues(edge_dofs, mode, addv=PETSc.InsertMode.INSERT)
            g.vector.assemblyBegin()
            g.vector.assemblyEnd()

            bc = data_factory.create_bc(g)
            bcs = [bc]

            # ### Assemble rhs
            with rhs.localForm() as r_loc:
                r_loc.set(0.0)
            dolfinx.fem.petsc.assemble_vector(rhs, L)
            dolfinx.fem.petsc.apply_lifting(rhs, [a], bcs=[bcs])
            rhs.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            dolfinx.fem.petsc.set_bc(rhs, bcs)

            solver.solve(rhs, ψ.vector)
            ψ.x.scatter_forward()

            extensions.append(ψ.vector.copy())
        psi[boundary] = source.make_array(extensions).to_numpy()

    timer.stop()
    logger.info(f"Extended pod modes in t={timer.dt}s.")

    # ### Compute coarse scale basis
    x_indices = dolfinx.cpp.mesh.entities_to_geometry(coarse_grid.grid, 2, coarse_grid.cells, False)
    x_points = coarse_grid.grid.geometry.x
    cell_vertex_coords = x_points[x_indices[cell_index]]

    quadrilateral = NumpyQuad(cell_vertex_coords)
    shape_functions = quadrilateral.interpolate(V)

    phi = []
    timer.start()
    for shape in shape_functions:
        g.x.array[:] = shape

        bc = data_factory.create_bc(g)
        bcs = [bc]

        # ### Assemble rhs
        with rhs.localForm() as r_loc:
            r_loc.set(0.0)
        dolfinx.fem.petsc.assemble_vector(rhs, L)
        dolfinx.fem.petsc.apply_lifting(rhs, [a], bcs=[bcs])
        rhs.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        dolfinx.fem.petsc.set_bc(rhs, bcs)

        solver.solve(rhs, ψ.vector)
        ψ.x.scatter_forward()
        phi.append(ψ.vector.copy())

    timer.stop()
    logger.info(f"Computed coarse scale basis in {timer.dt}s.")
    φ = source.make_array(phi).to_numpy()

    if output is not None:
        logger.info(f"Saving basis to file {output} ...")
        np.savez(output, phi=φ, **psi)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # common for all examples
    parser.add_argument("cell_index", type=int, help="The cell index.")
    parser.add_argument("distribution", type=str, help="The distribution used for sampling.")
    parser.add_argument("real", type=int, help="The i-th realization.")

    # optional, to be passed to ExampleConfig
    parser.add_argument("--degree", type=int)
    parser.add_argument("--material", type=str)
    parser.add_argument("--name", type=str)
    parser.add_argument("--num_real", type=int)
    parser.add_argument("--num_testvecs", type=int)
    parser.add_argument("--nx", type=int)
    parser.add_argument("--nxy", type=int)
    parser.add_argument("--ny", type=int)
    parser.add_argument("--range_product", type=str)
    parser.add_argument("--rce_type", type=int)
    parser.add_argument("--source_product", type=str)
    parser.add_argument("--ttol", type=float)

    # optional
    parser.add_argument("--output", type=str)
    parser.add_argument("--loglvl", type=str, help="The log level.")
    parser.add_argument("--logfile", type=str, help="The filename to write log to.")

    args = parser.parse_args(sys.argv[1:])

    extend_edge_modes(**args.__dict__)
