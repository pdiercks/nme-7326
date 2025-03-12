import sys
import argparse
import pathlib
import dolfinx
from mpi4py import MPI
from petsc4py import PETSc
import numpy as np

from multi.boundary import point_at
from multi.domain import RectangularDomain
from multi.problems import LinearElasticityProblem
from multi.dofmap import DofMap
from multi.bcs import apply_bcs, BoundaryDataFactory
from multi.shapes import NumpyQuad

from pymor.algorithms.projection import project
from pymor.bindings.fenicsx import FenicsxVectorSpace, FenicsxMatrixOperator
from pymor.operators.constructions import VectorOperator
from pymor.core.logger import getLogger
from pymor.tools.timing import Timer

from src.definitions import ExampleConfig
from src.beam.beam_problem import BeamProblem


def compute_coarse_rom_solution(
    output=None, loglvl="INFO", logfile=None, **kwargs
):
    """compute rom solution using coarse scale basis"""
    logger = getLogger(
        "src.coarse_rom.compute_coarse_rom_solution", level=loglvl, filename=logfile
    )
    timer = Timer("coarse_rom")

    example_config = ExampleConfig(**kwargs)

    beam_problem = BeamProblem(example_config.coarse_grid, example_config.fine_grid)
    beam_problem.degree = example_config.degree
    beam_problem.material = example_config.material
    beam_problem.setup_coarse_grid()
    beam_problem.setup_coarse_space()

    coarse_grid = beam_problem.coarse_grid

    mat = beam_problem.material
    E = mat["Material parameters"]["E"]["value"]
    NU = mat["Material parameters"]["NU"]["value"]
    plane_stress = mat["Constraints"]["plane_stress"]

    W = beam_problem.W  # coarse space
    fe_family = W.element.basix_element.family

    dofmap = DofMap(coarse_grid)
    dofmap.distribute_dofs(2, 0, 0)

    N = dofmap.num_dofs
    K = np.zeros((N, N))
    f_ext = np.zeros(N)
    rom_bcs = {}

    x_indices = dolfinx.cpp.mesh.entities_to_geometry(
        coarse_grid.grid, 2, coarse_grid.cells, False
    )
    x_points = coarse_grid.grid.geometry.x

    timer.start()
    for cell_index in range(coarse_grid.num_cells):
        # read subdomain grid
        subdomain_xdmf = example_config.subdomain_grid(cell_index)
        subdomain_facets_xdmf = example_config.subdomain_facets(cell_index)

        with dolfinx.io.XDMFFile(MPI.COMM_SELF, subdomain_xdmf.as_posix(), "r") as xdmf:
            subdomain = xdmf.read_mesh(name="Grid")
            subdomain_ct = xdmf.read_meshtags(subdomain, name="Grid")
        subdomain.topology.create_connectivity(
            subdomain.topology.dim, subdomain.topology.dim - 1
        )
        with dolfinx.io.XDMFFile(
            MPI.COMM_SELF, subdomain_facets_xdmf.as_posix(), "r"
        ) as xdmf:
            subdomain_ft = xdmf.read_meshtags(subdomain, name="Grid")

        V_i = dolfinx.fem.VectorFunctionSpace(
            subdomain, (fe_family, example_config.degree)
        )

        source = FenicsxVectorSpace(V_i)

        omega = RectangularDomain(subdomain, subdomain_ct, subdomain_ft)
        subdomain_problem = LinearElasticityProblem(
            omega, V_i, E=E, NU=NU, plane_stress=plane_stress
        )

        # ### setup solver and compute coarse scale basis
        a = dolfinx.fem.form(subdomain_problem.form_lhs)
        A = dolfinx.fem.petsc.create_matrix(a)

        L = dolfinx.fem.form(subdomain_problem.form_rhs)
        rhs = dolfinx.fem.petsc.create_vector(L)

        solver = PETSc.KSP().create(subdomain.comm)
        solver.setType("preonly")
        solver.getPC().setType("lu")
        solver.getPC().setFactorSolverType("mumps")
        solver.setOperators(A)

        data_factory = BoundaryDataFactory(subdomain, V_i)
        g = dolfinx.fem.Function(V_i)
        g.x.set(0.0)
        bc_zero = data_factory.create_bc(g)

        A.zeroEntries()
        dolfinx.fem.petsc.assemble_matrix(A, a, bcs=[bc_zero])
        A.assemble()

        cell_vertex_coords = x_points[x_indices[cell_index]]
        quadrilateral = NumpyQuad(cell_vertex_coords)
        shape_functions = quadrilateral.interpolate(V_i)

        phi = []
        φ = dolfinx.fem.Function(V_i)  # extended function psi
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

            solver.solve(rhs, φ.vector)
            φ.x.scatter_forward()
            phi.append(φ.vector.copy())

        timer.stop()
        logger.debug(f"Computed coarse scale basis in {timer.dt}s.")

        # ### Add neumann data
        # and assemble A and b for projection
        subdomain_problem.clear_bcs()
        traction_vector = dolfinx.fem.Function(V_i)
        rom_neumann_bc = beam_problem.get_neumann(cell_index)["online"]
        for neumann_bc in rom_neumann_bc:
            traction_vector.interpolate(neumann_bc["value"])
            neumann_bc.update({"value": traction_vector})
            subdomain_problem.add_neumann_bc(**neumann_bc)

        # ### Assemble full operators for projections
        A.zeroEntries()
        dolfinx.fem.petsc.assemble_matrix(A, a, bcs=[])
        A.assemble()
        Aop = FenicsxMatrixOperator(A, V_i, V_i)

        # recompile rhs because of neumann data
        L = dolfinx.fem.form(subdomain_problem.form_rhs)
        rhs = dolfinx.fem.petsc.create_vector(L)

        with rhs.localForm() as r_loc:
            r_loc.set(0.0)
        dolfinx.fem.petsc.assemble_vector(rhs, L)
        rhs_op = VectorOperator(source.make_array([rhs]))

        B = source.make_array(phi)
        a_local_op = project(Aop, B, B)
        b_local_op = project(rhs_op, B, None)
        a_local = a_local_op.matrix
        b_local = b_local_op.matrix

        dofs = dofmap.cell_dofs(cell_index)
        K[np.ix_(dofs, dofs)] += a_local
        f_ext[dofs] += b_local.flatten()

        _, on_left_boundary = beam_problem.boundaries["left"]
        left_boundary_vertices = coarse_grid.locate_entities_boundary(0, on_left_boundary)
        origin_vertex = coarse_grid.locate_entities_boundary(0, point_at([0.0, 0.0, 0.0]))

        # constrain x-component for each node
        for ent in left_boundary_vertices:
            dofs = dofmap.entity_dofs(0, ent)
            assert len(dofs) == 2
            rom_bcs.update({dofs[0]: 0.0})

        # constrain y-component for origin
        for ent in origin_vertex:
            dofs = dofmap.entity_dofs(0, ent)
            assert len(dofs) == 2
            rom_bcs.update({dofs[1]: 0.0})

    timer.stop()
    logger.info(f"Assembled ROM system of size {N=} in t={timer.dt}s.")

    apply_bcs(K, f_ext, list(rom_bcs.keys()), list(rom_bcs.values()))
    timer.start()
    Urom = np.linalg.solve(K, f_ext)
    timer.stop()
    logger.info(f"Computed ROM solution in t={timer.dt}s.")

    if output is not None:
        rom_sol = pathlib.Path(output)
        np.save(rom_sol.as_posix(), Urom)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
    compute_coarse_rom_solution(**args.__dict__)
