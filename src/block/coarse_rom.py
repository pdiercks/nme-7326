import sys
import argparse
import pathlib
import dolfinx
from mpi4py import MPI
from petsc4py import PETSc
import numpy as np

from multi.domain import Domain
from multi.problems import LinearElasticityProblem
from multi.dofmap import DofMap
from multi.interpolation import interpolate
from multi.bcs import apply_bcs, BoundaryDataFactory
from multi.shapes import NumpyQuad

from pymor.bindings.fenicsx import FenicsxVectorSpace, FenicsxMatrixOperator
from pymor.core.logger import getLogger
from pymor.tools.timing import Timer

from src.definitions import ExampleConfig
from src.block.block_problem import BlockProblem


def everywhere(x):
    return np.full(x[0].shape, True, dtype=bool)


def compute_coarse_rom_solution(
    output=None, loglvl="INFO", logfile=None, **kwargs
):
    """compute rom solution using coarse scale basis"""
    logger = getLogger(
        "src.coarse_rom.compute_coarse_rom_solution", level=loglvl, filename=logfile
    )
    timer = Timer("coarse_rom")

    example_config = ExampleConfig(**kwargs)

    block_problem = BlockProblem(example_config.coarse_grid, example_config.fine_grid)
    block_problem.degree = example_config.degree
    block_problem.material = example_config.material
    block_problem.setup_coarse_grid()
    block_problem.setup_coarse_space()

    coarse_grid = block_problem.coarse_grid

    mat = block_problem.material
    E = mat["Material parameters"]["E"]["value"]
    NU = mat["Material parameters"]["NU"]["value"]
    plane_stress = mat["Constraints"]["plane_stress"]

    W = block_problem.W  # coarse space
    fe_family = W.element.basix_element.family

    # Dirichlet data
    u_D = dolfinx.fem.Function(W)
    u_D_expr = block_problem.get_boundary_data_expression()
    u_D.interpolate(u_D_expr)

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
        with dolfinx.io.XDMFFile(MPI.COMM_SELF, subdomain_xdmf.as_posix(), "r") as xdmf:
            subdomain = xdmf.read_mesh(name="Grid")
            ct = xdmf.read_meshtags(subdomain, name="Grid")

        V_i = dolfinx.fem.VectorFunctionSpace(
            subdomain, (fe_family, example_config.degree)
        )

        source = FenicsxVectorSpace(V_i)

        omega = Domain(subdomain, ct)
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

        # ### define boundary condition
        data_factory = BoundaryDataFactory(subdomain, V_i)
        g = dolfinx.fem.Function(V_i)  # re-use g to define extension
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
        logger.info(f"Computed coarse scale basis in {timer.dt}s.")


        # ### Assemble full operator to compute local rom stiffness matrix
        A.zeroEntries()
        dolfinx.fem.petsc.assemble_matrix(A, a, bcs=[])
        A.assemble()
        Aop = FenicsxMatrixOperator(A, V_i, V_i)

        B = source.make_array(phi)
        a_local = Aop.apply2(B, B)

        dofs = dofmap.cell_dofs(cell_index)
        K[np.ix_(dofs, dofs)] += a_local

        boundary_vertices = coarse_grid.locate_entities_boundary(0, everywhere)
        x_verts = coarse_grid.get_entity_coordinates(0, boundary_vertices)
        coarse_values = interpolate(u_D, x_verts)

        for values, ent in zip(coarse_values, boundary_vertices):
            dofs = dofmap.entity_dofs(0, ent)
            assert len(values) == len(dofs)
            for k, v in zip(dofs, values):
                rom_bcs.update({k: v})

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
