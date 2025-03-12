import sys
import argparse
import dolfinx
from mpi4py import MPI
import numpy as np

from multi.boundary import point_at
from multi.domain import RectangularDomain
from multi.problems import LinearElasticityProblem
from multi.io import select_modes, BasesLoader
from multi.dofmap import DofMap
from multi.bcs import apply_bcs

from scipy.sparse import csc_array
from scipy.sparse.linalg import spsolve

from pymor.algorithms.projection import project
from pymor.bindings.fenicsx import FenicsxVectorSpace, FenicsxMatrixOperator
from pymor.operators.constructions import VectorOperator
from pymor.core.logger import getLogger
from pymor.tools.timing import Timer

from src.definitions import ExampleConfig
from src.lpanel.lpanel_problem import LpanelProblem


def lpanel_rom(
    distribution,
    real,
    product="l2",
    loglvl="INFO",
    logfile=None,
    **kwargs,
):
    """compute rom solution for lpanel example"""

    logger = getLogger(
        "src.lpanel_example.rom", level=loglvl, filename=logfile
    )
    timer = Timer("rom")

    example_config = ExampleConfig(**kwargs)

    lpanel_problem = LpanelProblem(example_config.coarse_grid, example_config.fine_grid)
    lpanel_problem.degree = example_config.degree
    lpanel_problem.material = example_config.material
    lpanel_problem.setup_coarse_grid()
    lpanel_problem.setup_coarse_space()

    coarse_grid = lpanel_problem.coarse_grid

    # ### points of the coarse grid
    x_points = coarse_grid.grid.geometry.x

    mat = lpanel_problem.material
    E = mat["Material parameters"]["E"]["value"]
    NU = mat["Material parameters"]["NU"]["value"]
    plane_stress = mat["Constraints"]["plane_stress"]

    W = lpanel_problem.W  # coarse space
    fe_family = W.element.basix_element.family

    bases_dir = example_config.basis(distribution, real, 0).parent
    loader = BasesLoader(bases_dir, coarse_grid.num_cells)
    bases, max_modes = loader.read_bases()

    local_max = np.amax(max_modes, axis=1)
    NUM_MAX_MODES = np.amax(local_max)
    logger.info(f"Global max number of modes per edge {NUM_MAX_MODES=}.")

    dofmap = DofMap(coarse_grid)

    # return values
    U_ROM = []
    NUM_MODES = []
    NUM_DOFS = []

    for num_modes in range(NUM_MAX_MODES + 1):
        logger.info(f"Computing ROM solution for {num_modes=}")

        dofs_per_edge = max_modes.copy()
        dofs_per_edge[max_modes > num_modes] = num_modes
        dofmap.distribute_dofs(2, dofs_per_edge, 0)

        N = dofmap.num_dofs
        K = np.zeros((N, N))
        f_ext = np.zeros(N)
        rom_bcs = {}

        _, on_bottom_boundary = lpanel_problem.boundaries["bottom"]
        bottom_boundary_vertices = coarse_grid.locate_entities_boundary(
            0, on_bottom_boundary
        )
        xmin = np.amin(x_points, axis=0)
        origin_vertex = coarse_grid.locate_entities_boundary(
            0, point_at(xmin)
        )

        # ### assemble high fidelity operators once
        cell_index = 0
        subdomain_xdmf = example_config.subdomain_grid(cell_index)
        subdomain_facets_xdmf = example_config.subdomain_facets(cell_index)

        with dolfinx.io.XDMFFile(
            MPI.COMM_SELF, subdomain_xdmf.as_posix(), "r"
        ) as xdmf:
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
        subdomain_problem.clear_bcs()
        subdomain_problem.setup_solver()
        subdomain_problem.assemble_matrix()
        subdomain_problem.assemble_vector()

        Aop = FenicsxMatrixOperator(subdomain_problem.A, V_i, V_i)
        rhs_op = VectorOperator(source.make_array([subdomain_problem.b]))

        # ### local contributions
        loc_A = []
        loc_b = []

        for cell_index in range(coarse_grid.num_cells):
            local_basis = select_modes(bases[cell_index], max_modes[cell_index],
                                       dofs_per_edge[cell_index])
            B = source.from_numpy(local_basis)
            a_local_op = project(Aop, B, B)
            b_local_op = project(rhs_op, B, None)
            a_local = a_local_op.matrix
            b_local = b_local_op.matrix
            loc_A.append(a_local)
            loc_b.append(b_local)


        timer.start()
        for cell_index in range(coarse_grid.num_cells):

            # local contributions
            lhs = loc_A[cell_index]
            rhs = loc_b[cell_index]

            rom_neumann_bc = lpanel_problem.get_neumann(cell_index)["online"]
            if len(rom_neumann_bc) > 0:
                subdomain_xdmf = example_config.subdomain_grid(cell_index)
                subdomain_facets_xdmf = example_config.subdomain_facets(cell_index)

                with dolfinx.io.XDMFFile(
                    MPI.COMM_SELF, subdomain_xdmf.as_posix(), "r"
                ) as xdmf:
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
                subdomain_problem.clear_bcs()

                traction_vector = dolfinx.fem.Function(V_i)
                for neumann_bc in rom_neumann_bc:
                    traction_vector.interpolate(neumann_bc["value"])
                    neumann_bc.update({"value": traction_vector})
                    subdomain_problem.add_neumann_bc(**neumann_bc)

                subdomain_problem.setup_solver()
                subdomain_problem.assemble_vector()
                rhs_op = VectorOperator(source.make_array([subdomain_problem.b]))
                local_basis = select_modes(bases[cell_index], max_modes[cell_index],
                                           dofs_per_edge[cell_index])
                B = source.from_numpy(local_basis)
                b_local_op = project(rhs_op, B, None)
                rhs = b_local_op.matrix

            dofs = dofmap.cell_dofs(cell_index)
            K[np.ix_(dofs, dofs)] += lhs
            f_ext[dofs] += rhs.flatten()

            # ### macroscopic bcs
            # constrain y-component for each node on bottom boundary
            for ent in bottom_boundary_vertices:
                dofs = dofmap.entity_dofs(0, ent)
                assert len(dofs) == 2
                rom_bcs.update({dofs[1]: 0.0})

            # ### macroscopic bcs
            # constrain x-component for origin
            for ent in origin_vertex:
                dofs = dofmap.entity_dofs(0, ent)
                assert len(dofs) == 2
                rom_bcs.update({dofs[0]: 0.0})

        timer.stop()
        logger.info(f"ROM system size {N=}.")
        logger.info(f"Assembled ROM in t={timer.dt}s.")

        apply_bcs(K, f_ext, list(rom_bcs.keys()), list(rom_bcs.values()))
        timer.start()
        Urom = spsolve(csc_array(K), f_ext)
        timer.stop()
        logger.info(f"Computed ROM solution in t={timer.dt}s.")

        U_ROM.append(Urom)
        NUM_MODES.append(num_modes)
        NUM_DOFS.append(N)

    rom_sol = example_config.rom_solution(distribution, real)
    # NOTE requires np.load(..., allow_pickle=True)
    np.savez(
        rom_sol.as_posix(),
        U=np.array(U_ROM, dtype=object),
        num_modes=NUM_MODES,
        max_modes=max_modes,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # common for all examples
    parser.add_argument("distribution", type=str, help="The distribution used for sampling.")
    parser.add_argument("real", type=int, help="The i-th realization.")
    parser.add_argument("--product", type=str, help="The inner product to use.")

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
    parser.add_argument("--loglvl", type=str, help="The log level.")
    parser.add_argument("--logfile", type=str, help="The filename to write log to.")

    args = parser.parse_args(sys.argv[1:])

    lpanel_rom(**args.__dict__)
