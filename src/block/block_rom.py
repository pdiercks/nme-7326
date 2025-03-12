import sys
import argparse
import dolfinx
from mpi4py import MPI
import numpy as np

from multi.domain import Domain
from multi.problems import LinearElasticityProblem
from multi.io import select_modes, BasesLoader
from multi.dofmap import DofMap
from multi.interpolation import interpolate
from multi.bcs import apply_bcs

from pymor.bindings.fenicsx import FenicsxVectorSpace, FenicsxMatrixOperator
from pymor.core.logger import getLogger
from pymor.tools.timing import Timer

from src.definitions import ExampleConfig
from src.block.block_problem import BlockProblem


def everywhere(x):
    return np.full(x[0].shape, True, dtype=bool)


def block_rom(
    distribution,
    real,
    product="l2",
    loglvl="INFO",
    logfile=None,
    **kwargs,
):
    """compute the rom solution for the block example"""

    logger = getLogger("src.block.rom", level=loglvl, filename=logfile)
    timer = Timer("rom")

    example_config = ExampleConfig(**kwargs)

    block_problem = BlockProblem(example_config.coarse_grid, example_config.fine_grid)
    block_problem.degree = example_config.degree
    block_problem.material = example_config.material
    block_problem.setup_coarse_grid()
    block_problem.setup_fine_grid()
    block_problem.setup_coarse_space()
    block_problem.setup_fine_space()

    coarse_grid = block_problem.coarse_grid

    mat = block_problem.material
    E = mat["Material parameters"]["E"]["value"]
    NU = mat["Material parameters"]["NU"]["value"]
    plane_stress = mat["Constraints"]["plane_stress"]

    W = block_problem.W  # coarse space
    V = block_problem.V  # fine space
    fe_family = W.element.basix_element.family

    # Dirichlet data
    u_D = dolfinx.fem.Function(V)
    u_D_expr = block_problem.get_boundary_data_expression()
    u_D.interpolate(u_D_expr)

    bases_dir = example_config.basis(distribution, real, 0).parent
    loader = BasesLoader(bases_dir, coarse_grid.num_cells)
    bases, max_modes = loader.read_bases()

    local_max = np.amax(max_modes, axis=1)
    NUM_MAX_MODES = np.amax(local_max)
    logger.info(f"Global max number of modes per edge {NUM_MAX_MODES=}.")

    dofmap = DofMap(coarse_grid)

    # ### data to define bcs
    boundary_vertices = coarse_grid.locate_entities_boundary(0, everywhere)
    x_verts = coarse_grid.get_entity_coordinates(0, boundary_vertices)
    u_coarse_values = interpolate(u_D, x_verts)

    # return values
    U_ROM = []
    NUM_MODES = []
    NUM_DOFS = []

    cell_index = 0
    subdomain_xdmf = example_config.subdomain_grid(cell_index)
    with dolfinx.io.XDMFFile(
        MPI.COMM_SELF, subdomain_xdmf.as_posix(), "r"
    ) as xdmf:
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

    a = dolfinx.fem.form(subdomain_problem.form_lhs)
    A = dolfinx.fem.petsc.create_matrix(a)

    A.zeroEntries()
    dolfinx.fem.petsc.assemble_matrix(A, a, bcs=[])
    A.assemble()
    Aop = FenicsxMatrixOperator(A, V_i, V_i)

    for num_modes in range(NUM_MAX_MODES + 1):
        logger.info(f"Computing ROM solution for {num_modes=}")

        dofs_per_edge = max_modes.copy()
        dofs_per_edge[max_modes > num_modes] = num_modes
        dofmap.distribute_dofs(2, dofs_per_edge, 0)

        N = dofmap.num_dofs
        K = np.zeros((N, N))
        f_ext = np.zeros(N)
        rom_bcs = {}

        # ### local contributions
        loc_A = []

        for cell_index in range(coarse_grid.num_cells):
            local_basis = select_modes(bases[cell_index], max_modes[cell_index],
                                       dofs_per_edge[cell_index])
            B = source.from_numpy(local_basis)
            a_local_op = Aop.apply2(B, B)
            loc_A.append(a_local_op)

        timer.start()
        for cell_index in range(coarse_grid.num_cells):

            lhs = loc_A[cell_index]
            dofs = dofmap.cell_dofs(cell_index)
            K[np.ix_(dofs, dofs)] += lhs

            for values, ent in zip(u_coarse_values, boundary_vertices):
                dofs = dofmap.entity_dofs(0, ent)
                assert len(values) == len(dofs)
                for k, v in zip(dofs, values):
                    rom_bcs.update({k: v})

            boundary_edges = coarse_grid.locate_entities_boundary(1, everywhere)
            for ent in boundary_edges:
                dofs = dofmap.entity_dofs(1, ent)
                assert len(dofs) <= 1
                for k in dofs:
                    rom_bcs.update({k: 1.0})

        timer.stop()
        logger.info(f"ROM system size {N=}.")
        logger.info(f"Assembled ROM in t={timer.dt}s.")

        apply_bcs(K, f_ext, list(rom_bcs.keys()), list(rom_bcs.values()))
        timer.start()
        Urom = np.linalg.solve(K, f_ext)
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

    block_rom(**args.__dict__)
