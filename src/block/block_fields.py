import sys
import argparse
import dolfinx
from mpi4py import MPI
import numpy as np

from multi.problems import LinearElasticityProblem
from multi.io import select_modes, BasesLoader
from multi.dofmap import DofMap

from pymor.core.logger import getLogger
from pymor.tools.timing import Timer

from src.definitions import ExampleConfig
from src.block.block_problem import BlockProblem


def everywhere(x):
    return np.full(x[0].shape, True, dtype=bool)


def block_fields(
    distribution,
    real,
    num_modes,
    loglvl="INFO",
    logfile=None,
    **kwargs,
):
    """write local fields (ufom, urom, uerr)"""

    logger = getLogger("src.block.fields", level=loglvl, filename=logfile)
    timer = Timer("fields")

    example_config = ExampleConfig(**kwargs)

    block_problem = BlockProblem(example_config.coarse_grid, example_config.fine_grid)
    block_problem.degree = example_config.degree
    block_problem.material = example_config.material
    block_problem.setup_coarse_grid()
    block_problem.setup_coarse_space()
    block_problem.setup_fine_grid()
    block_problem.setup_fine_space()

    coarse_grid = block_problem.coarse_grid
    fine_grid = block_problem.fine_grid

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

    # ### solve FOM problem
    fom_problem = LinearElasticityProblem(
        fine_grid, V, E=E, NU=NU, plane_stress=plane_stress
    )
    fom_domain_facets = dolfinx.mesh.locate_entities_boundary(
        fine_grid.grid, 1, everywhere
    )
    fom_problem.add_dirichlet_bc(
        u_D, boundary=fom_domain_facets, method="topological", entity_dim=1
    )

    # FOM Assembly
    petsc_options = {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}
    fom_problem.setup_solver(petsc_options=petsc_options)

    timer.start()
    fom_bcs = fom_problem.get_dirichlet_bcs()
    fom_problem.assemble_matrix(fom_bcs)
    fom_problem.assemble_vector(fom_bcs)
    timer.stop()
    N_delta = V.dofmap.bs * V.dofmap.index_map.size_global
    logger.info(f"FOM system size N={N_delta}.")
    logger.info(f"Assembled FOM in t={timer.dt}s.")

    # FOM solve
    u_fom = dolfinx.fem.Function(V)
    u_fom.name = "u_fom"
    fom_solver = fom_problem.solver
    timer.start()
    fom_solver.solve(fom_problem.b, u_fom.vector)
    timer.stop()
    u_fom.x.scatter_forward()

    logger.info(f"Computed FOM solution in t={timer.dt}s.")

    fom_xdmf = example_config.fom_solution(distribution, real)
    with dolfinx.io.XDMFFile(fine_grid.grid.comm, fom_xdmf.as_posix(), "w") as out:
        out.write_mesh(fine_grid.grid)
        out.write_function(u_fom)

    bases_dir = example_config.basis(distribution, real, 0).parent
    loader = BasesLoader(bases_dir, coarse_grid.num_cells)
    bases, max_modes = loader.read_bases()

    local_max = np.amax(max_modes, axis=1)
    NUM_MAX_MODES = np.amax(local_max)
    logger.info(f"Global max number of modes per edge {NUM_MAX_MODES=}.")

    dofmap = DofMap(coarse_grid)

    rom_solution_npz = example_config.rom_solution(distribution, real)
    rom_solution_data = np.load(rom_solution_npz, allow_pickle=True)
    U_ROM = rom_solution_data["U"]
    NUM_MODES = rom_solution_data["num_modes"]
    assert num_modes in NUM_MODES

    # postprocessing
    for cell_index in range(coarse_grid.num_cells):
        logger.debug(f"{cell_index=}")
        # read subdomain grid
        subdomain_xdmf = example_config.subdomain_grid(cell_index)
        with dolfinx.io.XDMFFile(
            MPI.COMM_SELF, subdomain_xdmf.as_posix(), "r"
        ) as xdmf:
            subdomain = xdmf.read_mesh(name="Grid")

        V_i = dolfinx.fem.VectorFunctionSpace(
            subdomain, (fe_family, example_config.degree)
        )

        u_fom_local = dolfinx.fem.Function(V_i)
        u_fom_local.name = "u_fom_local"
        u_fom_local.interpolate(u_fom)

        u_rom_local = dolfinx.fem.Function(V_i)
        u_rom_local.name = "u_rom_local"

        err_local = dolfinx.fem.Function(V_i)
        err_local.name = "u_err"
        err_vec = err_local.vector

        vtkfilename = example_config.fields_subdomain(distribution, real, cell_index)
        vtkfile = dolfinx.io.VTKFile(subdomain.comm, vtkfilename.as_posix(), "w")

        dofs_per_edge = max_modes.copy()
        dofs_per_edge[max_modes > num_modes] = num_modes
        dofmap.distribute_dofs(2, dofs_per_edge, 0)

        basis = select_modes(bases[cell_index], max_modes[cell_index], dofs_per_edge[cell_index])

        u_rom_vec = u_rom_local.vector
        dofs = dofmap.cell_dofs(cell_index)
        Urom = U_ROM[num_modes]
        u_rom_vec.array[:] = basis.T @ Urom[dofs]

        with err_vec.localForm() as e_loc:
            e_loc.set(0)
        err_vec.axpy(1.0, u_fom_local.vector)
        err_vec.axpy(-1.0, u_rom_vec)

        vtkfile.write_function([u_fom_local, u_rom_local, err_local], float(num_modes))
        vtkfile.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # common for all examples
    parser.add_argument("distribution", type=str, help="The distribution used for sampling.")
    parser.add_argument("real", type=int, help="The i-th realization.")
    parser.add_argument("num_modes", type=int, help="The number of fine scale edge modes.")

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

    block_fields(**args.__dict__)
