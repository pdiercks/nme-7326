"""offline phase"""

import re
import sys
import argparse
import pathlib
import dolfinx
import numpy as np
from dolfinx.io import XDMFFile
from mpi4py import MPI
from petsc4py import PETSc

from pymor.core.exceptions import ExtensionError
from pymor.bindings.fenicsx import FenicsxVectorSpace, FenicsxMatrixOperator
from pymor.core.logger import getLogger
from pymor.reductors.basic import extend_basis
from pymor.tools.random import get_random_state
from pymor.tools.timing import Timer

from scipy.spatial.distance import pdist, squareform

from multi.preprocessing import create_facet_tags
from multi.dofmap import QuadrilateralDofLayout
from multi.domain import RectangularDomain
from multi.misc import x_dofs_VectorFunctionSpace
from multi.problems import LinearElasticityProblem, TransferProblem
from multi.product import InnerProduct
from multi.projection import orthogonal_part, fine_scale_part
from multi.range_finder import adaptive_edge_rrf_normal, adaptive_edge_rrf_mvn
from multi.solver import build_nullspace

from src.definitions import ExampleConfig, get_random_seed


@Timer(
    "compute_pod_modes",
    getLogger("src.empirical_basis.compute_pod_modes", level="INFO"),
)
def compute_pod_modes(
    cell_index,
    distribution,
    real,
    output=None,
    loglvl="INFO",
    logfile=None,
    **kwargs,
):
    """compute pod modes on edges Γ_j ⊂ ∂Ω_i"""

    logger = getLogger(
        "src.empirical_basis.compute_pod_modes", level=loglvl, filename=logfile
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

    else:
        raise NotImplementedError

    multiscale_problem = ExampleProblem(
        example_config.coarse_grid, example_config.fine_grid
    )
    assert hasattr(multiscale_problem, "active_edges")

    # ### Common definitions of the problem
    multiscale_problem.material = example_config.material
    multiscale_problem.degree = example_config.degree
    multiscale_problem.setup_coarse_grid()
    multiscale_problem.setup_coarse_space()

    coarse_grid = multiscale_problem.coarse_grid
    material = multiscale_problem.material
    degree = multiscale_problem.degree

    W_global = multiscale_problem.W

    rom_solution = example_config.coarse_rom_solution
    if rom_solution.exists():
        # create global Function holding coarse ROM solution
        u_rom = np.load(rom_solution.as_posix())
        u_macro = dolfinx.fem.Function(W_global)
        try:
            u_macro.vector.array[:] = u_rom.flatten()
        except ValueError as err:
            raise err(
                "Shape of the coarse ROM solution and " "coarse FE space do not match."
            )

    if cell_index not in range(coarse_grid.num_cells):
        raise ValueError(f"The cell with {cell_index=} does not exist.")

    # ### read meshes
    coarse_patch_xdmf = example_config.coarse_patch(cell_index)
    fine_patch_xdmf = example_config.fine_patch(cell_index)
    subdomain_xdmf = example_config.subdomain_grid(cell_index)
    subdomain_facets_xdmf = example_config.subdomain_facets(cell_index)

    with XDMFFile(MPI.COMM_SELF, coarse_patch_xdmf.as_posix(), "r") as xdmf:
        coarse_patch = xdmf.read_mesh()
    with XDMFFile(MPI.COMM_SELF, fine_patch_xdmf.as_posix(), "r") as xdmf:
        fine_patch = xdmf.read_mesh(name="Grid")
        fine_patch_ct = xdmf.read_meshtags(fine_patch, name="Grid")
    with XDMFFile(MPI.COMM_SELF, subdomain_xdmf.as_posix(), "r") as xdmf:
        subdomain = xdmf.read_mesh(name="Grid")
        subdomain_ct = xdmf.read_meshtags(subdomain, name="Grid")
    subdomain.topology.create_connectivity(
        subdomain.topology.dim, subdomain.topology.dim - 1
    )
    with XDMFFile(MPI.COMM_SELF, subdomain_facets_xdmf.as_posix(), "r") as xdmf:
        subdomain_ft = xdmf.read_meshtags(subdomain, name="Grid")

    # ### domains
    # create facets for the oversampling domain if ∂Ω intersects
    # with the boundary of the global domain
    global_boundaries = multiscale_problem.boundaries
    fine_patch_ft, fine_patch_marked_boundary = create_facet_tags(
        fine_patch, global_boundaries
    )
    # oversampling domain
    Ω = RectangularDomain(
        fine_patch, cell_markers=fine_patch_ct, facet_markers=fine_patch_ft
    )
    Ω.set_coarse_grid(coarse_patch)

    # target subdomain
    Ω_i = RectangularDomain(
        subdomain,
        cell_markers=subdomain_ct,
        facet_markers=subdomain_ft,
        index=cell_index,
    )
    Ω_i.create_coarse_grid()
    Ω_i.create_edge_grids()

    # ### define Problems
    fe_family = W_global.element.basix_element.family
    V = dolfinx.fem.VectorFunctionSpace(fine_patch, (fe_family, degree))
    V_i = dolfinx.fem.VectorFunctionSpace(subdomain, (fe_family, degree))
    E = material["Material parameters"]["E"]["value"]
    NU = material["Material parameters"]["NU"]["value"]
    plane_stress = material["Constraints"]["plane_stress"]

    # oversampling problem
    oversampling_problem = LinearElasticityProblem(
        Ω, V, E=E, NU=NU, plane_stress=plane_stress
    )

    # subdomain problem
    subdomain_problem = LinearElasticityProblem(
        Ω_i, V_i, E=E, NU=NU, plane_stress=plane_stress
    )

    # setup spaces for later use
    subdomain_problem.setup_edge_spaces()
    subdomain_problem.setup_coarse_space()
    subdomain_problem.create_map_from_V_to_L()

    # problem and patch specific:
    gamma_out = multiscale_problem.get_gamma_out(cell_index)
    dirichlet = multiscale_problem.get_dirichlet(cell_index)
    neumann = multiscale_problem.get_neumann(cell_index)
    remove_kernel = multiscale_problem.get_remove_kernel(cell_index)

    logger.debug(f"{dirichlet=}")
    logger.debug(f"{neumann=}")

    source_product = example_config.source_product
    range_product = example_config.range_product

    logger.debug(f"{source_product=}")
    logger.debug(f"{range_product=}")
    if source_product == "energy":
        raise NotImplementedError
    if range_product == "energy":
        range_product_dict = {
            "product": subdomain_problem.form_lhs,
            "product_name": "energy",
        }
    else:
        range_product_dict = {"product": range_product, "product_name": None}
    source_product_dict = {"product": source_product}

    logger.info(f"Discretizing transfer problem for cell {cell_index:03}")
    transfer_problem = TransferProblem(
        oversampling_problem,
        subdomain_problem,
        gamma_out,
        dirichlet=dirichlet["homogeneous"],
        source_product=source_product_dict,
        range_product=range_product_dict,
        remove_kernel=remove_kernel,
    )
    logger.debug(
        "Transfer Problem: source space dim = "
        f"{transfer_problem.source_gamma_out.dim}"
    )
    logger.debug("Transfer Problem: range space dim = " f"{transfer_problem.range.dim}")
    logger.debug(f"{remove_kernel=}")

    logger.debug(f"{distribution=}")
    if distribution == "normal":
        # sample from gaussian normal distribution
        sampling_options = {"loc": 0.0, "scale": 1.0}

    elif distribution == "multivariate_normal":
        # do multivariate training around macro state

        # create coarse grid space
        W_local = dolfinx.fem.VectorFunctionSpace(coarse_patch, (fe_family, 1))
        w_local = dolfinx.fem.Function(W_local)

        # ### interpolate global solution onto oversampling domain
        w_local.interpolate(u_macro)

        # ### remove kernel from the training data (using pymor)
        # source space of transfer problem
        coarse_source = FenicsxVectorSpace(W_local)
        U = coarse_source.make_array([w_local.vector])
        # NOTE may also use `product="energy"` as a metric for
        # the training data
        inner_product = InnerProduct(W_local, product="l2")
        product_mat = inner_product.assemble_matrix()
        l2_product = FenicsxMatrixOperator(product_mat, W_local, W_local)
        kernel = build_nullspace(coarse_source, product=l2_product, gdim=2)
        U_proj = orthogonal_part(kernel, U, product=l2_product, orth=True)
        # update w_local
        w_local.vector.array[:] = U_proj.to_numpy().flatten()

        # ### create function for w_local in the fine grid space
        u_local = dolfinx.fem.Function(V)
        u_local.interpolate(w_local)
        # restrict function to values on Γ_out
        dofs_gamma_out = transfer_problem._bc_dofs_gamma_out
        train_data = u_local.vector.array[dofs_gamma_out]

        # ### covariance Σ
        xmin_cp = np.amin(coarse_patch.geometry.x, axis=0)
        xmax_cp = np.amax(coarse_patch.geometry.x, axis=0)
        correlation_length = np.linalg.norm(xmax_cp - xmin_cp)

        x_dofs = x_dofs_VectorFunctionSpace(V)
        points = x_dofs[dofs_gamma_out]
        distance = squareform(pdist(points, metric="euclidean"))

        sampling_options = {
            "distance": distance,
            "mean": train_data,
            "check_valid": "warn",
            "tol": 1e-8,
            "correlation_length": correlation_length,
        }
    else:
        raise NotImplementedError

    active_edges = multiscale_problem.get_active_edges(cell_index)
    seed = get_random_seed(example_config.name, real)
    random_state = get_random_state(seed=seed)
    dof_layout = QuadrilateralDofLayout()
    logger.debug(f"{active_edges=}")

    POD_BASES = {}

    if len(dirichlet["inhomogeneous"]) > 0:
        logger.info("Computing additional modes due to" " inhomogeneous Dirichlet bcs.")

        timer.start()
        for bc in dirichlet["inhomogeneous"]:

            sub = bc.get("sub")
            if sub is not None:
                raise NotImplementedError

            boundary_marker = bc["boundary"]
            for edge, edge_mesh in Ω_i.fine_edge_grid.items():
                facets = dolfinx.mesh.locate_entities(
                    edge_mesh, edge_mesh.topology.dim, boundary_marker
                )
                if facets.size > 0:
                    break
            assert edge in dof_layout.local_edge_index_map.keys()
            Lf = subdomain_problem.edge_spaces["fine"][edge]
            g = dolfinx.fem.Function(Lf)
            try:
                g.interpolate(bc["value"])
            except RuntimeError as err:
                raise err(
                    "Wrong value for inhomogeneous Dirichlet bc"
                    f" in {type(multiscale_problem)}"
                )
            Lm = subdomain_problem.edge_spaces["coarse"][edge]
            fine_scale_part(g, Lm, in_place=True)
            range_space = FenicsxVectorSpace(Lf)
            POD_BASES[edge] = range_space.make_array([g.vector.copy()])

            # remove edge
            active_edges.remove(edge)
        timer.stop()
        logger.info(
            "Computed additional mode due to inhomogeneous"
            f" Dirichlet BCs in t={timer.dt}s."
        )

    if len(active_edges) > 0:
        timer.start()
        if distribution == "normal":
            B, range_products, num_training_samples = adaptive_edge_rrf_normal(
                transfer_problem,
                random_state,
                active_edges,
                source_product=transfer_problem.source_product,
                range_product=example_config.range_product,
                error_tol=example_config.ttol,
                failure_tolerance=1e-15,
                num_testvecs=example_config.num_testvecs,
                **sampling_options,
            )
        elif distribution == "multivariate_normal":
            B, range_products, num_training_samples = adaptive_edge_rrf_mvn(
                transfer_problem,
                random_state,
                active_edges,
                source_product=transfer_problem.source_product,
                range_product=example_config.range_product,
                error_tol=example_config.ttol,
                failure_tolerance=1e-15,
                num_testvecs=example_config.num_testvecs,
                **sampling_options,
            )
        timer.stop()

        for k, v in B.items():
            POD_BASES[k] = v

        logger.info(f"Ran range finder algorithm in t={timer.dt}s.")
        logger.info(f"Number of training samples={num_training_samples}.")
    else:
        logger.info(f"No need to run range finder. {len(active_edges)=}.")

    if len(neumann["offline"]) > 0:
        logger.info("Computing additional modes due to Neumann BCs.")
        timer.start()
        # setup a new oversampling problem
        op = LinearElasticityProblem(Ω, V, E=E, NU=NU, plane_stress=plane_stress)

        for neumann_bc in neumann["offline"]:
            g = dolfinx.fem.Function(op.V)
            g.interpolate(neumann_bc["value"])
            neumann_bc.update({"value": g})
            op.add_neumann_bc(**neumann_bc)

        tdim = op.V.mesh.topology.dim
        fdim = tdim - 1
        facets_gamma_out = dolfinx.mesh.locate_entities_boundary(
            op.V.mesh, fdim, gamma_out
        )
        zeroes = np.array([0, 0], dtype=PETSc.ScalarType)
        bc_zero = {
            "value": zeroes,
            "boundary": facets_gamma_out,
            "method": "topological",
            "entity_dim": fdim,
        }
        op.add_dirichlet_bc(**bc_zero)

        bcs = op.get_dirichlet_bcs()
        petsc_options = {
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        }
        op.setup_solver(petsc_options=petsc_options)
        solver = op.solver

        op.assemble_matrix(bcs)
        op.assemble_vector(bcs)
        u = dolfinx.fem.Function(op.V)
        u.name = "u_neumann"

        solver.solve(op.b, u.vector)
        u.x.scatter_forward()

        op.setup_coarse_space()
        fine_scale_part(u, op.W, in_place=True)

        # restrict to target subdomain
        u_target = dolfinx.fem.Function(transfer_problem.range.V)
        u_target.interpolate(u)

        # restrict to active edges
        for edge in active_edges:
            dofs = transfer_problem.subproblem.V_to_L[edge]
            Lf = transfer_problem.subproblem.edge_spaces["fine"][edge]
            rs = FenicsxVectorSpace(Lf)
            neumann_mode = rs.from_numpy(u_target.vector.array[np.newaxis, dofs])

            try:
                extend_basis(
                    neumann_mode,
                    POD_BASES[edge],
                    product=range_products[edge],
                    method="pod",
                    pod_modes=1,
                )
            except ExtensionError:
                logger.info("Neumann mode already in span.")
        timer.stop()
        logger.info(
            "Extended POD basis by additional Neumann Mode in " f"t={timer.dt}s."
        )

    if output is not None:
        outdir = pathlib.Path(output).parent
        outdir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving POD bases for {cell_index=} to file {output} ...")
        out = {}
        for k, v in POD_BASES.items():
            out[k] = v.to_numpy()
            logger.info(f"Number of POD modes for edge {k}: {len(out[k])}.")
        np.savez(output, **out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # common for all examples
    parser.add_argument("cell_index", type=int, help="The cell index.")
    parser.add_argument(
        "distribution", type=str, help="The distribution used for sampling."
    )
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

    compute_pod_modes(**args.__dict__)
