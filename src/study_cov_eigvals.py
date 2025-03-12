"""study the eigenvalues of the covariance matrix
for different correlation lengths"""

import sys
import argparse

import dolfinx
import numpy as np
from mpi4py import MPI

from scipy.linalg import eigh
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import diags, csc_matrix
from scipy.sparse.linalg import eigsh

from multi.misc import x_dofs_VectorFunctionSpace
from multi.sampling import correlation_function
from multi.solver import build_nullspace
from multi.projection import orthogonal_part
from multi.product import InnerProduct

from pymor.bindings.fenicsx import FenicsxVectorSpace, FenicsxMatrixOperator
from pymor.tools.timing import Timer


def main(example, cell_index):
    timer = Timer("study")

    from definitions import ExampleConfig
    if example == "block":
        from block.block_problem import BlockProblem as ExampleProblem
        config = ExampleConfig(name="block_1", degree=2, nx=5, ny=5)
        cell_index = 12
    elif example == "beam":
        from beam.beam_problem import BeamProblem as ExampleProblem
        config = ExampleConfig(name="beam_1", degree=2, nx=50, ny=5)
        cell_index = 117
    else:
        raise NotImplementedError

    problem = ExampleProblem(config.coarse_grid, config.fine_grid)
    problem.degree = config.degree
    problem.setup_coarse_grid()
    problem.setup_coarse_space()

    fine_patch_xdmf = config.fine_patch(cell_index)
    with dolfinx.io.XDMFFile(
            MPI.COMM_SELF, fine_patch_xdmf.as_posix(), "r"
            ) as xdmf:
        domain = xdmf.read_mesh(name="Grid")
    V = dolfinx.fem.VectorFunctionSpace(domain, ("P", config.degree))

    gamma_out = problem.get_gamma_out(cell_index)
    facets_gamma_out = dolfinx.mesh.locate_entities_boundary(
            domain, 1, gamma_out)
    dofs = dolfinx.fem.locate_dofs_topological(V, 1, facets_gamma_out)
    dummy = dolfinx.fem.Function(V)
    bc = dolfinx.fem.dirichletbc(dummy, dofs)

    gamma_dofs = bc.dof_indices()[0]
    x_dofs = x_dofs_VectorFunctionSpace(V)
    points = x_dofs[gamma_dofs]

    rom_solution = config.coarse_rom_solution
    u_rom = np.load(rom_solution)
    W_global = problem.W
    u_macro = dolfinx.fem.Function(W_global)
    try:
        u_macro.vector.array[:] = u_rom.flatten()
    except ValueError as err:
        raise err(
                "Shape of the ROM solution and coarse FE space do not match.")

    coarse_patch_xdmf = config.coarse_patch(cell_index)
    with dolfinx.io.XDMFFile(
            MPI.COMM_WORLD, coarse_patch_xdmf.as_posix(), "r") as xdmf:
        coarse_patch = xdmf.read_mesh()
    fe_family = W_global.element.basix_element.family
    W_local = dolfinx.fem.VectorFunctionSpace(coarse_patch, (fe_family, 1))
    w_local = dolfinx.fem.Function(W_local)

    # ### interpolate global solution onto oversampling domain
    w_local.interpolate(u_macro)
    coarse_source = FenicsxVectorSpace(W_local)
    U = coarse_source.make_array([w_local.vector])
    inner_product = InnerProduct(W_local, product="l2")
    product_mat = inner_product.assemble_matrix()
    l2_product = FenicsxMatrixOperator(product_mat, W_local, W_local)
    kernel = build_nullspace(coarse_source, product=l2_product, gdim=2)
    U_proj = orthogonal_part(kernel, U, product=l2_product, orth=True)
    w_local.vector.array[:] = U_proj.to_numpy().flatten()

    # ### create function for w_local in the fine grid space
    u_local = dolfinx.fem.Function(V)
    u_local.interpolate(w_local)
    # restrict function to values on Γ_out
    mean = u_local.vector.array[gamma_dofs]
    D = diags(mean)

    distance = squareform(pdist(points, metric="euclidean"))

    xmin_cp = np.amin(coarse_patch.geometry.x, axis=0)
    xmax_cp = np.amax(coarse_patch.geometry.x, axis=0)
    Lcorr = np.linalg.norm(xmax_cp - xmin_cp)

    rtol = 5e-2

    def get_sigma_neig(dist, lc, D, rtol=5e-2):
        """return covariance matrix Σ and number of samples to be drawn
        using Σ""" 
        Σ_exp = correlation_function(
                dist, lc, function_type="exponential"
                )
        Σ = D.dot(csc_matrix(Σ_exp).dot(D))
        # get largest eigenvalue
        lambda_max = eigsh(Σ, k=1, which="LM", return_eigenvectors=False)
        # determine subset of eigenvalues by value
        eigvals = eigh(Σ.toarray(), eigvals_only=True, turbo=True,
                       subset_by_value=[lambda_max * rtol, np.inf])
        num_eigv = eigvals.size
        return Σ, num_eigv

    # timer.start()
    Σ, neig = get_sigma_neig(distance, Lcorr, D)
    # timer.stop()
    # print(f"t={timer.dt}")

    # timer.start()
    # Σ2, neig2 = get_sigma_neig(distance, Lcorr/2, D)
    # timer.stop()
    # print(f"t={timer.dt}")



    # TODO check if there is speed up in case of the beam (rce 02)
    # TODO run the range finder
    #   • use a uncorrelated test set
    #   • report the number of correlated training samples
    #     (compare to previous number of samples)
    #   • quality of the new basis? --> fastest way is to compare the
    #     training samples directly (if linear dependent, so is the solution)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("example", type=str, choices=("block", "beam"))
    PARSER.add_argument("cell_index", type=int)
    ARGS = PARSER.parse_args(sys.argv[1:])
    main(ARGS.example, ARGS.cell_index)
