import dolfinx
import ufl
from mpi4py import MPI
import numpy as np

from multi.bcs import BoundaryDataFactory
from multi.shapes import get_hierarchical_shape_functions
from multi.product import InnerProduct

from pymor.bindings.fenicsx import FenicsxVectorSpace, FenicsxMatrixOperator
from pymor.algorithms.gram_schmidt import gram_schmidt


def compute_hierarchical_modes(
    example=None,
    num_cells=None,
    max_degree=None,
    orthonormalize=False,
    product=None,
):
    """compute hierarchical edge modes

    Parameters
    ----------
    example : src.definitions.ExampleConfig
        A suitable definiton of the beam example.
    num_cells : int
        The number of cells in the (edge) grid.
    max_degree : int
        The maximum polynomial degree of the shape functions.
        Must be greater than or equal to 2.
    orthonormalize : bool, optional
        If True, orthonormalize the basis to inner ``product``.
    product : optional
        Inner product wrt to which the edge basis is orthonormalized
        if ``orthonormalize`` is True.

    """

    domain = dolfinx.mesh.create_unit_interval(MPI.COMM_SELF, num_cells)

    ve = ufl.VectorElement("P", domain.ufl_cell(), degree=example.degree, dim=2)
    V = dolfinx.fem.FunctionSpace(domain, ve)

    x_dofs = V.tabulate_dof_coordinates()

    basis = get_hierarchical_shape_functions(
        x_dofs[:, 0], max_degree, ncomp=ve.value_size()
    )
    source = FenicsxVectorSpace(V)
    B = source.from_numpy(basis)

    # ### build inner product for edge space
    data_factory = BoundaryDataFactory(domain, V)
    g = dolfinx.fem.Function(V)
    g.x.set(0.0)
    bc_zero = data_factory.create_bc(g)

    if product is not None:
        inner_product = InnerProduct(V, product, bcs=[bc_zero])
        product_mat = inner_product.assemble_matrix()
        product_op = FenicsxMatrixOperator(product_mat, V, V)
    else:
        product_op = None

    if orthonormalize:
        gram_schmidt(B, product=product_op, copy=False)


    basis = B.to_numpy()
    assert basis.shape == (2*(max_degree-1), source.dim)
    edge_modes = {"bottom": basis, "left": basis, "right": basis, "top": basis}

    real = 0  # there should only be one realization
    for cell_index in range(example.num_cells):
        # simply use existing structure
        # call distribution="hierarchical"
        out_path = example.pod_bases("hierarchical", real, cell_index)
        np.savez(out_path, **edge_modes)


if __name__ == "__main__":
    from src.definitions import ExampleConfig
    example = ExampleConfig(name="hbeam_1", nx=5, ny=3, num_real=1)
    compute_hierarchical_modes(
        example=example,
        num_cells=10,
        max_degree=7,
        orthonormalize=True,
        product="l2"
        )
