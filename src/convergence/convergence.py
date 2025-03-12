"""
convergence study for RVE mesh using arbitrary load case

Usage:
    convergence.py [options] PATH DEG MAT

Arguments:
    PATH      A filepath to search for meshes (.xdmf files).
    DEG       Degree of the FE space.
    MAT       Material specification file (.yaml).

Options:
    -h, --help               Show this message.
    -l LOG, --log=LOG        Set the log level [default: 30].
    --product=PROD           The inner product to use [default: energy].
    -o FILE, --output=FILE   Write error and mesh parameters to FILE.
"""

import sys
import dolfinx
from dolfinx.io import gmshio
from mpi4py import MPI
import numpy as np

import yaml
from pathlib import Path
from docopt import docopt
from pymor.core.logger import getLogger
from pymor.bindings.fenicsx import FenicsxMatrixOperator, FenicsxVectorSpace

from multi.problems import LinearElasticityProblem
from multi.domain import RectangularDomain
from multi.interpolation import interpolate
from multi.product import InnerProduct

CONVERGENCE = Path(__file__).parent
ROOT = CONVERGENCE.absolute().parent
BLOCK = ROOT / "block"
sys.path.append(BLOCK.absolute().as_posix())
from block_problem import boundary_expression_factory  # noqa: E402


def parse_arguments(args):
    args = docopt(__doc__, args)
    args["PATH"] = Path(args["PATH"])
    args["DEG"] = int(args["DEG"])
    args["MAT"] = Path(args["MAT"])
    args["--log"] = int(args["--log"])
    assert args["--product"] in ("energy", "mass", "stiffness", "h1", "l2", "h1-semi")
    return args


def main(args):
    args = parse_arguments(args)
    logger = getLogger("convergence")
    logger.setLevel(args["--log"])

    meshes = sorted(args["PATH"].glob("rce_*.msh"))

    ndofs = np.array([], dtype=int)
    h_mins = np.array([], dtype=float)
    h_maxs = np.array([], dtype=float)
    h_means = np.array([], dtype=float)
    errors = np.array([], dtype=float)

    # loop over meshes in reversed order
    logger.debug(f"mesh files are {meshes}")
    with logger.block("Starting convergence run ..."):
        logger.debug(f"Using inner product: {args['--product']}")
        ref_mesh = meshes[-1].as_posix()
        u_ref, Uref, prodref, (hmin, hmax), ndofs_ = run(
            ref_mesh,
            args["DEG"],
            args["MAT"],
            loglevel=args["--log"],
            product=args["--product"],
        )
        # prepare points of the reference mesh
        x_dofs_ref = Uref.space.V.tabulate_dof_coordinates()

        for mesh in meshes:
            u, U, product, (hmin, hmax), ndofs_ = run(
                mesh.as_posix(),
                args["DEG"],
                args["MAT"],
                loglevel=args["--log"],
                product=args["--product"],
            )
            logger.debug(f"hmin, hmax = {hmin}, {hmax}")
            logger.debug(f"hmax / hmin = {hmax / hmin}")
            logger.debug(f"hmean = {(hmax + hmin)/2}")
            ndofs = np.append(ndofs, ndofs_)
            h_mins = np.append(h_mins, hmin)
            h_maxs = np.append(h_maxs, hmax)
            h_means = np.append(h_means, (hmax + hmin) / 2)

            iu_arr = interpolate(u, x_dofs_ref)
            IntU = Uref.space.from_numpy(iu_arr.reshape(1, Uref.space.dim))
            # IntU = interpolate_fenics_vector(U, Uref.space)
            err = (Uref - IntU).norm(product=prodref)
            norm = Uref.norm(product=prodref)
            errors = np.append(errors, err / norm)

    if args["--output"] is not None:
        np.savetxt(
            args["--output"],
            np.column_stack((errors, h_mins, h_maxs, h_means, ndofs)),
            delimiter=",",
            header=f"rel err in {args['--product']} norm, hmin, hmax, hmean, ndofs",
        )


def run(meshfile, degree, material_file, loglevel=30, product="energy"):
    """fem solve"""
    logger = getLogger("FEM")
    logger.setLevel(loglevel)

    logger.info(f"Reading mesh from file {meshfile} ...")
    domain, cell_markers, facet_markers = gmshio.read_from_msh(
        meshfile, MPI.COMM_SELF, gdim=2
    )
    omega = RectangularDomain(domain, cell_markers, facet_markers, index=0)

    logger.debug(f"domain xmin: {omega.xmin}")
    logger.debug(f"domain xmax: {omega.xmax}")
    V = dolfinx.fem.VectorFunctionSpace(omega.grid, ("P", degree))
    Vdim = V.dofmap.bs * V.dofmap.index_map.size_global
    logger.debug(f"Number of DOFS: {Vdim}")

    with material_file.open("r") as instream:
        material = yaml.safe_load(instream)

    E = material["Material parameters"]["E"]["value"]
    NU = material["Material parameters"]["NU"]["value"]
    plane_stress = material["Constraints"]["plane_stress"]
    problem = LinearElasticityProblem(omega, V, E=E, NU=NU, plane_stress=plane_stress)

    if product == "energy":
        lhs = problem.form_lhs
        inner_product = InnerProduct(problem.V, lhs, bcs=(), name="energy")
    else:
        inner_product = InnerProduct(problem.V, product, bcs=())

    # FIXME not sure what exactly is going wrong, but
    # calling problem.form_lhs
    # AFTER problem.setup_solver gives RuntimeError
    # if one attempts to compile, i.e. fem.form(problem.form_lhs)

    unit_length = abs(omega.xmax[0] - omega.xmin[0])
    logger.debug(f"unit length: {unit_length}")

    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = dolfinx.mesh.exterior_facet_indices(domain.topology)

    boundary_data = boundary_expression_factory(
        omega.xmax[0], omega.xmax[1], degree=2, gdim=2, seed=66, scale=0.1 * unit_length
    )

    g = dolfinx.fem.Function(V)
    g.interpolate(boundary_data)
    problem.add_dirichlet_bc(g, boundary_facets, entity_dim=1)

    petsc_options = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    problem.setup_solver(petsc_options=petsc_options)

    u = problem.solve()
    hmin = 1
    hmax = 1

    p_mat = inner_product.assemble_matrix()
    product = FenicsxMatrixOperator(p_mat, V, V)
    S = FenicsxVectorSpace(V)
    U = S.make_array([u.vector])
    return u, U, product, (hmin, hmax), Vdim


if __name__ == "__main__":
    main(sys.argv[1:])
