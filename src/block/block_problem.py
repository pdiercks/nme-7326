import pathlib
import numpy as np
from petsc4py.PETSc import ScalarType
from multi.boundary import plane_at, within_range
from multi.problems import MultiscaleProblem

ROOT = pathlib.Path(__file__).parents[2]
SRC = ROOT / "src"


def boundary_expression_factory(xmax, ymax, degree=2, gdim=2, seed=None, scale=1.0):
    """define expression for boundary data on ∂Ω
    with Ω = [0, xmax] x [0, ymax]

        U_i = alpha_i0 + alpha_ij x_j + beta_ij x_j ^ 2
        where i = 1, 2, 3 and j = 1, 2, 3

    in 2d (i=j=1, 2) this leads to 10 parameters alpha_i0, alpha_ij and beta_ij.
    """
    random_state = np.random.RandomState(seed=seed)
    # set alpha_0 to zero, such that domain is not translated
    alpha_0 = np.zeros(gdim)
    alpha = random_state.rand(gdim, gdim)
    beta = random_state.rand(gdim, gdim)
    if degree == 1:
        beta = np.zeros_like(alpha)

    def f_x(x):
        return (
            alpha_0[0]
            + alpha[0, 0] * x[0]
            + alpha[0, 1] * x[1]
            + beta[0, 0] * x[0] ** 2
            + beta[0, 1] * x[1] ** 2
        )

    def f_y(x):
        return (
            alpha_0[1]
            + alpha[1, 0] * x[0]
            + alpha[1, 1] * x[1]
            + beta[1, 0] * x[0] ** 2
            + beta[1, 1] * x[1] ** 2
        )

    x = np.array([xmax, ymax])
    s_x = np.array([f_x(x), f_y(x)]) / scale

    def f(x):
        s = np.linalg.norm(s_x)
        return np.array([f_x(x) / s, f_y(x) / s])

    return f


class BlockProblem(MultiscaleProblem):
    def __init__(self, coarse_grid_path, fine_grid_path):
        super().__init__(coarse_grid_path, fine_grid_path)
        self.setup_coarse_grid()

        # build edge basis config
        cell_sets = self.cell_sets
        bottom = cell_sets["bottom"]
        left = cell_sets["left"]
        right = cell_sets["right"]
        top = cell_sets["top"]

        corners = set()
        corners.update(left.intersection(bottom.union(top)))
        corners.update(right.intersection(bottom.union(top)))
        inner_boundary = cell_sets["boundary"].difference(corners)
        cs = {
            "inner": cell_sets["inner"],
            "boundary": inner_boundary,
            "corners": corners,
        }
        self.build_edge_basis_config(cs)


    @property
    def cell_sets(self):
        cell_sets = {}

        coarse_grid = self.coarse_grid
        assert coarse_grid.facet_markers.values.size > 0
        bottom = set(coarse_grid.get_cells(1, coarse_grid.facet_markers.find(1)))
        left = set(coarse_grid.get_cells(1, coarse_grid.facet_markers.find(2)))
        right = set(coarse_grid.get_cells(1, coarse_grid.facet_markers.find(3)))
        top = set(coarse_grid.get_cells(1, coarse_grid.facet_markers.find(4)))

        boundary_layer = bottom.union(left.union(right.union(top)))
        inner = set(range(coarse_grid.num_cells)).difference(boundary_layer)

        corners = set()
        corners.update(left.intersection(bottom.union(top)))
        corners.update(right.intersection(bottom.union(top)))
        inner_boundary = boundary_layer.difference(corners)

        cell_sets = {
            "inner": inner,
            "boundary": boundary_layer,
            "bottom": bottom,
            "left": left,
            "right": right,
            "top": top,
            "dirichlet": boundary_layer,
            "neumann": set(),
            "corners": corners,
            "inner_boundary": inner_boundary,
        }

        return cell_sets

    @property
    def boundaries(self):
        # geometrical representation to define
        # (a) dirichet bcs
        # (b) facet tags
        # for the oversampling problem
        grid = self.coarse_grid
        x = grid.grid.geometry.x
        xmin = np.amin(x, axis=0)
        xmax = np.amax(x, axis=0)

        # dict: key=string, value=(marker, locator)
        return {
            "bottom": (11, plane_at(xmin[1], "y")),
            "left": (12, plane_at(xmin[0], "x")),
            "right": (13, plane_at(xmax[0], "x")),
            "top": (14, plane_at(xmax[1], "y")),
        }

    def get_boundary_data_expression(self):
        x = self.coarse_grid.grid.geometry.x
        xmax = np.amax(x, axis=0)
        g = boundary_expression_factory(xmax[0], xmax[1], degree=2, gdim=2, seed=17)
        return g

    def get_dirichlet(self, cell_index=None):
        """return (global definition of) Dirichlet bc

        Behaviour:
        (a) cell_index is None
        --> Return list of bc dict for 'homogeneous' and 'inhomogeneous'
            for entire structure (hom or inhom treated in the same way)
        (b) cell_index is int
        --> Return list of bc dict for 'homogeneous' and 'inhomogeneous'
            for current cell (oversampling (homogeneous) and extra mode (inhom))

        Returns
        -------
        bcs : dict
            A dict with keys 'inhomogeneous' and 'homogeneous' and
            a list of dict as value. Each element of the list defines
            a single Dirichlet bc
            (format as in multi.bcs.BoundaryConditions.add_dirichlet_bc).
        """

        g = self.get_boundary_data_expression()
        zero = np.array([0, 0], dtype=ScalarType)

        cell_sets = self.cell_sets
        boundaries = self.boundaries

        bcs = {}
        hom = []
        inhom = []

        if cell_index is not None:
            patch = self.coarse_grid.get_patch(cell_index)
            for boundary, (_, locator) in boundaries.items():
                if np.any([cell in cell_sets[boundary] for cell in patch]):
                    bc_zero = {
                        "value": zero,
                        "boundary": locator,
                        "method": "geometrical",
                    }
                    hom.append(bc_zero)
                if cell_index in cell_sets[boundary]:
                    bc = {"value": g, "boundary": locator, "method": "geometrical"}
                    inhom.append(bc)
        else:
            for boundary, (_, locator) in boundaries.items():
                bc = {"value": g, "boundary": locator, "method": "geometrical"}
                inhom.append(bc)

        bcs["homogeneous"] = hom
        bcs["inhomogeneous"] = inhom
        return bcs

    def get_remove_kernel(self, cell_index):
        gamma_out = self.get_gamma_out(cell_index)

        if gamma_out.__name__ == "everywhere":
            return True
        else:
            return False

    def get_gamma_out(self, cell_index):
        """returns marker to be used to determine boundary facets"""

        grid = self.coarse_grid
        patch = grid.get_patch(cell_index)
        x = grid.grid.geometry.x
        xmin = np.amin(x, axis=0)
        xmax = np.amax(x, axis=0)

        cell_sets = self.cell_sets
        tol = 1e-3

        start = [xmin[0], xmin[1], 0.0]
        end = [xmax[0], xmax[1], 0.0]

        # exclude bottom: start[1] + tol
        # exclude left: start[0] + tol
        # exclude right: end[0] - tol
        # exclude top: end[1] - tol

        def everywhere(x):
            return np.full(x[0].shape, True, dtype=bool)

        to_be_excluded = set()

        for boundary in ["bottom", "left", "right", "top"]:
            if np.any([cell in cell_sets[boundary] for cell in patch]):
                to_be_excluded.add(boundary)

        for edge in to_be_excluded:
            if edge == "bottom":
                start[1] += tol
            if edge == "left":
                start[0] += tol
            if edge == "right":
                end[0] -= tol
            if edge == "top":
                end[1] -= tol

        if len(to_be_excluded) < 1:
            return everywhere
        elif len(to_be_excluded) == 4:
            # special case: intersection of ∂Ω and Σ_D is the empty set
            return everywhere
        else:
            return within_range(start, end)

    def get_neumann(self, cell_index=None):
        """return neumann bc if any cell in the patch lies on Σ_N"""
        neumann = {}
        neumann["offline"] = []
        neumann["online"] = []
        neumann["fom"] = []
        return neumann


if __name__ == "__main__":
    ROOT = pathlib.Path("/mnt/paper")
    WORK = ROOT / "work"
    coarse_grid = WORK / "block_1/grids/global/coarse_grid.msh"
    fine_grid = WORK / "block_1/grids/global/fine_grid.xdmf"

    block = BlockProblem(coarse_grid.as_posix(), fine_grid.as_posix())
    block.material = ROOT / "src/material.yaml"
    block.degree = 2
    block.setup_fe_spaces()
