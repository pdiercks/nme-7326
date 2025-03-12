from pathlib import Path
import numpy as np
from petsc4py.PETSc import ScalarType
from multi.boundary import plane_at, point_at, within_range
from multi.problems import MultiscaleProblem

ROOT = Path(__file__).parents[2]
SRC = ROOT / "src"


class BeamProblem(MultiscaleProblem):
    def __init__(self, coarse_grid_path, fine_grid_path):
        super().__init__(coarse_grid_path, fine_grid_path)
        self.setup_coarse_grid()

        # build edge basis config such that it can
        # be used for the definition of tasks more convenient
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
        num_cells = coarse_grid.num_cells

        bottom = set(coarse_grid.get_cells(1, coarse_grid.facet_markers.find(1)))
        left = set(coarse_grid.get_cells(1, coarse_grid.facet_markers.find(2)))
        right = set(coarse_grid.get_cells(1, coarse_grid.facet_markers.find(3)))
        top = set(coarse_grid.get_cells(1, coarse_grid.facet_markers.find(4)))

        boundary_layer = bottom.union(left.union(right.union(top)))
        inner = set(range(num_cells)).difference(boundary_layer)

        cell_sets = {
            "inner": inner,
            "boundary": boundary_layer,
            "bottom": bottom,
            "left": left,
            "right": right,
            "top": top,
            "dirichlet": left,
            "neumann": right,
        }

        return cell_sets

    @property
    def boundaries(self):
        # geometrical representation to define
        # (a) dirichet bcs
        # (b) facet tags
        # for the oversampling problem
        x = self.coarse_grid.grid.geometry.x
        xmin = np.amin(x, axis=0)
        xmax = np.amax(x, axis=0)

        # dict: key=string, value=(marker, locator)
        return {
            "bottom": (11, plane_at(xmin[1], "y")),
            "left": (12, plane_at(xmin[0], "x")),
            "right": (13, plane_at(xmax[0], "x")),
            "top": (14, plane_at(xmax[1], "y")),
        }

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

        zeroes = np.array([0, 0], dtype=ScalarType)
        zero = ScalarType(0.0)

        cell_sets = self.cell_sets
        dirichlet = "left"
        _, on_left_boundary = self.boundaries[dirichlet]

        bcs = {}
        hom = []
        inhom = []

        fix_origin = {
            "value": zeroes,
            "boundary": point_at([0.0, 0.0, 0.0]),
            "method": "geometrical",
        }
        fix_ux = {
            "value": zero,
            "boundary": on_left_boundary,
            "sub": 0,
            "method": "geometrical",
            "entity_dim": 1,
        }

        if cell_index is not None:
            patch = self.coarse_grid.get_patch(cell_index)
            if np.any([cell in cell_sets[dirichlet] for cell in patch]):
                hom.append(fix_ux)
            if 0 in patch:
                hom.append(fix_origin)

        else:
            # global problem:
            # u_x = 0 at x=0 (left)
            # u_x = u_y = 0 at x=y=0 (origin)
            hom.append(fix_ux)
            hom.append(fix_origin)

        bcs["homogeneous"] = hom
        bcs["inhomogeneous"] = inhom
        return bcs

    def get_remove_kernel(self, cell_index):
        patch = self.coarse_grid.get_patch(cell_index)
        if 0 in patch:
            # oversampling problem kinematically determined due
            # to dirichlet bcs
            return False
        else:
            return True

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
        else:
            return within_range(start, end)

    def get_neumann(self, cell_index=None):
        """return neumann data if any cell in the patch is part
        of the neumann boundary"""

        grid = self.coarse_grid
        x = grid.grid.geometry.x
        xmax = np.amax(x, axis=0)

        def traction(x):
            f_x = 240.0 / xmax[1] * x[1] - 120.0
            f_y = np.zeros_like(f_x)
            return np.array([f_x, f_y])

        # always return traction to be interpolated
        # in the FE space (global space, oversampling space, ...)
        # However, the facet tag might change ...
        # single subdomain usually has facet markers defined
        # global fine grid has facet markers defined as well
        # the oversampling patch has its facet markers defined in
        # /src/compute_pod_modes.py based on multiscale_problem.boundaries !

        cell_sets = self.cell_sets
        (marker, locator) = self.boundaries["right"]

        global_bc = {"marker": marker, "value": traction}
        local_bc = {"marker": 3, "value": traction}

        neumann = {}
        offline = []
        online = []
        fom = []

        if cell_index is not None:
            patch = self.coarse_grid.get_patch(cell_index)
            if np.any([cell in cell_sets["neumann"] for cell in patch]):
                offline.append(global_bc)
            if cell_index in cell_sets["neumann"]:
                online.append(local_bc)

        else:
            fom.append(global_bc)

        neumann["fom"] = fom
        neumann["offline"] = offline
        neumann["online"] = online
        return neumann
