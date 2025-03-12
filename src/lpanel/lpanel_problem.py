import numpy as np
from dolfinx.mesh import locate_entities_boundary
from petsc4py.PETSc import ScalarType

from multi.boundary import point_at, plane_at, within_range
from multi.problems import MultiscaleProblem


class HatFunctionFactory(object):
    """factory for piece-wise linear functions"""

    def __init__(self, points, value):
        """initialize

        Parameters
        ----------
        points : list of floats
            The interpolation points.
        value : float
            The function value at the mid point.

        """
        self.p = points
        self.v = value

    def get_hat_function(self):
        """returns a callable that defines a function

                (v)
                / \
            f1 /   \ f2        # noqa: W605
              /     \
             /       \
            /         \
          (p0)  (p1)  (p2)

        """
        p = self.p
        v = self.v


        def hat(x):
            f_1 = v * (x[0] - p[0])
            f_2 = v * (p[2] - x[0])

            fx = np.zeros_like(x[0])
            fy = f_1 * np.heaviside(x[0] - p[0], 1) - f_1 * np.heaviside(x[0] - p[1], 1) + f_2 * np.heaviside(x[0] - p[1], 1)
            return np.array([fx, fy])

        return hat


class LpanelProblem(MultiscaleProblem):
    def __init__(self, coarse_grid_path, fine_grid_path):
        super().__init__(coarse_grid_path, fine_grid_path)
        self.setup_coarse_grid()

        cell_sets = self.cell_sets

        y0 = cell_sets["y0"]
        right = cell_sets["right"]
        top = cell_sets["top"]
        left = cell_sets["left"]
        bottom = cell_sets["bottom"]
        x0 = cell_sets["x0"]

        corners = set()
        corners.update(y0.intersection(right))
        corners.update(right.intersection(top))
        corners.update(top.intersection(left))
        corners.update(left.intersection(bottom))
        corners.update(bottom.intersection(x0))

        assert len(corners) == 5

        # recessed corner is part of inner
        # y0 and x0 do not intersect

        # boundary without corners and recessed corner
        boundary_layer = cell_sets["boundary"]
        boundary_layer.difference_update(corners)

        inner = cell_sets["inner"]

        cs = {"inner": inner, "boundary": boundary_layer, "corners": corners}

        # check
        Σ = 0
        for cset in cs.values():
            Σ += len(cset)

        assert np.isclose(Σ, self.coarse_grid.num_cells)

        self.build_edge_basis_config(cs)

    @property
    def cell_sets(self):
        cell_sets = {}

        coarse_grid = self.coarse_grid
        assert coarse_grid.facet_markers.values.size > 0
        num_cells = coarse_grid.num_cells

        partial_Ω = set()  # cell set for the whole boundary
        boundaries = self.boundaries
        facet_dim = 1
        for key, values in boundaries.items():
            marker = values[0]

            facets = coarse_grid.facet_markers.find(marker)
            cells = coarse_grid.get_cells(facet_dim, facets)
            cell_sets[key] = set(cells)  # use same name for cell set and boundary

            partial_Ω.update(set(cells))

        inner = set(range(num_cells)).difference(partial_Ω)
        cell_sets["boundary"] = partial_Ω
        cell_sets["inner"] = inner
        cell_sets["dirichlet"] = cell_sets["bottom"]

        # ### define neumann set
        # compute unit length from num_cells
        num_cells = coarse_grid.num_cells
        x = coarse_grid.grid.geometry.x
        xmin = np.amin(x, axis=0)
        xmax = np.amax(x, axis=0)
        n = np.sqrt(4*num_cells/3)
        ul = (xmax[0]-xmin[0])/n

        oy = xmin[1] + (xmax[1]-xmin[1])/2
        neumann_locator = within_range([xmax[0] - 2*ul, oy, 0.], [xmax[0], oy, 0.])
        neumann_facets = locate_entities_boundary(coarse_grid.grid, 1, neumann_locator)
        neumann_cells = coarse_grid.get_cells(1, neumann_facets)
        cell_sets["neumann"] = set(neumann_cells)

        return cell_sets

    @property
    def boundaries(self):
        x = self.coarse_grid.grid.geometry.x
        xmin = np.amin(x, axis=0)
        xmax = np.amax(x, axis=0)

        # COOS
        ox = xmin[0] + (xmax[0]-xmin[0])/2
        oy = xmin[1] + (xmax[1]-xmin[1])/2

        return {
                "y0": (11, within_range([ox, oy, 0.], [xmax[0], oy, 0.])),
                "right": (12, plane_at(xmax[0], "x")),
                "top": (13, plane_at(xmax[1], "y")),
                "left": (14, plane_at(xmin[0], "x")),
                "bottom": (15, plane_at(xmin[1], "y")),
                "x0": (16, within_range([ox, xmin[1], 0.], [ox, oy, 0.])),
                }

    def get_dirichlet(self, cell_index=None):
        """return dirichlet bcs"""
        zeroes = np.array([0., 0.], dtype=ScalarType)
        zero = ScalarType(0.0)

        cell_sets = self.cell_sets
        _, bottom_locator = self.boundaries["bottom"]

        fix_bottom_left = {
                "value": zeroes,
                "boundary": point_at([0.0, 0.0, 0.0]),
                "method": "geometrical"
                }
        fix_uy_bottom = {
                "value": zero,
                "boundary": bottom_locator,
                "sub": 1,
                "method": "geometrical",
                "entity_dim": 1,
                }
        bottom_left_cell = cell_sets["bottom"].intersection(
                cell_sets["left"]
                )

        hom = []
        if cell_index is not None:
            # offline phase
            patch = self.coarse_grid.get_patch(cell_index)
            if np.any([cell in cell_sets["bottom"] for cell in patch]):
                hom.append(fix_uy_bottom)
            if bottom_left_cell in patch:
                hom.append(fix_bottom_left)
            if cell_index in cell_sets["inner"]:
                hom.clear()
        else:
            # online (fom, rom)
            hom.append(fix_bottom_left)
            hom.append(fix_uy_bottom)

        bcs = {}
        bcs["homogeneous"] = hom
        bcs["inhomogeneous"] = []
        return bcs


    def get_remove_kernel(self, cell_index):
        cell_sets = self.cell_sets
        bottom_left_cell = cell_sets["bottom"].intersection(
                cell_sets["left"]
                )
        patch = self.coarse_grid.get_patch(cell_index)
        if bottom_left_cell in patch:
            return False
        else:
            return True

    def get_gamma_out(self, cell_index):
        """returns marker to be used with locate_enities_boundary"""

        grid = self.coarse_grid
        patch = grid.get_patch(cell_index)
        x = grid.grid.geometry.x
        xmin = np.amin(x, axis=0)
        xmax = np.amax(x, axis=0)

        cell_sets = self.cell_sets
        tol = 1e-3

        start = [xmin[0], xmin[1], 0.0]
        end = [xmax[0], xmax[1], 0.0]
        x0 = xmin[0] + (xmax[0]-xmin[0])/2
        y0 = xmin[1] + (xmax[1]-xmin[1])/2

        def everywhere(x):
            return np.full(x[0].shape, True, dtype=bool)

        to_be_excluded = set()

        for boundary in ["bottom", "left", "right", "top"]:
            if np.any([cell in cell_sets[boundary] for cell in patch]):
                to_be_excluded.add(boundary)

        for facets in to_be_excluded:
            if facets == "bottom":
                start[1] += tol
            elif facets == "left":
                start[0] += tol
            elif facets == "right":
                end[0] -= tol
            elif facets == "top":
                end[1] -= tol

        # if some cell in patch shares x0, but not y0
        if np.any([cell in cell_sets["x0"] for cell in patch]):
            if not np.any([cell in cell_sets["y0"] for cell in patch]):
                end[0] -= (x0 + tol)

                return within_range(start, end)

        # if some cell in patch shares y0, but not x0
        if np.any([cell in cell_sets["y0"] for cell in patch]):
            if not np.any([cell in cell_sets["x0"] for cell in patch]):
                start[1] += (y0 + tol)

                return within_range(start, end)

        # both x0 and y0
        if np.any([cell in cell_sets["x0"] for cell in patch]):
            if np.any([cell in cell_sets["y0"] for cell in patch]):
                end_range_1 = [xmax[0] - (x0 + tol), xmax[1], 0.0]
                start_range_2 = [xmin[0], xmin[1] + (y0 + tol), 0.0]

                def gamma(x):
                    r1 = within_range(start, end_range_1)  # excluded x0
                    r2 = within_range(start_range_2, end)  # excluded y0
                    return np.logical_or(r1(x), r2(x))

                return gamma

        if len(to_be_excluded) < 1:
            return everywhere
        else:
            return within_range(start, end)

    def get_neumann(self, cell_index=None):
        """return neumann data

        offline: 
            • apply traction if any cell in the patch is part of
              the set y0.
            • use the marker that locates boundary globally
              self.boundaries is used to create facet tags for the
              respective oversampling domain.
              See src.compute_pod_modes.py
        online:
            • FOM: simply pass the neumann bc defined globally
            • ROM: define neumann bc using local subdomain markers (b:1, l:2, r:3, t:4)
        """

        cell_sets = self.cell_sets

        # NOTE all facets on the line L=[ox, oy]x[xmax, oy]
        # are marked.
        # The Neumann boundary includes only the last two
        # cells on this line.
        # The reason for using "y0" is that I did not want to
        # split up the line in gmsh to be able to add
        # a physical group there.
        (marker, locator) = self.boundaries["y0"]

        # compute unit length from num_cells
        num_cells = self.coarse_grid.num_cells
        x = self.coarse_grid.grid.geometry.x
        xmin = np.amin(x, axis=0)
        xmax = np.amax(x, axis=0)
        n = np.sqrt(4*num_cells/3)
        ul = (xmax[0]-xmin[0])/n

        FORCE = 2e2 / ul
        hat_factory = HatFunctionFactory([xmax[0] - 2*ul, xmax[0] - ul, xmax[0]], FORCE)
        traction = hat_factory.get_hat_function()

        global_bc = {"marker": marker, "value": traction}
        local_bc = {"marker": 1, "value": traction}  # bottom

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

        neumann = {}
        neumann["fom"] = fom
        neumann["offline"] = offline
        neumann["online"] = online
        return neumann
