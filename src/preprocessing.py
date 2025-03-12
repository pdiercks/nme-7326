"""preprocessing routines specific to the paper"""

from enum import Enum
from dolfinx.io import XDMFFile, gmshio
from dolfinx.mesh import create_submesh
from mpi4py import MPI

from multi.domain import StructuredQuadGrid


class GridType(Enum):
    COARSE = 1
    FINE = 2


class DomainType(Enum):
    GLOBAL = 1
    PATCH = 2
    SUBDOMAIN = 3


def create_grid(
    coarse_grid_msh,
    grid_type,
    domain_type,
    output_path,
    cell_index=None,
    subdomain_grid_method=None,
    **grid_creation_param,
):
    """create coarse or fine scale grid for given domain type

    Parameters
    ----------
    coarse_grid_msh : str
        The coarse grid of the global domain.
    grid_type : GridType
        The grid type.
    domain_type : DomainType
        The domain type.
    output_path : str
        The path to write the grid to.
    cell_index : optional, int
        The cell index for which to create a grid based on given
        domain and grid type.
    subdomain_grid_method : optional, callable
        The method to generate the fine scale subdomain grid.
    grid_creation_param : optional
        Additional parameters to be handed down to the method
        for subdomain grid creation.
    """

    assert grid_type in GridType
    assert domain_type in DomainType

    domain, cell_markers, facet_markers = gmshio.read_from_msh(
        coarse_grid_msh, MPI.COMM_SELF, gdim=2
    )
    coarse_grid = StructuredQuadGrid(domain, cell_markers, facet_markers)
    if subdomain_grid_method is not None:
        coarse_grid.fine_grid_method = subdomain_grid_method

    if domain_type == DomainType.GLOBAL:
        coarse_grid.create_fine_grid(
            coarse_grid.cells, output_path, **grid_creation_param
        )
    elif domain_type == DomainType.PATCH:
        cells = coarse_grid.get_patch(cell_index)
        if grid_type == GridType.COARSE:
            coarse_patch = create_submesh(domain, domain.topology.dim, cells)[0]
            with XDMFFile(coarse_patch.comm, output_path, "w") as xdmf:
                xdmf.write_mesh(coarse_patch)
        else:
            coarse_grid.create_fine_grid(cells, output_path, **grid_creation_param)
    elif domain_type == DomainType.SUBDOMAIN:
        coarse_grid.create_fine_grid([cell_index], output_path, **grid_creation_param)
    else:
        raise NotImplementedError
