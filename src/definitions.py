from pathlib import Path
import dataclasses

ROOT = Path(__file__).parents[1]
SRC = ROOT / "src"


@dataclasses.dataclass
class ExampleConfig:
    degree: int = 2
    material: str = (SRC / "material.yaml").as_posix()
    name: str = "example"
    num_real: int = 1
    num_testvecs : int = 20
    nx : int = 0
    nxy : int = 0
    ny : int = 0
    range_product: str = "l2"
    rce_type : int = 0
    source_product: str = "l2"
    ttol: float = 1e-3

    @property
    def num_cells(self) -> int:
        n = self.nx * self.ny
        if n > 0:
            # block or beam example
            return n
        else:
            # lpanel example
            n = self.nxy
            return int(n ** 2 - (n / 2) ** 2)

    @property
    def rf(self) -> Path:
        """run folder"""
        return ROOT / "work" / f"{self.name}"

    @property
    def grid_path(self) -> Path:
        return self.rf / "grids"

    @property
    def coarse_grid(self) -> Path:
        return self.grid_path / "global/coarse_grid.msh"

    @property
    def fine_grid(self) -> Path:
        xdmf = self.grid_path / "global/fine_grid.xdmf"
        return xdmf

    def subdomain_grid(self, cell_index) -> Path:
        offline = self.grid_path / "subdomains"
        xdmf = offline / f"subdomain_{cell_index:03}.xdmf"
        return xdmf

    def subdomain_facets(self, cell_index) -> Path:
        offline = self.grid_path / "subdomains"
        xdmf = offline / f"subdomain_{cell_index:03}_facets.xdmf"
        return xdmf

    def coarse_patch(self, cell_index) -> Path:
        offline = self.grid_path / "patches"
        xdmf = offline / f"coarse_patch_{cell_index:03}.xdmf"
        return xdmf

    def fine_patch(self, cell_index) -> Path:
        offline = self.grid_path / "patches"
        xdmf = offline / f"fine_patch_{cell_index:03}.xdmf"
        return xdmf

    @property
    def coarse_rom_solution(self) -> Path:
        return self.rf / "multivariate_normal/coarse_rom/rom_solution.npy"

    @property
    def coarse_rom_log(self) -> Path:
        return self.rf / "multivariate_normal/coarse_rom/coarse_rom.log"

    def basis(self, distribution, real, cell_index) -> Path:
        folder = self.rf / f"{distribution}/real_{real:02}/bases"
        return folder / f"basis_{cell_index:03}.npz"

    def empirical_basis_log(self, distribution, real, cell_index) -> Path:
        """same log file for computation of pod edge modes and
        extension of the modes into the subdomain"""
        folder = self.rf / f"{distribution}/real_{real:02}/logs"
        return folder / f"empirical_basis_{cell_index:03}.log"

    def pod_bases(self, distribution, real, cell_index) -> Path:
        folder = self.rf / f"{distribution}/real_{real:02}/bases"
        return folder / f"pod_bases_{cell_index:03}.npz"

    def rom_log(self, distribution, real) -> Path:
        folder = self.rf / f"{distribution}/real_{real:02}/logs"
        return folder / f"{self.name}_rom.log"

    def error_log(self, distribution, real) -> Path:
        folder = self.rf / f"{distribution}/real_{real:02}/logs"
        return folder / f"{self.name}_error.log"

    def example_log(self, distribution, real) -> Path:
        folder = self.rf / f"{distribution}/real_{real:02}/logs"
        return folder / f"{self.name}_example.log"

    def fom_solution(self, distribution, real) -> Path:
        folder = self.rf / f"{distribution}/real_{real:02}/solutions"
        return folder / "fom.xdmf"

    def rom_solution(self, distribution, real) -> Path:
        folder = self.rf / f"{distribution}/real_{real:02}/solutions"
        return folder / "rom_solution.npz"

    def error(self, distribution, real) -> Path:
        folder = self.rf / f"{distribution}/real_{real:02}/solutions"
        return folder / "error.npz"

    def mean_error(self, distr) -> Path:
        return self.rf / f"{distr}/mean_error.npz"

    def std_error(self, distr) -> Path:
        return self.rf / f"{distr}/std_error.npz"

    def mean_log_data(self, distr) -> Path:
        return self.rf / f"{distr}/mean_log_data.npz"

    def fields_subdomain(self, distribution, real, cell_index) -> Path:
        folder = self.rf / f"{distribution}/real_{real:02}/solutions"
        return folder / f"fields_subdomain_{cell_index:03}.pvd"

    @property
    def error_plot(self) -> Path:
        return self.rf.parent / "tex" / f"{self.name}_error_plot.pdf"


def get_random_seed(example, real):
    """get a different random seed for each example problem
    and each realization of an example problem

    Parameters
    ----------
    example : str
        The name of the example.
    real : int
        The realization of this particular example.

    e.g.
    if there are two block examples like

    b1 = BlockConfig(name='block_1', ttol=1e-1, num_real=2)
    b2 = BlockConfig(name='block_2', ttol=1e-2, num_real=2)

    return 
    3710 block_1 real=0
    3711 block_1 real=1
    3720 block_2 real=0
    3721 block_2 real=1

    if there are 100 realizations would get 37199
    """

    abc = "abcdefghijklmnopqrstuvwxyz"
    numbers = "123456789"

    # naming convention: name_number
    # name = example.name.replace("_", "")
    name = example.replace("_", "")
    my_examples = ("block", "beam", "lpanel")
    assert any([name.startswith(e) for e in my_examples])
    assert any([i in name for i in numbers])
    summe = sum([abc.find(char) for char in name])
    base_seed = str(summe)

    for x in numbers:
        pos = name.find(x)
        if pos > -1:
            base_seed += x
            break

    return int(base_seed+str(real))
