"""
merge DataSets of given PVD files into one PVD file

Note that the target PVD file will always be written
in the same directory where the input files are located.

Usage:
    merge_pvtu_datasets.py [options] PVD...

Arguments:
    PVD       The PVD files.

Options:
    -h, --help         Show this message and exit.
    --output <file>    Target PVD file [default: merged.pvd].
"""

import sys
import pathlib
import textwrap
from docopt import docopt

if __name__ == "__main__":
    args = docopt(__doc__)
    input_files = [pathlib.Path(f) for f in args["PVD"]]
    assert all([f.suffix == ".pvd" for f in input_files])

    run_folder = input_files[0].parent
    target = run_folder / pathlib.Path(args["--output"]).name

    header = """\
    <?xml version="1.0"?>
    <VTKFile type="Collection" version="1.0">
    """

    with target.open("w") as outfile:
        outfile.write(textwrap.dedent(header))
        outfile.write(textwrap.indent("<Collection>", "  "))

        for f in input_files:
            with f.open("r") as handle:
                content = handle.read()
                collection = content.split("<Collection>")[1]
                datasets = collection.split("</Collection>")[0]

                outfile.write(datasets)

        outfile.write(textwrap.indent("</Collection>\n", ""))
        outfile.write("</VTKFile>")
