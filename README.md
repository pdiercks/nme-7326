# Source Code for *Multiscale modeling of linear elastic heterogeneous structures via localized model order reduction*

This repository contains the source code for the article [*Multiscale modeling of linear elastic heterogeneous structures via localized model order reduction*](https://doi.org/10.1002/nme.7326).
In this paper a multiscale approach to modeling linear elastic
heterogeneous structures is presented.
Key points are:
* additive split into coarse and fine scale solution,
* physically informed boundary conditions in the oversampling problem used to construct
local reduced spaces,
* continuos coupling of fine-scale edge modes,
* proof of concept for the proposed methodology using specific examples.

## Publication in IJNME (Wiley)
The manuscript is published in the International Journal for Numerical Methods in Engineering (Wiley)
under a CC-BY license and open access.

## Make this paper

Welcome! Besides all the code for the paper, 
the python package `multicode` and suitable software environments are necessary to
*reproduce* this paper.

The source code for the paper can be obtained via
```
git clone git@github.com:pdiercks/nme-7326.git
```
The source code for the `multi` package (multicode) can be obtained via
```
git clone git@github.com:pdiercks/multicode.git
```
Note that a specific version of the multicode package (tag [nme-7326](https://github.com/pdiercks/multicode/releases/tag/nme-7326)) is required.
If you downloaded an archive from [arXiv](https://arxiv.org/abs/2201.10374), then the folder `multicode` should already exist
besides the source code for the paper.

The workflow is divided into several
workflow implementations, which require different compute environments.
* `wf_preprocessing.py` mesh generation for all examples (docker image `pdiercks/multix:latest`),
* `wf_block.py` tasks for the block example (docker image `pdiercks/multix:latest`),
* `wf_beam.py` tasks for the beam example (docker image `pdiercks/multix:latest`),
* `wf_lpanel.py` tasks for the lpanel example (docker image `pdiercks/multix:latest`),
* `wf_postproc.py` containing tasks for postprocessing (docker image `pdiercks/multix:latest`),
* `wf_pv.py` postprocessing tasks using paraview (conda env, see `envs/paraview_v5.11.0.yaml`),
* `wf_tex.py` tex related tasks to finally compile the PDF (conda env, see `envs/tex.yaml`).

To build the paper run something like the following.
Using `udocker` first create the container.
```
udocker create --name=<container-name> <repo/image:tag> 
```
Then run the container (make sure to bind the source code for the paper and for the `multicode` package).
Assuming in the current working directory you have the code for the paper under `$PWD/paper`
and the code for the `multicode` package under `$PWD/multicode`
```
udocker run -v $PWD/paper:/mnt/paper -v $PWD/multicode:/mnt/multicode <container-name>
```
In the container we then install `multicode` first
```
$PYTHON -m pip install /mnt/multicode/multi
```
Each part of the workflow can be run either in `DEBUG` or in `PAPER` mode
for a fixed number of realizations. In the `DEBUG` mode the same
examples are run, but with a small number of coarse grid elements (subdomains).
First, build all grids with (assuming the root of the paper as
cwd, that is `/mnt/paper` in the container)
```
doit -f wf_preprocessing.py mode=PAPER nreal=10 run
```
The examples are run accordingly.
```
doit -f wf_block.py mode=PAPER nreal=10 run
doit -f wf_beam.py mode=PAPER nreal=10 run
doit -f wf_lpanel.py mode=PAPER nreal=10 run
```
This will take a while. 
The post-processing can be done with
```
doit -f wf_postproc.py mode=PAPER nreal=10 run
```
Now, we only need to make some plots using the paraview conda environment 
mentioned above
```
doit -f wf_pv.py run
```
and compile the final tex document.
```
doit -f wf_tex.py run
```
