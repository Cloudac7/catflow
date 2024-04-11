# CatFlow

> Parts of the code to be open source.

[![Python package](https://github.com/Cloudac7/CatFlow/actions/workflows/ci.yml/badge.svg)](https://github.com/Cloudac7/CatFlow/actions/workflows/ci.yml)
[![Coverage Status](https://coveralls.io/repos/github/Cloudac7/CatFlow/badge.svg?branch=master)](https://coveralls.io/github/Cloudac7/CatFlow?branch=master)


Machine learning aided catalysis reaction free energy calculation and post-analysis workflow, thus, analyzer for catalysis.

As is known to all, cat is fluid and thus cat flows. 🐱

> Former Miko-Analyzer and Miko-Tasker
> This repository is a temporary branch of original CatFlow.
> It would be merged into main repo after active refactor.

## Analyzer

### Installation

To install, clone the repository:

```
git clone https://github.com/chenggroup/catflow.git
```

and then install with `pip`:

```
cd catflow
pip install .
```

### Acknowledgement
This project is inspired by and built upon the following projects:
- [ai2-kit](https://github.com/chenggroup/ai2-kit): A toolkit featured artificial intelligence × ab initio for computational chemistry research.
- [DP-GEN](https://github.com/deepmodeling/dpgen): A concurrent learning platform for the generation of reliable deep learning based potential energy models.
- [ASE](https://wiki.fysik.dtu.dk/ase/): Atomic Simulation Environment.
- [DPDispatcher](https://github.com/deepmodeling/dpdispatcher): Generate and submit HPC jobs.
- [Metadynminer](https://github.com/spiwokv/metadynminer): Reading, analysis and visualization of metadynamics HILLS files produced by Plumed. As well as its Python implementation [Metadynminer.py](https://github.com/Jan8be/metadynminer.py).
- [stringmethod](https://github.com/apallath/stringmethod): Python implementation of the string method to compute the minimum energy path.

## Tasker

### Potential of Mean Force Calculation

A simple workflow designed for free energy calculation from Potential of Mean Force (PMF).

### Usage

#### Commandline

First, prepare a yaml file for workflow settings in detial. For example, `config.yaml`.


```yaml
job_config:
  work_path: "/some/place"
  machine_name: "machine_name"
  resources:
    number_node: 1
    cpu_per_node: 1
    gpu_per_node: 1
    queue_name: gpu
    group_size: 1
    module_list:
      - ...
    envs:
      ...
  command: "cp2k.ssmp -i input.inp"

  reaction_pair: [0, 1] # select indexes of atoms who would be constrained
  steps: 10000000 # MD steps
  timestep: 0.5 # unit: fs
  restart_steps: 10000000 # extra steps run in each restart
  dump_freq: 100 # dump frequency
  cell: [24.0, 24.0, 24.0] # set box size for initial structure
  type_map: # should be unified with DeePMD potential
    O: 0
    Pt: 1
  model_path: "/place/of/your/graph.pb"
  backward_files:
    - ...

flow_config:
  coordinates: ... # a list of coordinations to be constrained at
  t_min: 300.0 # under limit of simulation temperature
  cluster_component:
    - Pt # select elements of cluster
  lindemann_n_last_frames: 20000 # use last 20000 steps to judge convergence by calculate Lindemann index
  init_artifact:
    - coordinate: 1.4
      structure_path: "/place/of/your/initial_structure.xyz"
    - coordinate: 3.8
      structure_path: "/place/of/your/initial_structure.cif"
job_type: "dp_pmf" # dp_pmf when using DeePMD
```

Then, just type command like this:

```bash
catflow tasker pmf config.yaml
```

And enjoy it!
