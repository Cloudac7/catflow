# miko-tasker
A scheduler plugin for miko-analyzer to generate tasks.

## Installation

```bash
git clone https://github.com/Cloudac7/miko-tasker.git
pip install .
```

## Potential of Mean Force Calculation

A simple workflow designed for free energy calculation from Potential of Mean Force (PMF).

### Usage

For a quick start, just configure your machine and write the Python script, naming as `run.py` for example, like below:

```python
from multiprocessing import Pool
from ase.io import read
from miko_tasker.workflow.pmf import DPPMFCalculation
import numpy as np


resource_dict = {
    "number_node": 1,
    "cpu_per_node": 4,
    "gpu_per_node": 1,
    "kwargs": {
      "gpu_usage": True,
      "gpu_new_syntax": True,
      "gpu_exclusive": True
    },
    "custom_flags": [
      "#BSUB -J PMF",
      "#BSUB -W 240:00",
      "#BSUB -m mgt"
    ],
    "queue_name": "gpu",
    "para_deg": 2,
    "group_size": 2,
    "module_list": ["deepmd/2.0"]
}

input_dict = {
    "GLOBAL": {
        "PROJECT": 'pmf',
        "RUN_TYPE": "MD"
    },
    "FORCE_EVAL": {
        "METHOD": "FIST",
        "PRINT": {
            "FORCES": {
                "_": "ON",
                "EACH": {}
            }
        },
        "MM": {
            "FORCEFIELD": {
                "CHARGE": [],
                "NONBONDED": {
                    "DEEPMD": [
                    ]
                },
                "IGNORE_MISSING_CRITICAL_PARAMS": True
            },
            "POISSON": {
                "EWALD": {
                    "EWALD_TYPE": "none"
                }
            }
        },
        "SUBSYS": {
            "COLVAR": {
                "DISTANCE": {
                    "ATOMS": None
                }
            },
            "CELL": {
                "ABC": None
            },
            "TOPOLOGY": {}
        }
    },
    "MOTION": {
        "CONSTRAINT": {
            "COLLECTIVE": {
                "TARGET": None,
                "INTERMOLECULAR": True,
                "COLVAR": 1
            },
            "LAGRANGE_MULTIPLIERS": {
                "_": "ON",
                "COMMON_ITERATION_LEVELS": 8000000
            }
        },
        "MD": {
            "ENSEMBLE": "NVT",
            "STEPS": 8000000,
            "TIMESTEP": 0.5,
            "TEMPERATURE": None,
            "THERMOSTAT": {
                "TYPE": "CSVR",
                "CSVR": {
                    "TIMECON": 1000,
                }
            }
        },
        "PRINT": {
            "TRAJECTORY": {
                "EACH": {
                    "MD": 1
                }
            },
            "FORCES": {
                "EACH": {
                    "MD": 1
                }
            },
            "RESTART_HISTORY": {
                "EACH": {
                    "MD": 500000
                }
            }
        }
    }
}
temperatures = [500.0, 600.0, 700.0, 750.0, 800.0, 900.0, 1000.0]
reaction_coords = [3.8, 3.7, 3.6, 3.4, 3.2, 3.0, 2.8, 2.6, 2.4, 2.2, 2.0, 1.9, 1.8, 1.7, 1.6, 1.5, 1.4]

wf = DPPMFCalculation(
    reaction_coords=reaction_coords,
    temperatures=temperatures,
    reaction_pair=(0, 1),
    init_structure=read('POSCAR'),
    work_base='3.8',
    machine_name='ChenglabHPC',
    resource_dict=resource_dict,
    command='cp2k.sopt -i input.inp',
    input_dict=input_dict,
    model_name="graph.000.pb"
)
wf.type_map = {"O": 0, "Pt": 1}
wf.run_workflow()
```

Here, `reaction_pair` means the index of the atoms pair that the reaction happens between and `init_structure` is a `ase.Atoms` instance. `input_dict` could be provided with some special parameters changed or even not given for workflow to pick it from default dict.

With everything prepaired, we could just run:

```bash
python run.py
```
to start the workflow runs.

Enjoy it!
