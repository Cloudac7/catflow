import os
import numpy as np

from ase.io import read, write
from collections import deque
from dpdata.unit import *
from multiprocessing import Pool
from pymatgen.core.periodic_table import Element

from miko.utils import logger
from miko.resources.submit import JobFactory
from miko_tasker.utils.cp2k import Cp2kInput


class PMFCalculation(object):
    """
    Workflow for potential of mean force calculation.
    """

    def __init__(
            self,
            reaction_coords,
            temperatures,
            reaction_pair,
            init_structure,
            work_base: str,
            machine_name: str,
            resource_dict: dict,
            command: str,
            input_dict=None,
            **kwargs
    ):
        self.reaction_coords = reaction_coords
        self.temperatures = temperatures
        self.reaction_pair = reaction_pair
        self.init_structure = init_structure
        self.work_base = work_base
        self.machine_name = machine_name
        self.resource_dict = resource_dict
        self.command = command
        self.input_dict = self.set_input_dict(input_dict)
        self.kwargs = kwargs
        self.task_map = self._task_map()

    @property
    def cell(self):
        cell = self.init_structure.cell.cellpar()[:3]
        if np.array_equiv(cell, np.array([0., 0., 0.])):
            return self.kwargs.get('cell')
        else:
            return cell

    def _task_map(self):
        len_temps, len_coords = len(self.temperatures), len(self.reaction_coords)
        return np.zeros(len_temps, len_coords)

    def run_workflow(self, **kwargs):
        work_path = os.path.abspath(self.work_base)
        if os.path.exists(os.path.join(work_path, 'task_map.out')):
            try:
                self.task_map = np.loadtxt('task_map.txt')
            except Exception:
                pass
        p = Pool(len(self.temperatures))
        for i, temp in enumerate(self.temperatures):
            logger.info(f'Generating subprocesses for {temp} K')
            p.apply_async(
                self.task_iteration,
                kwds={
                    "work_path": work_path,
                    "temp_index": i,
                    "structure": self.init_structure,
                    "kwargs": kwargs
                }
            )
        logger.info('All subprocesses submitted')
        p.close()
        p.join()
        logger.info('All subprocesses done.')

    def task_iteration(self, work_path, structure, temp_index, coord_index=0, **kwargs):
        np.savetxt(self.task_map, os.path.join(work_path, 'task_map.out'))
        coordinate = self.reaction_coords[coord_index]
        temperature = self.temperatures[temp_index]
        num_atoms = len(structure.numbers)
        init_task_name = f'task.{coordinate}_{temperature}'
        init_task_path = os.path.join(work_path, init_task_name)

        if self.task_map[temp_index, coord_index] == 0:
            # first loop
            coordinate = self.reaction_coords[coord_index]
            temperature = self.temperatures[temp_index]
            self._task_generate(init_task_path, coordinate, temperature, structure, **kwargs)
            self.task_map[temp_index, coord_index] = 1

        if self.task_map[temp_index, coord_index] == 1:
            # run tasks from
            task_list = [self._get_task_generated(init_task_path, **kwargs)]
            job = self.job_generator(task_list)
            logger.info("First pile of tasks submitting.")
            job.run_submission()
            self.task_map[temp_index, coord_index] = 2

        if self.task_map[temp_index, coord_index] == 2:
            # prepare for the next loop
            with open(os.path.join(init_task_path, 'pmf-pos-1.xyz')) as f:
                last = deque(f, num_atoms + 2)
            with open(os.path.join(init_task_path, 'last.xyz'), 'w') as output:
                for q in last:
                    output.write(str(q))
            self.task_map[temp_index, coord_index] = 3

        if self.task_map[temp_index, coord_index] == 3:
            # turn to the next loop
            next_structure = read(os.path.join(init_task_path, 'last.xyz'))
            coord_index += 1
            try:
                self.task_iteration(work_path, temp_index, next_structure, coord_index, **kwargs)
            except IndexError:
                logger.info('End loop')
                logger.info('Workflow finished!')

    def _task_generate(self, init_task_path, coordinate, temperature, structure, **kwargs):
        os.makedirs(init_task_path, exist_ok=True)
        write(os.path.join(init_task_path, 'init.xyz'), structure)
        logger.info(f"Writing init.xyz to {init_task_path}")
        conv = LengthConversion("angstrom", "bohr")
        coord = coordinate * conv.value()

        input_dict = self.input_dict
        input_dict["MOTION"]["CONSTRAINT"]["TARGET"] = coord
        input_dict["MOTION"]["MD"]["TEMPERATURE"] = temperature
        input_dict["MOTION"]["CONSTRAINT"]["INTERMOLECULAR"] = ' '.join(self.reaction_pair)
        cell = self.cell
        input_dict["FORCE_EVAL"]["SUBSYS"]["CELL"]["ABC"] = ' '.join(cell)
        input_dict["FORCE_EVAL"]["SUBSYS"]["TOPOLOGY"] = {
            "COORD_FILE_FORMAT": "XYZ",
            "COORD_FILE_NAME": "init.xyz"
        }
        steps = kwargs.get('steps', 0)
        timestep = kwargs.get('timestep', 0)
        if steps:
            input_dict["MOTION"]["MD"]["STEPS"] = steps
            input_dict["MOTION"]["CONSTRAINT"]["LAGRANGE_MULTIPLIERS"]["COMMON_ITERATION_LEVELS"] = steps
        if timestep:
            input_dict["MOTION"]["MD"]["TIMESTEP"] = timestep * 1e3

        with open(os.path.join(init_task_path, 'input.inp'), 'w') as f:
            output = Cp2kInput(params=input_dict).render()
            f.write(output)
        logger.info(f"Writing input.inp to {init_task_path}")

    def _get_task_generated(self, task_name, **kwargs):
        forward_files = kwargs.get('forward_files', [])
        forward_files += ['input.inp', 'init.xyz']
        backward_files = kwargs.get(
            'backward_files',
            ['pmf-1.ener', 'pmf.LagrangeMultLog', 'pmf-pos-1.xyz', 'pmf-frc-1.xyz', 'output']
        )
        task_dict = {
            "command": self.command,
            "task_work_path": task_name,
            "forward_files": forward_files,
            "backward_files": backward_files,
            "outlog": kwargs.get('outlog', 'output'),
            "errlog": kwargs.get('errlog', 'err.log'),
        }
        logger.info(f"Task {task_dict} generating")
        return task_dict

    def set_input_dict(self, input_dict):
        _default_input_dict = {
            "GLOBAL": {
                "PROJECT": self.kwargs.get('project_name', 'pmf'),
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
                            "DEEPMD": []
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
                        "COMMON_ITERATION_LEVELS": 20000000
                    }
                },
                "MD": {
                    "ENSEMBLE": "NVT",
                    "STEPS": 20000000,
                    "TIMESTEP": 0.5,
                    "TEMPERATURE": None,
                    "THERMOSTAT": {
                        "NOSE": {
                            "LENGTH": 3,
                            "YOSHIDA": 3,
                            "TIMECON": 1000,
                            "MTS": 2
                        }
                    }
                },
                "PRINT": {
                    "TRAJECTORY": {
                        "EACH": {
                            "MD": 1
                        }
                    },
                    "FORCES": {},
                    "RESTART_HISTORY": {
                        "EACH": {
                            "MD": 200000
                        }
                    }
                }
            }
        }
        if input_dict is None:
            input_dict = _default_input_dict
        return input_dict

    def job_generator(self, task_dict_list):
        submission_dict = {
            "work_base": self.work_base,
            "forward_common_files": self.kwargs.get('forward_common_files', []),
            "backward_common_files": self.kwargs.get('backward_common_files', [])
        }
        return JobFactory(task_dict_list, submission_dict, self.machine_name, self.resource_dict)

    def check_point(self):
        pass
