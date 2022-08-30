import json
import os
import numpy as np

from pathlib import Path
from yaml import load, SafeLoader

from ase import Atoms
from ase.io import read, write
from collections import deque
from dpdata.unit import *
from pymatgen.core.periodic_table import Element

from miko.utils import logger
from miko.resources.submit import JobFactory
from miko_tasker.utils.file import tail
from miko_tasker.utils.cp2k import Cp2kInput, Cp2kInputToDict

_default_input_dict = {
    "GLOBAL": {
        "PROJECT": "pmf",
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


class PMFFactory(object):
    def __init__(
            self,
            param_file: str,
            machine_pool: str,
            conf_file: str
    ):
        self.param_file = Path(param_file).resolve()
        self.machine_pool = Path(machine_pool).resolve()
        self.conf_file = Path(conf_file).resolve()

        self.params = self.get_data(param_file)
        input_dict = self.params
        if input_dict is None:
            input_dict = _default_input_dict
        self.input_dict = input_dict

        self.machine = self.get_data(machine_pool)
        self.machine_name = self.machine["machine_name"]
        self.resource_dict = self.machine["resource"]
        self.command = self.machine["command"]

        workflow_settings = self.set_workflow(self.conf_file)
        self.workflow_settings = workflow_settings
        for key in workflow_settings.keys():
            setattr(self, key, workflow_settings[key])
        
        self.kwargs = workflow_settings.get("kwargs", {})
        self.task_map = self._task_map()

    @staticmethod
    def get_data(json_file):
        with open(json_file, 'r', encoding='utf-8') as fp:
            return json.load(fp)

    def set_workflow(self, conf_file):
        with open(conf_file) as f:
            conf = load(f, Loader=SafeLoader)
        return conf

    @property
    def init_structure(self):
        return read(self.init_structure_file)

    def generate(self):
        wf = PMFCalculation(
            reaction_coords=self.reaction_coords,
            temperatures=self.temperatures,
            reaction_pair=self.reaction_pair,
            init_structure=self.init_structure,
            work_base=self.work_base,
            machine_name=self.machine_name,
            resource_dict=self.resource_dict,
            command=self.command,
            input_dict=self.input_dict,
            **self.kwargs
        )
        return wf


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
        if kwargs:
            self.kwargs = kwargs
        else:
            self.kwargs = {}
        if input_dict is None:
            input_dict = _default_input_dict
        self.input_dict = input_dict
        self.task_map = self._task_map()

    @property
    def cell(self):
        cell = self.init_structure.cell.cellpar()[:3]
        if np.array_equiv(cell, np.array([0., 0., 0.])):
            return self.kwargs.get('cell')
        else:
            return cell

    def _task_map(self):
        len_temps, len_coords = len(
            self.temperatures), len(self.reaction_coords)
        return np.zeros([len_temps, len_coords], dtype=int)

    def run_workflow(self, **kwargs):
        work_path = os.path.abspath(self.work_base)
        if os.path.exists(os.path.join(work_path, 'task_map.out')):
            try:
                self.task_map = np.loadtxt(os.path.join(work_path, 'task_map.out'), dtype=int)
            except Exception:
                pass
        self._preprocess(**kwargs)
        self.task_iteration(work_path, kwargs.get('structures', self.init_structure), **kwargs)
        self._postprocess(**kwargs)

    def task_iteration(self, work_path, structures, coord_index=0, **kwargs):
        logger.info(f'Start generating task for {self.reaction_coords[coord_index]}')
        if isinstance(structures, Atoms):
            structures = [structures for i in self.temperatures]
        coordinate = self.reaction_coords[coord_index]
        np.savetxt(os.path.join(work_path, 'task_map.out'), self.task_map, fmt="%d")

        for i, temperature in enumerate(self.temperatures):
            if self.task_map[i, coord_index] == 0:
                # first loop
                coordinate = self.reaction_coords[coord_index]

                structure = structures[i]
                task_name = f'task.{coordinate}_{temperature}'
                init_task_path = os.path.join(work_path, task_name)
                self._task_preprocess(init_task_path, coordinate,
                                      temperature, structure, **kwargs)
                self._task_generate(init_task_path, coordinate,
                                    temperature, structure, **kwargs)
                self._task_postprocess(init_task_path, coordinate,
                                       temperature, structure, **kwargs)
                self.task_map[i, coord_index] = 1
            np.savetxt(os.path.join(work_path, 'task_map.out'), self.task_map, fmt="%d")

        task_list = []
        for i, temperature in enumerate(self.temperatures):
            if self.task_map[i, coord_index] == 1:
                # run tasks from generated
                task_name = f'task.{coordinate}_{temperature}'
                task_list.append(self._get_task_generated(task_name, **kwargs))
        
        if len(task_list) != 0:
            job = self.job_generator(task_list)
            logger.info("New Tasks submitting.")
            job.run_submission()
            for i, temperature in enumerate(self.temperatures):
                if self.task_map[i, coord_index] == 1:
                    self.task_map[:, coord_index] = 2
            np.savetxt(os.path.join(work_path, 'task_map.out'), self.task_map, fmt="%d")

        for i, temperature in enumerate(self.temperatures):
            if self.task_map[i, coord_index] == 2:
                num_atoms = len(structures[0].numbers)
                task_name = f'task.{coordinate}_{temperature}'
                init_task_path = os.path.join(work_path, task_name)
                logger.info('Reading end of file')
                with open(os.path.join(init_task_path, 'pmf-pos-1.xyz'), 'rb') as f:
                    last = tail(f, num_atoms + 2)
                logger.info('Making new file')
                with open(os.path.join(init_task_path, 'last.xyz'), 'wb') as output:
                    output.write(last)
                self.task_map[i, coord_index] = 3
            np.savetxt(os.path.join(work_path, 'task_map.out'), self.task_map, fmt="%d")

        next_structures = []
        for i, temperature in enumerate(self.temperatures):
            if self.task_map[i, coord_index] == 3:
                # turn to the next loop
                task_name = f'task.{coordinate}_{temperature}'
                init_task_path = os.path.join(work_path, task_name)
                next_structure = read(os.path.join(init_task_path, 'last.xyz'))
                next_structures.append(next_structure)
        logger.debug(f"{next_structures}")
        coord_index += 1
        try:
            self.task_iteration(
                work_path, next_structures, coord_index, **kwargs)
        except IndexError:
            logger.info('End loop')
            logger.info('Workflow finished!')
        self._postprocess()

    def _task_generate(self, init_task_path, coordinate, temperature, structure, **kwargs):
        os.makedirs(init_task_path, exist_ok=True)
        write(os.path.join(init_task_path, 'init.xyz'), structure)
        logger.info(f"Writing init.xyz to {init_task_path}")
        conv = LengthConversion("angstrom", "bohr")
        coord = coordinate * conv.value()

        input_dict = self.input_dict
        input_dict["MOTION"]["CONSTRAINT"]["COLLECTIVE"]["TARGET"] = coord
        input_dict["MOTION"]["MD"]["TEMPERATURE"] = temperature
        input_dict["FORCE_EVAL"]["SUBSYS"]["COLVAR"]["DISTANCE"]["ATOMS"] = ' '.join(
            [str(i + 1) for i in self.reaction_pair])
        cell = self.cell
        input_dict["FORCE_EVAL"]["SUBSYS"]["CELL"]["ABC"] = ' '.join(
            [str(i) for i in cell])
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
            ['pmf-1.ener', 'pmf.LagrangeMultLog',
             'pmf-pos-1.xyz', 'pmf-frc-1.xyz', 'output']
        )
        task_dict = {
            "command": self.command,
            "task_work_path": task_name,
            "forward_files": forward_files,
            "backward_files": backward_files,
            "outlog": kwargs.get('outlog', 'output'),
            "errlog": kwargs.get('errlog', 'err.log'),
        }
        logger.info(f"{task_name} generating")
        return task_dict

    def _preprocess(self, **kwargs):
        pass

    def _postprocess(self, **kwargs):
        pass

    def _task_preprocess(self, task_path, coordinate, temperature, structure, **kwargs):
        pass

    def _task_postprocess(self, task_path, coordinate, temperature, structure, **kwargs):
        pass

    def job_generator(self, task_dict_list):
        submission_dict = {
            "work_base": self.work_base,
            "forward_common_files": self.kwargs.get('forward_common_files', []),
            "backward_common_files": self.kwargs.get('backward_common_files', [])
        }
        return JobFactory(task_dict_list, submission_dict, self.machine_name, self.resource_dict)

    def check_point(self):
        pass

    def input_dict_from_file(self, template_file):
        _cp2k_input = Cp2kInputToDict(template_file)
        _input_dict = _cp2k_input.get_tree()
        self.input_dict = _input_dict


class DPPMFFactory(PMFFactory):
    def generate(self):
        wf = DPPMFCalculation(
            reaction_coords=self.reaction_coords,
            temperatures=self.temperatures,
            reaction_pair=self.reaction_pair,
            init_structure=self.init_structure,
            work_base=self.work_base,
            machine_name=self.machine_name,
            resource_dict=self.resource_dict,
            command=self.command,
            input_dict=self.input_dict,
            **self.kwargs
        )
        wf.type_map = self.workflow_settings["type_map"]
        return wf


class DPPMFCalculation(PMFCalculation):
    @property
    def type_map(self):
        return self._type_map

    @type_map.setter
    def type_map(self, type_map: dict):
        """set type map for workflow.

        Parameters
        ----------
        type_map : dict
            map dict from each element to type index from DP potential file, e.g.: {"H": 0, "O": 1}.
        Returns
        -------

        """
        self._type_map = type_map

    def _link_model(self, model_path):
        model_abs_path = os.path.abspath(model_path)
        work_abs_base = os.path.abspath(self.work_base)
        os.symlink(model_abs_path, os.path.join(work_abs_base, 'graph.pb'))
        fwd_files_flag = self.kwargs.get("forward_common_files", False)
        if fwd_files_flag is False:
            self.kwargs["forward_common_files"] = ['graph.pb']
        else:
            self.kwargs["forward_common_files"] += ['graph.pb']

    def _make_charge_dict(self, structure):
        element_set = set(structure.symbols)
        self.input_dict["FORCE_EVAL"]["MM"]["FORCEFIELD"]["CHARGE"] = []
        for element in element_set:
            if Element(element).is_metal:
                _charge_dict = {
                    "ATOM": element,
                    "CHARGE": 0.0
                }
                self.input_dict["FORCE_EVAL"]["MM"]["FORCEFIELD"]["CHARGE"].append(_charge_dict)

    def _make_deepmd_dict(self, structure):
        element_set = set(structure.symbols)
        self.input_dict["FORCE_EVAL"]["MM"]["FORCEFIELD"]["NONBONDED"]["DEEPMD"] = []
        for element in element_set:
            try:
                _deepmd_dict = {
                    "ATOMS": element + ' ' + element,
                    "POT_FILE_NAME": "../graph.pb",
                    "ATOM_DEEPMD_TYPE": self.type_map[element]
                }
            except KeyError:
                raise KeyError("Please provide type_map property")
            self.input_dict["FORCE_EVAL"]["MM"]["FORCEFIELD"]["NONBONDED"]["DEEPMD"].append(_deepmd_dict)

    def _preprocess(self, **kwargs):
        self._link_model(self.kwargs.get('model_name'))

    def _task_preprocess(self, task_path, coordinate, temperature, structure, **kwargs):
        self._make_charge_dict(structure)
        self._make_deepmd_dict(structure)
