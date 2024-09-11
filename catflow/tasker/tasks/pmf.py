import asyncio
import json
import os
import numpy as np

from collections import deque, defaultdict
from pathlib import Path
from ruamel.yaml import YAML
from typing import Union, List, Optional, Dict, Iterable

from ase import Atoms
from ase.io import read, write
from dpdata.unit import LengthConversion
from pymatgen.core.periodic_table import Element
from pydantic import BaseModel
from ai2_kit.core.executor import HpcExecutor

from catflow.utils import logger
from catflow.tasker.resources.submit import JobFactory
from catflow.utils.file import tail
from catflow.utils.cp2k import Cp2kInput, Cp2kInputToDict


def pmf_analyzer(
    task_path: Path,
    restart_count: int = 0
):
    from catflow.utils.cp2k import lagrange_mult_log_parser
    from catflow.utils.statistics import block_average

    coordinate = task_path.name.split('-')[2]
    temperature = task_path.name.split('-')[3]
    lagrange_mults = []
    
    if restart_count > 0:
        for i in range(restart_count):
            _task_path = \
                task_path.parent / f"task-{str(i).zfill(3)}-{coordinate}-{temperature}"
            lagrange_mult_log_path = _task_path / "pmf.LagrangeMultLog"
            new_lagrange_mults = lagrange_mult_log_parser(lagrange_mult_log_path)
            lagrange_mults += new_lagrange_mults
    
    mean, var = block_average(
        lagrange_mults[1:], int(len(lagrange_mults[1:])/5)
    )
    return mean, var


class PMFJobFactory(JobFactory):

    def process_input(self, base_dir: Path, **inputs) -> List[Path]:
        restart_count = inputs.get('restart_count', 0)
        task_dir = base_dir / \
            f"task-{restart_count}-{inputs['coordinate']}-{inputs['temperature']}"
        task_dir.mkdir(exist_ok=True)

        if restart_count == 0:
            # create new task
            # write init.xyz
            structure = inputs["structure"]
            structure.write(task_dir / 'init.xyz')
            logger.info(f"Writing init.xyz to {task_dir}")

            # write input.inp
            with open(task_dir / 'input.inp', 'w') as f:
                output = Cp2kInput(params=inputs['input_dict']).render()
                f.write(output)
        else:
            import shutil

            # restart from previous task
            restart_file_list = ['init.xyz', 'pmf-1.restart']
            last_task_path = \
                base_dir / f"task-{restart_count - 1}-{inputs['coordinate']}-{inputs['temperature']}"
            for file_name in restart_file_list:
                shutil.copy(last_task_path / file_name, task_dir / file_name)
            restart_file = task_dir / 'pmf-1.restart'
            restart_file.rename(task_dir / 'pmf-1.original.restart')
            logger.info("Copyed files from previous task")

            input_dict = inputs['input_dict']

            input_dict["MOTION"]["MD"]["STEPS"] = inputs.get(
                'restart_steps', 10000000)
            if input_dict.get("EXT_RESTART"):
                input_dict["EXT_RESTART"]["RESTART_FILE_NAME"] = 'pmf-1.original.restart'
            else:
                input_dict["EXT_RESTART"] = {
                    "RESTART_FILE_NAME": 'pmf-1.original.restart'
                }
            with open(task_dir / 'input.inp', 'w') as f:
                output = Cp2kInput(params=input_dict).render()
                f.write(output)
        logger.info(f"Writing input.inp to {task_dir}")

        if inputs.get('common_files'):
            common_files = inputs['common_files']
            for file in common_files:
                shutil.copy(file, task_dir)
                logger.info(f"Copying {file} to {task_dir}")

        return [task_dir]

    def process_output(self, task_dirs: List[Path], **inputs):
        results = []
        for task_dir in task_dirs:
            result = {}
            result['task_dir'] = task_dir

            self.get_last_frame(task_dir)
            result['last_frame'] = task_dir / 'last.xyz'

            # parse pmf.LagrangeMultLog
            mean, var = pmf_analyzer(task_dir, restart_count=inputs.get('restart_count', 0))
            result['mean'] = mean
            result['var'] = var

            results.append(result)
        return results

    @staticmethod
    def get_last_frame(task_dir: Path):
        init_structure = read(task_dir / 'init.xyz')
        num_atoms = init_structure.get_number_of_atoms() # type: ignore
        logger.info('Reading end of file')
        with open(task_dir / 'pmf-pos-1.xyz', 'rb') as f:
            last = tail(f, num_atoms + 2)
        last_path = task_dir / 'last.xyz'
        logger.info(f'Writing last frame to {last_path}')
        with open(last_path, 'wb') as output:
            output.write(last)

class PMFTask(object):
    """
    Workflow for potential of mean force calculation.
    """

    def __init__(
            self,
            coordinate,
            temperature,
            init_structure_path: str,
            reaction_pair: Union[list, tuple],
            work_path: str,
            executor: Dict,
            command: str,
            input_dict=None,
            script_header: str = "",
            restart_count: int = 0,
            task_prefix: str = 'pmf',
            concurrency: int = 1,
            **kwargs
    ):
        self.coordinate = coordinate
        self.temperature = temperature
        self.reaction_pair = reaction_pair
        self.init_structure_path = Path(init_structure_path).resolve()
        self.init_structure = read(self.init_structure_path)
        self.work_path = Path(work_path).resolve()
        if not self.work_path.exists():
            os.makedirs(self.work_path)
        self.executor = executor
        self.command = command
        if kwargs:
            self.kwargs = kwargs
        else:
            self.kwargs = {}
        if input_dict is None:
            from .defaults.pmf import _default_input_dict
            input_dict = _default_input_dict
        self.input_dict = input_dict
        self.script_header = script_header
        self.restart_count = restart_count
        self.task_prefix = task_prefix
        self.concurrency = concurrency

    @property
    def cell(self):
        init_structure = self.init_structure
        if isinstance(init_structure, list):
            raise TypeError("Initial structure should be an Atoms instance")
        # define cell with x, y, z and alpha, beta, gamma
        if self.kwargs.get('cell') is None:
            cell = init_structure.cell.cellpar()
        else:
            cell = self.kwargs.get('cell')
            cell = np.array(cell)
        if cell.shape == (6,):
            pass
        elif cell.shape == (3, 3):
            from ase.geometry import Cell
            cell = Cell(cell).cellpar()
        elif cell.shape == (3,):
            # if cell is defined with x, y, z
            # set alpha, beta, gamma to 90
            cell = np.append(cell, [90, 90, 90])
        else:
            raise ValueError(
                'Cell should be defined with x, y, z (and alpha, beta, gamma).')
        return cell

    def task_generate(self):
        structure = self.init_structure
        coordinate = self.coordinate
        temperature = self.temperature
        input_dict = self._generate_input(coordinate, temperature)

        job = PMFJobFactory(executor=HpcExecutor.from_config(self.executor))
        return job

    def _generate_input(self, coordinate, temperature):
        conv = LengthConversion("angstrom", "bohr")
        coord = coordinate * conv.value()

        input_dict = self.input_dict

        # set job name
        input_dict["GLOBAL"]["PROJECT"] = "pmf"

        # set motion
        input_dict["MOTION"]["CONSTRAINT"]["COLLECTIVE"]["TARGET"] = coord
        input_dict["MOTION"]["MD"]["TEMPERATURE"] = temperature
        input_dict["FORCE_EVAL"]["SUBSYS"]["COLVAR"]["DISTANCE"]["ATOMS"] = \
            ' '.join([str(i + 1) for i in self.reaction_pair])
        a_b_c = self.cell[:3]
        alpha_beta_gamma = self.cell[3:]
        input_dict["FORCE_EVAL"]["SUBSYS"]["CELL"] = {
            "ABC": " ".join([str(i) for i in a_b_c]),
            "ALPHA_BETA_GAMMA": " ".join([str(i) for i in alpha_beta_gamma])
        }
        input_dict["FORCE_EVAL"]["SUBSYS"]["TOPOLOGY"] = {
            "COORD_FILE_FORMAT": "XYZ",
            "COORD_FILE_NAME": "init.xyz"
        }
        steps = self.kwargs.get('steps')
        timestep = self.kwargs.get('timestep')
        if steps:
            input_dict["MOTION"]["MD"]["STEPS"] = steps
            input_dict["MOTION"]["CONSTRAINT"]["LAGRANGE_MULTIPLIERS"][
                "COMMON_ITERATION_LEVELS"] = steps
        if timestep:
            input_dict["MOTION"]["MD"]["TIMESTEP"] = timestep  # unit: fs

        dump_freq = self.kwargs.get('dump_freq')
        if dump_freq:
            md_print = input_dict.setdefault(
                "MOTION", {}).setdefault("PRINT", {})
            md_print.setdefault("TRAJECTORY", {}).setdefault(
                "EACH", {}).setdefault("MD", dump_freq)
            md_print.setdefault("FORCES", {}).setdefault(
                "EACH", {}).setdefault("MD", dump_freq)
            input_dict["MOTION"]["PRINT"]["TRAJECTORY"]["EACH"]["MD"] = dump_freq
            input_dict["MOTION"]["PRINT"]["FORCES"]["EACH"]["MD"] = dump_freq
        return input_dict

    def check_point(self):
        pass

    def input_dict_from_file(self, template_file):
        _cp2k_input = Cp2kInputToDict(template_file)
        _input_dict = _cp2k_input.get_tree()
        self.input_dict = _input_dict


class DPPMFTask(PMFTask):
    """calculate the free energy of a reaction coordinate using DeePMD-kit
    """

    def type_map(self) -> dict:
        type_map = self.kwargs.get("type_map")
        if isinstance(type_map, dict):
            return type_map
        else:
            structure = self.init_structure
            if isinstance(structure, list):
                raise TypeError(
                    "Initial structure should be an Atoms instance")
            element_set = set(structure.symbols)
            type_map = {}
            for i, element in enumerate(element_set):
                type_map[element] = i
        return type_map

    def _link_model(self, model_path):
        model_abs_path = Path(model_path).resolve()
        graph_file = self.work_path / 'graph.pb'
        if not graph_file.exists():
            graph_file.symlink_to(model_abs_path)
        else:
            logger.info("model symlink exists")
        fwd_files_flag = self.kwargs.get("forward_common_files")
        if fwd_files_flag:
            self.kwargs["forward_common_files"] += ['graph.pb']
        else:
            self.kwargs["forward_common_files"] = ['graph.pb']

    def _make_charge_dict(self, structure):
        element_set = set(structure.symbols)
        self.input_dict["FORCE_EVAL"]["MM"]["FORCEFIELD"]["CHARGE"] = []
        for element in element_set:
            if Element(element).is_metal:
                _charge_dict = {
                    "ATOM": element,
                    "CHARGE": 0.0
                }
                self.input_dict["FORCE_EVAL"]["MM"]["FORCEFIELD"]["CHARGE"].append(
                    _charge_dict)

    def _make_deepmd_dict(self, structure):
        element_set = set(structure.symbols)
        self.input_dict["FORCE_EVAL"]["MM"]["FORCEFIELD"]["NONBONDED"][
            "DEEPMD"] = []
        for element in element_set:
            try:
                type_map = self.type_map()
                _deepmd_dict = {
                    "ATOMS": element + ' ' + element,
                    "POT_FILE_NAME": "../graph.pb",
                    "ATOM_DEEPMD_TYPE": type_map[element]
                }
            except KeyError:
                raise KeyError("Please provide type_map property")
            self.input_dict["FORCE_EVAL"]["MM"]["FORCEFIELD"]["NONBONDED"][
                "DEEPMD"].append(_deepmd_dict)

    def _task_preprocess(self):
        structure = self.init_structure
        self._make_charge_dict(structure)
        self._make_deepmd_dict(structure)

    def _task_postprocess(self):
        self._link_model(self.kwargs.get('model_path'))
