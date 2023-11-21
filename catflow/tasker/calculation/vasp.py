import os
import shutil

from copy import deepcopy
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from catflow.utils.log_factory import logger
from catflow.tasker.resources.submit import JobFactory


def trajectory_checkrun_vasp(
        traj_file: str, 
        work_path: str, 
        chemical_symbol: Optional[str] = None, 
        cell: Optional[Union[List, np.ndarray]] = None,
        index: str = "::",
        **task_generation_params
    ):
    """Exports VASP POSCAR files from a trajectory.

    Args:
        traj_file (str): The trajectory file to read.
        work_path (str): The working directory to write the POSCAR files to.
        chemical_symbol (str): The chemical symbol to use for the atoms in the POSCAR files.
        index (str): The index of the frame to use.

    Returns:
        None
    """
    from ase.io import iread, write

    stcs = iread(os.path.abspath(traj_file), index=index)    
    os.makedirs(work_path, exist_ok=True)

    for i, j in enumerate(stcs):
        if chemical_symbol:
            j.set_chemical_symbols(chemical_symbol)
        if cell is not None:
            j.set_cell(cell)
        task_path = os.path.join(work_path, f'task.{str(i).zfill(6)}')
        os.makedirs(task_path, exist_ok=True)
        write(os.path.join(task_path, 'POSCAR'), j, vasp5=True) # type: ignore
    if task_generation_params:
        submission = multi_fp_task(work_path=work_path, **task_generation_params)
        return submission


def multi_fp_task(
        work_path, 
        fp_command, 
        machine_name, 
        resource_dict, 
        files_to_forward: Optional[List[str]] = None,
        **kwargs
    ):
    """Run multiple VASP tasks in parallel.

    Args:
        work_path (_type_): _description_
        fp_command (_type_): _description_
        machine_name (_type_): _description_
        resource_dict (_type_): _description_
    """
    forward_files = kwargs.get('forward_files', [])
    forward_files += ['POSCAR', 'INCAR', 'POTCAR']
    forward_files = list(set(forward_files))

    backward_files = kwargs.get('backward_files', [])
    backward_files += ['OUTCAR', 'vasprun.xml', 'fp.log', 'fp.err']
    backward_files = list(set(backward_files))

    task_dir_pattern = kwargs.get('task_dir_pattern', 'task.*')
    fp_tasks = glob(os.path.join(work_path, task_dir_pattern))
    fp_tasks.sort()
    if len(fp_tasks) == 0:
        return
    if files_to_forward is not None:
        for ii in fp_tasks:
            for jj in files_to_forward:
                shutil.copy(jj, ii)
    fp_run_tasks = fp_tasks
    run_tasks = [os.path.basename(ii) for ii in fp_run_tasks]

    task_dict_list = [
        {
            "command": fp_command,
            "task_work_path": task,
            "forward_files": forward_files,
            "backward_files": backward_files,
            "outlog": "fp.log",
            "errlog": "fp.err",
        } for task in run_tasks
    ]

    submission_dict = {
        "work_base": work_path,
        "forward_common_files": kwargs.get('forward_common_files', []),
        "backward_common_files": kwargs.get('backward_common_files', [])
    }
    job = JobFactory(task_dict_list, submission_dict, machine_name, resource_dict, group_size=1)
    submission = job.submission
    return submission.run_submission(exit_on_submit=True)


def cell_tests(
        init_structure_path,
        work_path,
        cell_list,
        cubic=True,
        required_files=None,
        **task_generation_params
):
    """Run VASP tasks with different cell sizes.
    """

    from ase.io import read, write

    work_path = Path(work_path).resolve()
    if required_files is None:
        required_files = {
            "incar_path": os.path.join(work_path, "INCAR"),
            "potcar_path": os.path.join(work_path, "POTCAR")
        }
    if cubic is False:
        for c in cell_list:
            if np.array(c).shape != (3, 3):
                raise Exception("should provide 3 vectors as cell")
    init_structure = read(os.path.abspath(init_structure_path))
    for idx, cell in enumerate(cell_list):
        task_path = os.path.join(os.path.abspath(work_path), f'task.{str(idx).zfill(3)}')
        os.makedirs(task_path, exist_ok=True)
        s = deepcopy(init_structure)
        s.set_cell(cell) # type: ignore
        s.set_pbc([1, 1, 1]) # type: ignore
        write(os.path.join(task_path, 'POSCAR'), s, vasp5=True) # type: ignore
        assert ('incar_path' in required_files.keys()) & ('potcar_path' in required_files.keys())
        for path in required_files.values():
            shutil.copy(
                path,
                os.path.join(task_path, os.path.basename(path))
            )
    multi_fp_task(work_path=work_path, **task_generation_params)


"""
import json
import logging
import time
import uuid

def fp_tasks(ori_fp_tasks, work_path, machine_data, group_size=1):
    forward_files = ['POSCAR', 'INCAR', 'POTCAR']
    backward_files = ['OUTCAR', 'vasprun.xml', 'fp.log', 'fp.err', 'tag_0_finished']
    forward_common_files = []
    if len(ori_fp_tasks) == 0:
        return
    fp_run_tasks = []
    for task in ori_fp_tasks:
        _task_group = glob(task)
        _task_group.sort()
        fp_run_tasks += _task_group
    work_path = os.path.join(work_path, 'groups')
    if os.path.exists(os.path.join(work_path, 'groups.json')):
        with open(os.path.join(work_path, 'groups.json'), 'r') as f:
            _groups = json.load(f)
    else:
        _group_num = 0
        _groups = []
        _groups_index = 0
        _uuid = str(uuid.uuid4())
        _work_dir = os.path.join(work_path, _uuid)
        os.makedirs(os.path.join(_work_dir), exist_ok=True)
        for tt in fp_run_tasks:
            if not os.path.exists(os.path.join(tt, 'tag_0_finished')):
                if _group_num >= group_size:
                    _group_num = 0
                    _groups_index += 1
                    _uuid = str(uuid.uuid4())
                    _work_dir = os.path.join(work_path, _uuid)
                    os.makedirs(os.path.join(_work_dir), exist_ok=True)
                _base_name = os.path.basename(tt)
                os.symlink(tt, os.path.join(_work_dir, _base_name))
                try:
                    _groups[_groups_index]['run_tasks'].append(_base_name)
                except IndexError:
                    _group_item = {
                        "uuid": _uuid,
                        "work_dir": _work_dir,
                        "run_tasks": []
                    }
                    _groups.append(_group_item)
                    _groups[_groups_index]['run_tasks'].append(_base_name)
                _group_num += 1
        with open(os.path.join(work_path, 'groups.json'), 'w') as f:
            json.dump(_groups, f)
    with daemon.DaemonContext():
        p = Pool(len(_groups))
        for i, item in enumerate(_groups):
            time.sleep(i)
            p.apply_async(fp_await_submit, args=(
                item,
                forward_common_files,
                forward_files,
                backward_files,
                machine_data
            ))
        logger.info('Waiting for all tasks done...')
        p.close()
        p.join()
        shutil.rmtree(work_path)


def fp_await_submit(item, forward_common_files=None, forward_files=None, backward_files=None, machine_data=None):
    print(f'Task {item["uuid"]} was submitted.')
    os.chdir(item["work_dir"])
    # machine_data = decide_fp_machine(machine_data)
    fp_submit(
        item["work_dir"],
        item["run_tasks"],
        forward_common_files,
        forward_files,
        backward_files,
        machine_data
    )
    logger.info(f'Task {item["uuid"]} finished.')


def fp_submit(work_path, run_tasks,
              forward_common_files=None, forward_files=None, backward_files=None, machine_data=None):
    fp_command = machine_data['fp_command']
    fp_group_size = machine_data['fp_group_size']
    fp_resources = machine_data['fp_resources']
    mark_failure = fp_resources.get('mark_failure', False)
    dispatcher = make_dispatcher(
        machine_data['fp_machine'], machine_data['fp_resources'], work_path, run_tasks, fp_group_size
    )
    for i in range(10):
        try:
            dispatcher.run_jobs(fp_resources,
                                [fp_command],
                                work_path,
                                run_tasks,
                                fp_group_size,
                                forward_common_files,
                                forward_files,
                                backward_files,
                                mark_failure=mark_failure,
                                outlog='fp.log',
                                errlog='fp.err')
        except (Exception, SSHException):
            if i < 9:
                time.sleep(0.5)
            else:
                time.sleep(0.1)
                break
"""
