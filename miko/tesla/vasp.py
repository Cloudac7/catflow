# import json
# import logging
import os
# import shutil
# import time
# import uuid

from glob import glob
from ase.io import iread, write
# from multiprocessing import Pool
# from paramiko import SSHException

from miko.utils import logger
from miko.resources.submit import JobFactory


def traj_fp_vasp(traj_file, work_path, chemical_symbol=None, index="::"):
    """
    Export VASP POSCAR files from trajectory.
    :param traj_file:
    :param work_path:
    :param chemical_symbol:
    :param index:
    :return:
    """
    stcs = iread(os.path.abspath(traj_file), index=index)
    os.makedirs(work_path, exist_ok=True)
    for i, j in enumerate(stcs):
        if chemical_symbol:
            j.set_chemical_symbols(chemical_symbol)
        task_path = os.path.join(work_path, f'task.{str(i).zfill(6)}')
        os.makedirs(task_path, exist_ok=True)
        write(os.path.join(task_path, 'POSCAR'), j, vasp5=True)


def multi_fp_task(work_path, fp_command, machine_name, resource_dict, **kwargs):
    forward_files = ['POSCAR', 'INCAR', 'POTCAR']
    backward_files = ['OUTCAR', 'vasprun.xml', 'fp.log', 'fp.err']
    task_dir_pattern = kwargs.get('task_dir_pattern', 'task.*')
    fp_tasks = glob(os.path.join(work_path, task_dir_pattern))
    fp_tasks.sort()
    if len(fp_tasks) == 0:
        return
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
    job.run_submission()


# TODO: fix fp_tasks for DPDispatcher
"""
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
    # TODO: decide which machine to use according to resources
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
