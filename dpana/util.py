import json
import logging
import os
import shutil
import time
import uuid
import daemon
import seaborn as sns
from glob import glob
from ase.io import iread, write
from multiprocessing import Pool
from dpgen.dispatcher.Dispatcher import make_dispatcher, Dispatcher
from dpgen.remote.decide_machine import decide_train_machine, decide_fp_machine
from paramiko import SSHException

model_dict = {
    "machine": {
        "batch": "lsf",
        "hostname": "localhost",
        "port": 22,
        "username": "username",
        "work_path": "/remote/work/path"
    },
    "resources": {
        "cvasp": False,
        "task_per_node": 24,
        "numb_node": 1,
        "node_cpu": 24,
        "exclude_list": [],
        "with_mpi": True,
        "source_list": [
        ],
        "module_list": [
            "intel/17u5",
            "mpi/intel/17u5"
        ],
        "time_limit": "12:00:00",
        "partition": "medium",
        "_comment": "that's Bel"
    },
    "command": "/some/work/path/vasp_std",
    "group_size": 25
}


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


def multi_fp_task(work_path, machine_data=None):
    if machine_data is None:
        machine_data = model_dict
    forward_files = ['POSCAR', 'INCAR', 'POTCAR']
    backward_files = ['OUTCAR', 'vasprun.xml', 'fp.log', 'fp.err']
    forward_common_files = []
    fp_command = machine_data['command']
    fp_group_size = machine_data['group_size']
    fp_resources = machine_data['resources']
    mark_failure = fp_resources.get('mark_failure', False)
    fp_tasks = glob(os.path.join(work_path, 'task.*'))
    fp_tasks.sort()
    if len(fp_tasks) == 0:
        return
    fp_run_tasks = fp_tasks
    run_tasks = [os.path.basename(ii) for ii in fp_run_tasks]
    dispatcher = make_dispatcher(
        machine_data['machine'],
        machine_data['resources'],
        work_path,
        run_tasks,
        fp_group_size)
    dispatcher.run_jobs(machine_data['resources'],
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


def fp_tasks(ori_fp_tasks, work_path, machine_data=None, group_size=1):
    """Submit single point energy tasks at one time.

    Submitting multiple VASP tasks in the work directory to remote or local servers.

    Args:
        ori_fp_tasks: A list of path containing `INPUT FILES`.
        work_path: A local path for creating work directory.
        machine_data: The machine data, read as a dict.
            For example:
                {
                    "machine": {
                        "batch": "lsf",
                        "hostname": "localhost",
                        "port": 22,
                        "username": "username",
                        "work_path": "/remote/work/path"
                    },
                    "resources": {
                        "cvasp": False,
                        "task_per_node": 24,
                        "numb_node": 1,
                        "node_cpu": 24,
                        "exclude_list": [],
                        "with_mpi": True,
                        "source_list": [
                        ],
                        "module_list": [
                            "intel/17u5",
                            "mpi/intel/17u5"
                        ],
                        "time_limit": "12:00:00",
                        "partition": "medium",
                        "_comment": "that's Bel"
                    },
                    "command": "/some/work/path/vasp_std",
                    "group_size": 25
                }
        group_size: Set the group size for tasks.
    """
    if machine_data is None:
        machine_data = model_dict
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
        logging.info('Waiting for all tasks done...')
        p.close()
        p.join()
        shutil.rmtree(work_path)


def fp_await_submit(item, forward_common_files=None, forward_files=None, backward_files=None, machine_data=None):
    print(f'Task {item["uuid"]} was submitted.')
    os.chdir(item["work_dir"])
    machine_data = decide_fp_machine(machine_data)
    fp_submit(
        item["work_dir"],
        item["run_tasks"],
        forward_common_files,
        forward_files,
        backward_files,
        machine_data
    )
    logging.info(f'Task {item["uuid"]} finished.')


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


def _make_dispatcher(mdata, mdata_resource=None, work_path=None, run_tasks=None, group_size=None, **kwargs):
    if 'cloud_resources' in mdata:
        if mdata['cloud_resources']['cloud_platform'] == 'ali':
            from dpgen.dispatcher.ALI import ALI
            dispatcher = ALI(mdata, mdata_resource, work_path, run_tasks, group_size, mdata['cloud_resources'])
            dispatcher.init()
            return dispatcher
        elif mdata['cloud_resources']['cloud_platform'] == 'ucloud':
            pass
    else:
        hostname = mdata.get('hostname', None)
        if hostname:
            context_type = 'ssh'
        else:
            context_type = 'local'
        try:
            batch_type = mdata['batch']
        except KeyError:
            batch_type = mdata['machine_type']
        lazy_local = (mdata.get('lazy-local', False)) or (mdata.get('lazy_local', False))
        if lazy_local and context_type == 'local':
            context_type = 'lazy-local'
        disp = Dispatcher(
            remote_profile=mdata,
            context_type=context_type,
            batch_type=batch_type,
            job_record=kwargs.get('job_record', 'jr.json')
        )
        return disp


def canvas_style(context='paper'):
    sns.set_theme()
    sns.set_style('white')
    sns.set_context(context)
