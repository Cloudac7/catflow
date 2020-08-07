import os
import asyncio
import uuid
from glob import glob
from dpgen.dispatcher.Dispatcher import make_dispatcher, Dispatcher

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
    if machine_data is None:
        machine_data = model_dict
    forward_files = ['POSCAR', 'INCAR', 'POTCAR']
    backward_files = ['OUTCAR', 'vasprun.xml', 'fp.log', 'fp.err']
    forward_common_files = []
    if len(ori_fp_tasks) == 0:
        return
    fp_run_tasks = []
    for task in ori_fp_tasks:
        _task_group = glob(task)
        _task_group.sort()
        fp_run_tasks += _task_group
    _group_num = 0
    _groups = []
    _groups_index = 0
    _uuid = str(uuid.uuid1())
    _work_dir = os.path.join(work_path, _uuid)
    os.makedirs(os.path.join(_work_dir), exist_ok=True)
    for tt in fp_run_tasks:
        if _group_num >= group_size:
            _group_num = 0
            _groups_index += 1
            _uuid = str(uuid.uuid1())
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
    loop = asyncio.get_event_loop()
    tasks = [fp_await_submit(
        item,
        forward_common_files,
        forward_files,
        backward_files,
        machine_data) for item in _groups]
    loop.run_until_complete(asyncio.wait(tasks))
    loop.close()


async def fp_await_submit(item,
                          forward_common_files=None, forward_files=None, backward_files=None, machine_data=None):
    print(f'Task {item["uuid"]} was submitted.')
    await fp_submit(
        item["work_dir"],
        item["run_tasks"],
        forward_common_files,
        forward_files,
        backward_files,
        machine_data
    )
    print(f'Task {item["uuid"]} finished.')


async def fp_submit(work_path, run_tasks,
                    forward_common_files=None, forward_files=None, backward_files=None, machine_data=None):
    dispatcher = _make_dispatcher(
        mdata=machine_data['machine'],
        job_record='jr.json'
    )
    fp_command = machine_data['command']
    fp_group_size = machine_data['group_size']
    fp_resources = machine_data['resources']
    mark_failure = fp_resources.get('mark_failure', False)
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
            job_record=kwargs.get('job_record', 'jr.json'))
        return disp
