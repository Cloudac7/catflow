import os
from glob import glob
from dpgen.dispatcher.Dispatcher import make_dispatcher

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
