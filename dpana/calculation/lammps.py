import os
from glob import glob
from . import from_yaml, from_json


def multi_deepmd_task(
        work_path,
        machine_conf,
        model_path=None,
        numb_models=4,
        outlog=None,
        errlog=None,
        **kwargs
):
    """
    Submit your own md task with the help of DPDispatcher.

    Parameters
    ----------
        work_path: The dir contains your md tasks.
        machine_conf: Configuration file of machine and resources.
        model_path: The path of models contained for calculation.
        numb_models: The number of models selected.
        outlog : filename of output file
        errlog : filename of error file

    Returns
    -------
        CalulationTask object.
    """

    folder_list = kwargs.get('folder_list', ["task.*"])
    all_task = []
    for i in folder_list:
        _task = glob(os.path.join(work_path, i))
        _task.sort()
        all_task += _task
    run_tasks_ = all_task
    run_tasks = [os.path.basename(ii) for ii in run_tasks_]
    work_base = os.path.abspath(work_path)

    model_names = kwargs.get('model_names', [f'graph.{str(i).zfill(3)}.pb' for i in range(numb_models)])
    if model_path is None:
        model_path = work_path
    for ii in model_names:
        if not os.path.exists(os.path.join(work_path, ii)):
            os.symlink(os.path.join(model_path, ii), os.path.join(work_path, ii))

    forward_files = kwargs.get('forward_files', ['conf.lmp', 'input.lammps', 'traj'])
    backward_files = kwargs.get('backward_files', ['model_devi.out', 'model_devi.log', 'traj'])
    if outlog is not None:
        backward_files.append(outlog)
    if errlog is not None:
        backward_files.append(errlog)
    machine_conf = os.path.abspath(machine_conf)

    if os.path.basename(machine_conf).split('.')[-1] in ['yaml', 'yml']:
        calc = from_yaml(
            machine_conf,
            work_base=work_base,
            task_list=run_tasks,
            forward_common_files=model_names,
            forward_files=forward_files,
            backward_files=backward_files,
        )
    else:
        calc = from_json(
            machine_conf,
            work_base=work_base,
            task_list=run_tasks,
            forward_common_files=model_names,
            forward_files=forward_files,
            backward_files=backward_files,
        )
    return calc
