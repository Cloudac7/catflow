import os
from glob import glob
from . import from_yaml, from_json
from . import CalculationTask


def model_devi_calc(conf_file, work_path, model_path=None, numb_models=4, **kwargs):
    """
    Submit your own md task with the help of dpgen.

    Parameters
    ----------
        conf_file: Configuration file of machine and resources.
        work_path: The dir contains your md tasks.
        model_path: The path of models contained for calculation.
        numb_models: The number of models selected.

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
    conf_file = os.path.abspath(conf_file)

    if os.path.basename(conf_file).split('.')[-1] in ['yaml', 'yml']:
        calc = from_yaml(
            conf_file,
            work_base=work_base,
            task_list=run_tasks,
            forward_common_files=model_names,
            forward_files=forward_files,
            backward_files=backward_files,
        )
    else:
        calc = from_json(
            conf_file,
            work_base=work_base,
            task_list=run_tasks,
            forward_common_files=model_names,
            forward_files=forward_files,
            backward_files=backward_files,
        )
    return calc
