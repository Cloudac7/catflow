from catflow.utils.config import load_yaml_configs
from catflow.tasker.calculation.vasp import multi_fp_task, trajectory_checkrun_vasp


def vasprun(input_settings):
    """Command line for submitting VASP tasks."""
    settings = load_yaml_configs(input_settings)
    multi_fp_task(**settings)


def trajcheck(input_settings):
    """Command line for submitting VASP tasks."""
    settings = load_yaml_configs(input_settings)
    trajectory_checkrun_vasp(**settings)
