import click
import time

from miko_tasker.utils.config import load_yaml_configs
from miko_tasker.calculation.vasp import multi_fp_task, trajectory_checkrun_vasp

@click.command()
@click.option('--input-settings', '-i', type=click.Path(exists=True), required=True,
              help='A json containing input parameters of the fp calculations.')
def vasprun(input_settings):
    """Command line for submitting VASP tasks."""
    settings = load_yaml_configs(input_settings)
    multi_fp_task(**settings)

@click.command()
@click.option('--input-settings', '-i', type=click.Path(exists=True), required=True,
              help='A json containing input parameters of the fp calculations.')
def trajcheck(input_settings):
    """Command line for submitting VASP tasks."""
    settings = load_yaml_configs(input_settings)
    trajectory_checkrun_vasp(**settings)
