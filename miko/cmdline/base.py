import click
import logging
from importlib.metadata import entry_points

from .simu import simu_cli
from .fprun import fprun_cli
from .tasker import tasker_cli


cli = click.CommandCollection(sources=[
    simu_cli,
    fprun_cli,
    tasker_cli
])
