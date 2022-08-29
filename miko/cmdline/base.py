import click
import logging
from importlib.metadata import entry_points

from .simu import simu_cli
from .fprun import fprun_cli


miko_eps = entry_points().get('miko.cmdline', [])

cli = click.CommandCollection(sources=[
    simu_cli,
    fprun_cli,
    *miko_eps
])
