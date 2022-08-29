import click
from importlib.metadata import entry_points

try:
    tasker_eps = entry_points()['miko.cmdline.tasker']
    tasker_cli = tasker_eps[0].load()
except (KeyError, IndexError):
    @click.group()
    def tasker_cli():
        pass
