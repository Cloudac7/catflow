import click
from importlib.metadata import entry_points

tasker_eps = entry_points(group='miko.cmdline.tasker:tasker_cli')
try:
    tasker_cli = tasker_eps[0].load()
except IndexError:
    @click.group()
    def tasker_cli():
        pass
