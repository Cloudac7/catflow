import click
import logging
from importlib.metadata import entry_points


miko_eps = entry_points().get('miko.cmdline', [])
if miko_eps != []:
    miko_eps_cli = [i.load() for i in miko_eps]
else:
    miko_eps_cli =[]

cli = click.CommandCollection(sources=[
    *miko_eps_cli
])
