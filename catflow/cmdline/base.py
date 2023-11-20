import click
import logging
from importlib.metadata import entry_points


entries = entry_points().get('catflow.cmdline', [])
if entries != []:
    entries_cli = [i.load() for i in entries]
else:
    entries_cli =[]

cli = click.CommandCollection(sources=[
    *entries_cli
])
