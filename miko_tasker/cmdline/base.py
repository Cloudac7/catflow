import click
from .workflow.cli_pmf import pmf, dppmf
from .workflow.cli_tesla import tesla, tesla_cluster, tesla_metad


@click.group()
def tasker_cli():
    pass


@tasker_cli.group()
def tasker():
    """Start workflow runs with miko-tasker."""
    pass


tasker.add_command(pmf)
tasker.add_command(dppmf)
tasker.add_command(tesla)
tasker.add_command(tesla_cluster)
tasker.add_command(tesla_metad)
