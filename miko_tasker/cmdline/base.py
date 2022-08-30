import click


@click.group()
def tasker_cli():
    pass


@tasker_cli.group()
def tasker():
    """Start workflow runs with miko-tasker."""
    pass
