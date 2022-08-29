import click
from miko_tasker.workflow.tesla import CLWorkFlow, ClusterReactionWorkflow


@click.group()
def tasker_cli():
    pass


@tasker_cli.group()
def tasker():
    pass


@tasker.command()
@click.argument('param', type=click.Path(exists=True), required=True,
                help='params required for DP-GEN loop')
@click.argument('machine', type=click.Path(exists=True), required=True,
                help='machine pool for tasks to be submitted.')
@click.argument('configure', type=click.Path(exists=True), required=True,
                help='A yaml containing input parameters of the workflow.')
@click.argument('record', type=click.Path(exists=True), default='record.tesla')
def tesla(param, machine, configure, record):
    task = CLWorkFlow(
        param_file=param,
        machine_pool=machine,
        conf_file=configure
    )
    task.run_loop(record=record)


@tasker.command()
@click.argument('param', type=click.Path(exists=True), required=True,
                help='params required for DP-GEN loop')
@click.argument('machine', type=click.Path(exists=True), required=True,
                help='machine pool for tasks to be submitted.')
@click.argument('configure', type=click.Path(exists=True), required=True,
                help='A yaml containing input parameters of the workflow.')
@click.argument('record', type=click.Path(exists=True), default='record.tesla')
def tesla_cluster(param, machine, configure, record):
    task = ClusterReactionWorkflow(
        param_file=param,
        machine_pool=machine,
        conf_file=configure
    )
    task.run_loop(record=record)
