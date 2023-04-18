import click
from miko_tasker.tasks.tesla import CLWorkflow, ClusterReactionWorkflow, MetadynReactionWorkflow


@click.command()
@click.argument('param', type=click.Path(exists=True), required=True)
@click.argument('machine', type=click.Path(exists=True), required=True)
@click.argument('record', type=click.Path(exists=True), default='miko.record')
def tesla(param, machine, record="miko.record"):
    """Start TESLA workflow run. \f

    Args: 
        param (Path): Param file, just like DP-GEN param.json.
        machine (Path): Machine file, just like DP-GEN param.json.
        record (Path): Record. Default: `miko.record`
    """
    task = CLWorkflow(
        param_file=param,
        machine_pool=machine
    )
    task.run_loop(record=record)


@click.command()
@click.argument('param', type=click.Path(exists=True), required=True)
@click.argument('machine', type=click.Path(exists=True), required=True)
@click.argument('configure', type=click.Path(exists=True), required=True)
@click.argument('record', type=click.Path(), default='miko.record')
def tesla_cluster(param, machine, configure, record="miko.record"):
    """Start TESLA workflow run for reaction at clusters. \f

    Args:
        param (Path): Param file, just like DP-GEN param.json.
        machine (Path): Machine file, just like DP-GEN param.json.
        configure (Path): Yaml file containing configurations for workflow run.
        record (Path): Record. Default: `miko.record`
    """
    task = ClusterReactionWorkflow(
        param_file=param,
        machine_pool=machine,
        conf_file=configure
    )
    task.run_loop(record=record)

@click.command()
@click.argument('param', type=click.Path(exists=True), required=True)
@click.argument('machine', type=click.Path(exists=True), required=True)
@click.argument('configure', type=click.Path(exists=True), required=True)
@click.argument('record', type=click.Path(), default='miko.record')
def tesla_metad(param, machine, configure, record="miko.record"):
    """Start TESLA workflow run for metadynamics. \f

    Args:
        param (Path): Param file, just like DP-GEN param.json.
        machine (Path): Machine file, just like DP-GEN param.json.
        configure (Path): Yaml file containing configurations for workflow run.
        record (Path): Record. Default: `miko.record`
    """
    task = MetadynReactionWorkflow(
        param_file=param,
        machine_pool=machine,
        conf_file=configure
    )
    task.run_loop(record=record)
