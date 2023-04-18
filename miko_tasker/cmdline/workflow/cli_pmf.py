import click


@click.command()
@click.argument('configure', type=click.Path(exists=True), required=True)
@click.argument('output', type=click.Path(exists=True), required=False)
def pmf(configure, output):
    """Run potential of mean force calculation.

    Args:
        configure (Path): Configure files containing the input parameters.
        output (Path): Output files containing the output parameters to be reused.
    """
    import asyncio
    from miko_tasker.utils.config import load_yaml_configs
    from miko_tasker.flows.pmf_flow import PMFInput, PMFOutput
    from miko_tasker.flows.pmf_flow import flow_pmf_calculation

    flow_input = PMFInput.parse_obj(load_yaml_configs(configure))
    if output is not None:
        flow_output = PMFOutput.parse_obj(load_yaml_configs(output))
    else:
        flow_output = PMFOutput(pmf_tasks=[])

    loop = asyncio.get_event_loop()
    loop.run_until_complete(flow_pmf_calculation(flow_input, flow_output))
