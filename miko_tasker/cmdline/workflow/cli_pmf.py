import click
from miko_tasker.workflow.pmf import PMFFactory, DPPMFFactory


@click.command()
@click.argument('param', type=click.Path(exists=True), required=True)
@click.argument('machine', type=click.Path(exists=True), required=True)
@click.argument('configure', type=click.Path(exists=True), required=True)
def pmf(param, machine, configure):
    """Run potential of mean force calculation."""
    wf = DPPMFFactory(param_file=param, machine_pool=machine,
                      conf_file=configure)
    wf.run_workflow()


@click.command()
@click.argument('param', type=click.Path(exists=True), required=True)
@click.argument('machine', type=click.Path(exists=True), required=True)
@click.argument('configure', type=click.Path(exists=True), required=True)
def dppmf(param, machine, configure):
    """Run potential of mean force calculation based on DP."""
    wf_factory = DPPMFFactory(param_file=param, machine_pool=machine,
                      conf_file=configure)
    wf = wf_factory.generate()
    wf.run_workflow()
