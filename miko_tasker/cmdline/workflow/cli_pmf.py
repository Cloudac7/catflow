import click
from miko_tasker.workflow.pmf import PMFFactory, DPPMFFactory
from ..base import tasker_cli


@tasker_cli.command()
@click.argument('param', type=click.Path(exists=True), required=True)
@click.argument('machine', type=click.Path(exists=True), required=True)
@click.argument('configure', type=click.Path(exists=True), required=True)
def pmf(param, machine, configure):
    wf = DPPMFFactory(param_file=param, machine_pool=machine, conf_file=configure)
    wf.run_workflow()


@tasker_cli.command()
@click.argument('param', type=click.Path(exists=True), required=True)
@click.argument('machine', type=click.Path(exists=True), required=True)
@click.argument('configure', type=click.Path(exists=True), required=True)
def dppmf(param, machine, configure):
    wf = DPPMFFactory(param_file=param, machine_pool=machine, conf_file=configure)
    wf.run_workflow()
