import click
import logging
import time
import json
from dpana.dpgen import DPTask
from dpana.util import fp_tasks
from dpana.message import task_reminder


def read_params(task_path, param='param.json', machine='machine.json', record='record.dpgen'):
    """
    Load the DP-GEN task
    :param task_path:
    :param param:
    :param machine:
    :param record:
    :return:
    """
    long_task = DPTask(
        path=task_path,
        param_file=param,
        machine_file=machine,
        record_file=record
    )
    return long_task


@click.group()
def simu_cli():
    pass


@simu_cli.command()
@click.option('--input-settings', '-i', type=click.Path(exists=True), required=True,
              help='A json containing input parameters of the simulation.')
@click.argument('task_path', type=click.Path(exists=True), required=True)
@click.argument('param', default='param.json')
@click.argument('machine', default='machine.json')
@click.argument('record', default='record.dpgen')
def simu(input_settings, task_path, param, machine, record):
    """
    Start a simulation with selected parameters \n
    input_settings: The JSON file input.\n
    task_path: Path of DP-GEN task.\n
    param: param file name\n
    machine: machine file name\n
    record: record file name\n
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    logging.getLogger("paramiko").setLevel(logging.WARNING)
    with open(input_settings) as f:
        settings = json.load(f)
    params = settings['params']
    long_task = read_params(task_path, param, machine, record)
    long_task.train_model_test(
        iteration=settings['iteration'],
        params=params,
        files=settings['input'],
        forward_files=settings['forward_files'],
        backward_files=settings['backward_files']
    )
    mes_text = "# 完成情况 \n\n"
    mes_text += "您的长训练MD任务已经全部完成，请登陆服务器查看\n\n"
    mes_text += "时间：" + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    task_reminder(
        webhook=settings['webhook'],
        secret=settings['secret'],
        text=mes_text
    )


@click.group()
def fprun_cli():
    pass


@fprun_cli.command()
@click.option('--input-settings', '-i', type=click.Path(exists=True), required=True,
              help='A json containing input parameters of the fp calculations.')
def fprun(input_settings):
    """Command line for submitting VASP tasks."""
    with open(input_settings) as f:
        settings = json.load(f)
    fp_tasks(
        ori_fp_tasks=settings['ori_fp_tasks'],
        work_path=settings['work_path'],
        machine_data=settings,
        group_size=settings['group_size']
    )

    webhook = settings.get('webhook', None)
    if webhook is not None:
        mes_text = "# 完成情况 \n\n"
        mes_text += "您的单点能任务已经全部完成，请登陆服务器查看\n\n"
        mes_text += "时间：" + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        task_reminder(
            webhook=settings['webhook'],
            secret=settings['secret'],
            text=mes_text
        )


cli = click.CommandCollection(sources=[simu_cli, fprun_cli])

if __name__ == '__main__':
    cli()
