import click
import time
import json

from miko.tesla.dpgen import DPTask
from miko.tesla.dpgen.exploration import DPExplorationAnalyzer
from miko.utils.log_factory import logger
from miko.utils.message import task_reminder

def read_params(task_path, param='param.json', machine='machine.json', record='record.tesla'):
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
    long_task_analyzer = DPExplorationAnalyzer(long_task)
    return long_task_analyzer


@click.group()
def simu_cli():
    pass


@simu_cli.command()
@click.option('--input-settings', '-i', type=click.Path(exists=True), required=True,
              help='A json containing input parameters of the simulation.')
@click.argument('task_path', type=click.Path(exists=True), required=True)
@click.argument('param', default='param.json')
@click.argument('machine', default='machine.json')
@click.argument('record', default='record.tesla')
def simu(input_settings, task_path, param, machine, record):
    """
    Start a simulation with selected parameters \n
    input_settings: The JSON file input.\n
    task_path: Path of DP-GEN task.\n
    param: param file name\n
    machine: machine file name\n
    record: record file name\n
    """
    logger.info("Loading tasks...")
    with open(input_settings) as f:
        settings = json.load(f)
    params = settings['params']
    long_task_ana = read_params(task_path, param, machine, record)
    long_task_ana.train_model_test(
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
