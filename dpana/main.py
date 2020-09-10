import click
import time
import json
from dpana.dpgen import DPTask
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
@simu_cli.option('--input-settings', '-i', type=click.Path(exists=True), required=True)
@click.argument('task_path', type=click.Path(exists=True), required=True)
@click.argument('param', default='param.json')
@click.argument('machine', default='machine.json')
@click.argument('record', default='record.dpgen')
def simu(input_settings, task_path, param, machine, record):
    """
    Start a long MD simulation
    :param input_settings:
    :param task_path:
    :param param:
    :param machine:
    :param record:
    :return:
    """
    with open(input_settings) as f:
        settings = json.load(f)
    params = settings['params']
    long_task = read_params(task_path, param, machine, record)
    long_task.train_model_test(
        iteration=11,
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


cli = click.CommandCollection(sources=[simu_cli])

if __name__ == '__main__':
    cli()
