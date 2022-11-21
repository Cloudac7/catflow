import click
import json
import time
from miko_tasker.calculation.vasp import multi_fp_task
from miko_tasker.utils.messager import dingtalk_reminder

@click.command()
@click.option('--input-settings', '-i', type=click.Path(exists=True), required=True,
              help='A json containing input parameters of the fp calculations.')
def vasprun(input_settings):
    """Command line for submitting VASP tasks."""
    with open(input_settings) as f:
        settings = json.load(f)
    multi_fp_task(
        work_path=settings['work_path'],
        machine_name=settings['machine_name'],
        resource_dict=settings['resource_dict'],
        **settings['kwargs']
    )

    webhook = settings.get('webhook', None)
    if webhook is not None:
        mes_text = "# 完成情况 \n\n"
        mes_text += "您的单点能任务已经全部完成，请登陆服务器查看\n\n"
        mes_text += "时间：" + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        dingtalk_reminder(
            webhook=settings['webhook'],
            secret=settings['secret'],
            text=mes_text
        )
