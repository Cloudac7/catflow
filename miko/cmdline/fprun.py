import click
# from miko.tesla.vasp import fp_tasks

@click.group()
def fprun_cli():
    pass

#TODO: to be refactored
# @fprun_cli.command()
# @click.option('--input-settings', '-i', type=click.Path(exists=True), required=True,
#               help='A json containing input parameters of the fp calculations.')
# def fprun(input_settings):
#     """Command line for submitting VASP tasks."""
#     with open(input_settings) as f:
#         settings = json.load(f)
#     fp_tasks(
#         ori_fp_tasks=settings['ori_fp_tasks'],
#         work_path=settings['work_path'],
#         machine_data=settings,
#         group_size=settings['group_size']
#     )
#
#     webhook = settings.get('webhook', None)
#     if webhook is not None:
#         mes_text = "# 完成情况 \n\n"
#         mes_text += "您的单点能任务已经全部完成，请登陆服务器查看\n\n"
#         mes_text += "时间：" + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
#         task_reminder(
#             webhook=settings['webhook'],
#             secret=settings['secret'],
#             text=mes_text
#         )
