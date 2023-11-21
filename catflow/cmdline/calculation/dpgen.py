import click
import time
import json

from catflow.analyzer.tesla.dpgen.task import DPTask
from catflow.utils.log_factory import logger
from catflow.tasker.calculation.dpgen import DPCheck


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
    long_task_analyzer = DPCheck(long_task)
    return long_task_analyzer


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
    machine_config = settings['machine']
    long_task_ana = read_params(task_path, param, machine, record)
    long_task_ana.train_model_test(
        machine_name=machine_config['machine_name'],
        resource_dict=machine_config['resources'],
        iteration=settings['iteration'],
        params=params,
        files=settings['input'],
        forward_files=settings['forward_files'],
        backward_files=settings['backward_files']
    )
