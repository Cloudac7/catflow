import json
import os
import sys

import numpy as np
from miko.utils import logger
from miko.resources.submit import settings
# from dpgen.remote.decide_machine import convert_mdata
from dpgen.generator.run import *


class WorkStep(object):
    def __init__(self, params, step_code, machine):
        self.params = params
        self.step_code = step_code
        self.machine = machine

    @property
    def sub_step_dict(self):
        return {
            0: self.make,
            1: self.run,
            2: self.post
        }

    def make(self):
        pass

    def run(self):
        pass

    def post(self):
        pass


# TODO: refine each step of whole workflow.
class DPTrain(WorkStep):
    def make(self):
        return make_train(self.step_code, self.params, self.machine)

    def run(self):
        return run_train(self.step_code, self.params, self.machine)

    def post(self):
        return post_train(self.step_code, self.params, self.machine)


class DPExploration(WorkStep):
    def make(self):
        return make_model_devi(self.step_code, self.params, self.machine)

    def run(self):
        return run_model_devi(self.step_code, self.params, self.machine)

    def post(self):
        return post_train(self.step_code, self.params, self.machine)


class FPCalculation(WorkStep):
    def make(self):
        return make_fp(self.step_code, self.params, self.machine)

    def run(self):
        return run_fp(self.step_code, self.params, self.machine)

    def post(self):
        return post_fp(self.step_code, self.params)


class LongTrain(WorkStep):
    pass


class CLWorkFlow(object):
    def __init__(self, param_file, machine_pool):
        """initialize the concurrent learning workflow

        Parameters
        ----------
        param_file : a file containing all the parameters for workflow runs
        machine_pool : a file with machine for each step to use
        """
        self.param_file = param_file
        self.machine_pool = machine_pool
        self.stage = 0
        self.step = 0

    @staticmethod
    def get_data(json_file):
        with open(json_file, 'r') as fp:
            return json.load(fp)

    @staticmethod
    def record_stage(record_file_path, stage, step):
        record_array = np.array([stage, step])
        np.savetxt(record_file_path, record_array, fmt="%d")

    @property
    def params(self):
        return self.get_data(self.param_file)

    @property
    def machine(self):
        return self.get_data(self.machine_pool)

    @property
    def main_step(self):
        return self.step / 3

    @property
    def real_step(self):
        return self.step % 3

    @property
    def main_step_dict(self):
        return {
            0: DPTrain,
            1: DPExploration,
            2: FPCalculation,
        }

    def get_work_step(self, step_code):
        step = self.main_step_dict.get(step_code)
        return step

    def read_record(self, record="miko.record"):
        record_file_path = os.path.abspath(record)
        try:
            stage_rec = np.loadtxt(record_file_path)
            self.stage = stage_rec[0]
            self.step = stage_rec[1]
            logger.info("continue from stage {0:03d} step {1:02d}".format(self.stage, self.step))
        except FileNotFoundError or NameError:
            logger.debug("record file not found")
            logger.debug("creating record file {0}".format(record))
            self.record_stage(record_file_path, self.stage, self.step)

    def run_step(self):
        stage_class = self.get_work_step(self.main_step)
        stage_task = stage_class(self.params, self.stage, self.machine_pool)
        step_task = stage_task.sub_step_dict.get(self.real_step)
        step_task()

        if self.step != 8:
            self.step += 1
        else:
            self.stage += 1
            self.step = 0

    def check_converge(self):
        if self.main_step == 1:
            if self.real_step == 0:
                model_devi_jobs = self.params['model_devi_jobs']
                if self.stage >= len(model_devi_jobs):
                    return False
            else:
                return True
        else:
            return True

    def run_loop(self, record="miko.record"):
        self.read_record(record)
        while self.check_converge():
            self.run_step()


class ClusterReactionWorkflow(CLWorkFlow):
    def check_converge(self):
        if self.main_step == 1:
            if self.real_step == 0:
                last_model_devi_job = self.params.get('model_devi_jobs')[-1]
                last_sys_idx = last_model_devi_job.get('sys_idx')
                conv_flags = np.zeros_like(last_sys_idx, dtype=int)
                for i, idx in enumerate(last_sys_idx):
                    logger.info(f'Checking convergancy for iteration {self.stage}')
                    accu_ratio = self._check_index_converge(idx)
                    logger.info(f'idx {idx} reach accuracy ratio: {accu_ratio}')
                    if accu_ratio >= 0.97:
                        conv_flags[i] = 1
                if 1 in conv_flags:
                    logger.info('Not all idxs reach 97% accuracy')
                    logger.info('Continue training process')
                    self.update_params()
                    return True
                else:
                    logger.info('Model accuracy converged.')
                    return False
            else:
                return True
        else:
            return True

    def _check_index_converge(self, index):
        # TODO: check accu in each sys_idx
        return 1

    def update_params(self):
        # TODO: update self.params, generating new model_devi_jobs
        pass
