import os
import re

import numpy as np
from pathlib import Path
from typing import Literal

from ase.io import read
from matplotlib import pyplot as plt

from miko.utils.log_factory import logger
from miko.utils.log_factory import LogFactory
from miko.tesla.base.labeling import LabelingAnalyzer
from miko.tesla.ai2_kit.task import CllAnalyzer, CllTask
from miko.graph.plotting import canvas_style



class CllLabelingAnalyzer(LabelingAnalyzer, CllAnalyzer):

    def get_fp_tasks(self, iteration: int = 0):
        n_iter = self._iteration_dir(iteration=iteration)
        stage_path = self.dp_task.path / n_iter / 'label-vasp/tasks'
        all_data = []
        task_files = [
            item for item in stage_path.iterdir() if re.search(r'^\d+$', str(item))
        ]
        return task_files
