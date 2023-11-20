import re

from catflow.analyzer.tesla.base.labeling import LabelingAnalyzer
from catflow.analyzer.tesla.ai2_kit.task import CllAnalyzer
from catflow.analyzer.graph.plotting import canvas_style


class CllLabelingAnalyzer(LabelingAnalyzer, CllAnalyzer):

    def get_fp_tasks(self, iteration: int = 0):
        n_iter = self._iteration_dir(iteration=iteration)
        stage_path = self.dp_task.path / n_iter / 'label-vasp/tasks'
        task_files = [
            item for item in stage_path.iterdir() if re.search(r'^\d+$', str(item))
        ]
        return task_files
