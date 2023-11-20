from typing import Optional

from catflow.analyzer.tesla.base.labeling import LabelingAnalyzer
from catflow.analyzer.tesla.dpgen.task import DPTask, DPAnalyzer


class DPLabelingAnalyzer(LabelingAnalyzer, DPAnalyzer):

    def __init__(
        self, 
        dp_task: DPTask, 
        fp_style: Optional[str] = None
    ) -> None:
        super().__init__(dp_task)
        if fp_style is None:
            fp_style = self.dp_task.param_data['fp_style']
        self.fp_style = fp_style

    def get_fp_tasks(self, iteration: int = 0):
        _iteration_dir = self._iteration_dir(iteration=iteration)
        stage_path = self.dp_task.path / _iteration_dir / '02.fp'
        task_files = [list(stage_path.glob("task.*"))]
        return task_files
