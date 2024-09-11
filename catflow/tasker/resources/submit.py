import asyncio
from pathlib import Path
from typing import List, Dict, Optional

from ai2_kit.core.executor import HpcExecutor
from ai2_kit.core.script import BashScript, BashStep, BashTemplate
from ai2_kit.core.util import list_split
from ai2_kit.core.job import gather_jobs
from ai2_kit.core.checkpoint import set_checkpoint_dir, apply_checkpoint


class JobFactory(object):
    def __init__(self,
                 executor: HpcExecutor):
        """Initialize the JobFactory with an executor."""
        self.executor = executor

    @property
    def job_name(self):
        return self.__class__.__name__

    def process_input(self, base_dir: Path, **inputs) -> List[Path]:
        """Process the input and return a list of directories for tasks."""
        return []

    def process_output(self, task_dirs: List[Path], **inputs):
        """Process the output from task directories and return the results."""
        pass

    async def workstep(
            self,
            command: str,
            path_prefix: str,
            script_header: str,
            concurrency: int = 5,
            task_parameters: Optional[Dict] = None
    ):
        # create base_dir to store input and output
        # it is suggested to use a unique path_prefix for each workflow
        executor = self.executor
        base_dir = Path(executor.work_dir) / path_prefix
        executor.mkdir(str(base_dir))

        if task_parameters is None:
            task_parameters = {}

        # run pre process
        task_dirs = executor.run_python_fn(self.process_input)(
            base_dir=base_dir, **task_parameters)

        # build commands to calculate square and save to output
        steps = [BashStep(cmd=command,
                          cwd=str(task_dir)) for task_dir in task_dirs]
        # create script according to concurrency limit and submit
        jobs = []
        for group in list_split(steps, concurrency):
            script = BashScript(
                template=BashTemplate(header=script_header),
                steps=group,
            )
            job = executor.submit(script.render(), cwd=str(base_dir))
            jobs.append(job)

        # wait for all jobs to complete
        await gather_jobs(jobs, max_tries=2, raise_error=True)

        # post process
        result = executor.run_python_fn(self.process_output)(
            task_dirs=task_dirs, **task_parameters)
        return result

    def run(
            self,
            command: str,
            path_prefix: str,
            checkpoint_dir: str,
            script_header: str,
            concurrency: int = 5,
            task_parameters: Optional[Dict] = None
    ):
        self.executor.init()
        set_checkpoint_dir(checkpoint_dir)
        result = asyncio.run(
            apply_checkpoint(f"{checkpoint_dir}/{self.job_name}")(self.workstep)(
                command=command,
                path_prefix=path_prefix,
                script_header=script_header,
                concurrency=concurrency,
                task_parameters=task_parameters
            ))
        return result
