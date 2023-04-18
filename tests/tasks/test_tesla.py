import json
from pathlib import Path
import numpy as np
from miko_tasker.tasks.tesla import ClusterReactionWorkflow, ClusterReactionUpdater


class MockClusterReactionWorkflow(object):
    def __init__(self, param_file=None):
        self.workflow_settings = {
            "IS": {"coordination": 1.4, "sys_idx": 0},
            "FS": {"coordination": 4.0, "sys_idx": 0}
        }
        self.IS = {"coordination": 1.4, "sys_idx": 0}
        self.FS = {"coordination": 4.0, "sys_idx": 0}
        self.stage = 0
        self.step = 1
        self.reaction_atoms_pair = [0, 1]
        self.params = {
            "sys_configs_prefix": "/",
            "sys_configs": ["a", "b"],
            "sys_batch_size": ["auto", "auto"],
            "model_devi_jobs": [
                {
                    "template": {
                        "lmp": "lmp/input-restrain.lammps",
                        "plm": "lmp/input.plumed"
                    },
                    "sys_idx": [0, 1, 2],
                    "traj_freq": 600,
                    "_idx": 0,
                    "rev_mat": {
                        "lmp": {
                            "V_NSTEPS": [30000],
                            "V_TEMP": [200, 400, 600, 800, 1000, 1200, 1400]
                        }
                    },
                    "sys_rev_mat": {
                        "0": {"lmp": {"V_DIS1": [1.4], "V_DIS2": [1.6], "V_FORCE": [10]}, "_type": "IS"},
                        "1": {"lmp": {"V_DIS1": [4.0], "V_DIS2": [3.8], "V_FORCE": [10]}, "_type": "FS"}
                    },
                    "model_devi_f_trust_lo": 0.2,
                    "model_devi_f_trust_hi": 0.6
                }
            ]
        }
        if param_file is not None:
            self.param_file = param_file / "param.json"

    @property
    def work_path(self):
        return Path("/")


def test_render_params(shared_datadir):
    param_file = shared_datadir / "params.json"
    wf = ClusterReactionWorkflow(
        param_file=param_file,
        machine_pool=shared_datadir / "machine.json",
        conf_file=shared_datadir / "workflow_settings.yml"
    )
    with open(shared_datadir / "params_new.json") as f:
        param_ref = json.load(f)
    assert wf.params == param_ref
    print(wf.param_file)
    wf.render_params()
    with open(param_file) as f:
        param_dumped = json.load(f)
    assert param_dumped["model_devi_jobs"] == param_ref["model_devi_jobs"]


def test_cluster_reaction_updater(mocker):
    wf = MockClusterReactionWorkflow()
    updater = ClusterReactionUpdater(wf)

    def _quick_check():
        while True:
            wf.params["sys_configs"].append("/")
            yield len(wf.params["sys_configs"])

    get_mock = mocker.patch.object(
        ClusterReactionUpdater, '_pick_new_structure')
    get_mock.side_effect = _quick_check()
    for i in range(10):
        wf.stage = i
        updater.model_devi_job_generator()
        assert len(wf.params["model_devi_jobs"]) == wf.stage + 1
        print(wf.params["model_devi_jobs"][-1]["sys_rev_mat"])
    assert len([i['lmp']['V_DIS1'][0]
                for i in wf.params["model_devi_jobs"][-1]['sys_rev_mat'].values()]) == len(np.arange(1.4, 4.1, 0.1))
