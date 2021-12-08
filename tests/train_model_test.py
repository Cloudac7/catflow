import unittest
from miko.tesla import DPTask

class MyTestCase(unittest.TestCase):
    def test_train_model_test(self):
        """Test method `_train_generate_md_test`."""
        params = {
            "type_map": ["Au", "O"],
            "mass_map": [196, 16],
            "init_data_prefix": "data",
            "init_data_sys": ["deepmd"],
            "init_batch_size": [16],
            "sys_configs_prefix": os.getcwd(),
            "sys_configs": [
                ["data/al.fcc.02x02x02/01.scale_pert/sys-0032/scale*/000001/POSCAR"],
                ["data/al.fcc.02x02x02/01.scale_pert/sys-0032/scale*/000000/POSCAR"]
            ],
            "numb_models": 4,
            "shuffle_poscar": False,
            "model_devi_f_trust_lo": 0.050,
            "model_devi_f_trust_hi": 0.150,
            "model_devi_e_trust_lo": 1e10,
            "model_devi_e_trust_hi": 1e10,
            "model_devi_plumed": True,
            "model_devi_jobs": [
                {"sys_idx": [0, 1], 'traj_freq': 10, "template": {"lmp": "lmp/input.lammps", "plm": "lmp/input.plumed"},
                 "rev_mat": {
                     "lmp": {"V_NSTEPS": [1000], "V_TEMP": [50, 100], "V_PRES": [1, 10]},
                     "plm": {"V_DIST0": [3, 4], "V_DIST1": [5, 6]}
                 }}
            ]
        }

        self.assertEqual(len(tasks), (len(jdata['model_devi_jobs'][0]['rev_mat']['lmp']['V_NSTEPS']) *
                                      len(jdata['model_devi_jobs'][0]['rev_mat']['lmp']['V_TEMP']) *
                                      len(jdata['model_devi_jobs'][0]['rev_mat']['lmp']['V_PRES']) *
                                      len(jdata['model_devi_jobs'][0]['rev_mat']['plm']['V_DIST0']) *
                                      len(jdata['model_devi_jobs'][0]['rev_mat']['plm']['V_DIST1']) *
                                      4))


if __name__ == '__main__':
    unittest.main()
