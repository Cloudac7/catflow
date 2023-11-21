from catflow.tasker.utils.config import load_yaml_config
from catflow.tasker.utils.config import load_yaml_configs

def test_load_yaml_config(shared_datadir, tmp_path):
    config = load_yaml_config(shared_datadir / "first.yaml")
    config_ref = {
        "a": {
            "b": "cde",
            "f": {
                "g": [0, 1, 2]
            }
        },
    }
    assert config == config_ref

def test_load_yaml_configs(shared_datadir, tmp_path):
    config = load_yaml_configs(
        shared_datadir / "first.yaml",
        shared_datadir / "second.yaml"
    )
    config_ref = {
        "a": {
            "b": "cde",
            "f": {
                "g": [0, 1, 2]
            }
        },
        "h": "ijk",
        "l": [0, 1, 2]
    }
    assert config == config_ref