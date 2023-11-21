from typing import Tuple
from pathlib import Path
from collections import defaultdict
from collections import ChainMap
from ruamel.yaml import YAML
from pydantic import BaseModel


def load_yaml_config(path: Path):
    yaml = YAML(typ='safe')
    with open(path) as f:
        config = yaml.load(f)
    return config


def load_yaml_configs(*paths: Path):
    config = ChainMap(*[load_yaml_config(path) for path in paths])
    return config

def get_item_from_list(my_list, **kwargs):
    for index, item in enumerate(my_list):
        if all(dict(item).get(key) == value for key, value in kwargs.items()):
            return index, item
        
def dump_yaml_config(path: Path, config: BaseModel):
    yaml = YAML(typ='safe')
    data = yaml.load(config.json())
    with open(path, 'w') as f:
        yaml.dump(data, f)
