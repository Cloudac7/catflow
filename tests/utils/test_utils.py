import json
from miko.utils.lammps import *


def test_dict_lists_combination():
    ori_dict = {
        "param1": [0, 1, 2],
        "param2": [3, 4],
        "param3": [5, 6]
    }
    prod_dict_list = [
        {"param1": 0, "param2": 3, "param3": 5},
        {"param1": 0, "param2": 3, "param3": 6},
        {"param1": 0, "param2": 4, "param3": 5},
        {"param1": 0, "param2": 4, "param3": 6},
        {"param1": 1, "param2": 3, "param3": 5},
        {"param1": 1, "param2": 3, "param3": 6},
        {"param1": 1, "param2": 4, "param3": 5},
        {"param1": 1, "param2": 4, "param3": 6},
        {"param1": 2, "param2": 3, "param3": 5},
        {"param1": 2, "param2": 3, "param3": 6},
        {"param1": 2, "param2": 4, "param3": 5},
        {"param1": 2, "param2": 4, "param3": 6},
    ]
    assert dict_lists_combination(ori_dict) == prod_dict_list


def test_parse_template(datadir):
    contents = (datadir / 'conf-lammps.json').read_text()
    conf_dict = json.loads(contents)
    print(conf_dict)


def test_check_keywords():
    pass


def test_substitute_keywords():
    pass


def test_insert_plm_into_lmp():
    pass


def test_convert_init_structures():
    pass
