import os
import re
from pathlib import Path
from itertools import product

import numpy as np

from catflow.utils.log_factory import logger

def lammps_variable_parser(input_file):
    from pymatgen.io.lammps.inputs import LammpsInputFile

    lammps_input = LammpsInputFile.from_file(input_file)
    keys, values = [], []
    for k, v in lammps_input.stages[0]['commands']:
        if k == "variable":
            keys.append(v.split()[0])
            try:
                values.append(float(v.split()[2]))
            except ValueError:
                values.append(v.split()[2])
    variable_dict = {k: v for k, v in zip(keys, values)}
    return variable_dict

def dict_lists_combination(ori_dict: dict):
    keys = ori_dict.keys()
    lists = ori_dict.values()
    return [dict(zip(keys, values)) for values in product(*lists)]


def parse_template(test_job, sys_idx):
    templates = test_job['template']
    rev_mat = test_job['rev_mat']
    sys_rev_mat = test_job['sys_rev_mat'][str(sys_idx)]
    total_rev_mat = {**rev_mat, **sys_rev_mat}
    template_list = templates.values()
    revise_matrix = dict_lists_combination(total_rev_mat)
    return template_list, revise_matrix


def check_keywords(lmp_lines: list, key: list):
    found = []
    regex_pattern = r'^' + r'\s+'.join(key)
    for idx, line in enumerate(lmp_lines):
        words = line.split()
        if re.match(regex_pattern, line):
            found.append(idx)
    if len(found) > 1:
        raise RuntimeError('%d keywords %s found' % (len(found), key))
    elif len(found) == 0:
        raise RuntimeError('keyword %s not found' % key)
    else:
        return found[0]


def substitute_keywords(lines: list, sub_dict: dict):
    for i in range(len(lines)):
        for k, v in sub_dict.items():
            lines[i] = lines[i].replace(k, str(v))
    return lines


def insert_plm_into_lmp(lmp_lines, in_plm, out_plm='output.plumed'):
    idx = check_keywords(lmp_lines, ['fix', 'dpgen_plm'])
    lmp_lines[idx] = "fix            dpgen_plm all plumed plumedfile %s outfile %s\n" % (in_plm, out_plm)
    return lmp_lines


def convert_init_structures(test_job, sys_idx):
    from ase.io import read
    init_stc_path = os.path.join(test_job['md_sys_configs_prefix'], test_job['md_sys_configs'][sys_idx])
    stc = read(init_stc_path, format=test_job['md_sys_configs_format'])
    return stc


def read_model_deviation(model_devi_path: Path):
    model_devi_path = model_devi_path.resolve()
    try:
        steps = np.loadtxt(model_devi_path, usecols=0)
        max_devi_f = np.loadtxt(model_devi_path, usecols=4)
        max_devi_e = np.loadtxt(model_devi_path, usecols=3)
    except FileNotFoundError as err:
        logger.error('Please select an existing model_devi.out')
        raise err
    return steps, max_devi_f, max_devi_e

def read_dump_energy(lammps_log_path: Path):
    start, final = 0, 0
    with open(lammps_log_path, 'r') as f:
        for i, line in enumerate(f):
            key_line = line.strip()
            if 'Step Temp' in key_line:
                start = i + 1
            elif 'Loop time of' in key_line:
                final = i
    with open(lammps_log_path, 'r') as f:
        lines = f.readlines()[start:final]
    pot_energy = np.array(
        [p.split()[2] for p in lines if 'WARNING' not in p]).astype('float')
    kin_energy = np.array(
        [p.split()[3] for p in lines if 'WARNING' not in p]).astype('float')
    tot_energy = np.array(
        [p.split()[4] for p in lines if 'WARNING' not in p]).astype('float')
    return pot_energy, kin_energy, tot_energy
