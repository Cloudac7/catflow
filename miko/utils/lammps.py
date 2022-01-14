import os
import re
from itertools import product


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
