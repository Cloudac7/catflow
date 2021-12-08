import copy


def parse_cur_job_revmat(cur_job, use_plm=False):
    templates = [cur_job['template']['lmp']]
    if use_plm:
        templates.append(cur_job['template']['plm'])
    revise_keys = []
    revise_values = []
    if 'rev_mat' not in cur_job.keys():
        cur_job['rev_mat'] = {}
    if 'lmp' not in cur_job['rev_mat'].keys():
        cur_job['rev_mat']['lmp'] = {}
    for ii in cur_job['rev_mat']['lmp'].keys():
        revise_keys.append(ii)
        revise_values.append(cur_job['rev_mat']['lmp'][ii])
    n_lmp_keys = len(revise_keys)
    if use_plm:
        if 'plm' not in cur_job['rev_mat'].keys():
            cur_job['rev_mat']['plm'] = {}
        for ii in cur_job['rev_mat']['plm'].keys():
            revise_keys.append(ii)
            revise_values.append(cur_job['rev_mat']['plm'][ii])
    revise_matrix = expand_matrix_values(revise_values)
    return revise_keys, revise_matrix, n_lmp_keys


def expand_matrix_values(target_list, cur_idx=0):
    nvar = len(target_list)
    if cur_idx == nvar:
        return [[]]
    else:
        res = []
        prev = expand_matrix_values(target_list, cur_idx + 1)
        for ii in target_list[cur_idx]:
            tmp = copy.deepcopy(prev)
            for jj in tmp:
                jj.insert(0, ii)
                res.append(jj)
        return res


def find_only_one_key(lmp_lines, key):
    found = []
    for idx in range(len(lmp_lines)):
        words = lmp_lines[idx].split()
        nkey = len(key)
        if len(words) >= nkey and words[:nkey] == key :
            found.append(idx)
    if len(found) > 1:
        raise RuntimeError('found %d keywords %s' % (len(found), key))
    if len(found) == 0:
        raise RuntimeError('failed to find keyword %s' % key)
    return found[0]


def revise_by_keys(lmp_lines, keys, values):
    for kk, vv in zip(keys, values):
        for ii in range(len(lmp_lines)):
            lmp_lines[ii] = lmp_lines[ii].replace(kk, str(vv))
    return lmp_lines


def revise_lmp_input_plm(lmp_lines, in_plm, out_plm='output.plumed'):
    idx = find_only_one_key(lmp_lines, ['fix', 'dpgen_plm'])
    lmp_lines[idx] = "fix            dpgen_plm all plumed plumedfile %s outfile %s\n" % (in_plm, out_plm)
    return lmp_lines