#!/usr/bin/python3
from ase.cluster import Icosahedron
from ase.cluster import Decahedron
from math import sqrt
from ase import Atoms, Atom
from ase.io import write, read
from itertools import combinations
from ase.utils.structure_comparator import SymmetryEquivalenceCheck


def all_m13_cluster_adsoption(mele='Cu', absorption='O', dist_from_cent=2.5):
    # construct a cluster
    cluster = Icosahedron('Cu', 2)
    cluster.set_cell((0, 0, 0))

    # find the face for adsorption
    face_list = []
    tot_atom = len(cluster.get_positions())
    face_len = cluster.get_distance(9, 10)
    for i in range(1, tot_atom-2):
        for j in range(i+1, tot_atom-1):
            for k in range(j+1, tot_atom):
                if (cluster.get_distance(i, j) == face_len) and (cluster.get_distance(i, k) == face_len) and (cluster.get_distance(j, k) == face_len):
                    face_list.append([i, j, k])

    # generate all adsorbate site (present by oxygen)
    dist_from_cent = 2.5
    for i in range(len(face_list)):
        v1 = cluster.get_positions()[face_list[i][1]] - \
            cluster.get_positions()[face_list[i][0]]
        v2 = cluster.get_positions()[face_list[i][2]] - \
            cluster.get_positions()[face_list[i][0]]
        v_dia = v1+v2
        v_dia = v_dia/3
        v_dia = cluster.get_positions()[face_list[i][0]]+v_dia
        v_cent = v_dia-cluster.get_positions()[0]
        v_cent_len = sqrt(v_cent[0]**2+v_cent[1]**2+v_cent[2]**2)
        v_cent = v_cent*dist_from_cent/v_cent_len
        v_cent = v_cent+cluster.get_positions()[0]
        o = Atom(absorption, v_cent)
        cluster += o
    return cluster


def reduced_adsorption_cluster(element='Cu', absorption='O', dist_from_cent=2.5, select=3):
    cluster_ori = Icosahedron('Cu', 2)
    cluster_ori.set_cell((15,15,15))
    cluster = all_m13_cluster_adsoption(element, absorption, dist_from_cent)

    # find all combination of three oxygen
    ad_item = list(range(13, len(cluster.numbers)))
    cluster_list = [cluster_ori + cluster[i] for i in combinations(ad_item, select)]
    print(len(cluster_list))

    # delete the repeated structure
    ad_len_list = []
    for i in cluster_list:
        x = [i.get_all_distances()[j] for j in combinations(range(13,13+select), select)]
        x.sort()
        ad_len_list.append(['%.6f' % elem for elem in x])

    no_repeat_list = [0]
    for i in range(1, len(ad_len_list)):
        x = [(ad_len_list[j] == ad_len_list[i]) for j in no_repeat_list]
        if all(judge is False for judge in x):
            no_repeat_list.append(i)
    print(no_repeat_list)
    # write all the unique structure into xyz file
    for i in no_repeat_list:
        write('{0}13{1}{2}-{3}.xyz'.format(element,
                                           absorption, select, i), cluster_list[i])

reduced_adsorption_cluster(element='Cu', absorption='O', dist_from_cent=4, select=2)