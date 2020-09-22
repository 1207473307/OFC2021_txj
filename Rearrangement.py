import numpy as np
import RSA_methods as RM

def get_hops(path_tree, s, d):
    for path,_,_ in path_tree:
        if path[-1] == d:
            if path[0] == s:
                return len(path) - 1
            else:
                return get_hops(path_tree, s, path[0]) + len(path) - 1

def DTS(service_list):
    hops_list = []
    rea_list = []
    for path_tree, source, destination, _, _ in service_list:
        hops = 0
        for d in destination:
            hops = max(hops, get_hops(path_tree, source, d))
        hops_list.append(hops)
    hops_ave = np.mean(hops_list)
    for i in range(len(hops_list)):
        if hops_list[i] > hops_ave:
            rea_list.append(i)
    return rea_list


def QTS(service_list, G):
    q_list = []
    rea_list = []
    for path_tree, source, destination, bandwidth, _ in service_list:
        path_tree_new = RM.SFMOR(G, source, destination, bandwidth)
        if len(path_tree_new) == 0:  #临时
            path_tree_new = path_tree
        hops = 0
        hidx = 0
        hops_new = 0
        hidx_new = 0
        for d in destination:
            hops = max(hops, get_hops(path_tree, source, d))
            hops_new = max(hops_new, get_hops(path_tree_new, source, d))
        for path,len_fs,start_f in path_tree:
            hidx = max(hidx, len_fs + start_f)
        for path,len_fs,start_f in path_tree_new:
            hidx_new = max(hidx_new, len_fs + start_f)
        q = (hops_new*hidx_new)/(hops*hidx)
        q_list.append(q)
    q_ave = np.mean(q_list)
    for q in q_list:
        if q < q_ave:
            rea_list.append(q_list.index(q))
    return rea_list


def Full_rearrangement():
    pass

def Partial_rearrangement():
    pass

def DTS_F():
    pass

def DTS_P():
    pass

def QTS_F():
    pass

def QTS_P():
    pass