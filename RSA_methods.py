import Net_Graph as NG
import numpy as np
import math
path = '/home/txj/OFC2021/OFC2021_txj/'
path_map = np.load(path+'path_map.npy', allow_pickle=True)
K_path_map = np.load(path + 'K_path_map.npy', allow_pickle=True)

def updata_fs(G, path, len_fs, start_f):
    if len(path) <= 1:
        return
    for i in range(len_fs):
        for j in range(len(path) - 1):
            G[path[j]][path[j + 1]]['fs'][start_f + i] = 0


def release_fs(G, path, len_fs, start_f):
    if len(path) <= 1:
        return
    for i in range(len_fs):
        for j in range(len(path) - 1):
            G[path[j]][path[j + 1]]['fs'][start_f + i] = 1

def modulation_level(bandwidth, len_path):
    if len_path <= 625:
        len_fs = math.ceil(bandwidth / 12.5 / 4)
    elif len_path <= 1250:
        len_fs = math.ceil(bandwidth / 12.5 / 3)
    elif len_path <= 2500:
        len_fs = math.ceil(bandwidth / 12.5 / 2)
    else:
        len_fs = math.ceil(bandwidth / 12.5 / 1)
    return len_fs

def SP_FF(G, path, len_fs, num_fs=NG.num_fs):  # Shortest Paths——First Fit
    block = 1
    start_f = -1
    for i in range(num_fs - len_fs + 1):
        flag1 = 0
        for j in range(len_fs):
            for k in range(len(path) - 1):
                if G[path[k]][path[k + 1]]['fs'][i + j] == 0:
                    flag1 = 1
                    break
            if flag1 == 1:
                break
        if flag1 == 0:
            start_f = i
            block = 0
            break
    if block == 0:
        return start_f
    else:
        return -1

def SPT(G, source, destination, bandwidth,  num_fs = NG.num_fs):
    tree = []
    block = 0
    for d in destination:
        path = path_map[source, d][0]
        len_path = path_map[source, d][1]
        len_fs = modulation_level(bandwidth, len_path)
        start_f = SP_FF(G, path, len_fs)
        if start_f == -1:
            block = 1
            break
        updata_fs(G, path, len_fs, start_f)
        tree.append([path, len_fs, start_f])

    for path, len_fs, start_f in tree:
        release_fs(G, path, len_fs, start_f)
    if block == 1:
        return []
    else:
        return tree

def MST(G, source, destination, bandwidth,  num_fs = NG.num_fs):
    tree = []

def cal_min_cut_num(G, path, len_fs, num_fs = NG.num_fs):
    block = 1
    min_n_cut = float('inf')
    start_f = -1
    available_block = []
    len_block = 0
    for i in range(num_fs):
        flag1 = 0
        for k in range(len(path) - 1):
            if G[path[k]][path[k + 1]]['fs'][i] == 0:
                flag1 = 1
                if len_block >= len_fs:
                    block = 0
                    available_block.append([i - len_block, i-1])
                len_block = 0
                break
        if flag1 == 0:
            len_block = len_block + 1
    if len_block >= len_fs:
        block = 0
        available_block.append([num_fs - len_block, num_fs - 1])

    if not block:
        for start, end in available_block:
            if start == 0:
                min_n_cut = 0
                start_f = 0
                break
            elif end == num_fs-1:
                min_n_cut = 0
                start_f = num_fs - len_fs
                break
            else:
                n_cut = 0
                for k in range(len(path) - 1):
                    if G[path[k]][path[k + 1]]['fs'][start-1] == 1:
                        n_cut += 1
                if n_cut < min_n_cut:
                    min_n_cut = n_cut
                    start_f = start
                n_cut = 0
                for k in range(len(path) - 1):
                    if G[path[k]][path[k + 1]]['fs'][end + 1] == 1:
                        n_cut += 1
                if n_cut < min_n_cut:
                    min_n_cut = n_cut
                    start_f = end - len_fs + 1

    return block, min_n_cut, start_f

def get_path_len(G, path):
    l = 0
    for i in range(len(path) - 1):
        l += G[path[i]][path[i+1]]['weight']

    return l

def KSP_FA(G, path_list, bandwidth, num_fs = NG.num_fs):
    block = 1
    start_f = -1
    path_min_cut = float("inf")
    selected_path = []
    selected_path_len = float("inf")
    for path in path_list:
        path_len = get_path_len(G,path)
        block, tmp_path_min_cut, tmp_start_f = cal_min_cut_num(G, path, modulation_level(bandwidth, path_len))
        if tmp_path_min_cut < path_min_cut and not block:
            path_min_cut = tmp_path_min_cut
            selected_path = path
            start_f = tmp_start_f
            selected_path_len = path_len
    return selected_path, start_f, path_min_cut, selected_path_len

def SFMOR(G, source, destination, bandwidth,  num_fs = NG.num_fs):
    tree = []
    block = 0
    Vin = [source]
    Vout = destination.copy()
    while len(Vout) != 0:
        p = []
        for i in Vin:
            for j in Vout:
                p.append(KSP_FA(G, K_path_map[i][j], bandwidth))
        p = sorted(p, key=lambda x:(x[2],x[3]))
        path ,start_f,_, len_path = p[0]
        # path = p[0][0]
        # len_path = p[0][1]
        #len_path = get_path_len(G, path)
        len_fs = modulation_level(bandwidth, len_path)
        #start_f = SP_FF(G, path, len_fs)
        if start_f == -1:
            block = 1
            break
        updata_fs(G, path, len_fs, start_f)
        tree.append([path, len_fs, start_f])
        Vin.append(path[-1])
        Vout.remove(path[-1])
    for path, len_fs, start_f in tree:
        release_fs(G, path, len_fs, start_f)
    if block == 1:
        return []
    else:
        return tree

def SFMOR_FF(G, source, destination, bandwidth,  num_fs = NG.num_fs):
    tree = []
    block = 0
    Vin = [source]
    Vout = destination.copy()
    while len(Vout) != 0:
        p = []
        for i in Vin:
            for j in Vout:
                for path in K_path_map[i][j]:
                    p.append([path, get_path_len(G,path)])
        p = sorted(p, key=lambda x:x[1])
        flag = 1
        for path, len_path in p:
            len_fs = modulation_level(bandwidth, len_path)
            start_f = SP_FF(G, path, len_fs)
            if start_f != -1:
                flag = 0
                updata_fs(G, path, len_fs, start_f)
                tree.append([path, len_fs, start_f])
                Vin.append(path[-1])
                Vout.remove(path[-1])
                break
        if flag == 1:
            block = 1
            break

    for path, len_fs, start_f in tree:
        release_fs(G, path, len_fs, start_f)
    if block == 1:
        return []
    else:
        return tree