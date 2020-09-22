import networkx as nx
import Net_Graph as NG
import numpy as np

path_map = [[[]for i in range(15)]for i in range(14)]
for a in NG.G.nodes:
    for b in NG.G.nodes:
        if a != b:
            path_map[a][b] = [nx.dijkstra_path(NG.G, a, b, weight='weight'), nx.dijkstra_path_length(NG.G, a, b, weight='weight')]
        else:
            path_map[a][b] = [[a], 0]

np.save('path_map.npy',path_map)


K = 3
K_path_map = [[[]for i in range(15)]for i in range(14)]

def YenKSP(G, source, target, K):
    path_list = []
    path_list.append(nx.dijkstra_path(G, source, target, weight='weight'))


    for k in range(K-1):
        temp_path = []
        for i in range(len(path_list[k])-1):
            tempG = G.copy() #复制一份图 供删减操作
            spurNode = path_list[k][i]
            rootpath = path_list[k][:i+1]
            len_rootpath = nx.dijkstra_path_length(tempG, source, spurNode, weight='weight')

            for p in path_list:
                if rootpath == p[:i+1]:
                    if tempG.has_edge(p[i], p[i+1]):
                        tempG.remove_edge(p[i], p[i+1])  #防止与祖先状态重复
            tempG.remove_nodes_from(rootpath[:-1])  #防止出现环路
            if not(nx.has_path(tempG, spurNode, target)):
                continue  #如果无法联通，跳过该偏移路径

            spurpath = nx.dijkstra_path(tempG, spurNode, target, weight='weight')
            len_spurpath = nx.dijkstra_path_length(tempG, spurNode, target, weight='weight')

            totalpath = rootpath[:-1] + spurpath
            len_totalpath = len_rootpath + len_spurpath
            temp_path.append([totalpath, len_totalpath])
        if len(temp_path)==0:
            break

        temp_path.sort(key = (lambda x:[x[1], len(x[0])]))  #按路程长度为第一关键字，节点数为第二关键字，升序排列
        path_list.append(temp_path[0][0])

    return path_list

for a in NG.G.nodes:
    for b in NG.G.nodes:
        if a != b:
            K_path_map[a][b] = YenKSP(NG.G, a, b, K)
        else:
            K_path_map[a][b] = []

np.save('K_path_map.npy', K_path_map)
