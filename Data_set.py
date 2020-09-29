import dgl
import torch


edge_list = [(0, 1), (0, 2), (0, 7), (1, 2), (1, 3), (2, 5),
             (3, 4), (3, 10), (4, 5), (4, 6), (5, 9), (5, 13),
             (6, 7), (6, 9), (7, 8), (8, 9), (8, 11), (8, 12),
             (10, 11), (10, 12), (11, 13), (12, 13),
             (1, 0), (2, 0), (7, 0), (2, 1), (3, 1), (5, 2),
             (4, 3), (10, 3), (5, 4), (6, 4), (9, 5), (13, 5),
             (7, 6), (9, 6), (8, 7), (9, 8), (11, 8), (12, 8),
             (11, 10), (12, 10), (13, 11), (13, 12)]

def get_edge_features(G):
    edge_feature = []
    for src,dst in edge_list:
        fs = G[src][dst]['fs']
        num = 0
        block = 0
        flag = 0
        fist_avaliable = -1
        for i in range(len(fs)):
            if fs[i] == 1:
                fist_avaliable = i
                break
        for i in fs:
            if i == 1:
                num += 1
                if flag == 0:
                    block += 1
                    flag = 1
            else:
                flag = 0
        if num == 0:
            #edge_feature.append([0, 0, fist_avaliable, 0])
            # edge_feature.append([0, 0, fist_avaliable])  # 防止出现除数为0
            #edge_feature.append([0, 0])
            edge_feature.append([0])
        else:
            #edge_feature.append([num, num / block, fist_avaliable, 0])
            #edge_feature.append([num, num / block, fist_avaliable])
            #edge_feature.append([num, num / block])
            edge_feature.append([num])
    return edge_feature
    # return [[0] for i in range(44)]

def get_edge_features_2(G, path_tree):
    # edge_feature = [[0, 0] for i in range(len(edge_list))]
    edge_feature = [[0] for i in range(len(edge_list))]
    for path, len_fs, start_f in path_tree:
        for i in range(len(path) - 1):
            # edge_feature[edge_list.index((path[i],path[i+1]))] = [len_fs, start_f]
            edge_feature[edge_list.index((path[i], path[i + 1]))] = [len_fs]
    return edge_feature

def creat_g(edge_list):
    # g = dgl.DGLGraph()
    # g.add_nodes(14)
    #
    # for src, dst in edge_list:
    #     g.add_edges(src, dst)
    g = dgl.graph(edge_list)


    return g

def data_set(service, G):
    path_tree, source, destination, bandwidth, time = service
    g = creat_g(edge_list)
    #node_tags = [0 for i in range(14)]  # 0:not in the tree ; 1:source ; 2：destination ; 3:leave node ; 4:intermediate node
    node_features = [[1, 0, 0, 0, 0] for i in range(14)]
    edge_features = get_edge_features(G)

    #g.add_nodes(14)
    #node_features[source -1].append(1)
    node_features[source] = [0, 1, 0, 0, 0]
    for d in destination:
        #node_features[d - 1].append(2)
        node_features[d] = [0, 0, 1, 0, 0]

    for path,_,_ in path_tree:

        if path[0] not in [source] + destination:
            #node_features[path[0] - 1].append(3)
            node_features[path[0]] = [0, 0, 0, 1, 0]
        if path[-1] not in [source] + destination:
            #node_features[path[0] - 1].append(3)
            node_features[path[0]] = [0, 0, 0, 1, 0]
        for i in range(len(path) - 1):
            edge_features[edge_list.index((path[i],path[i+1]))][-1] = 1
            if i != 0:
                #node_features[path[i] - 1].append(4)
                node_features[path[i]] = [0, 0, 0, 0, 1]
            #g.add_edge(path[i], path[i+1])
            #edge_features.append(get_edge_features(G, path[i], path[i+1]))

    g.ndata['feat'] = torch.tensor(node_features, dtype=torch.float)
    g.edata['feat'] = torch.tensor(edge_features, dtype=torch.float)
    return g

def data_set_2(service, G, device):
    path_tree, source, destination, bandwidth, time = service
    g1 = creat_g(edge_list)#.to(device)

    edge_features1 = get_edge_features(G)
    g1.edata['feat'] = torch.tensor(edge_features1, dtype=torch.float)
    g1 = g1.to(device)
    # edge_list2, edge_features2 = [], []
    # for path, len_fs, start_f in path_tree:
    #     for i in range(len(path) - 1):
    #         edge_list2.append([path[i],path[i+1]])
    #         edge_features2.append([len_fs, start_f])

    g2 = creat_g(edge_list)
    edge_features2 = get_edge_features_2(G, path_tree)
    g2.edata['feat'] = torch.tensor(edge_features2, dtype=torch.float)
    g2 = g2.to(device)
    return [g1, g1.edata['feat'], g2, g2.edata['feat']]