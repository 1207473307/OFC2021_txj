import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from Data_set import edge_list
# 主要定义message方法和reduce方法
# NOTE: 为了易于理解，整个教程忽略了归一化的步骤

# def gcn_message(edges):
#     # 参数：batch of edges
#     # 得到计算后的batch of edges的信息，这里直接返回边的源节点的feature.
#     return {'msg' : edges.src['h']}


# def gcn_reduce(nodes):
#     # 参数：batch of nodes.
#     # 得到计算后batch of nodes的信息，这里返回每个节点mailbox里的msg的和
#     return {'h' : torch.sum(nodes.mailbox['msg'], dim=1)}

# Define the GCNLayer module
# class GCNLayer(nn.Module):
#     def __init__(self, in_feats, out_feats):
#         super(GCNLayer, self).__init__()
#         self.linear = nn.Linear(in_feats, out_feats)
#
#     def forward(self, g, inputs):
#         # g 为图对象； inputs 为节点特征矩阵
#         # 设置图的节点特征
#         g.ndata['h'] = inputs
#         # 触发边的信息传递
#         g.send(g.edges(), gcn_message)
#         # 触发节点的聚合函数
#         g.recv(g.nodes(), gcn_reduce)
#         # 取得节点向量
#         h = g.ndata.pop('h')
#         # 线性变换
#         return self.linear(h)


def gcn_message(edges):
    #msg = torch.mean(nodes.edges['h'])
    return {'msg' : edges.src['h']}

def gcn_reduce(nodes):

    return {'h' : torch.sum(nodes.mailbox['msg'], dim=1)}

class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.in_feats = in_feats
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g, inputs):
        # g 为图对象； inputs 为节点特征矩阵
        # 设置图的节点特征
        g.edata['h'] = inputs
        for i in range(14):
            g.nodes[i].data['h'] = torch.mean(g.edges[g.in_edges(i)].data['h'], dim=0).view(1, self.in_feats)

        # 触发边的信息传递
        g.send(g.edges(), gcn_message)
        # 触发节点的聚合函数
        g.recv(g.nodes(), gcn_reduce)
        # 取得节点向量
        h = g.ndata.pop('h')
        for src, dst in edge_list:
            g.edges[src, dst].data['h'] = torch.mean(torch.index_select(h, dim=0, index=torch.tensor([src, dst])), dim=0).view(1, self.in_feats)
        # 线性变换
        return self.linear(g.edata['h'])

class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super(GCN, self).__init__()
        #self.n_hidden = 4
        self.in_feats = in_feats
        self.gcn1 = GCNLayer(in_feats, hidden_size)
        self.gcn2 = GCNLayer(hidden_size, hidden_size)
        self.gcn3 = GCNLayer(hidden_size, hidden_size)
        self.gcn4 = GCNLayer(hidden_size, hidden_size)
        self.gcn5 = GCNLayer(hidden_size, num_classes)

    def forward(self, g, inputs):
        h = self.gcn1(g, inputs)
        h = torch.relu(h)
        h = self.gcn2(g, h)
        h = torch.relu(h)
        h = self.gcn3(g, h)
        h = torch.relu(h)
        h = self.gcn4(g, h)
        h = torch.relu(h)
        h = self.gcn5(g, h)
        return h

    def get_action(self, g, greedy=True):
        for src, dst in edge_list:
            g.edges[src, dst].data['h'] = torch.mean(torch.index_select(g.ndata['feat'], dim=0, index=torch.tensor([src, dst])), dim=0).view(1, self.in_feats)

        probs = self(g, g.edata['h'])
        #probs = F.softmax(probs)
        # probs[probs < 10**-3] = 10**-3
        # probs[probs > 0.95] = 0.95

        if greedy:
            act_id = torch.multinomial(probs, 1)
            return act_id
        else:
            prob, act_id = torch.topk(probs, 1, dim= 1)
            return act_id


