import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from Data_set import data_set,edge_list
import os

# os.environ["OMP_NUM_THREADS"] = "1"
random.seed(0)
torch.manual_seed(0)

# UPDATE_GLOBAL_ITER = 5
# GAMMA = 0.9
# MAX_EP = 2000
# batch_size = 32
# state_dim = 1
# hidden_size = 128
# action_dim = 2
#
# class SharedAdam(torch.optim.Adam):
#     def __init__(self, params, lr=1e-5, betas=(0.9, 0.99), eps=1e-8,
#                  weight_decay=0):
#         super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
#         # State initialization
#         for group in self.param_groups:
#             for p in group['params']:
#                 state = self.state[p]
#                 state['step'] = 0
#                 state['exp_avg'] = torch.zeros_like(p.data)
#                 state['exp_avg_sq'] = torch.zeros_like(p.data)
#
#                 # share in memory
#                 state['exp_avg'].share_memory_()
#                 state['exp_avg_sq'].share_memory_()


def gcn_message(edges):
    #msg = torch.mean(nodes.edges['h'])
    # b = edges.data["feat"][:,-1].detach().squeeze()
    # c = torch.eye(44,44)
    #
    # for i in range(len(b)):
    #     c[i][i] = b[i]
    return {'msg' : edges.src['h']}
    # msg = torch.mm(c,edges.src['h'])
    # return {'msg': torch.mm(c,edges.src['h'])}


def gcn_reduce(nodes):

    return {'h' : torch.sum(nodes.mailbox['msg'], dim=1)}


class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, device):
        super(GCNLayer, self).__init__()
        self.in_feats = in_feats
        self.linear = nn.Linear(in_feats, out_feats)
        self.device = device

    def forward(self, g, inputs):

        #g.ndata['h'] = inputs
        g.edata['h'] = inputs
        for i in range(14):
            # print(i)
            # a = g.edges[g.in_edges(i)].data['h']
            # a = torch.mean(a, dim=0)
            # a = a.view(1, self.in_feats)
            # g.nodes[i].data['h'] = a
            g.nodes[i].data['h'] = torch.mean(g.edges[g.in_edges(i)].data['h'], dim=0).view(1, self.in_feats)

        #g.send_and_recv([i for i in range(44)], gcn_message, gcn_reduce)
        g.update_all(gcn_message, gcn_reduce)

        # # 触发边的信息传递
        # g.send(g.edges(), gcn_message)
        # # 触发节点的聚合函数
        # g.recv(g.nodes(), gcn_reduce)
        # 取得节点向量
        h = g.ndata.pop('h')
        for src, dst in edge_list:
            g.edges[src, dst].data['h'] = torch.mean(torch.index_select(h, dim=0, index=torch.tensor([src, dst]).to(self.device)), dim=0).view(1, self.in_feats)
        h = g.edata['h']
        # # 线性变换
        return self.linear(h)

class Net(nn.Module):
    def __init__(self, s_dim_1, s_dim_2, hidden_size, out_dim, action_dim = 0,device = 'cpu'):
        super(Net, self).__init__()
        self.s_dim_1 = s_dim_1
        self.s_dim_2 = s_dim_2
        self.out_dim = out_dim
        #self.in_feats = s_dim
        self.activation = torch.nn.LeakyReLU(negative_slope=0.01, inplace=False)
        self.device = device
        #self.activation = torch.nn.PReLU(num_parameters=1,init=0.25)

        self.a11 = GCNLayer(s_dim_1, hidden_size, self.device)
        self.a12 = GCNLayer(hidden_size, hidden_size, self.device)
        self.a21 = GCNLayer(s_dim_2, hidden_size, self.device)
        self.a22 = GCNLayer(hidden_size, hidden_size, self.device)
        self.a3 = nn.Linear(hidden_size * 2 + action_dim, out_dim)

        # self.v11 = GCNLayer(s_dim_1, hidden_size)
        # self.v12 = GCNLayer(hidden_size, hidden_size)
        # self.v21 = GCNLayer(s_dim_2, hidden_size)
        # self.v22 = GCNLayer(hidden_size, hidden_size)
        # self.v3 = nn.Linear(hidden_size * 2, 1)
        #set_init([self.a1, self.a2, self.v1, self.v2])
        self.distribution = torch.distributions.Categorical

    def forward(self, state, action = None):
        g1, inputs1, g2, inputs2 = state
        # for src, dst in edge_list:
        #     g.edges[src, dst].data['h'] = torch.mean(torch.index_select(g.ndata['feat'], dim=0, index=torch.tensor([src, dst])), dim=0).view(1, self.in_feats)
        # inputs = g.edata['h']
        h1 = self.a11(g1, inputs1)
        h1 = self.activation(h1)
        h1 = self.a12(g1, h1)
        h1 = self.activation(h1)
        h1 = torch.mean(h1, dim=0)

        h2 = self.a21(g2, inputs2)
        h2 = self.activation(h2)
        h2 = self.a22(g2, h2)
        h2 = self.activation(h2)
        h2 = torch.mean(h2, dim=0)

        if action != None:
            out = torch.cat((h1, h2, action), dim=0)
        else:
            out = torch.cat((h1, h2), dim=0)
        out = self.a3(out)

        return out



    # def get_action(self, g, inputs, greedy=True):
    #     # for src, dst in edge_list:
    #     #     g.edges[src, dst].data['h'] = torch.mean(torch.index_select(g.ndata['feat'], dim=0, index=torch.tensor([src, dst])), dim=0).view(1, self.in_feats)
    #     #
    #     # probs,_ = self(g, g.edata['h'])
    #     probs, _ = self(g, inputs)
    #     #probs,_ = self(g)
    #     probs = F.softmax(probs, dim=1)
    #     probs[probs < 0.05] = 0.05
    #     probs[probs > 0.95] = 0.95
    #
    #     if greedy:
    #         act_id = torch.multinomial(probs, 1)
    #         return act_id
    #     else:
    #         prob, act_id = torch.topk(probs, 1, dim= 1)
    #         return act_id
    #
    # def choose_action(self, s):
    #     self.eval()
    #     logits, _ = self.forward(s)
    #     prob = F.softmax(logits, dim=1).data
    #     m = self.distribution(prob)
    #     return m.sample().numpy()[0]

    # def loss_func(self, s, a, v_t):
    #     self.train()
    #     logits, values = [],[]
    #     for g,inputs in s:
    #         l, v = self.forward(g, inputs)
    #         logits.append(l)
    #         values.append(torch.mean(v))
    #     #logits = torch.cat(logits, dim=1)
    #     #values = torch.from_numpy(np.array(values))
    #     # values = torch.cat(values, dim=1)
    #     # values = torch.mean(values, dim=0).view(batch_size,1)
    #     values = torch.tensor(values)
    #
    #     td = v_t.detach().squeeze() - values
    #     c_loss = td.pow(2)
    #
    #     X = []
    #     for l,action in zip(logits,a):
    #         probs = F.softmax(l, dim=1)
    #         m = self.distribution(probs)
    #         x = m.log_prob(torch.squeeze(action))
    #         X.append(torch.mean(x))
    #     X = torch.tensor(X)
    #     y = td.detach().squeeze()
    #     exp_v = X * td.detach().squeeze()
    #     a_loss = -exp_v
    #     total_loss = (c_loss + a_loss).mean()
    #     return total_loss
