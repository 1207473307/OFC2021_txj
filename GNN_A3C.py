import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from utils import v_wrap, set_init, push_and_pull, record
import RSA_methods as RM
import Net_Graph as NG
import random
import numpy as np
from RSA_methods import KSP_FA
from Data_set import data_set,edge_list
from visdom import Visdom
import time
import os

os.environ["OMP_NUM_THREADS"] = "1"
random.seed(0)
torch.manual_seed(0)

UPDATE_GLOBAL_ITER = 5
GAMMA = 0.9
MAX_EP = 2000
batch_size = 32
state_dim = 1
hidden_size = 128
action_dim = 2

class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-5, betas=(0.9, 0.99), eps=1e-8,
                 weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        # State initialization
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                # share in memory
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

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
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.in_feats = in_feats
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g, inputs):

        #g.ndata['h'] = inputs
        g.edata['h'] = inputs
        for i in range(14):
            g.nodes[i].data['h'] = torch.mean(g.edges[g.in_edges(i)].data['h'], dim=0).view(1, self.in_feats)

        g.send_and_recv([i for i in range(44)], gcn_message, gcn_reduce)

        # # 触发边的信息传递
        # g.send(g.edges(), gcn_message)
        # # 触发节点的聚合函数
        # g.recv(g.nodes(), gcn_reduce)
        # 取得节点向量
        h = g.ndata.pop('h')
        for src, dst in edge_list:
            g.edges[src, dst].data['h'] = torch.mean(torch.index_select(h, dim=0, index=torch.tensor([src, dst])), dim=0).view(1, self.in_feats)
        h = g.edata['h']
        # # 线性变换
        return self.linear(h)

class Net(nn.Module):
    def __init__(self, s_dim, hidden_size,a_dim):
        super(Net, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.in_feats = s_dim
        self.activation = torch.nn.LeakyReLU(negative_slope=0.01, inplace=False)
        #self.activation = torch.nn.PReLU(num_parameters=1,init=0.25)

        self.a1 = GCNLayer(s_dim, hidden_size)
        self.a2 = GCNLayer(hidden_size, hidden_size)
        self.a3 = GCNLayer(hidden_size, hidden_size)
        self.a4 = GCNLayer(hidden_size, hidden_size)
        self.a5 = GCNLayer(hidden_size, a_dim)

        self.v1 = GCNLayer(s_dim, hidden_size)
        self.v2 = GCNLayer(hidden_size, hidden_size)
        self.v3 = GCNLayer(hidden_size, hidden_size)
        self.v4 = GCNLayer(hidden_size, hidden_size)
        self.v5 = GCNLayer(hidden_size, 1)
        #set_init([self.a1, self.a2, self.v1, self.v2])
        self.distribution = torch.distributions.Categorical

    def forward(self, g, inputs):
        # for src, dst in edge_list:
        #     g.edges[src, dst].data['h'] = torch.mean(torch.index_select(g.ndata['feat'], dim=0, index=torch.tensor([src, dst])), dim=0).view(1, self.in_feats)
        # inputs = g.edata['h']
        h1 = self.a1(g, inputs)
        h1 = self.activation(h1)
        h1 = self.a2(g, h1)
        h1 = self.activation(h1)
        # h1 = self.a3(g, h1)
        # h1 = self.activation(h1)
        # h1 = self.a4(g, h1)
        # h1 = self.activation(h1)
        h1 = self.a5(g, h1)
        #h1 = F.softmax(h1, dim =1)

        h2 = self.v1(g, inputs)
        h2 = self.activation(h2)
        h2 = self.v2(g, h2)
        h2 = self.activation(h2)
        # h2 = self.v3(g, h2)
        # h2 = self.activation(h2)
        # h2 = self.v4(g, h2)
        # h2 = self.activation(h2)
        h2 = self.v5(g, h2)

        return h1, h2



    def get_action(self, g, inputs, greedy=True):
        # for src, dst in edge_list:
        #     g.edges[src, dst].data['h'] = torch.mean(torch.index_select(g.ndata['feat'], dim=0, index=torch.tensor([src, dst])), dim=0).view(1, self.in_feats)
        #
        # probs,_ = self(g, g.edata['h'])
        probs, _ = self(g, inputs)
        #probs,_ = self(g)
        probs = F.softmax(probs, dim=1)
        probs[probs < 0.05] = 0.05
        probs[probs > 0.95] = 0.95

        if greedy:
            act_id = torch.multinomial(probs, 1)
            return act_id
        else:
            prob, act_id = torch.topk(probs, 1, dim= 1)
            return act_id
    #
    # def choose_action(self, s):
    #     self.eval()
    #     logits, _ = self.forward(s)
    #     prob = F.softmax(logits, dim=1).data
    #     m = self.distribution(prob)
    #     return m.sample().numpy()[0]

    def loss_func(self, s, a, v_t):
        self.train()
        logits, values = [],[]
        for g,inputs in s:
            l, v = self.forward(g, inputs)
            logits.append(l)
            values.append(torch.mean(v))
        #logits = torch.cat(logits, dim=1)
        #values = torch.from_numpy(np.array(values))
        # values = torch.cat(values, dim=1)
        # values = torch.mean(values, dim=0).view(batch_size,1)
        values = torch.tensor(values)

        td = v_t.detach().squeeze() - values
        c_loss = td.pow(2)

        X = []
        for l,action in zip(logits,a):
            probs = F.softmax(l, dim=1)
            m = self.distribution(probs)
            x = m.log_prob(torch.squeeze(action))
            X.append(torch.mean(x))
        X = torch.tensor(X)
        y = td.detach().squeeze()
        exp_v = X * td.detach().squeeze()
        a_loss = -exp_v
        total_loss = (c_loss + a_loss).mean()
        return total_loss


class GNN_agent(mp.Process):
    def __init__(self, net_graph,  state_dim, action_dim, gnet, opt, global_ep, global_ep_r, res_queue, name,
                 model_path=None, summary_writer=None):
        super(GNN_agent, self).__init__()
        self.name = 'w%02i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt = gnet, opt
        self.lnet = Net(state_dim, hidden_size, action_dim)  # local network
        self.G = net_graph.copy()
        self.method = RM.SFMOR
        self.service_list = []
        #self.criterion = criterion
        # self.model = model
        # self.device = device
        # self.optimizer = optimizer
        self.episodes = 1000
        self.batch_size = batch_size
        self.path_map = np.load('path_map.npy', allow_pickle=True)
        self.K_path_map = np.load('K_path_map.npy', allow_pickle=True)

    def random_request(self):
        source = random.randint(0, 13)
        num_destination = random.randint(2, 5)
        destination = []
        for i in range(num_destination):
            d = random.randint(0, 13)
            while d == source or d in destination:
                d = random.randint(0, 13)
            destination.append(d)
        # len_fs = random.randint(1, 20)
        bandwidth = random.randint(1, 500)
        time = random.randint(1, 100)

        # return source, destination, len_fs, time
        return source, destination, bandwidth, time

    def node_join(self):
        if len(self.service_list) == 0:
            return -1, -1

        i = random.randint(0, len(self.service_list) - 1)
        source = self.service_list[i][1]
        destination = self.service_list[i][2]
        if len(destination) > 10:
            return -1, -1
        d = random.randint(0, 13)
        while d == source or d in destination:
            d = random.randint(0, 13)
        return i, d

    def node_leave(self):
        if len(self.service_list) == 0:
            return -1, -1

        i = random.randint(0, len(self.service_list) - 1)
        # while len(self.service_list[i][2]) <= 1:
        #     i = random.randint(0, len(self.service_list) - 1)
        source = self.service_list[i][1]
        destination = self.service_list[i][2]
        if len(destination) <= 1:
            return -1, -1
        d = random.choice(destination)

        return i, d

    def update_fs(self, path, len_fs: int, start_f: int):
        if len(path) <= 1:
            return
        for i in range(len_fs):
            for j in range(len(path) - 1):
                self.G[path[j]][path[j + 1]]['fs'][start_f + i] = 0

    def release_fs(self, path, len_fs: int, start_f: int):
        if len(path) <= 1:
            return
        for i in range(len_fs):
            for j in range(len(path) - 1):
                self.G[path[j]][path[j + 1]]['fs'][start_f + i] = 1

    def update_request(self, path_tree):
        for path, len_fs, start_f in path_tree:
            self.update_fs(path, len_fs, start_f)

    def release_request(self, time_to):
        remove_list = []
        flag = 0
        for r in self.service_list:
            r[-1] = r[-1] - time_to
            if r[-1] <= 0:
                flag = 1
                for path, len_fs, start_f in r[0]:
                    self.release_fs(path, len_fs, start_f)
                remove_list.append(r)
        for i in remove_list:
            self.service_list.remove(i)
        return flag

    def attempt(self, service, action):
        path_tree, source, destination, bandwidth, t = service
        edge_tree = []
        edge_state = []
        block = 0
        for path in path_tree:
            for i in range(len(path[0]) - 1):
                edge_tree.append((path[0][i], path[0][i + 1]))

        for i in range(len(edge_list)):
            if edge_list[i] in edge_tree:
                edge_state.append(1)
            else:
                edge_state.append(0)
        for i in range(len(action)):
            if action[i] == 1:
                if edge_list[i] not in edge_tree:
                    block += 1
                else:
                    block -= 1
            else:
                if edge_list[i] in edge_tree:
                    block += 1
                else:
                    block -= 1

        return block,edge_state

    def attempt_n(self, service, action):
        path_tree, source, destination, bandwidth, t = service
        block = 0
        for i in range(len(action)):
            #if i in destination:
            if action[i] not in [source]+destination or action[i] == i:
                block += 1
            else:
                block -= 1
        return block

    def run(self):
        episode_size = 1000
        num_episode = 0
        num_block = 0
        num_request = 0
        ep_block = 0
        blocking_rate_list = []
        r_a = []
        r_s = []

        # 将窗口类实例化
        viz = Visdom()
        # 创建窗口并初始化
        viz.line([0.], [0], win='train_loss', opts=dict(title='train_loss'))

        buffer_s, buffer_a, buffer_r, buffer_b, buffer_n = [], [], [], [], []
        while self.g_ep.value < MAX_EP:
            # while num_episode < 10:
            time_to = 1
            for i in range(len(buffer_n)):
                buffer_n[i] += 1
            flag = self.release_request(time_to)
            if flag == 1:  # rearrangement
                for i in range(len(self.service_list)):
                    g = data_set(self.service_list[i], self.G)
                    # for src, dst in edge_list:
                    #     g.edges[src, dst].data['h'] = torch.sum(
                    #         torch.index_select(g.ndata['feat'], dim=0, index=torch.tensor([src, dst])), dim=0).view(1,state_dim)
                    # inputs = g.edata['h']

                    #inputs = g.ndata['feat']
                    inputs = g.edata['feat']

                    buffer_s.append([g, inputs])
                    action = self.lnet.get_action(g, inputs)
                    block,edge_state = self.attempt(self.service_list[i], action)
                    #block = self.attempt_n(self.service_list[i], action)
                    #print('block:',block)
                    r_a.append(action.numpy())
                    r_s.append(edge_state)
                    #r_s.append([self.service_list[i][1:3]])
                    buffer_a.append(action)
                    buffer_b.append(block)
                    buffer_r.append(-block)
                    buffer_n.append(1)

            num_request += 1
            #print(num_request)
            mode = random.randint(0, 2)
            #mode = 0

            if mode == 0:  # a multicast session  first appears
                source, destination, bandwidth, t = self.random_request()
                #source, destination, bandwidth, t = 0, [1,2], 1, random.randint(1,100)
                path_tree = self.method(self.G, source, destination, bandwidth)
                if len(path_tree) == 0:
                    num_block += 1
                    for i in range(len(buffer_b)):
                        buffer_b[i] += 1
                    ep_block += 1
                else:
                    self.service_list.append([path_tree, source, destination, bandwidth, t])
                    self.update_request(path_tree)
            elif mode == 1:  # a new member d to join
                i, d = self.node_join()
                if i != -1:
                    tree, source, destination, bandwidth, _ = self.service_list[i]
                    flag = 0
                    for path, _, _ in tree:
                        if d == path[0] or d == path[-1]:
                            flag = 1
                    if flag == 1:
                        self.service_list[i][2].append(d)
                    else:
                        p = []
                        for u in ([source] + destination):
                            # p.append(self.path_map[u, d])
                            p.append(KSP_FA(self.G, self.K_path_map[u][d], bandwidth))
                        p = sorted(p, key=lambda x: (x[2], x[3]))
                        path, start_f, _, len_path = p[0]
                        # path = p[0][0]
                        # len_path = p[0][1]
                        # len_path = RM.get_path_len(self.G, path)
                        len_fs = RM.modulation_level(bandwidth, len_path)
                        # start_f = RM.SP_FF(self.G, path, len_fs)
                        if start_f == -1:
                            num_block += 1
                            for i in range(len(buffer_b)):
                                buffer_b[i] += 1
                            ep_block += 1
                        else:
                            self.update_fs(path, len_fs, start_f)
                            self.service_list[i][0].append([path, len_fs, start_f])
                            self.service_list[i][2].append(d)
            else:  # a member d to leave
                i, d = self.node_leave()
                if i != -1:
                    tree, source, destination, bandwidth, _ = self.service_list[i]
                    flag = 0
                    for path, _, _ in tree:
                        if path[0] == d:
                            flag = 1

                    if flag == 1:  # d has downstream members
                        self.service_list[i][2].remove(d)
                    else:
                        self.service_list[i][2].remove(d)
                        for path, len_fs, start_f in tree:
                            if path[-1] == d:
                                self.release_fs(path, len_fs, start_f)
                                self.service_list[i][0].remove([path, len_fs, start_f])

            if len(buffer_s) >= 2 * self.batch_size - 1:
                # for i in range(self.batch_size):
                #     buffer_b[i] = buffer_b[i] / buffer_n[i]

                loss = push_and_pull(self.opt, self.lnet, self.gnet,  buffer_s[:self.batch_size], buffer_a[:self.batch_size], buffer_r[:self.batch_size], GAMMA)
                #record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)
                viz.line([[float(loss), np.mean(buffer_r[:self.batch_size])]], [self.g_ep.value], win='train_loss', update='append')
                del buffer_s[:self.batch_size]
                del buffer_a[:self.batch_size]
                del buffer_r[:self.batch_size]
                del buffer_b[:self.batch_size]
                del buffer_n[:self.batch_size]
                with self.g_ep.get_lock():
                    self.g_ep.value += 1

                time.sleep(0.5)

            if num_request % episode_size == 0:
                num_episode += 1
                print("Ep: {}, Blocking P: {},  Ep Bp: {}".format(num_episode,
                                                                  num_block / num_request,
                                                                  ep_block / episode_size))
                blocking_rate_list.append(num_block / num_request)
                ep_block = 0
        self.release_request(1000)
        self.res_queue.put(None)
        np.save("training_logs/r_a"+self.name, r_a)
        np.save("training_logs/r_s"+self.name, r_s)
        return num_block / num_request

num_works = 1
#num_works = mp.cpu_count()

if __name__ == "__main__":
    gnet = Net(state_dim, hidden_size, action_dim)  # global network
    gnet.share_memory()  # share the global parameters in multiprocessing
    opt = SharedAdam(gnet.parameters(), lr=1e-3, betas=(0.92, 0.999))  # global optimizer
    #opt = torch.optim.Adam(gnet.parameters(), lr=1e-3)
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

    # parallel training
    workers = [GNN_agent(NG.G, state_dim, action_dim, gnet, opt, global_ep, global_ep_r, res_queue, i) for i in range(num_works)]
    [w.start() for w in workers]
    res = []  # record episode reward to plot
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break
    [w.join() for w in workers]

    import matplotlib.pyplot as plt

    plt.plot(res)
    plt.ylabel('Moving average ep reward')
    plt.xlabel('Step')
    plt.show()
