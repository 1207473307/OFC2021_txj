import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from utils import v_wrap, set_init, push_and_pull, record, discount_reward
from RSA_methods import KSP_FA, SFMOR, modulation_level
from Rearrangement import get_hops
import Net_Graph as NG
import random
import numpy as np
from RSA_methods import KSP_FA
from Data_set import data_set,edge_list
from visdom import Visdom
import time
import os
import math

os.environ["OMP_NUM_THREADS"] = "1"
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

UPDATE_GLOBAL_ITER = 5
GAMMA = 0.9
MAX_EP = 2000
batch_size = 32
state_dim = 4
hidden_size = 128
action_dim = 1

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

class Net(nn.Module):
    # def __init__(self, s_dim, hidden_size,a_dim):
    #     super(Net, self).__init__()
    #     self.s_dim = s_dim
    #     self.a_dim = a_dim
    #     self.in_feats = s_dim
    #     self.activation = torch.nn.LeakyReLU(negative_slope=0.01, inplace=False)
    #     #self.activation = torch.nn.PReLU(num_parameters=1,init=0.25)
    #
    #     self.a1 = nn.Linear(s_dim, hidden_size)
    #     self.a2 = nn.Linear(hidden_size, hidden_size)
    #     self.a3 = nn.Linear(hidden_size, hidden_size)
    #     self.a4 = nn.Linear(hidden_size, hidden_size)
    #     self.a5 = nn.Linear(hidden_size, a_dim)
    #
    #     self.v1 = nn.Linear(s_dim, hidden_size)
    #     self.v2 = nn.Linear(hidden_size, hidden_size)
    #     self.v3 = nn.Linear(hidden_size, hidden_size)
    #     self.v4 = nn.Linear(hidden_size, hidden_size)
    #     self.v5 = nn.Linear(hidden_size, 1)
    #
    #     set_init([self.a1, self.a2,self.a3, self.a4, self.a5, self.v1, self.v2, self.v3, self.v4, self.v5])
    #     self.distribution = torch.distributions.Categorical
    #
    # def forward(self,inputs):
    #     # for src, dst in edge_list:
    #     #     g.edges[src, dst].data['h'] = torch.mean(torch.index_select(g.ndata['feat'], dim=0, index=torch.tensor([src, dst])), dim=0).view(1, self.in_feats)
    #     # inputs = g.edata['h']
    #     h1 = self.a1(inputs)
    #     h1 = self.activation(h1)
    #     h1 = self.a2(h1)
    #     h1 = self.activation(h1)
    #     # h1 = self.a3(g, h1)
    #     # h1 = self.activation(h1)
    #     # h1 = self.a4(g, h1)
    #     # h1 = self.activation(h1)
    #     h1 = self.a5(h1)
    #     #h1 = F.softmax(h1, dim =1)
    #
    #     h2 = self.v1(inputs)
    #     h2 = self.activation(h2)
    #     h2 = self.v2(h2)
    #     h2 = self.activation(h2)
    #     # h2 = self.v3(g, h2)
    #     # h2 = self.activation(h2)
    #     # h2 = self.v4(g, h2)
    #     # h2 = self.activation(h2)
    #     h2 = self.v5(h2)
    #
    #     return h1, h2
    def __init__(self, s_dim, hidden_dim, a_dim):
        super(Net, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.a1 = nn.Linear(s_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, a_dim)
        self.sigma = nn.Linear(hidden_dim, a_dim)
        self.c1 = nn.Linear(s_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1)
        set_init([self.a1, self.mu, self.sigma, self.c1, self.v])
        self.distribution = torch.distributions.Normal

    def forward(self, x):
        a = self.a1(x)
        a1 = F.relu6(a)
        mu = 2 * F.tanh(self.mu(a1))
        sigma = F.softplus(self.sigma(a1)) + 0.001      # avoid 0
        c1 = F.relu6(self.c1(x))
        values = self.v(c1)
        return mu, sigma, values

    #
    # def get_action(self, inputs, greedy=True):
    #     # for src, dst in edge_list:
    #     #     g.edges[src, dst].data['h'] = torch.mean(torch.index_select(g.ndata['feat'], dim=0, index=torch.tensor([src, dst])), dim=0).view(1, self.in_feats)
    #     #
    #     # probs,_ = self(g, g.edata['h'])
    #     probs, _ = self(inputs)
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

    # def choose_action(self, s):
    #     self.training = False
    #     h1 = self.forward(s)
    #     return h1

    # def loss_func(self, s, a, v_t):
    #     # self.train()
    #     # logits, values = [],[]
    #     # for g,inputs in s:
    #     #     l, v = self.forward(inputs)
    #     #     logits.append(l)
    #     #     values.append(torch.mean(v))
    #     # #logits = torch.cat(logits, dim=1)
    #     # #values = torch.from_numpy(np.array(values))
    #     # # values = torch.cat(values, dim=1)
    #     # # values = torch.mean(values, dim=0).view(batch_size,1)
    #     # values = torch.tensor(values)
    #     #
    #     # # logits, values = self.forward(s)
    #     #
    #     # td = v_t.detach().squeeze() - values
    #     # c_loss = td.pow(2)
    #     #
    #     # X = []
    #     # for l,action in zip(logits,a):
    #     #     probs = F.softmax(l, dim=1)
    #     #     m = self.distribution(probs)
    #     #     x = m.log_prob(torch.squeeze(action))
    #     #     X.append(torch.mean(x))
    #     # X = torch.tensor(X)
    #     # y = td.detach().squeeze()
    #     # exp_v = X * td.detach().squeeze()
    #     # a_loss = -exp_v
    #     # total_loss = (c_loss + a_loss).mean()
    #     # return total_loss
    #
    #     self.train()
    #     logits, values = self.forward(s)
    #     td = v_t - values
    #     c_loss = td.pow(2)
    #
    #     probs = F.softmax(logits, dim=1)
    #     m = self.distribution(probs)
    #
    #     exp_v = m.log_prob(a) * td.detach().squeeze()
    #     a_loss = -exp_v
    #     total_loss = (c_loss + a_loss).mean()
    #     return total_loss

    def choose_action(self, s):
        self.training = False
        mu, sigma, _ = self.forward(s)
        m = self.distribution(mu.view(1, ).data, sigma.view(1, ).data)
        a = m.sample()#.numpy()
        # print(a)
        return torch.sigmoid(a)

    def loss_func(self, s, a, v_t):
        self.train()
        mu, sigma, values = self.forward(s)
        td = v_t - values
        c_loss = td.pow(2)

        m = self.distribution(mu, sigma)
        log_prob = m.log_prob(a)
        entropy = 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(m.scale)  # exploration
        exp_v = log_prob * td.detach() + 0.005 * entropy
        a_loss = -exp_v
        total_loss = (a_loss + c_loss).mean()
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
        self.method = SFMOR
        self.service_list = []
        #self.criterion = criterion
        # self.model = model
        # self.device = device
        # self.optimizer = optimizer
        self.episodes = 1000
        self.batch_size = batch_size
        self.path = 'E:\OFC2021_txj'
        self.path_map = np.load(self.path + '/path_map.npy', allow_pickle=True)
        self.K_path_map = np.load(self.path + '/K_path_map.npy', allow_pickle=True)

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
        time = random.randint(1, 1000)

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

    def new_request(self):
        block = 0
        source, destination, bandwidth, time = self.random_request()
        path_tree = self.method(self.G, source, destination, bandwidth)
        if len(path_tree) == 0:
            block = 1
        else:
            # num_session += 1
            self.service_list.append([path_tree, source, destination, bandwidth, time])
            self.update_request(path_tree)
        return block

    def request_join(self):
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
                len_fs = modulation_level(bandwidth, len_path)
                # start_f = RM.SP_FF(self.G, path, len_fs)
                if start_f == -1:
                    pass
                    # num_block += 1
                    # ep_block += 1
                else:
                    self.update_fs(path, len_fs, start_f)
                    self.service_list[i][0].append([path, len_fs, start_f])
                    self.service_list[i][2].append(d)

    def request_leave(self):
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

    def Full_rearrangement(self, rea_list):
        num_rerouting = 0
        for r in rea_list:
            path_tree, source, destination, bandwidth, time = self.service_list[r]
            for path, len_fs, start_f in path_tree:
                self.release_fs(path, len_fs, start_f)
            path_tree_new = self.method(self.G, source, destination, bandwidth)
            if len(path_tree_new) != 0:
                self.update_request(path_tree_new)
                self.service_list[r][0] = path_tree_new
                num_rerouting += len(path_tree_new)
                # for path, len_fs, start_f in path_tree:
                #     self.release_fs(path, len_fs, start_f)
            else:
                for path, len_fs, start_f in path_tree:
                    self.update_fs(path, len_fs, start_f)

        return num_rerouting

    def statistical(self):
        num = 0
        block = 0
        for src, dst in edge_list:
            fs = self.G[src][dst]['fs']
            flag = 0
            for i in fs:
                if i == 1:
                    num += 1
                    if flag == 0:
                        block += 1
                        flag = 1
                else:
                    flag = 0
        return num/len(edge_list), num/block

    def conduct(self, action, q_list, q_list_new):
        # K = 1e-3
        n = round(len(q_list)*action.item())
        rea_list = []
        for i in range(n):
            rea_list.append(q_list.index(q_list_new[i]))
        num_rerouting = self.Full_rearrangement(rea_list)
        return num_rerouting

    def get_state(self, service_list, G):
        q_list = []
        rea_list = []
        for path_tree, source, destination, bandwidth, _ in service_list:
            path_tree_new = SFMOR(G, source, destination, bandwidth)
            if len(path_tree_new) == 0:  # 临时
                path_tree_new = path_tree
            hops = 0
            hidx = 0
            hops_new = 0
            hidx_new = 0
            for d in destination:
                hops = max(hops, get_hops(path_tree, source, d))
                hops_new = max(hops_new, get_hops(path_tree_new, source, d))
            for path, len_fs, start_f in path_tree:
                hidx = max(hidx, len_fs + start_f)
            for path, len_fs, start_f in path_tree_new:
                hidx_new = max(hidx_new, len_fs + start_f)
            q = (hops_new * hidx_new) / (hops * hidx)
            q_list.append(q)
        #q_ave = np.mean(q_list)
        q_list_new = sorted(q_list)
        # for q in q_list:
        #     if q < q_ave:
        #         rea_list.append(q_list.index(q))
        # return rea_list
        state = [np.mean(q_list), np.var(q_list), np.max(q_list), np.min(q_list)]
        return state, q_list, q_list_new

    def run(self):
        episode_size = 1000
        num_episode = 0
        num_block = 0
        num_request = 0
        ep_block = 0
        blocking_rate_list = []
        t = 0
        num_session = 0
        num_session_ep = 0
        num_rerouting = 0
        num_rerouting_ep = 0
        occup_rate = []
        fragment = []
        r_a = []
        r_s = []

        # 将窗口类实例化
        viz = Visdom()
        # 创建窗口并初始化
        # viz.line([0.], [0], win='train_loss', opts=dict(title='train_loss'))

        buffer_s, buffer_a, buffer_r, buffer_b, buffer_n = [], [], [], [], []
        b = 0
        n = 0
        while self.g_ep.value < MAX_EP:
            # while num_episode < 10:
            time_to = round(np.random.exponential(30))
            t += time_to
            flag = self.release_request(time_to)
            #if flag == 1:  # rearrangement
            if t >= 1000:
                if len(buffer_r) != 0 :
                    buffer_r[-1] = 1-b/n
                    b = 0
                    n = 0
                state, q_list, q_list_new = self.get_state(self.service_list, self.G)
                action = self.lnet.choose_action(torch.tensor(state, dtype=torch.float32))
                num_r = self.conduct(action, q_list, q_list_new)
                num_rerouting += num_r
                num_rerouting_ep += num_r
                buffer_s.append(state)
                buffer_a.append(action)
                buffer_r.append(0)

                # buffer_b.append(0)
                # buffer_n.append(0)
                s_,_ ,_  = self.get_state(self.service_list, self.G)
                t -= 1000

            num_request += 1
            n += 1
            block = self.new_request()
            b += block
            # for i in range(len(buffer_b)):
            #     buffer_b[i] += block
            #     buffer_n[i] += 1
            num_session += (1 - block)
            num_session_ep += (1 - block)
            num_block += block
            ep_block += block

            self.request_join()
            self.request_leave()


            if len(buffer_s) >= 2 * self.batch_size - 1:
                # for i in range(self.batch_size):
                #     buffer_r[i] = 1 - buffer_b[i] / buffer_n[i]
                br = discount_reward(buffer_r, GAMMA, self.batch_size)

                loss = push_and_pull(self.opt, self.lnet, self.gnet, s_, buffer_s[:self.batch_size], buffer_a[:self.batch_size], br, GAMMA)
                #record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)
                viz.line([float(loss)], [self.g_ep.value], win='train_loss', update='append', opts=dict(title='train_loss', legend=['train_loss']))
                viz.line([np.mean(buffer_r[:self.batch_size])], [self.g_ep.value], win='reward', update='append', opts=dict(title='reward', legend=['reward']))
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
                o, f = self.statistical()
                occup_rate.append(o)
                fragment.append(f)
                print("         occupancy rate : {},  Ep: {}".format(np.mean(occup_rate), o))
                print("         fragment: {},  Ep: {}".format(np.mean(fragment), f))
                print("         num_rerouting: {},  Ep: {}".format(num_rerouting / num_session,
                                                                   num_rerouting_ep / num_session_ep))
                #print("         r_0  r_1:{}".format(self.agent.sta_0_1(r_a)))
                #r_a.clear()
                ep_block = 0
                num_rerouting_ep = 0
                num_session_ep = 0
        self.release_request(1000)
        self.res_queue.put(None)
        # np.save("training_logs/r_a"+self.name, r_a)
        # np.save("training_logs/r_s"+self.name, r_s)
        return num_block / num_request

#num_works = 1
num_works = mp.cpu_count()

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
