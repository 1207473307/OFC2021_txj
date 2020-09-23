import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import random
import numpy as np
from RSA_methods import KSP_FA, SFMOR, modulation_level
from Data_set import data_set, data_set_2, edge_list
from visdom import Visdom
import time
import os

os.environ["OMP_NUM_THREADS"] = "1"
random.seed(0)
torch.manual_seed(0)


class Multicast_Env(object):
    def __init__(self, config, agent): #net_graph,  state_dim, action_dim, gnet, opt, global_ep, global_ep_r, res_queue, name,model_path=None, summary_writer=None):
        super(Multicast_Env, self).__init__()
        #self.name = 'w%02i' % name
        #self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.agent = agent
        #self.lnet = Net(state_dim, hidden_size, action_dim)  # local network
        self.G = config.G.copy()
        self.method = SFMOR
        self.service_list = []
        #self.criterion = criterion
        # self.model = model
        # self.device = device
        # self.optimizer = optimizer
        self.episodes = config.num_episodes_to_run
        #self.ep = 0
        # self.batch_size = batch_size
        self.path_map = np.load(config.path + 'path_map.npy', allow_pickle=True)
        self.K_path_map = np.load(config.path + 'K_path_map.npy', allow_pickle=True)
        self.device = config.device

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

    def Full_rearrangement(self, s):
        num_rerouting = 0
        path_tree, source, destination, bandwidth, time = self.service_list[s]
        for path, len_fs, start_f in path_tree:
            self.release_fs(path, len_fs, start_f)
        path_tree_new = self.method(self.G, source, destination, bandwidth)
        if len(path_tree_new) != 0:
            self.update_request(path_tree_new)
            self.service_list[s][0] = path_tree_new
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

    def conduct(self, s, action):
        k1 = 1
        k2 = 1
        if action:
            l1, f1 = self.statistical()
            num_rerouting = self.Full_rearrangement(s)
            l2, f2 = self.statistical()
            if num_rerouting == 0 :
                return 0, 0
            else:
                return (l2-l1) + k1*(f1 - f2) - k2*num_rerouting, num_rerouting
        else:
            return 0, 0

    def run(self):
        episode_size = 1000
        num_episode = 0
        num_block = 0
        num_request = 0
        ep_block = 0
        blocking_rate_list = []

        num_session = 0
        num_rerouting = 0
        num_rerouting_ep = 0
        occup_rate = []
        fragment = []
        r_a = []
        r_s = []

        # # 将窗口类实例化
        # viz = Visdom()
        # # 创建窗口并初始化
        # viz.line([0.], [0], win='train_loss', opts=dict(title='train_loss'))

        while self.agent.global_step_number < self.episodes:
            # while num_episode < 10:
            time_to = round(np.random.exponential(30))
            flag = self.release_request(time_to)
            if flag == 1:  # rearrangement
                for i in range(len(self.service_list)):
                    # g = data_set(self.service_list[i], self.G)
                    # inputs = g.edata['feat']
                    state = data_set_2(self.service_list[i], self.G, self.device)

                    #buffer_s.append([g1, inputs1, g2, inputs2])
                    #action = self.agent.pick_action(state)
                    action = self.agent.get_action(state)
                    reward, num_r = self.conduct(i, action)
                    #num_rerouting += num_r
                    num_rerouting_ep += num_r
                    next_state = data_set_2(self.service_list[i], self.G, self.device)

                    # r_a.append(action.numpy())
                    # r_s.append(edge_state)
                    #r_s.append([self.service_list[i][1:3]])
                    # buffer_a.append(action)
                    # buffer_r.append(reward)
                    #self.agent.save_experience(self.agent.memory, experience=(state, action.to(self.device), torch.tensor(reward).to(self.device), next_state, torch.tensor(0)))
                    self.agent.save_experience(self.agent.memory, experience=(
                    state, action, reward, next_state, 0))

            num_request += 1
            print(num_request)
            block = self.new_request()
            num_block += block
            ep_block += block

            self.request_join()
            self.request_leave()

            if self.agent.time_for_critic_and_actor_to_learn():
                self.agent.learn()

                self.agent.global_step_number += 1


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
                print("         num_rerouting: {},  Ep: {}".format(np.mean(fragment), f))
                ep_block = 0
                num_rerouting_ep = 0
        self.release_request(1000)
        # self.res_queue.put(None)
        # np.save("training_logs/r_a"+self.name, r_a)
        # np.save("training_logs/r_s"+self.name, r_s)
        return num_block / num_request

