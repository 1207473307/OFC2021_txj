import matplotlib.pyplot as plt
import torch
import RSA_methods as RM
import Rearrangement as Rea
import Net_Graph as NG
import random
import numpy as np
import torch.multiprocessing as mp
from RSA_methods import KSP_FA
from Data_set import data_set,edge_list
from GNN_A3C import Net

random.seed(0)

class GNN_agent(mp.Process):
    def __init__(self,  net_graph, criterion,state_dim, action_dim, gnet, opt, global_ep, global_ep_r, res_queue, name, model_path = None, summary_writer = None):
        super(GNN_agent, self).__init__()
        self.name = 'w%02i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt = gnet, opt
        self.lnet = Net(state_dim, action_dim)  # local network
        self.G = net_graph.copy()
        self.method = RM.SFMOR
        self.service_list = []
        self.criterion = criterion
        # self.model = model
        # self.device = device
        # self.optimizer = optimizer
        self.episodes = 1000
        self.batch_size = 32
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
        #len_fs = random.randint(1, 20)
        bandwidth = random.randint(1, 500)
        time = random.randint(1, 100)

        #return source, destination, len_fs, time
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
                    self.release_fs(path,  len_fs, start_f)
                remove_list.append(r)
        for i in remove_list:
            self.service_list.remove(i)
        return flag

    def attempt(self, service, action):
        path_tree, source, destination, bandwidth, time = service
        edge_tree = []
        block = 0
        for path in path_tree:
            for i in range(len(path) - 1):
                edge_tree.append((path[i], path[i+1]))
        for i in range(len(action)):
            if action[i] == 1:
                if edge_list[i] not in edge_tree:
                    block += 1
            else:
                if edge_list[i] in edge_tree:
                    block += 1

        return block



    def run(self):
            episode_size = 1000
            num_episode = 0
            #service_list = []
            num_block = 0
            num_request = 0
            ep_block = 0
            blocking_rate_list = []
            #
            # buffer_g = []
            # buffer_a = []
            # buffer_b = []
            # buffer_n = []
            buffer_g, buffer_a, buffer_b, buffer_n = [], [], [], []
            while self.g_ep.value < MAX_EP:
            #while num_episode < 10:
                time_to = 1
                for i in range(len(buffer_n)):
                    buffer_n[i] += 1
                flag = self.release_request(time_to)
                if flag == 1:   #rearrangement
                    for i in range(len(self.service_list)):
                        g = data_set(self.service_list[i], self.G)
                        buffer_g.append(g)
                        #a = self.model(g, g.edata['feat'])
                        action = self.lnet.get_action(g)
                        block = self.attempt(self.service_list[i], action)
                        buffer_b.append(block)
                        buffer_n.append(1)

                num_request += 1
                #print(num_request)
                mode = random.randint(0, 2)

                if mode == 0 :  #a multicast session  first appears
                    source, destination, bandwidth, time = self.random_request()
                    path_tree = self.method(self.G, source, destination, bandwidth)
                    if len(path_tree) == 0:
                        num_block += 1
                        for i in range(len(buffer_b)):
                            buffer_b[i] += 1
                        ep_block += 1
                    else:
                        self.service_list.append([path_tree, source, destination, bandwidth, time])
                        self.update_request(path_tree)
                elif mode == 1:  #a new member d to join
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
                                #p.append(self.path_map[u, d])
                                p.append(KSP_FA(self.G, self.K_path_map[u][d], bandwidth))
                            p = sorted(p, key=lambda x:(x[2],x[3]))
                            path, start_f, _,len_path = p[0]
                            # path = p[0][0]
                            # len_path = p[0][1]
                            #len_path = RM.get_path_len(self.G, path)
                            len_fs = RM.modulation_level(bandwidth, len_path)
                            #start_f = RM.SP_FF(self.G, path, len_fs)
                            if start_f == -1:
                                num_block += 1
                                for i in range(len(buffer_b)):
                                    buffer_b[i] += 1
                                ep_block += 1
                            else:
                                self.update_fs(path, len_fs, start_f)
                                self.service_list[i][0].append([path, len_fs, start_f])
                                self.service_list[i][2].append(d)
                else:            #a member d to leave
                    i, d = self.node_leave()
                    if i != -1:
                        tree, source, destination, bandwidth, _ = self.service_list[i]
                        flag = 0
                        for path,_,_ in tree:
                            if path[0] == d:
                                flag = 1

                        if flag == 1:  #d has downstream members
                            self.service_list[i][2].remove(d)
                        else:
                            self.service_list[i][2].remove(d)
                            for path, len_fs, start_f in tree:
                                if path[-1] == d:
                                    self.release_fs(path, len_fs, start_f)
                                    self.service_list[i][0].remove([path, len_fs, start_f])

                if len(buffer_g) >= 2 * self.batch_size - 1:
                    for i in range(self.batch_size):
                        buffer_b[i] = buffer_b[i] / buffer_n[i]
                    # self.train(self.model, self.device, buffer_g[:self.batch_size], buffer_a[:self.batch_size],
                    #            buffer_b[:self.batch_size], self.optimizer)
                    push_and_pull(self.opt, self.lnet, self.gnet, done, s_, buffer_s, buffer_a, buffer_r, GAMMA)
                    record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)
                    del buffer_g[:self.batch_size]
                    del buffer_a[:self.batch_size]
                    del buffer_b[:self.batch_size]
                    del buffer_n[:self.batch_size]

                if num_request % episode_size == 0:
                    num_episode += 1
                    print("Ep: {}, Blocking P: {},  Ep Bp: {}".format(num_episode,
                                                                      num_block / num_request,
                                                                      ep_block / episode_size))
                    blocking_rate_list.append(num_block / num_request)
                    ep_block = 0
            self.release_request(1000)
            self.res_queue.put(None)
            return num_block / num_request
