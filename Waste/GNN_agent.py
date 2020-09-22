import numpy as np
import math
import networkx
import scipy.signal
import torch
import torch.nn as nn
import RSA_methods as RM
import random
from util import data_reduction
import Rearrangement as Rea
from tqdm import tqdm
import datetime


random.seed(0)
method = [RM.SPT, RM.MST, RM.SFMOR]

class GNN_agent():
    def __init__(self,  net_graph, criterion, model, device, optimizer, model_path = None, summary_writer = None):
        # self.name = 'agent_' + str(id)
        # self.name = name
        # self.opt_a = opt_a
        # self.opt_c = opt_c
        # self.globalAC = globalAC
        # self.coord = coord
        # self.globalTrainStep = globalTrainStep
        # self.maxTrainStep = maxTrainStep
        # self.net_graph = net_graph.copy()
        # self.model_path = model_path
        # self.summary_writer = summary_writer
        # self.gamma = 0.95
        #self.name = 'agent_' + str(id)
        #self.name = name
        self.G = net_graph.copy()
        self.method = RM.SFMOR
        self.service_list = []
        self.criterion = criterion
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.episodes = 1000
        self.batch_size = 32
        self.path_map = np.load('path_map.npy', allow_pickle=True)

    def random_request(self):
        source = random.randint(1, 14)
        num_destination = random.randint(2, 5)
        destination = []
        for i in range(num_destination):
            d = random.randint(1, 14)
            while d == source or d in destination:
                d = random.randint(1, 14)
            destination.append(d)
        # len_fs = random.randint(1, 20)
        bandwidth = random.randint(1, 500)
        time = random.randint(1, 100)

        # return source, destination, len_fs, time
        return source, destination, bandwidth, time

    def join(self):
        if len(self.service_list) == 0:
            return -1, -1

        i = random.randint(0, len(self.service_list) - 1)
        source = self.service_list[i][1]
        destination = self.service_list[i][2]
        if len(destination) > 10:
            return -1, -1
        d = random.randint(1, 14)
        while d == source or d in destination:
            d = random.randint(1, 14)
        return i, d

    def leave(self):
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

    def get_down(self, destination, child, i):
        down = []
        for c in child[i]:
            down.append(c)
            down.extend(self.get_down(destination, child, destination.index(c)))
        return down

    def Full_rearrangement(self, r):

        path_tree, source, destination, bandwidth, time = self.service_list[r]
        for path, len_fs, start_f in path_tree:
            self.release_fs(path, len_fs, start_f)
        path_tree_new = self.method(self.G, source, destination, bandwidth)
        if len(path_tree_new) != 0:
            self.update_request(path_tree_new)
            self.service_list[r][0] = path_tree_new
            return 0
            # for path, len_fs, start_f in path_tree:
            #     self.release_fs(path, len_fs, start_f)
        else:
            for path, len_fs, start_f in path_tree:
                self.update_fs(path, len_fs, start_f)
            return 1

    def Partial_rearrangement(self, r):

        path_tree, source, destination, bandwidth, time = self.service_list[r]
        up_member = [source]
        down_member = []
        remove_list = []
        append_list = []
        block = 0
        for path, len_fs, start_f in path_tree:
            if (path[0] not in [source] + destination) and (path[-1] not in [source] + destination):
                remove_list.append([path, len_fs, start_f])
            elif (path[0] not in [source] + destination):
                remove_list.append([path, len_fs, start_f])
                down_member.append(path[-1])
            elif (path[-1] not in [source] + destination):
                # self.service_list[r][0].remove([path, len_fs, start_f])
                remove_list.append([path, len_fs, start_f])
                flag = 0
                for p in path_tree:
                    if path[-1] == p[0] and p[-1] in destination:
                        flag = 1
                if flag == 1:
                    up_member.append(path[0])

        while len(down_member) != 0:
            p = []
            for i in up_member:
                for j in down_member:
                    p.append(self.path_map[i, j])
            p = sorted(p, key=lambda x: x[1])
            flag = 1
            for path_1, len_path_n in p:
                # path = p[0][0]
                # len_path = p[0][1]

                len_fs_1 = RM.modulation_level(bandwidth, len_path_n)
                start_f_1 = RM.SP_FF(self.G, path_1, len_fs_1)
                if start_f_1 != -1:
                    append_list.append([path_1, len_fs_1, start_f_1])
                    down_member.remove(path_1[-1])
                    flag = 0
                    break
            if flag == 1:
                block = 1
                return 1

            # Vin.append(path[-1])

        if block == 0:
            for path, len_fs, start_f in append_list:
                self.update_fs(path, len_fs, start_f)
                self.service_list[r][0].append([path, len_fs, start_f])
            for path, len_fs, start_f in remove_list:
                self.release_fs(path, len_fs, start_f)
                self.service_list[r][0].remove([path, len_fs, start_f])

            # reroute
            path_tree, source, destination, bandwidth, time = self.service_list[r]
            hid_list = [0 for i in range(len(destination))]
            hop_list = [0 for i in range(len(destination))]
            # parent = [[] for i in range(len(destination))]
            child = [[] for i in range(len(destination))]
            down = [[] for i in range(len(destination))]
            for path, len_fs, start_f in path_tree:
                hid_list[destination.index(path[-1])] = start_f + len_fs - 1
                hop_list[destination.index(path[-1])] = Rea.get_hops(path_tree, source, path[-1])

                if path[0] != source:
                    # parent[destination.index(path[-1])].append(path[0])
                    child[destination.index(path[0])].append(path[-1])
            for i in range(len(destination)):
                down[i] = self.get_down(destination, child, i)
            Cth = max(hid_list) * sum(hop_list) / len(destination)
            for i in range(len(destination)):
                cost = hid_list[i] * hop_list[i]
                if cost > Cth:
                    p = []
                    for j in [source] + destination:
                        if j != destination[i] and j not in down[i]:
                            #p.append(self.path_map[j, destination[i]])
                            path_n, len_path_n = self.path_map[j, destination[i]]
                            len_fs_n = RM.modulation_level(bandwidth, len_path_n)
                            _, cut_n, start_f = RM.cal_min_cut_num(self.G, path_n, len_fs_n)
                            p.append([path_n, len_path_n, len_fs_n, cut_n, start_f])
                    p = sorted(p, key=lambda x: x[3])
                    for path_n, len_path_n, len_fs_n, cut_n, start_f_n in p:
                        block = 1
                        #len_fs_1 = RM.modulation_level(bandwidth, len_path_n)
                        #start_f_1 = RM.SP_FF(self.G, path_1, len_fs_1)
                        if start_f_n != -1:
                            block = 0
                            for path, len_fs, start_f in path_tree:
                                if path[-1] == destination[i]:
                                    if path != path_n:
                                        self.service_list[r][0].remove([path, len_fs, start_f])
                                        self.release_fs(path, len_fs, start_f)
                                        self.service_list[r][0].append([path_n, len_fs_n, start_f_n])
                                        self.update_fs(path_n, len_fs_n, start_f_n)
                                        for j in range(len(destination)):
                                            down[j].append(destination[i])
                                            down[j].extend(down[i])
                                    break
                            break
                    if block == 1:
                        return 1
        return 0



    def train(self, model, device, buffer_g, buffer_a, buffer_b, optimizer):
        model.train()

        batch_graph = buffer_g
        output = buffer_a

        #labels = torch.LongTensor([graph.label for graph in batch_graph]).to(device)

        # compute loss
        loss = self.criterion(buffer_b, 0)

        # backprop
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss = loss.detach().cpu().numpy()

        print("loss training: %f" % (loss))

        return loss

    def rmsa(self):
        episode_size = 1000
        num_episode = 0
        num_block = 0
        num_request = 0
        ep_block = 0
        blocking_rate_list = []

        buffer_g = []
        buffer_a = []
        buffer_b = []
        buffer_n = []

        while num_episode < self.episodes:
            time_to = 1
            flag = self.release_request(time_to)
            if flag == 1:  # rearrangement
                for i in range(len(self.service_list)):
                    g = data_reduction(self.service_list[i], self.G)
                    buffer_g.append(g)
                    action = self.model([g])
                    buffer_a.append(action)
                    if action == 0:
                        buffer_b.append(0)
                        buffer_n.append(0)
                    elif action == 1:
                        buffer_b.append(self.Full_rearrangement(i))
                        buffer_n.append(1)
                    elif action == 2:
                        buffer_b.append(self.Partial_rearrangement(i))
                        buffer_n.append(1)

            num_request += 1
            for i in range(len(buffer_n)):
                buffer_n[i] += 1
            # print(num_request)
            mode = random.randint(0, 2)

            if mode == 0:  # a multicast session  first appears
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
            elif mode == 1:  # a new member d to join
                i, d = self.join()
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
                            p.append(self.path_map[u, d])
                        p = sorted(p, key=lambda x: x[1])
                        path = p[0][0]
                        len_path = p[0][1]
                        len_fs = RM.modulation_level(bandwidth, len_path)
                        start_f = RM.SP_FF(self.G, path, len_fs)
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
                i, d = self.leave()
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

            if num_request % episode_size == 0:
                num_episode += 1
                print("Ep: {}, Blocking P: {},  Ep Bp: {}".format(num_episode,
                                                                  num_block / num_request,
                                                                  ep_block / episode_size))
                blocking_rate_list.append(num_block / num_request)
                ep_block = 0

            if len(buffer_g) == 2 * self.batch_size - 1:
                # buffer_v_target = tl.rein.discount_episode_rewards(np.asarray(buffer_r[i]), self.gamma, mode=1)
                # self.agents[i].update_global(np.vstack(buffer_s[i][:self.batch_size]),
                #                              np.vstack(buffer_a[i][:self.batch_size]),
                #                              np.vstack(buffer_v_target[:self.batch_size]),
                #                              self.globalAC[i],
                #                              num_episode)
                for i in range(self.batch_size):
                    buffer_b[i] = buffer_b[i]/buffer_n[i]
                self.train(self.model, self.device, buffer_g[:self.batch_size], buffer_a[:self.batch_size], buffer_b[:self.batch_size], self.optimizer)
                del buffer_g[:self.batch_size]
                del buffer_a[:self.batch_size]
                del buffer_b[:self.batch_size]
                del buffer_n[:self.batch_size]

        self.release_request(1000)
        return num_block / num_request