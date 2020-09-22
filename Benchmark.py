import matplotlib.pyplot as plt
import RSA_methods as RM
import Net_Graph as NG
import random
import numpy as np

random.seed(0)
method = [RM.SPT, RM.MST, RM.SFMOR]
class RMSA():
    def __init__(self, net_graph, method, name=None):
        self.name = 'agent_' + str(id)
        self.name = name
        self.G = net_graph.copy()
        self.path_map = np.load('path_map.npy', allow_pickle=True)
        self.method = method
        self.service_list = []

    def random_request(self):
        source = random.randint(1, 14)
        num_destination = random.randint(2, 5)
        destination = []
        for i in range(num_destination):
            d = random.randint(1, 14)
            while d == source or d in destination:
                d = random.randint(1, 14)
            destination.append(d)
        #len_fs = random.randint(1, 20)
        bandwidth = random.randint(1, 250)
        time = random.randint(1, 20)

        #return source, destination, len_fs, time
        return source, destination, bandwidth, time

    def updata_fs(self, path, len_fs: int, start_f: int):
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
            self.updata_fs(path, len_fs, start_f)

    def release_request(self, time_to):
        remove_list = []
        for r in self.service_list:
            r[-1] = r[-1] - time_to
            if r[-1] <= 0:
                for path, len_fs, start_f in r[0]:
                    self.release_fs(path,  len_fs, start_f)
                remove_list.append(r)
        for i in remove_list:
            self.service_list.remove(i)

    def rmsa(self):
            episode_size = 1000
            num_episode = 0
            #service_list = []
            num_block = 0
            num_request = 0
            ep_block = 0
            blocking_rate_list = []

            while num_episode < 100:

                num_request += 1
                source, destination, bandwidth, time = self.random_request()
                time_to = 1
                self.release_request(time_to)

                path_tree = self.method(self.G, source, destination, bandwidth)

                if len(path_tree) == 0:
                    num_block += 1
                    ep_block += 1

                else:
                    self.service_list.append([path_tree,  time])
                    self.update_request(path_tree)


                if num_request % episode_size == 0:
                    num_episode += 1
                    print("Ep: {}, Blocking P: {},  Ep Bp: {}".format(num_episode,
                                                                      num_block / num_request,
                                                                      ep_block / episode_size))
                    blocking_rate_list.append(num_block / num_request)
                    ep_block = 0
            #return num_block / num_request
Benchmark = RMSA(NG.G, method=method[2])
Benchmark.rmsa()