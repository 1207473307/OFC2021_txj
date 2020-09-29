import matplotlib.pyplot as plt
import RSA_methods as RM
import Rearrangement as Rea
import Net_Graph as NG
import random
import numpy as np
from Data_set import edge_list
from RSA_methods import KSP_FA, get_path_len

random.seed(0)
np.random.seed(0)
method = [RM.SPT, RM.MST, RM.SFMOR, RM.SFMOR_FF]
load = [300, 500, 700, 900, 1100]
class RMSA():
    def __init__(self, net_graph, method, traffic_load, name=None):
        self.name = 'agent_' + str(id)
        self.name = name
        self.G = net_graph.copy()
        #self.path_map = np.load('path_map.npy', allow_pickle=True)
        self.K_path_map = np.load('K_path_map.npy', allow_pickle=True)
        self.method = method
        self.service_list = []
        self.traffic_load = traffic_load

    def random_request(self, max_bandwidth):
        source = random.randint(0, 13)
        num_destination = random.randint(2, 5)
        destination = []
        for i in range(num_destination):
            d = random.randint(0, 13)
            while d == source or d in destination:
                d = random.randint(0, 13)
            destination.append(d)
        #len_fs = random.randint(1, 20)
        bandwidth = random.randint(1, max_bandwidth)
        time = random.randint(1, 1000)

        #return source, destination, len_fs, time
        return source, destination, bandwidth, time

    def join(self):
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
                    self.release_fs(path,  len_fs, start_f)
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

    def down(self, path_tree, source, destination):
        child = [[] for i in range(len(destination))]
        down = [[] for i in range(len(destination))]
        for path, len_fs, start_f in path_tree:
            if path[0] != source:
                # parent[destination.index(path[-1])].append(path[0])
                child[destination.index(path[0])].append(path[-1])
        for i in range(len(destination)):
            down[i] = self.get_down(destination, child, i)
        return down

    def get_cut(self, path ,len_fs, start_f, num_fs = NG.num_fs):
        #self.release_fs(path, len_fs, start_f)
        start = start_f
        end = start_f + len_fs - 1
        if start == 0:
            n_cut = 0
        elif end == num_fs-1:
            n_cut = 0
        else:
            n_cut_1 = 0
            for k in range(len(path) - 1):
                if self.G[path[k]][path[k + 1]]['fs'][start-1] == 1:
                    n_cut_1 += 1
            n_cut_2 = 0
            for k in range(len(path) - 1):
                if self.G[path[k]][path[k + 1]]['fs'][end + 1] == 1:
                    n_cut_2 += 1
            n_cut = min(n_cut_1, n_cut_2)
        #self.update_fs(path, len_fs, start_f)
        return n_cut

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

    def Partial_rearrangement(self, rea_list):
        num_rerouting = 0
        for r in rea_list:
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

            for path, len_fs, start_f in remove_list:
                self.release_fs(path, len_fs, start_f)
                self.service_list[r][0].remove([path, len_fs, start_f])

            while len(down_member) != 0:
                p = []
                for i in up_member:
                    for j in down_member:
                        p.append(KSP_FA(self.G, self.K_path_map[i][j], bandwidth))
                p = sorted(p, key=lambda x: (x[2], x[3]))
                #path, start_f, _, len_path = p[0]
                flag = 1
                for path_n, start_f_n, _, len_path_n in p:
                    # path = p[0][0]
                    # len_path = p[0][1]

                    len_fs_n = RM.modulation_level(bandwidth, len_path_n)
                    #start_f_1 = RM.SP_FF(self.G, path_1, len_fs_1)
                    if start_f_n != -1:
                        append_list.append([path_n, len_fs_n, start_f_n])
                        down_member.remove(path_n[-1])
                        flag = 0
                        break
                if flag == 1:
                    block = 1
                    break

                # Vin.append(path[-1])

            if block == 0:
                for path, len_fs, start_f in append_list:
                    self.update_fs(path, len_fs, start_f)
                    self.service_list[r][0].append([path, len_fs, start_f])
                # for path, len_fs, start_f in remove_list:
                #     self.release_fs(path, len_fs, start_f)
                #     self.service_list[r][0].remove([path, len_fs, start_f])

                # reroute
                path_tree, source, destination, bandwidth, time = self.service_list[r]
                hid_list = [0 for i in range(len(destination))]
                hop_list = [0 for i in range(len(destination))]
                cut_list = [0 for i in range(len(destination))]
                len_list = [0 for i in range(len(destination))]
                # parent = [[] for i in range(len(destination))]
                # child = [[] for i in range(len(destination))]
                # down = [[] for i in range(len(destination))]
                down = self.down(path_tree, source, destination)
                for path, len_fs, start_f in path_tree:
                    hid_list[destination.index(path[-1])] = start_f + len_fs - 1
                    hop_list[destination.index(path[-1])] = Rea.get_hops(path_tree, source, path[-1])
                    cut_list[destination.index(path[-1])] = self.get_cut(path, len_fs, start_f)
                    len_list[destination.index(path[-1])] = get_path_len(self.G, path)
                    #     if path[0] != source:
                #         # parent[destination.index(path[-1])].append(path[0])
                #         child[destination.index(path[0])].append(path[-1])
                # for i in range(len(destination)):
                #     down[i] = self.get_down(destination, child, i)

                #Cth = max(hid_list) * sum(hop_list) / len(destination)
                #Cth = max(cut_list) * sum(hop_list) / len(destination)
                #Cth = np.mean(cut_list) * sum(hop_list) / len(destination)
                Cth = np.mean(cut_list) * sum(len_list) / len(destination)
                for i in range(len(destination)):
                    #cost = hid_list[i] * hop_list[i]
                    #cost = cut_list[i] * hop_list[i]
                    cost = cut_list[i] * len_list[i]
                    if cost > Cth:

                        for path, len_fs, start_f in path_tree:
                            if path[-1] == destination[i]:
                                #self.service_list[r][0].remove([path, len_fs, start_f])
                                self.release_fs(path, len_fs, start_f)
                                break
                        p = []
                        num_rerouting += 1
                        for j in [source] + destination:
                            if j != destination[i] and j not in down[i]:

                                # path_n, len_path_n = self.path_map[j, destination[i]]
                                # len_fs_n = RM.modulation_level(bandwidth, len_path_n)
                                # _, cut_n, start_f = RM.cal_min_cut_num(self.G, path_n, len_fs_n)
                                # p.append([path_n, len_path_n, len_fs_n, cut_n, start_f])
                                p.append(KSP_FA(self.G, self.K_path_map[j][destination[i]], bandwidth))
                        p = sorted(p, key=lambda x:(x[2],x[3]))
                        path_n, start_f_n, _, len_path_n = p[0]
                        block = 1
                        len_fs_n = RM.modulation_level(bandwidth, len_path_n)
                        #len_fs_1 = RM.modulation_level(bandwidth, len_path_n)
                        #start_f_1 = RM.SP_FF(self.G, path_1, len_fs_1)
                        if start_f_n != -1:
                            block = 0
                            for path, len_fs, start_f in path_tree:
                                if path[-1] == destination[i]:
                                    #if path != path_n:
                                    self.service_list[r][0].remove([path, len_fs, start_f])
                                    #self.release_fs(path, len_fs, start_f)
                                    self.service_list[r][0].append([path_n, len_fs_n, start_f_n])
                                    self.update_fs(path_n, len_fs_n, start_f_n)

                                    down = self.down(self.service_list[r][0], source, destination)
                                        # down[j].append(destination[i])
                                        # down[j].extend(down[i])
                                    break
                        if block == 1:
                            for path, len_fs, start_f in path_tree:
                                if path[-1] == destination[i]:
                                    # self.service_list[r][0].remove([path, len_fs, start_f])
                                    self.update_fs(path, len_fs, start_f)
                                    break
            else:
                for path, len_fs, start_f in remove_list:
                    self.update_fs(path, len_fs, start_f)
                    self.service_list[r][0].append([path, len_fs, start_f])
        return num_rerouting

    def crowded_rearrangement(self):
        num_rerouting = 0
        edge_feature = []
        for src, dst in edge_list:
            fs = self.G[src][dst]['fs']
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
                edge_feature.append([[src,dst], 0, 0, fist_avaliable])
            else:
                edge_feature.append([[src,dst], num, num / block, fist_avaliable])
        edge_feature.sort(key=lambda x:(x[1], x[2]))

        n = 20
        crowd = []
        for i in range(n):
            crowd.append(edge_feature[i][0])

        for r,(path_tree, source, destination, bandwidth, time) in enumerate(self.service_list):
            for path, len_fs, start_f in path_tree:
                flag = 0
                for i in range(len(path) - 1):
                    if [path[i],path[i+1]] in crowd:
                        flag = 1
                        break
                if flag == 1:
                    self.release_fs(path, len_fs, start_f)
                    self.service_list[r][0].remove([path, len_fs, start_f])

                    path_list = self.K_path_map[path[0]][path[-1]]

                    remove_list = []
                    for path in path_list:
                        for i in range(len(path) - 1):
                            if [path[i], path[i + 1]] in crowd:
                                remove_list.append(path)
                                break
                    for path in remove_list:
                        path_list.remove(path)

                    p = KSP_FA(self.G, path_list, bandwidth)
                    #p.sord(key=lambda x: (x[2], x[3]))
                    path_n, start_f_n, _, len_path_n = p
                    block = 1
                    len_fs_n = RM.modulation_level(bandwidth, len_path_n)
                    # len_fs_1 = RM.modulation_level(bandwidth, len_path_n)
                    # start_f_1 = RM.SP_FF(self.G, path_1, len_fs_1)
                    if start_f_n != -1:
                        block = 0
                        self.service_list[r][0].append([path_n, len_fs_n, start_f_n])
                        self.update_fs(path_n, len_fs_n, start_f_n)
                        num_rerouting += 1
                    if block == 1:
                        self.update_fs(path, len_fs, start_f)
                        self.service_list[r][0].append([path, len_fs, start_f])
        return num_rerouting

    def all_rearrangement(self, rea_list):
        num_rerouting = 0
        for r in rea_list:
            path_tree, source, destination, bandwidth, time = self.service_list[r]
            for path, len_fs, start_f in path_tree:
                self.release_fs(path, len_fs, start_f)

        for r in rea_list:
            path_tree, source, destination, bandwidth, time = self.service_list[r]
            path_tree_new = self.method(self.G, source, destination, bandwidth)
            if len(path_tree_new) != 0:
                self.update_request(path_tree_new)
                self.service_list[r][0] = path_tree_new
                num_rerouting += len(path_tree_new)

            else:
                print("block!")
                # for path, len_fs, start_f in path_tree:
                #     self.update_fs(path, len_fs, start_f)

        return num_rerouting

    def DTS_F(self):
        rea_list = Rea.DTS(self.service_list)
        return self.Full_rearrangement(rea_list)

    def DTS_P(self):
        rea_list = Rea.DTS(self.service_list)
        return self.Partial_rearrangement(rea_list)

    def QTS_F(self):
        rea_list = Rea.QTS(self.service_list, self.G)
        return self.Full_rearrangement(rea_list)

    def QTS_P(self):
        rea_list = Rea.QTS(self.service_list, self.G)
        return self.Partial_rearrangement(rea_list)

    def DTS_A(self):
        rea_list = Rea.DTS(self.service_list)
        return self.all_rearrangement(rea_list)

    def QTS_A(self):
        rea_list = Rea.QTS(self.service_list, self.G)
        return self.all_rearrangement(rea_list)

    def request_join(self):
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
                    pass
                    # num_block += 1
                    # ep_block += 1
                else:
                    self.update_fs(path, len_fs, start_f)
                    self.service_list[i][0].append([path, len_fs, start_f])
                    self.service_list[i][2].append(d)

    def request_leave(self):
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

    def statistical(self):
        num = 0
        block = 0
        for src, dst in edge_list:
            fs = self.G[src][dst]['fs']
            flag = 0
            # fist_avaliable = -1
            # for i in range(len(fs)):
            #     if fs[i] == 1:
            #         fist_avaliable = i
            #         break
            for i in fs:
                if i == 1:
                    num += 1
                    if flag == 0:
                        block += 1
                        flag = 1
                else:
                    flag = 0

        return num/(len(edge_list)*NG.num_fs), num/block

    def rmsa(self):
            episode_size = 1000
            num_episode = 0
            #service_list = []
            num_block = 0
            num_request = 0
            ep_block = 0
            blocking_rate_list = []
            num_session = 0
            num_rerouting = 0
            occup_rate = []
            fragment = []
            t = 0

            while num_episode < 25:
                time_to = round(np.random.exponential(30))
                t += time_to
                flag = self.release_request(time_to)
                if t >= 1000:
                    #num_rerouting += self.DTS_F()
                    #num_rerouting += self.DTS_P()
                    #num_rerouting += self.QTS_F()
                    #num_rerouting += self.QTS_P()
                    t -= 1000
                # if flag == 1:   #rearrangement
                #     #pass
                #     print(len(self.service_list))
                #     num_rerouting += self.DTS_F()
                #     #num_rerouting += self.DTS_P()
                #     #num_rerouting += self.QTS_F()
                #     #num_rerouting += self.QTS_P()
                #     #num_rerouting += self.crowded_rearrangement()
                #     #self.DTS_A()
                #     #self.QTS_A()

                num_request += 1
                #print(num_request)
                # mode = random.randint(0, 2)
                # mode = 0

                # if mode == 0 :  #a multicast session  first appears
                source, destination, bandwidth, time = self.random_request(self.traffic_load)
                path_tree = self.method(self.G, source, destination, bandwidth)
                if len(path_tree) == 0:
                    num_block += 1
                    ep_block += 1
                else:
                    num_session += 1
                    self.service_list.append([path_tree, source, destination, bandwidth, time])
                    self.update_request(path_tree)

                self.request_join()
                self.request_leave()
                #elif mode == 1:  #a new member d to join
                    # i, d = self.join()
                    # if i != -1:
                    #     tree, source, destination, bandwidth, _ = self.service_list[i]
                    #     flag = 0
                    #     for path, _, _ in tree:
                    #         if d == path[0] or d == path[-1]:
                    #             flag = 1
                    #     if flag == 1:
                    #         self.service_list[i][2].append(d)
                    #     else:
                    #         p = []
                    #         for u in ([source] + destination):
                    #             #p.append(self.path_map[u, d])
                    #             p.append(KSP_FA(self.G, self.K_path_map[u][d], bandwidth))
                    #         p = sorted(p, key=lambda x:(x[2],x[3]))
                    #         path, start_f, _,len_path = p[0]
                    #         # path = p[0][0]
                    #         # len_path = p[0][1]
                    #         #len_path = RM.get_path_len(self.G, path)
                    #         len_fs = RM.modulation_level(bandwidth, len_path)
                    #         #start_f = RM.SP_FF(self.G, path, len_fs)
                    #         if start_f == -1:
                    #             num_block += 1
                    #             ep_block += 1
                    #         else:
                    #             self.update_fs(path, len_fs, start_f)
                    #             self.service_list[i][0].append([path, len_fs, start_f])
                    #             self.service_list[i][2].append(d)
                #else:            #a member d to leave
                    # i, d = self.leave()
                    # if i != -1:
                    #     tree, source, destination, bandwidth, _ = self.service_list[i]
                    #     flag = 0
                    #     for path,_,_ in tree:
                    #         if path[0] == d:
                    #             flag = 1
                    #
                    #     if flag == 1:  #d has downstream members
                    #         self.service_list[i][2].remove(d)
                    #     else:
                    #         self.service_list[i][2].remove(d)
                    #         for path, len_fs, start_f in tree:
                    #             if path[-1] == d:
                    #                 self.release_fs(path, len_fs, start_f)
                    #                 self.service_list[i][0].remove([path, len_fs, start_f])

                if num_request % episode_size == 0:
                    num_episode += 1
                    print("Ep: {}, Blocking P: {},  Ep Bp: {}".format(num_episode,
                                                                      num_block / num_request,
                                                                      ep_block / episode_size))
                    o, f = self.statistical()
                    occup_rate.append(o)
                    fragment.append(f)
                    print("         occupancy rate : {},  Ep: {}".format(np.mean(occup_rate), o))
                    print("         fragment: {},  Ep: {}".format(np.mean(fragment), f))
                    blocking_rate_list.append(num_block / num_request)
                    ep_block = 0
            self.release_request(1000)
            print("Average rerouting number:",num_rerouting/num_session)
            return num_block / num_request, num_rerouting/num_session

# load_f_list = []
# rerouting_list = []
# for l in load:
#     print('load:', l)
#     Benchmark = RMSA(NG.G, method=method[2], traffic_load=l)
#     blocking_rate,rerouting_number = Benchmark.rmsa()
#     load_f_list.append(blocking_rate)
#     rerouting_list.append(rerouting_number)
# np.save("logs/load_f_along_traffic_load", load_f_list)
# np.save("logs/rerouting_number", rerouting_list)
Benchmark = RMSA(NG.G, method=method[2], traffic_load=500)
Benchmark.rmsa()