import networkx as nx
import matplotlib.pyplot as plt

def creat_G():
    G = nx.DiGraph()
    for i in range(14):
        G.add_node(i)
    # G.add_weighted_edges_from([(1, 2, 1050), (1, 3, 1500), (1, 8, 2400), (2, 3, 600), (2, 4, 750), (3, 6, 1800),
    #                             (4, 5, 600), (4, 11, 1950), (5, 6, 1200), (5, 7, 600), (6, 10, 1050), (6, 14, 1800),
    #                             (7, 8, 750), (7, 10, 1350), (8, 9, 750), (9, 10, 750), (9, 12, 300), (9, 13, 300),
    #                             (11, 12, 600), (11, 13, 750), (12, 14, 300), (13, 14, 150),
    #                            (2, 1, 1050), (3, 1, 1500), (8, 1, 2400), (3, 2, 600), (4, 2, 750), (6, 3, 1800),
    #                            (5, 4, 600), (11, 4, 1950), (6, 5, 1200), (7, 5, 600), (10, 6, 1050), (14, 6, 1800),
    #                            (8, 7, 750), (10, 7, 1350), (9, 8, 750), (10, 9, 750), (12, 9, 300), (13, 9, 300),
    #                            (12, 11, 600), (13, 11, 750), (14, 12, 300), (14, 13, 150)])
    G.add_weighted_edges_from([(0, 1, 1050), (0, 2, 1500), (0, 7, 2400), (1, 2, 600), (1, 3, 750), (2, 5, 1800),
                               (3, 4, 600), (3, 10, 1950), (4, 5, 1200), (4, 6, 600), (5, 9, 1050), (5, 13, 1800),
                               (6, 7, 750), (6, 9, 1350), (7, 8, 750), (8, 9, 750), (8, 11, 300), (8, 12, 300),
                               (10, 11, 600), (10, 12, 750), (11, 13, 300), (12, 13, 150),
                               (1, 0, 1050), (2, 0, 1500), (7, 0, 2400), (2, 1, 600), (3, 1, 750), (5, 2, 1800),
                               (4, 3, 600), (10, 3, 1950), (5, 4, 1200), (6, 4, 600), (9, 5, 1050), (13, 5, 1800),
                               (7, 6, 750), (9, 6, 1350), (8, 7, 750), (9, 8, 750), (11, 8, 300), (12, 8, 300),
                               (11, 10, 600), (12, 10, 750), (13, 11, 300), (13, 12, 150)])
    return G

def show_G(G):
    print('number_of_nodes:', nx.number_of_nodes(G))
    print('nodes:', G.nodes())
    print('number_of_edges:', nx.number_of_edges(G))
    print('edges:', G.edges(data=True))

    pos = nx.spring_layout(G)
    #pos = nx.circular_layout(G)
    nx.draw_networkx(G, pos, with_labels=True)
    #nx.draw_networkx_edge_labels(G, pos, labels='weight')
    plt.show()

G = creat_G()

num_fs = 100
for u,v in G.edges():
    #print(u,v)
    #print(e)
    #print(G[e[0]][e[1]])
    G[u][v]['fs'] = [1 for i in range(num_fs)]
#show_G(G)