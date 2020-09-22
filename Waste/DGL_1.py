import torch
import dgl

G = dgl.DGLGraph()
G.add_nodes(3)
G.ndata['x'] = torch.zeros((3, 5))  # init 3 nodes with zero vector(len=5)
print(G.ndata)
