import datetime
import os
import threading
import Net_Graph as NG
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from GNN_A3C import Net
from GNN_model import GCN
from agent import GNN_agent
#criterion = nn.CrossEntropyLoss()

criterion = nn.L1Loss(reduction='mean')

net_graph = NG.G
model_path = "saved_model/"

device = 0
iters_per_epoch = 50
lr = 0.01
seed = 0
in_feats = 1
hidden_size = 64
num_classes = 2

# set up seeds and gpu device
torch.manual_seed(0)
np.random.seed(0)
device = torch.device("cuda:" + str(device)) if torch.cuda.is_available() else torch.device("cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)


# model = GCN(in_feats, hidden_size, num_classes)
# optimizer = optim.Adam(model.parameters(), lr=lr)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
#
# agent = GNN_agent(net_graph, criterion, model, device, optimizer)
# agent.rmsa()

global_model = Net(in_feats, num_classes)

n_worker = 1

