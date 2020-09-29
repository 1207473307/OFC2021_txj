import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from Data_set import data_set,edge_list
from torch.nn import init

random.seed(0)
torch.manual_seed(0)

class Net(nn.Module):
    pass