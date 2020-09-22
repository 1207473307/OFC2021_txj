import numpy as np
import torch

# action = np.load("training_logs/r_aw00.npy",allow_pickle=True)
# state = np.load("training_logs/r_sw00.npy",allow_pickle=True)
# a = np.squeeze(action)
# s = np.squeeze(state)
# b = 1
#
# a = [[0,1,2,3,4,5],[6,7,8,9,0,1]]
# for i,(a1,_,_,_,_,_) in enumerate(a):
#     print(a1)
print(torch.cuda.is_available())