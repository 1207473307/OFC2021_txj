import numpy as np
import matplotlib.pyplot as plt

x = [300, 500, 700, 900, 1100]
NR = np.load("logs/NR3.npy", allow_pickle=True)
DTS_F = np.load("logs/DTS_F3.npy", allow_pickle=True)
DTS_P = np.load("logs/DTS_P3.npy", allow_pickle=True)
QTS_F = np.load("logs/QTS_F3.npy", allow_pickle=True)
QTS_P = np.load("logs/QTS_P3.npy", allow_pickle=True)

plt.figure()
plt.xlabel('traffic load')
plt.ylabel('blocking rate')


#plt.plot(x, y, color='red', label='FF')
# plt.plot(x, DRL, color='red', label='DRL')
# plt.plot(x, y[0], color='blue', label='FF')
# plt.plot(x, y[1], color='blue', linestyle='--', label='FLF')
# plt.plot(x, y[2], color='green', label='LU')
# plt.plot(x, y[3], color='green', linestyle='--', label='MU')
# plt.plot(x, y[4], color='black', label='EF')

plt.semilogy(x, NR, color='red', label='NR')
plt.semilogy(x, DTS_F, color='blue', label='DTS_F')
plt.semilogy(x, DTS_P, color='blue', linestyle='--', label='DTS_P')
plt.semilogy(x, QTS_F, color='green', label='QTS_F')
plt.semilogy(x, QTS_P, color='green', linestyle='--', label='QTS_P')


plt.legend()
plt.show()