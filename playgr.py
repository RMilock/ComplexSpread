#%%

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import itertools
import random
from definitions import sir, itermean_sir
import networkx as nx

numb_classes = 3; numb_iter = 30

G = nx.connected_watts_strogatz_graph( n = 200, k = 200, p = 1, seed = 1) #k is the number of near linked nodes
nx.draw_circular(G)
plt.show()
plt.close()
#print("\nlen_traj", np.shape(trajectories[0]), "traj", trajectories[0], "\navg", avg[0])
prev, rec, cum = sir(G, seed=True)
plt.plot(prev, "r", rec, "b--", cum, "mo--", ms = 2)
plt.show()

#%%
plt.close()
trajectories, avg = itermean_sir(G, numb_iter=numb_iter, k = 10)
plt.show()
# %%
for i in range(numb_classes):
    for j in range(numb_iter):
        plt.plot(trajectories[i][j], color="wheat")
for i in range(numb_classes): plt.plot(avg[i])
# %%
def rhu(n, decimals=0): #round_half_up
  import math
  multiplier = 10 ** decimals
  return math.floor(n*multiplier + 0.5) / multiplier
rhu(0.001,3)

# %%
x = np.linspace(0.2,1,9)
print(x)
# %%

# %%
