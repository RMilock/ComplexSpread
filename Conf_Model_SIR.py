# Commented out IPython magic to ensure Python compatibility.
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
# %matplotlib inline
from itertools import product
import os #to create a folder
from definitions import ws_sir, plot_sir, infos_sorted_nodes, plot_G_degdist_adjmat_sir, \
remove_loops_parallel_edges, check_loops_parallel_edges, config_pois_model, replace_edges_from, \
rhu, nearest_neighbors_pois_net

p_max = 0; N = int(1e3)

"""## Configurational Model with poissonian degree:
1) Question: 
  1. using this meth, loops and parallel edges are present leading 
  to a adj_matrix not normalized to 1. Since neighbors are involved in sir 
  and contagious is made by (fut = S, curr = S) - nodes, I may leave them, 
  but there're not so in line of "social contacts". 
  So, I prefer to remove them, but I loose a lot of precision if $D / N !<< 1$. 
  <br>Ex., $D = 50 = N, <k> ~ 28$. For $N= 1000 \text{ and } D = 3 \textrm{ or } 8, 
  <k> \textrm{is acceptable.}$
  2. To avoid the picking of $0$, choice on a poisson sequence

TODO: implement the idea of a pruning factor as in ws_sir
"""



"""## NN_rewiring: Pb with D = 8"""

'test != kind of '
k_prog = np.arange(2,10,2)
p_prog = np.linspace(0,p_max,int(p_max*10)+1)
mu_prog = np.linspace(0.001,1,15)
beta_prog = np.linspace(0.001,1,15)
p_prog = [0]
'try only with p = 0.1'
total_iterations = 0
for D,mu,p,beta in product(k_prog, mu_prog, p_prog, beta_prog):  
  if .5 < beta*D/mu < 16:
    total_iterations+=1
print("Total Iterations:", total_iterations)
done_iterations = 0
for D,mu,p,beta in product(k_prog, mu_prog, p_prog, beta_prog):  
  if .5 < beta*D/mu < 16:
    done_iterations+=1
    print("Iterations left: %s" % ( total_iterations - done_iterations ) )
    G = config_pois_model(N,D, beta = beta, mu = mu, plot_all = False)
    #infos_sorted_nodes(G, True)
    nearest_neighbors_pois_net(G, D = D, beta = beta, mu = mu, plot_all=False)