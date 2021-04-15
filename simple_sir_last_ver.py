# Commented out IPython magic to ensure Python compatibility.
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
# %matplotlib inline
from itertools import product
import os #to create a folder
from definitions import ws_sir, infos_sorted_nodes, \
  remove_loops_parallel_edges, check_loops_parallel_edges, config_pois_model, replace_edges_from, \
    rhu, pow_max

"""##CONNECTED_Watts-Strogatz Net!

This model works only if k = even, i.e. if k=odd then k-1 is choosen. 
So, k-1 neighbors are connected + I've choosen that if D = float 
-> round_half_up(D) = D and let the watts-strogatz model go ahead with the pruning. 
I "round_half_up" since, as for N = 15, k = N/2 = 8, in many cases 
it drives to a nearer pruning.
"""

'rewire all the edges with a probability of p'
N = int(1e3)

def even_int(x):
  if int(x) % 2 != 0: return int(x-1)
  return int(x)

for pruning in [True, False]: #if 1 needed: add ``break``

  if pruning == True:
    p_max = .2
    p_prog = np.linspace(0,p_max,int(p_max*10)+1)
    print("I'm pruning!")
    betas = [2e-3]

    'In WS model, if D = odd, D = D - 1. So, convert it now'
    k_prog = [even_int(N/x) for x in \
              [2**i for i in range(0,pow_max(N, num_iter = "all"))]]*len(betas) #if pow_max +1 --> error of connectivity: D = k_odd - 1
    beta_prog = [beta*N/k for beta in betas for k in k_prog[:len(set(k_prog))]]
    mu_prog = np.linspace(0.016,1,25) #[0.467, 0.385, 0.631]
    
    folder = "WS_Pruned" 

    print("k_prog %s, len: %s" % (k_prog, len(k_prog)),
    "\nbeta_prog %s, len: %s" % (beta_prog, len(beta_prog)), 
    "\nmu_prog %s, len: %s" % (mu_prog, len(mu_prog)), )

    total_iterations = 0
    for mu, p in product(mu_prog, p_prog):
        for beta, D in zip(beta_prog, k_prog):
          #print("R0: ", beta*D/mu)
          if  0 < beta*D/mu < 7 and beta <= 1:
            total_iterations += 1
    print("Total Iterations:", total_iterations)
    

    done_iterations = 0; saved_nets = []
    for D, beta in zip(k_prog, beta_prog):
      
      #print("beta %s ; mu: %s; beta_1.2: %s" % (beta, mu, beta) )
      'With p = 1 and <k>/N ~ 0, degree distr is sim to a Poissonian'
      G = nx.connected_watts_strogatz_graph( n = N, k = D, p = p, seed = 1 ) #k is the number of near linked nodes
      for mu, p in product(mu_prog, p_prog): 
          if 0 < beta*D/mu < 7 and beta <= 1:
            done_iterations+=1
            print("Iterations left: %s" % ( total_iterations - done_iterations ) )
            #print("beta", beta, "D", D, "R0", beta*D/mu, "mu", mu) 
            ws_sir(G, saved_nets=saved_nets, pruning = pruning, folder = folder, D = D, p = p, beta = beta, mu = mu)

  if pruning == False:
    'test != kind of '
    print("-----------------I'm NOT pruning!")
    p_prog = [0.1]#np.linspace(0,p_max,int(p_max*10)+1)
    mu_prog = np.linspace(0.001,1,10)
    beta_prog = np.linspace(0.7,1,10)
    k_prog = [2]#np.arange(2,18,4)
    folder = "WS_Epids_D2"

    total_iterations = 0
    for beta, D, mu, p in product(beta_prog, k_prog, mu_prog, p_prog):
        if  .3 < beta*D/mu < 8:
          total_iterations += 1
    print("Total Iterations:", total_iterations)
    done_iterations = 0
    
    saved_nets = []
    for D, p in product(k_prog, p_prog):
      G = nx.connected_watts_strogatz_graph( n = N, k = D, p = p, seed = 1 ) #k is the number of near linked nodes
      for mu, beta in product(mu_prog, beta_prog): 
        if  0.3 < beta*D/mu < 8:   
          done_iterations+=1
          print("Iterations left: %s" % ( total_iterations - done_iterations ) )
          ws_sir(G, saved_nets = saved_nets, folder = folder, pruning = pruning, D = D, p = p, beta = beta, mu = mu)