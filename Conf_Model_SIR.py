import random
import math
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
# %matplotlib inline
from itertools import product
import os #to create a folder
from definitions import rhu, plot_save_net, plot_save_sir, config_pois_model, NN_pois_net, pois_pos_degrees

'save scaled version for better visualization'
def scaled_conf_pois(G,D,cut_off=30):    
  scaled_N = int(G.number_of_nodes()/cut_off) # int(D/cut_off)
  #if int(D) >= 2: 
  #  print("The rescaled one has N: %s and D: %s" % (int(N/cut_off), int(D/cut_off)) )
  return config_pois_model(scaled_N, D)

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

p_max = 0; N = int(1e3)

'progression of net-parameters'
k_prog = np.arange(10,36,2)
p_prog = np.linspace(0,p_max,int(p_max*10)+1)
mu_prog = np.linspace(0.001,1,15)
beta_prog = np.linspace(0.001,1,15)
p_prog = [0]
R0_max = 5; R_min = 0.3

'try only with p = 0.1'
total_iterations = 0
for D,mu,p,beta in product(k_prog, mu_prog, p_prog, beta_prog):  
  if R_min < beta*D/mu < R0_max:
    total_iterations+=1
print("Total Iterations:", total_iterations)
done_iterations = 0

saved_nets = []
for D,mu,p,beta in product(k_prog, mu_prog, p_prog, beta_prog):  
  if R_min < beta*D/mu < R0_max:
    done_iterations+=1
    print("Iterations left: %s" % ( total_iterations - done_iterations ) )

    folder = "Config_Model"
    G = config_pois_model(N, D)
  
    
    scaled_G = scaled_conf_pois(G, D = D)
    
    'plot G, degree distribution and the adiaciency matrix and save them'
    #Config_SIR def: D = 8, beta, mu = 0.1, 0.05
    #plot_save_net(G, scaled_G = scaled_G, folder = folder, D = D, p = 0)
    #plot_save_sir(G, folder = folder, beta = beta, D = D, mu = mu, p = 0)

    #infos_sorted_nodes(G, True)
    G = NN_pois_net(G, D = D)
    folder = "NNR_Conf_Model"

    plot_save_sir(G, folder = folder, beta = beta, D = D, mu = mu, p = p_max)

    'possibly save scaled version for better visualization'
    scaled_G = NN_pois_net(G = scaled_G, D = D)

    if "N%s_D%s_p%s"% (N,D,rhu(p,3)) not in saved_nets: 
      plot_save_net(G = G, scaled_G = G, folder = folder, D = D, p = p)
      saved_nets.append("N%s_D%s_p%s"% (N,D,rhu(p,3)))
      print(saved_nets)