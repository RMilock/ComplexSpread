# -*- coding: utf-8 -*-
"""Simple_SIR_ver25.3(8:32).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/16RB0AyULIeW0eD0auju7M_9synQ_9Z7R

Version17.3: without "Create a Fully Connected Network

"""## Importation"""

# Commented out IPython magic to ensure Python compatibility.
import random
import math
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
from itertools import product
import os #to create a folder
from definitions import ws_sir, plot_sir, infos_sorted_nodes, plot_G_degdist_adjmat_sir, \
  remove_loops_parallel_edges, check_loops_parallel_edges, config_pois_model, replace_edges_from, \
    rhu

np.set_printoptions(precision=4, suppress=True)
#make the axes white
params = {"ytick.color" : "k",
          "xtick.color" : "k",
          "axes.labelcolor" : "k",
          "axes.edgecolor" : "k",
        "axes.titlecolor":"k"}
plt.rcParams.update(params)

"""##CONNECTED_Watts-Strogatz Net!

This model works only if k = even, i.e. if k=odd then k-1 is choosen. 
So, k-1 neighbors are connected + I've choosen that if k_ws = float 
-> round_half_up(k_ws) = k_ws and let the watts-strogatz model go ahead. 
I "round_half_up" since, as for N = 15, k = N/2 = 8, in many cases 
it drives to a nearer pruning.
"""

'rewire all the edges with a probability of p'
N = int(30); p_max = 1

'excecute the code'
'max pow-2'
#k_prog = [int(N/x) for x in \
#          [2**i for i in range(0,pow_max(N, num_iter = "all"))]] #if pow_max +1 --> error of connectivity: k_ws = k_odd - 1

'test != kind of '
k_prog = np.arange(4,10,2)
p_prog = np.linspace(0,p_max,int(p_max*10)+1)
mu_prog = np.linspace(0.01,1,10)
beta_prog = np.linspace(0.01,1,10)
p_prog = [0]
'try only with p = 0.1'
total_iterations = len(k_prog)*len(p_prog)*len(mu_prog)*len(beta_prog)
print("Total Iterations:", total_iterations)
done_iterations = 0
for k_ws,mu,p,beta in product(k_prog, mu_prog, p_prog, beta_prog): 
  done_iterations+=1
  print("Iterations left: %s" % ( total_iterations - done_iterations ) )
  if beta*k_ws/mu < 16 and beta*k_ws/mu > 0.5:
    ws_sir(N, k_ws = k_ws, p = p, beta = beta, mu = mu, plot_all=True) 


#from google.colab import files
#!rm -r /content/WS_plots/*.pdf
#!rm -rf WS_plots/
#!zip -r /content/WS_plots.zip /content/WS_plots
#files.download("/content/WS_plots.zip")
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















