# -*- coding: utf-8 -*-
"""Simple_SIR_ver18.3(20:00).ipynb

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
#comment for vsc
#commented on vsc from ssh-tunneling from colab
# %matplotlib inline
from itertools import product
import os #to create a folder
#from definitions import sir, plot_sir, infos_sorted_nodes, plot_G_degdist_adjmat_sir, \
#  remove_loops_parallel_edges, check_loops_parallel_edges

from definitions import *

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
N = int(100); p_max = 1

'excecute the code'
'max pow-2'
#k_prog = [int(N/x) for x in \
#          [2**i for i in range(0,pow_max(N, num_iter = "all"))]] #if pow_max +1 --> error of connectivity: k_ws = k_odd - 1
'test != kind of '
k_prog = np.arange(5,11)
p_prog = np.linspace(0,p_max,int(p_max*10)+1)
mu_prog = np.linspace(0.01,1,10)
beta = np.linspace(0.01,1,10)
k_prog = [3,8]; mu_prog = [0.16]
'try only with p = 0.1'
for k_ws,mu,p in product(k_prog, mu_prog,p_prog):  
  ws_sir(N, k_ws = k_ws, p = p, beta = 0.1, mu = mu) 


#from google.colab import files
#!rm -r /content/WS_plots/*.pdf
#!rm -rf WS_plots/
#!zip -r /content/WS_plots.zip /content/WS_plots
#files.download("/content/WS_plots.zip")
print("end")
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

#from google.colab import files
#!rm -r /content/Config_plots/*.pdf

D = 3; seed=123; p = 0; N = int(1e3)

"""## NN_rewiring: Pb with D = 8"""

G = config_pois_model(N,D,visual = False)

infos_sorted_nodes(G)

verbose = False

def verboseprint(*args):
  if verbose == True:
      print(*args)
  elif verbose == False:
     None

'''for random rewiring with p'''
l_nodes = [x for x in G.nodes()]

edges = set() #avoid to put same link twice (+ unordered)
nodes_degree = {}

'list of the nodes sorted by their degree'
for node in G.nodes():
  nodes_degree[node] = G.degree(node)
sorted_nodes_degree = {k: v for k, v in sorted(nodes_degree.items(), key=lambda item: item[1])}
sorted_nodes = [node for node in sorted_nodes_degree.keys()]
verboseprint("There are the sorted_nodes", sorted_nodes) #, "\n", sorted_nodes_degree.values())

'cancel all the edges'
replace_edges_from(G)


'------ Start of Rewiring with NNR! ---------'
'''Hint: create edges rewiring from ascending degree'''
def get_var_name(my_name):
  variables = dict(globals())
  for name in variables:
      if variables[name] is my_name:
          #verboseprint("v[n]", variables[name], "my_n", my_name)
          return name
def ls_nodes_remove(node): l_nodes.remove(node); sorted_nodes.remove(node)
def zero_deg_remove(node): 
  if nodes_degree[node] == 0 and node in l_nodes and node in sorted_nodes: ls_nodes_remove(node); verboseprint("\n", get_var_name(node), "is node", node, "geremoved via if deg == 0")

verboseprint("\nStart of the wiring:")
while( len(l_nodes) > 1 ):
  node = sorted_nodes[0]
  verboseprint("---------------------------")
  verboseprint("Wire node", node, " with degree", nodes_degree[node], \
        "\nto l_nodes left for rew:", l_nodes, "\nlen(nodes left)", len(l_nodes))
  
  L = len(l_nodes)
  #try:
  aa_attached = l_nodes[(l_nodes.index(node)+1)%L]; verboseprint("aa_attached is ", aa_attached)
  #except: verboseprint("error in the anti_anti_clock-wise attachment -- break"); break
  
  if node in l_nodes:
    'if degreees[node] > 1, forced-oscillation-wiring'
    for j in range(1,nodes_degree[node]//2+1): #neighbors attachment and no self-loops "1"
        if len(l_nodes) == 1: break
        verboseprint("entered for j:",j)
        idx = l_nodes.index(node)
        verboseprint("idx_node:", idx)
        a_attached = l_nodes[(idx+j)%L] #anticlockwise-linked node
        c_attached = l_nodes[(idx-j)%L]
        aa_attached = l_nodes[(idx-nodes_degree[node]//2+1)%L]
        verboseprint(node,a_attached); verboseprint(node,c_attached)
        if node != a_attached: edges.add((node,a_attached)); nodes_degree[a_attached]-=1; \
        verboseprint("deg[%s] = %s" % (a_attached, nodes_degree[a_attached]))
        if node != c_attached: edges.add((node,c_attached)); nodes_degree[c_attached]-=1; \
        verboseprint("deg[%s] = %s"%(c_attached,nodes_degree[c_attached]))

        '''remove node whenever its degree is = 0:'''
        zero_deg_remove(a_attached)
        zero_deg_remove(c_attached)

    if len(l_nodes) == 1: break       
    '''if nodes_degree[i] is odd  and the aa_attached, present in l_nodes, has a stub avaible, then, +1 anticlock-wise'''
    if nodes_degree[node] % 2 != 0 and nodes_degree[aa_attached] != 0 : edges.add((node, aa_attached)); nodes_degree[aa_attached]-=1; \
                            verboseprint("edge with aa added: ", node, aa_attached, "and deg_aa_att[%s] = %s"%(aa_attached,nodes_degree[aa_attached]))
    
    'aa_attached == 0 should not raise error since it should be always present in l_n and s_n'
    if nodes_degree[aa_attached] == 0: ls_nodes_remove(aa_attached); verboseprint("\naa_attached node", aa_attached, "  removed via if deg == 0")
    if node in l_nodes: ls_nodes_remove(node);  verboseprint(node, "is removed via last if statement")
    if len(l_nodes)==1: verboseprint("I will stop here"); break

verboseprint("End of wiring")

replace_edges_from(G, edges)

check_loops_parallel_edges(G)
infos_sorted_nodes(G, num_nodes=False)


plot_G_degdist_adjmat_sir(G, beta = beta_eff, mu = mu_eff, log = True)


try:
      plt.savefig("Config_plots/NNR_Conf_SIR_N%s_D%s_p%s_beta%s_mu%s" % (N,D,p, rhu(beta_eff,3), rhu(mu_eff,3)) + ".pdf")
except:
      os.mkdir("Config_plots")
      plt.savefig("Config_plots/NNR_Config_SIR_N%s_D%s_p%s_beta%s_mu%s" % (N,D,p, rhu(beta_eff,3), rhu(mu_eff,3)) + ".pdf")

#from google.colab import files
#!rm -r /content/WS_plots/*.pdf
#!rm -rf WS_plots/
#!zip -r /content/Config_plots.zip /content/Config_plots
#files.download("/content/Config_plots.zip")















"""## Old version of the above cell"""

'''Nearest-Neighbors Rewiring (NNR) by ascending degree order'''
nodes_rew = [x for x in G.nodes()]

'''for random rewiring with p'''
l_nodes = nodes_rew.copy() 


edges = set() #avoid to put same link twice (+ unordered)
nodes_degree = {}

'list of the nodes sorted by their degree'
for node in G.nodes():
  nodes_degree[node] = G.degree(node)
sorted_nodes_degree = {k: v for k, v in sorted(nodes_degree.items(), key=lambda item: item[1])}
sorted_nodes = [node for node in sorted_nodes_degree.keys()]
print("There are the sorted_nodes", sorted_nodes) #, "\n", sorted_nodes_degree.values())

'cancel all the edges'
replace_edges_from(G)

print("\nStart of the rewiring:")
for node in sorted_nodes:
  print("---------------------------")
  print("wiring node", node, " with degree", nodes_degree[node], "\nto l_nodes left for rew:", l_nodes, "\nlen(nodes left)", len(l_nodes))
  #print(len(l_nodes))
  L = len(l_nodes)
  try:
    aa_attached = l_nodes[(l_nodes.index(node)+1)%L]; print("aa_attached is ", aa_attached)
    if aa_attached == node: aa_attached = l_nodes[1]; print("aa_attached has been changed from %s to %s"%(l_nodes[0], l_nodes[1]))
  except: print("l_nodes:", l_nodes, "was 0 after",node, "So, break it" ); break
  
  if node in l_nodes:
    '''create edges rewiring from ascending degree'''

    'if degreees[node] > 1, forced-oscillation-wiring'
    for j in range(1,nodes_degree[node]//2+1): #neighbors attachment and no self-loops "1"
        
        print("entered for j:",j)
        idx = l_nodes.index(node)
        print("idx_node:", idx)
        a_attached = l_nodes[(idx+j)%L] #anticlockwise-linked node
        c_attached = l_nodes[(idx-j)%L]
        aa_attached = l_nodes[(idx-nodes_degree[node]//2+1)%L]
        #print(node,a_attached); print(node,c_attached);
        if node != a_attached: edges.add((node,a_attached)); nodes_degree[a_attached]-=1 \
        #print("deg[%s] = %s" % (a_attached, nodes_degree[a_attached]))
        if node != c_attached: edges.add((node,c_attached)); nodes_degree[c_attached]-=1 \
        #print("deg[%s] = %s"%(c_attached,nodes_degree[c_attached]))

        '''remove node whenever its degree is = 0:'''
        try: 
          if nodes_degree[a_attached] == 0: l_nodes.remove(a_attached)
        except: 
          print("error for a_att:", a_attached)
        try: 
          if nodes_degree[c_attached] == 0: l_nodes.remove(c_attached)
        except: 
          print("error for c_att:", c_attached)
          
    '''if nodes_degree[i] is odd  and the aa_attached, present in l_nodes, has a stub avaible, then, +1 anticlock-wise'''
    if nodes_degree[node] % 2 != 0 and nodes_degree[aa_attached] != 0 : edges.add((node, aa_attached)); nodes_degree[aa_attached]-=1; \
                            print("edge added: ", node, aa_attached); print("deg[%s] = %s"%(aa_attached,nodes_degree[aa_attached]))
    
    'aa_attached == 0 shold be always present'
    if nodes_degree[aa_attached] == 0: l_nodes.remove(aa_attached); print("\naa_attached node", aa_attached, "  removed via if deg == 0")
    if node in l_nodes: l_nodes.remove(node);  print(node, "is removed via last if statement")

  if len(l_nodes)==1: print("stop!"); break

print(G.edges())
replace_edges_from(G, edges)

check_loops_parallel_edges(G)
infos_sorted_nodes(G)

plot_G_degdist_adjmat_sir(G, log = True)

"""NB: seed = 125; Taken comments on plot sir to cure the warning"""