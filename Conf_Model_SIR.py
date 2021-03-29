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

#from google.colab import files
#!rm -r /content/Config_plots/*.pdf

p_max = 0; N = int(30)

"""## NN_rewiring: Pb with D = 8"""

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
    G = config_pois_model(N,k_ws, beta_eff = beta, mu_eff = mu, visual = True)
    #infos_sorted_nodes(G, True)
    nearest_neighbors_pois_net(G, D = k_ws, beta_eff = beta, mu_eff = mu)
    plt.show()





plt.show()
'''
todo: 
insert the NNR conf model + save_in + uniform the proression done in ws also with this
+ halve the ws w/ eps = 0.1 as many as possible as in the 17.3 version
'''


#from google.colab import files
#!rm -r /content/WS_plots/*.pdf
#!rm -rf WS_plots/
#!zip -r /content/Config_plots.zip /content/Config_plots
#files.download("/content/Config_plots.zip")


'''


"""## Old version of the above cell"""

'Nearest-Neighbors Rewiring (NNR) by ascending degree order'
nodes_rew = [x for x in G.nodes()]

'for random rewiring with p'
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
    'create edges rewiring from ascending degree'

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

        'remove node whenever its degree is = 0:'
        try: 
          if nodes_degree[a_attached] == 0: l_nodes.remove(a_attached)
        except: 
          print("error for a_att:", a_attached)
        try: 
          if nodes_degree[c_attached] == 0: l_nodes.remove(c_attached)
        except: 
          print("error for c_att:", c_attached)
          
    'if nodes_degree[i] is odd  and the aa_attached, present in l_nodes, has a stub avaible, then, +1 anticlock-wise'
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

'''