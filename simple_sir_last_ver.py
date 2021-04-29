<<<<<<< HEAD
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
    rhu, pow_max, save_log_params, my_dir

'''
CONNECTED_Watts-Strogatz Net!
This model works only if k = even, i.e. if k=odd then k-1 is choosen. 
So, k-1 neighbors are connected + I've choosen that if D = float 
-> round_half_up(D) = D and let the watts-strogatz model go ahead with the pruning. 
I "round_half_up" since, as for N = 15, k = N/2 = 8, in many cases 
it drives to a nearer pruning.
'''

'rewire all the edges with a probability of p'
N = int(50)

def even_int(x):
  if int(x) % 2 != 0: return int(x-1)
  return int(x)

for pruning in [False]: #if 1 needed: add ``break``
  if pruning == True:
    p_max = 0.2
    p_prog = np.linspace(0,p_max,int(p_max*10)+1)
    print("---I'm pruning!")
    betas = [1e-3, 2e-3, 1e-4]

    'In WS model, if D = odd, D = D - 1. So, convert it now'
    k_prog = [even_int(N/x) for x in \
              [2**i for i in range(0,pow_max(N, num_iter = "all"))]]*len(betas) #if pow_max +1 --> error of connectivity: D = k_odd - 1
    beta_prog = [beta*N/k for beta in betas for k in k_prog[:len(set(k_prog))]]
    mu_prog = np.linspace(0.16,1,10) #[0.467, 0.385, 0.631]
    R0_min = 0; R0_max = 7
    
    folder = "WS_Pruned" 
    text = "N %s;\n k_prog %s, len: %s;\np_prog %s, len: %s;\nbeta_prog %s, len: %s;\nmu_prog %s, len: %s;\nR0_min %s, R0_max %s\n---\n" \
          % (N, k_prog, len(k_prog), p_prog, len(p_prog), beta_prog, len(beta_prog), \
          mu_prog, len(mu_prog), R0_min, R0_max)
    print(text)    

    total_iterations = 0
    for mu, p in product(mu_prog, p_prog):
        for beta, D in zip(beta_prog, k_prog):
          #print("R0: ", beta*D/mu)
          if  R0_min < beta*D/mu < R0_max and beta <= 1:
            total_iterations += 1
    print("Total Iterations:", total_iterations)

    done_iterations = 0; saved_nets = []
    for D, beta in zip(k_prog, beta_prog):
      for p in p_prog:
        G = nx.connected_watts_strogatz_graph( n = N, k = D, p = p, seed = 1 ) #k is the number of near linked nodes  
        for mu in mu_prog:
          'With p = 1 and <k>/N ~ 0, degree distr is sim to a Poissonian'
          if R0_min < beta*D/mu < R0_max and beta <= 1:
            done_iterations+=1
            print("Iterations left: %s" % ( total_iterations - done_iterations ) )
            print("beta", rhu(beta,3), "D", D, "R0", rhu(beta*D/mu,3), \
                  "mu", rhu(mu,3), "p", p) 
            ws_sir(G, saved_nets=saved_nets, pruning = pruning, folder = folder, D = D, p = p, beta = beta, mu = mu, done_iterations = done_iterations)
    
    save_log_params(folder = folder, text = text)

  if pruning == False:
    'test != kind of '
    print("---I'm NOT pruning!")
    'if p_prog has sequence save_it like ./R0_0-1/R0_0.087'
    p_prog = np.concatenate((np.array([0.001]), np.linspace(0.012,0.1,10)))
    mu_prog = np.linspace(0.1,1,7)
    beta_prog = np.linspace(0.1,1,7)
    k_prog = np.arange(2,18,4)
    R0_min = 0.3; R0_max = 5
    
    folder = "WS_Epids"
    
    text = "N %s;\n k_prog %s, len: %s;\np_prog %s, len: %s;\nbeta_prog %s, len: %s;\nmu_prog %s, len: %s;\nR0_min %s, R0_max %s\n---\n" \
          % (N, k_prog, len(k_prog), p_prog, len(p_prog), beta_prog, len(beta_prog), \
          mu_prog, len(mu_prog), R0_min, R0_max)
    print(text)

    save_log_params(folder = folder, text = text)

    total_iterations = 0
    for D, p in product(k_prog, p_prog):
      for beta, mu in product(beta_prog, mu_prog):
        if  R0_min < beta*D/mu < R0_max:
          total_iterations += 1
    
    '''
    if os.path.exists("/home/hal21/MEGAsync/Tour_Physics2.0/Thesis/NetSciThesis/Project/Plots/Test/WS_Epids/WS_Epids_log_saved_nets.txt"):
      with open( "/home/hal21/MEGAsync/Tour_Physics2.0/Thesis/NetSciThesis/Project/Plots/Test/WS_Epids/WS_Epids_log_saved_nets.txt", "r" ) as r:
        nonempty_lines = [line.strip("\n") for line in r if line != "\n" if line[0] == N]
        line_count = len(nonempty_lines)
    else: line_count = 0
    print("New AdjMat", line_count)
    '''
    
    print("Total SIR Iterations:", total_iterations)
    
    done_iterations = 0
    saved_nets = []
    for D, p in product(k_prog, p_prog):
      G = nx.connected_watts_strogatz_graph( n = N, k = D, p = p, seed = 1 ) #k is the number of near linked nodes
      for beta, mu in product(beta_prog, mu_prog): 
        if  R0_min < beta*D/mu < R0_max:   
          done_iterations+=1
          print("Iterations left: %s" % ( total_iterations - done_iterations ) )
          ws_sir(G, saved_nets = saved_nets, folder = folder, pruning = pruning, D = D, p = p, beta = beta, mu = mu, done_iterations = done_iterations)
    
=======
import numpy as np
from itertools import product
from definitions import save_log_params, rhu, plot_save_net, plot_save_sir, config_pois_model, NN_pois_net, pois_pos_degrees

'save scaled version for better visualization'
def scaled_conf_pois(G,D,cut_off=30):    
  scaled_N = int(G.number_of_nodes()/cut_off) # int(D/cut_off)
  return config_pois_model(scaled_N, D)

'''Configurational Model with poissonian degree:
1) Question: 
  1. using this meth, loops and parallel edges are present leading 
  to a adj_matrix not normalized to 1. Since neighbors are involved in sir 
  and contagious is made by (fut = S, curr = S) - nodes, I may leave them, 
  but there're not so in line of "social contacts". 
  So, I prefer to remove them, but I loose a lot of precision if $D / N !<< 1$. 
  <br>Ex., $D = 50 = N, <k> ~ 28$. For $N= 1000 \text{ and } D = 3 \textrm{ or } 8, 
  <k> \textrm{is acceptable.}$
'''
p_max = 0; N = int(1e3)

'progression of net-parameters'
k_prog = np.arange(2,10,2)
p_prog = np.linspace(0,p_max,int(p_max*10)+1)
mu_prog = np.linspace(0.01,1,15)
beta_prog = np.linspace(0.01,1,15)
p_prog = [0]
R0_min = 0.5; R0_max = 16


'try only with p = 0.1'
total_iterations = 0
for D,mu,p,beta in product(k_prog, mu_prog, p_prog, beta_prog):  
  if R0_min < beta*D/mu < R0_max:
    total_iterations+=1
print("Total Iterations:", total_iterations)
done_iterations = 0

saved_nets = []
for D,mu,p,beta in product(k_prog, mu_prog, p_prog, beta_prog):  
  if R0_min < beta*D/mu < R0_max:
    done_iterations+=1
    print("Iterations left: %s" % ( total_iterations - done_iterations ) )

    folder = "Config_Model"
    G = config_pois_model(N, D)
    'plot G, degree distribution and the adiaciency matrix and save them'
    G = NN_pois_net(G, D = D)
    
    folder = "NNR_Conf_Model"
    text = "N %s;\n k_prog %s, len: %s;\np_prog %s, len: %s;\nbeta_prog %s, len: %s;\nmu_prog %s, len: %s;\nR0_min %s, R0_max %s\n---\n" \
            % (N, k_prog, len(k_prog), p_prog, len(p_prog), beta_prog, len(beta_prog), \
            mu_prog, len(mu_prog),  R0_min, R0_max)

    if "N%s_D%s_p%s"% (N,D,rhu(p,3)) not in saved_nets: 
      plot_save_net(G = G, scaled_G = G, folder = folder, D = D, p = p, done_iterations = done_iterations)
      saved_nets.append("N%s_D%s_p%s"% (N,D,rhu(p,3)))
      print(saved_nets)
    plot_save_sir(G, folder = folder, beta = beta, D = D, mu = mu, p = p_max, done_iterations = done_iterations)

<<<<<<<< HEAD:simple_sir_last_ver.py
    save_log_file(folder = folder, text = text)
========
    save_log_params(folder = folder, text = text)
>>>>>>>> 8b3505985b01c5e60db5688db20ca89235ec914f:Conf_Model_SIR.py
>>>>>>> 8b3505985b01c5e60db5688db20ca89235ec914f
