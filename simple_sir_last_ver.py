# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
# %matplotlib inline
from itertools import product
import os #to create a folder
from definitions import plot_save_nes, pow_max, save_log_params, parameters_net_and_sir

'''
CONNECTED_Watts-Strogatz Net!
This model works only if k = even, i.e. if k=odd then k-1 is choosen. 
So, k-1 neighbors are connected + I've choosen that if D = float 
-> round_half_up(D) = D and let the watts-strogatz model go ahead with the pruning. 
I "round_half_up" since, as for N = 15, k = N/2 = 8, in many cases 
it drives to a nearer pruning.
'''

'rewire all the edges with a probability of p'
N = int(1e3); p_max = 0.2

def even_int(x):
  if int(x) % 2 != 0: return int(x-1)
  return int(x)

for pruning in [True, False]: 
  if pruning == True:
    folder = "WS_Pruned"
    
    _, p_prog, _, mu_prog, R0_min, R0_max =  parameters_net_and_sir(folder = folder, p_max = p_max) 
    #old mu_prog: np.linspace(0.16,1,10)
    #R0_min = 0; R0_max = 4
    #p_prog = np.linspace(0,p_max,int(p_max*10)+1)
    print("---I'm pruning!")
    betas = [1e-3, 2e-3]

    'In WS model, if D = odd, D = D - 1. So, convert it now'
    k_prog = [even_int(N/x) for x in \
              [2**i for i in range(0,pow_max(N, num_iter = "all"))]]*len(betas) #if pow_max +1 --> error of connectivity: D = k_odd - 1
    beta_prog = [beta*N/k for beta in betas for k in k_prog[:len(set(k_prog))]] 

    total_iterations = 0
    for mu, p in product(mu_prog, p_prog):
        for beta, D in zip(beta_prog, k_prog):
          #print("R0: ", beta*D/mu)
          if  R0_min < beta*D/mu < R0_max and beta <= 1:
            total_iterations += 1
    print("Total SIR Pruned Iterations:", total_iterations)

    'save parameters'
    text = "N %s;\nk_prog %s, len: %s;\np_prog %s, len: %s;\nbeta_prog %s, len: %s;\nmu_prog %s, len: %s;\nR0_min %s, R0_max %s; \nTotal Iterations: %s;\n---\n" \
            % (N, k_prog, len(k_prog), p_prog, len(p_prog), beta_prog, len(beta_prog), \
            mu_prog, len(mu_prog),  R0_min, R0_max, total_iterations)
    save_log_params(folder = folder, text = text)    

    done_iterations = 0; saved_nets = []
    for D, beta in zip(k_prog, beta_prog):
      for p, mu in product(p_prog, mu_prog):
        'With p = 1 and <k>/N ~ 0, degree distr is sim to a Poissonian'
        if R0_min < beta*D/mu < R0_max and beta <= 1:
          done_iterations+=1
          print("Iterations left: %s" % ( total_iterations - done_iterations ) )


          plot_save_nes(G = nx.connected_watts_strogatz_graph( n = N, k = D, p = p, seed = 1 ), 
          p = p, folder = folder, adj_or_sir="AdjMat", done_iterations=done_iterations)
          plot_save_nes(G = nx.connected_watts_strogatz_graph( n = N, k = D, p = p, seed = 1 ),
          p = p, folder = folder, adj_or_sir="SIR", R0_max = R0_max, beta = beta, mu = mu, done_iterations=done_iterations)
          print("---")
    
    save_log_params(folder = folder, text = text)

  if pruning == False:
    'test != kind of '
    print("---I'm NOT pruning!")
    'if p_prog has sequence save_it like ./R0_0-1/R0_0.087'
    '''
    p_prog = np.concatenate((np.array([0.001]), np.linspace(0.012,0.1,10)))
    p_prog = [0,0.1]
    mu_prog = np.linspace(0.99,1,6)
    beta_prog = np.linspace(0.99,1,6)
    k_prog = np.arange(2,18,2)
    R0_min = 0.3; R0_max = 5
    '''
    folder = "WS_Epids"; p_max = 0.2
    k_prog, p_prog, beta_prog, mu_prog, R0_min, R0_max =  parameters_net_and_sir(folder = folder, p_max = p_max) 
    #p_prog = np.concatenate((np.array([0.001]), np.linspace(0.01,0.1,5)))

    total_iterations = 0
    for D, p, beta, mu in product(k_prog, p_prog, beta_prog, mu_prog):
        if  R0_min < beta*D/mu < R0_max:
          total_iterations += 1
    print("Total SIR Epids Iterations:", total_iterations)

    'save parameters'
    text = "N %s;\nk_prog %s, len: %s;\np_prog %s, len: %s;\nbeta_prog %s, len: %s;\nmu_prog %s, len: %s;\nR0_min %s, R0_max %s; \nTotal Iterations: %s;\n---\n" \
            % (N, k_prog, len(k_prog), p_prog, len(p_prog), beta_prog, len(beta_prog), \
            mu_prog, len(mu_prog),  R0_min, R0_max, total_iterations)
    save_log_params(folder = folder, text = text)    
    
    '''
    if os.path.exists("/home/hal21/MEGAsync/Tour_Physics2.0/Thesis/NetSciThesis/Project/Plots/Test/WS_Epids/WS_Epids_log_saved_nets.txt"):
      with open( "/home/hal21/MEGAsync/Tour_Physics2.0/Thesis/NetSciThesis/Project/Plots/Test/WS_Epids/WS_Epids_log_saved_nets.txt", "r" ) as r:
        nonempty_lines = [line.strip("\n") for line in r if line != "\n" if line[0] == N]
        line_count = len(nonempty_lines)
    else: line_count = 0
    print("New AdjMat", line_count)
    '''
    done_iterations = 0; saved_nets = []
    for D, p, beta, mu in product(k_prog, p_prog, beta_prog, mu_prog): 
        'With p = 1 and <k>/N ~ 0, degree distr is sim to a Poissonian'
        if R0_min < beta*D/mu < R0_max and beta <= 1:
          done_iterations+=1
          print("Iterations left: %s" % ( total_iterations - done_iterations ) )

          plot_save_nes(G = nx.connected_watts_strogatz_graph( n = N, k = D, p = p, seed = 1 ), 
          p = p, folder = folder, adj_or_sir="AdjMat", done_iterations=done_iterations)
          plot_save_nes(G = nx.connected_watts_strogatz_graph( n = N, k = D, p = p, seed = 1 ),
          p = p, folder = folder, adj_or_sir="SIR", beta = beta, mu = mu, done_iterations=done_iterations)
          print("---")