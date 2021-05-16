import networkx as nx
import random
import matplotlib.pyplot as plt
from functools import reduce
import numpy as np
from itertools import product
from definitions import save_log_params, plot_save_nes, bam
   
def plot_save_net_sir(G, folder, N, k_prog, p_prog, beta_prog, mu_prog, R0_min, R0_max):

  'try only with p = 0.1'
  total_iterations = 0
  for D,mu,p,beta in product(k_prog, mu_prog, p_prog, beta_prog):  
    if R0_min < beta*D/mu < R0_max:
      total_iterations+=1
  print("Total Iterations:", total_iterations)
  done_iterations = 0

  text = "N %s;\nk_prog %s, len: %s;\np_prog %s, len: %s;\nbeta_prog %s, len: %s;\nmu_prog %s, len: %s;\nR0_min %s, R0_max %s; \nTotal Iterations: %s;\n---\n" \
              % (N, k_prog, len(k_prog), p_prog, len(p_prog), beta_prog, len(beta_prog), \
              mu_prog, len(mu_prog),  R0_min, R0_max, total_iterations)
  save_log_params(folder = folder, text = text)

  saved_nets = []
  for D,mu,p,beta in product(k_prog, mu_prog, p_prog, beta_prog):  
    'since D_real ~ 2*D (D here is fixing only the m and N0), R0_max-folder ~ 2*R0_max'
    if R0_min < beta*D/mu < R0_max: 

      if folder == "B-A_Model": 
        m, N0 = D,D; G = bam(N, m = m, N0 = N0)
      done_iterations+=1
      
      print("\nIterations left: %s" % ( total_iterations - done_iterations ) )
      

      plot_save_nes(G, m = m, N0 = N0,
      p = p, folder = folder, adj_or_sir="AdjMat", done_iterations=done_iterations)
      plot_save_nes(G, m = m, N0 = N0,
      p = p, folder = folder, adj_or_sir="SIR", beta = beta, mu = mu, done_iterations=done_iterations)

add_edges_only = True
folders = ["B-A_Model", f"Overlapping_Rew_Add_{add_edges_only}"]

'progression of net-parameters'
N = int(1e3)
k_prog = np.arange(2,18,2) # these are the fully connected initial cliques
p_max = 0.1
p_prog = np.linspace(0,p_max,int(p_max*10)+1)
beta_prog = np.linspace(0.001,1,15)
mu_prog = np.linspace(0.01,1,13)
R0_min = 0; R0_max = 3    

for folder in folders:
  if folder == "B-A_Model": p_prog = [0]
  plot_save_net_sir( G = None, folder = folder, N = N, k_prog = k_prog, p_prog = p_prog, \
    beta_prog = beta_prog, mu_prog = mu_prog, R0_min = R0_min, R0_max = R0_max )