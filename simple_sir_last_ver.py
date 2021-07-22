# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
# %matplotlib inline
from itertools import product
import os #to create a folder
from definitions import plot_save_nes, pow_max, save_log_params, parameters_net_and_sir
from definitions import save_log_params, plot_save_nes, \
    NestedDict, jsonKeys2int, my_dir, rhu
from itertools import product
import networkx as nx
import json

'''
CONNECTED_Watts-Strogatz Net!
This model works only if k = even, i.e. if k=odd then k-1 is choosen. 
So, k-1 neighbors are connected + I've choosen that if D = float 
-> round_half_up(D) = D and let the watts-strogatz model go ahead with the pruning. 
I "round_half_up" since, as for N = 15, k = N/2 = 8, in many cases 
it drives to a nearer pruning.
'''

'rewire all the edges with a probability of p'
N = int(1e3); p_max = 0.3

def even_int(x):
  if int(x) % 2 != 0: return int(x-1)
  return int(x)

for pruning in [False]: 
  if pruning == True:
    folder = "WS_Pruned"

    'load a dic to save D-order parameter'
    ordp_pmbD_dic = NestedDict()  
    _, p_prog, _, mu_prog, R0_min, R0_max =  parameters_net_and_sir(folder = folder, p_max = p_max) 
    #old mu_prog: np.linspace(0.16,1,10)
    #R0_min = 0; R0_max = 4
    #p_prog = np.linspace(0,p_max,int(p_max*10)+1)
    print("---I'm pruning!")
    betas = [1e-3]

    'In WS model, if D = odd, D = D - 1. So, convert it now'
    k_prog = [even_int(N/x) for x in \
              [2**i for i in range(0,pow_max(N, num_iter = "all"))]]*len(betas) #if pow_max +1 --> error of connectivity: D = k_odd - 1
    beta_prog = [beta*N/k for beta in betas for k in k_prog[:len(set(k_prog))] if beta*N/k <= 1] 

    print("kprog, betaprog, zip",  k_prog, "\n", beta_prog, "\n", list(zip(k_prog,beta_prog))[3:])

    zipped = list(zip(k_prog,beta_prog))[3:] #old: zip(k_prog, beta_prog)

    total_iterations = 0
    for mu, p in product(mu_prog, p_prog):
        for D, beta in zipped:
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
    for D, beta in zipped:
      for mu, p in product(mu_prog, p_prog):
        'With p = 1 and <k>/N ~ 0, degree distr is sim to a Poissonian'
        if R0_min < beta*D/mu < R0_max and beta <= 1:
          done_iterations+=1
          print("Iterations left: %s" % ( total_iterations - done_iterations ) )

          if folder == "WS_Pruned":
            ordp_path = f"{my_dir()}{folder}/OrdParam/p{rhu(p,3)}/mu{rhu(mu,3)}/"
          #print(ordp_path)
          #ordp_path = "".join((my_dir(), folder, "/OrdParam/p%s/mu%s/" % (rhu(p,3),rhu(mu,3)) ))
          #print(ordp_path)
          #ordp_path = "".join((my_dir(), folder, "/OrdParam/"))#p%s/beta_%s/" % (rhu(p,3),rhu(beta,3)) ))
          ordp_path = "".join( (ordp_path, "saved_ordp_dict.txt"))
          if os.path.exists(ordp_path): 
            with open(ordp_path,"r") as f:
              ordp_pmbD_dic = json.loads(f.read(), object_hook=jsonKeys2int)
  
          pp_ordp_pmbD_dic = json.dumps(ordp_pmbD_dic, sort_keys=False, indent=4)
          if done_iterations == 1: print(pp_ordp_pmbD_dic)
          
          m, N0 = 0,0
          pos, partition = None, None

          G = nx.connected_watts_strogatz_graph( n = N, k = D, p = p, seed = 1 )

          import datetime as dt
          start_time = dt.datetime.now()       
          plot_save_nes(G, m = m, N0 = N0, pos = pos, partition = partition,
          p = p, folder = folder, adj_or_sir="AdjMat", done_iterations=done_iterations)
          print("\nThe total-time of one main-loop of one AdjMat is", dt.datetime.now()-start_time)
  
          start_time_total = dt.datetime.now()       
          plot_save_nes(G, m = m, N0 = N0,
          p = p, folder = folder, adj_or_sir="SIR", R0_max = R0_max, beta = beta, mu = mu, 
          ordp_pmbD_dic = ordp_pmbD_dic, done_iterations=done_iterations)
          print("\nThe total-time of one main-loop of one SIR is", dt.datetime.now()-start_time)
    

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
    folder = "WS_Epids"
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