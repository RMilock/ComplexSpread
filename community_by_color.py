#https://stackoverflow.com/questions/65069624/networkx-cluster-nodes-in-a-circular-formation-based-on-node-color

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from definitions import save_log_params, plot_sir, \
check_loops_parallel_edges, remove_loops_parallel_edges, \
caveman_defs
    
'start of the main()'
from itertools import product
from definitions import rhu, plot_save_net, plot_save_sir

partition_layout, comm_caveman_relink = caveman_defs()

p_max = 0.1; N = int(1e3)

'progression of net-parameters'
k_prog = np.arange(2,34,2)
p_prog = np.linspace(0.1,p_max,4)
mu_prog = np.linspace(0.01,1,7)
beta_prog = np.linspace(0.01,1,7)
R0_min = 0.5; R0_max = 10

'try only with p = 0.1'
total_iterations = 0
for D,mu,p,beta in product(k_prog, mu_prog, p_prog, beta_prog):  
  if R0_min < beta*D/mu < R0_max:
    total_iterations+=1
print("Total Iterations:", total_iterations)
done_iterations = 0

saved_nets = []
for D,p,beta,mu in product(k_prog, p_prog, beta_prog, mu_prog):  
  if R0_min < beta*D/mu < R0_max:
    done_iterations+=1
    print("Iterations left: %s" % ( total_iterations - done_iterations ) )
    
    folder = "Caveman_Model"
    clique_size = D; cliques = int(N/D)
    G = comm_caveman_relink(cliques=cliques, clique_size = D, 
                            p = p, relink_rnd = D, numb_rel_inring = 1)
    
    partition = {node : np.int(node/clique_size) for node in range(cliques * clique_size)}
    pos = partition_layout(G, partition, ratio=clique_size/cliques*0.1)

    text = "N %s;\n k_prog %s, len: %s;\np_prog %s, len: %s;\nbeta_prog %s, len: %s;\nmu_prog %s, len: %s;\nR0_min %s, R0_max %s\n---\n" \
            % (N, k_prog, len(k_prog), p_prog, len(p_prog), beta_prog, len(beta_prog), \
            mu_prog, len(mu_prog),  R0_min, R0_max)

    if "N%s_D%s_p%s"% (N,D,rhu(p,3)) not in saved_nets: 
      print("This is p", p)
      plot_save_net(G = G, pos = pos, partition = partition, folder = folder, D = D, p = p, done_iterations = done_iterations)
      saved_nets.append("N%s_D%s_p%s"% (N,D,rhu(p,3)))
      print(saved_nets)
    plot_save_sir(G, folder = folder, beta = beta, D = D, mu = mu, p = p, done_iterations = done_iterations)
    
    save_log_params(folder = folder, text = text)

'''
ax = plt.subplot()
plot_sir(G, ax = ax, D = D, beta = beta, mu = mu) #mf! expected
plt.show()'''