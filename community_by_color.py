#https://stackoverflow.com/questions/65069624/networkx-cluster-nodes-in-a-circular-formation-based-on-node-color

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from definitions import save_log_params, parameters_net_and_sir, caveman_defs, main, rmv_folder
    
    
'start of the main()'
from itertools import product
from definitions import rhu, save_nes

partition_layout, comm_caveman_relink = caveman_defs()
   
N = int(1e2); folder = "Caveman_Model"
rmv_folder(folder, True)

'progression of cliquesnet-parameters'
k_prog, p_prog, beta_prog, mu_prog, R0_min, R0_max, start_inf =  parameters_net_and_sir(folder = folder) 

main(folder = folder, N = N, k_prog = k_prog, p_prog = p_prog, \
  beta_prog = beta_prog, mu_prog = mu_prog, R0_min = R0_min, R0_max = R0_max, start_inf = start_inf)