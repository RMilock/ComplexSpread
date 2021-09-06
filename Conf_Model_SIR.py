import numpy as np
from itertools import product
import networkx as nx
from definitions import main, save_log_params, save_nes, \
NN_pois_net, parameters_net_and_sir, \
NestedDict, my_dir, jsonKeys2int
import os; import json

'''Configurational Model with poissonian degree:
Look for multi // and loops edges. Are affecting G.neighbors(i)? If so, remove them!
'''
N = int(1e3); folder = "NN_Conf_Model"

k_prog, p_prog, beta_prog, mu_prog, R0_min, R0_max \
  = parameters_net_and_sir(folder)
main(folder = folder, N = N, k_prog = k_prog, p_prog = p_prog, \
  beta_prog = beta_prog, mu_prog = mu_prog, R0_min = R0_min, R0_max = R0_max)