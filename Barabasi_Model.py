from definitions import parameters_net_and_sir, main, rmv_folder
from numba import config

N = int(1e3)
#print("Number of cpus used", config.NUMBA_DEFAULT_NUM_THREADS)
  
'progression of net-parameters'
folder = "BA_Model"
k_prog, p_prog, beta_prog, mu_prog, R0_min, R0_max, start_inf \
  = parameters_net_and_sir(folder)

rmv_folder(folder, True)

#beta_prog = [0.015]; mu_prog = [1/14]; p_prog = [0]

main(folder = folder, N = N, k_prog = k_prog, p_prog = p_prog, \
  beta_prog = beta_prog, mu_prog = mu_prog, R0_min = R0_min, R0_max = R0_max, start_inf = start_inf)