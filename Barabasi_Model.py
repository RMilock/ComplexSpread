from definitions import parameters_net_and_sir, main
from numba import config

N = int(303); p_max = 0
#print("Number of cpus used", config.NUMBA_DEFAULT_NUM_THREADS)

'progression of net-parameters'
folder = "BA_Model"
k_prog, p_prog, beta_prog, mu_prog, R0_min, R0_max \
  = parameters_net_and_sir(folder, p_max = p_max)

main(folder = folder, N = N, k_prog = k_prog, p_prog = p_prog, \
  beta_prog = beta_prog, mu_prog = mu_prog, R0_min = R0_min, R0_max = R0_max)