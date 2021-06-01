import numpy as np
from itertools import product
from definitions import NN_Overl_pois_net, save_log_params, plot_save_nes , parameters_net_and_sir

N = int(1e3); p_max = 0.1; add_edges_only = True

'progression of net-parameters'
'''
k_prog = np.arange(2,18,2)
p_prog = np.linspace(0,p_max,int(p_max*10)+1)
beta_prog = np.linspace(0.01,1,10)
mu_prog = np.linspace(0.01,1,8)
R0_min = 0; R0_max = 6'''

folder = f"Overlapping_Rew_Add_{add_edges_only}"
k_prog, p_prog, beta_prog, mu_prog, R0_min, R0_max =  parameters_net_and_sir(folder = folder, p_max = p_max) 
'try only with p = 0.1 -- since NN_Overl_add_edge augment D, we are overestimating tot_number'
total_iterations = 0
for D,p,beta,mu in product(k_prog, p_prog, beta_prog, mu_prog):  
  if R0_min < beta*D/mu < R0_max:
    total_iterations+=1
print("Total Iterations:", total_iterations)
done_iterations = 0

'save parameters'
text = "N %s;\nk_prog %s, len: %s;\np_prog %s, len: %s;\nbeta_prog %s, len: %s;\nmu_prog %s, len: %s;\nR0_min %s, R0_max %s; \nTotal Iterations: %s;\n---\n" \
        % (N, k_prog, len(k_prog), p_prog, len(p_prog), beta_prog, len(beta_prog), \
        mu_prog, len(mu_prog),  R0_min, R0_max, total_iterations)
save_log_params(folder = folder, text = text)

saved_nets = []
for D,p,beta,mu in product(k_prog, p_prog, beta_prog, mu_prog,):  
  if R0_min < beta*D/mu < R0_max:
    done_iterations+=1
    print("\nIterations left: %s" % ( total_iterations - done_iterations ) )  
    plot_save_nes(G = NN_Overl_pois_net(N, D, p = p, add_edges_only = add_edges_only), 
    p = p, folder = folder, adj_or_sir="AdjMat", done_iterations=done_iterations)
    plot_save_nes(G = NN_Overl_pois_net(N, D, p = p, add_edges_only = add_edges_only), 
    p = p, folder = folder, adj_or_sir="SIR", R0_max = R0_max, beta = beta, mu = mu, done_iterations=done_iterations)