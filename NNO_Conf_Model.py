#import numpy as np
#from itertools import product
from definitions import main, parameters_net_and_sir, rmv_folder
#NNOverl_pois_net, save_log_params, plot_save_nes, \
#  , NestedDict, my_dir, jsonKeys2int
#import os; import json
#import matplotlib.pylab as plt

N = int(1e3)
folder = f"NNO_Conf_Model" #add edges instead of delete&rewiring
rmv_folder(folder, False)

k_prog, p_prog, beta_prog, mu_prog, R0_min, R0_max, start_inf =  parameters_net_and_sir(folder = folder) 

main(folder = folder, N = N, k_prog = k_prog, p_prog = p_prog, \
  beta_prog = beta_prog, mu_prog = mu_prog, R0_min = R0_min, R0_max = R0_max, start_inf = start_inf)

'''
'progression of net-parameters'

p_max = 0.2
k_prog = np.concatenate(([0,1,2],np.arange(3,40,2)))
p_prog = np.linspace(0,p_max,int(p_max*10)+1)
beta_prog = np.linspace(0.01,1,10)
mu_prog = np.linspace(0.01,1,8)
R0_min = 0; R0_max = 30

ordp_pmbD_dic = NestedDict()
ordp_path = "".join( (my_dir(),folder,"/OrdParam/saved_ordp_dict.txt") )
if os.path.exists(ordp_path): 
  with open(ordp_path,"r") as f:
    ordp_pmbD_dic = json.loads(f.read(), object_hook=jsonKeys2int)

k_prog, p_prog, beta_prog, mu_prog, R0_min, R0_max =  \
  parameters_net_and_sir(folder = folder) 

'try only with p = 0.1 -- since NN_Overl_add_edge augment D, we are overestimating tot_number'
total_iterations = 0
for D,p,beta,mu in product(k_prog, p_prog, beta_prog, mu_prog):  
  if R0_min <= beta*D/mu <= R0_max:
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
  if R0_min <= beta*D/mu <= R0_max:
    done_iterations+=1
    print("\nIterations left: %s" % ( total_iterations - done_iterations ) )  
    if os.path.exists(ordp_path): 
      with open(ordp_path,"r") as f:
        ordp_pmbD_dic = json.loads(f.read(), object_hook=jsonKeys2int)


    plot_save_nes(G = NNOverl_pois_net(N, D, p = p, add_edges_only = add_edges_only), 
    p = p, folder = folder, adj_or_sir="AdjMat", done_iterations=done_iterations)
    plot_save_nes(G = NNOverl_pois_net(N, D, p = p, add_edges_only = add_edges_only), 
    p = p, folder = folder, adj_or_sir="SIR", R0_max = R0_max, beta = beta, mu = mu, 
    ordp_pmbD_dic = ordp_pmbD_dic, done_iterations=done_iterations)

'''