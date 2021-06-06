import numpy as np
from itertools import product
from definitions import NN_Overl_pois_net, save_log_params, plot_save_nes, \
  parameters_net_and_sir, NestedDict, my_dir
import matplotlib.pylab as plt

N = int(1e3); add_edges_only = True

'progression of net-parameters'
'''
p_max = 0.2
k_prog = np.concatenate(([0,1,2],np.arange(3,40,2)))
p_prog = np.linspace(0,p_max,int(p_max*10)+1)
beta_prog = np.linspace(0.01,1,10)
mu_prog = np.linspace(0.01,1,8)
R0_min = 0; R0_max = 30'''


folder = f"Overlapping_Rew_Add_{add_edges_only}"
std_pmbD_dic = NestedDict()
my_dir = my_dir()
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
    plot_save_nes(G = NN_Overl_pois_net(N, D, p = p, add_edges_only = add_edges_only), 
    p = p, folder = folder, adj_or_sir="AdjMat", done_iterations=done_iterations)
    plot_save_nes(G = NN_Overl_pois_net(N, D, p = p, add_edges_only = add_edges_only), 
    p = p, folder = folder, adj_or_sir="SIR", R0_max = R0_max, beta = beta, mu = mu, 
    std_pmbD_dic = std_pmbD_dic, done_iterations=done_iterations)

'''
import json as js
print(js.dumps(std_pmbD_dic, sort_keys=False, indent=4))

for D,mu,p,beta in product(k_prog, mu_prog, p_prog, beta_prog):  
  'since D_real ~ 2*D (D here is fixing only the m and N0), R0_max-folder ~ 2*R0_max'
  if R0_min <= beta*D/mu <= R0_max: 
    fixed_std = std_pmbD_dic[p][mu][beta]
    print("std_pmbD_dic, p0, mu0, beta0, fixed_std", \
      std_pmbD_dic, p, mu, beta, fixed_std)
    x = sorted(fixed_std.keys())
    print("x", x)
    y = [fixed_std[i] for i in x]
    print("y", y)
    plt.plot(x,y,'-*')
    std_path = my_dir + folder + "/Std/"
    if not os.path.exists(std_path): os.makedirs(std_path)
    plt.savefig("".join((std_path,"std_p%s_mu%s_beta%s.png" % (p,mu,beta))))
    plt.close()
'''