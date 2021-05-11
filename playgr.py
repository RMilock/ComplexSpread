import numpy as np
from itertools import product
import networkx as nx
import os
from definitions import save_log_params, rhu, plot_save_net, \
    remove_loops_parallel_edges, plot_save_sir, config_pois_model, replace_edges_from, \
    infos_sorted_nodes, my_dir

N = int(20); p_max = 0.1; add_edges_only = True

'progression of net-parameters'
k_prog = np.arange(2,18,2)
p_prog = np.linspace(0,p_max,int(p_max*10)+1)
beta_prog = np.linspace(0.01,1,10)
mu_prog = np.linspace(0.01,1,8)
R0_min = 0; R0_max = 6

folder = f"Overlapping_Rew_Add_{add_edges_only}"
def already_saved_list(folder, adj_or_sir, chr_min, chr_max = None, my_print = True):
    log_upper_path = "".join((my_dir(),folder,"/")) #../Plots/Test/Overlapping.../
    log_path = "".join((log_upper_path, folder, f"_log_saved_{adj_or_sir}.txt"))
    saved_list = []
    if os.path.exists(log_path):
      with open(log_path, "r") as file:
          if chr_max == None: saved_list = [l.rstrip("\n") for l in file]
          else: saved_list = [l.rstrip("\n")[chr_min:] for l in file]
    if my_print: print(f"\nThe already saved {adj_or_sir} are", saved_list)

    return saved_list




'try only with p = 0.1'
total_iterations = 0
for D,mu,p,beta in product(k_prog, mu_prog, p_prog, beta_prog):  
  if R0_min < beta*D/mu < R0_max:
    total_iterations+=1
print("Total Iterations:", total_iterations)
done_iterations = 0

saved_nets = []

for D,mu,p,beta in product(k_prog, mu_prog, p_prog, beta_prog):  
  if R0_min < beta*D/mu < R0_max:

    done_iterations+=1
    print("\nIterations left: %s" % ( total_iterations - done_iterations ) )
    
    'save parameters'
    text = "N %s;\n k_prog %s, len: %s;\np_prog %s, len: %s;\nbeta_prog %s, len: %s;\nmu_prog %s, len: %s;\nR0_min %s, R0_max %s\n---\n" \
            % (N, k_prog, len(k_prog), p_prog, len(p_prog), beta_prog, len(beta_prog), \
            mu_prog, len(mu_prog),  R0_min, R0_max)
    save_log_params(folder = folder, text = text, done_iterations = done_iterations)


    G = nx.watts_strogatz_graph(N, k = D, p = p)
    
    def plot_save_not_ex(G, D, p, adj_or_sir, my_print = True):
      'save net only if does not exist'
      N = G.number_of_nodes()
      if adj_or_sir == "AdjMat": 
        saved_files = already_saved_list(folder, adj_or_sir, 0, my_print= my_print)
        file_name = folder + "_AdjMat_N%s_D%s_p%s.png" % (N,rhu(D,1),rhu(p,3)) 
      if adj_or_sir == "SIR": 
        saved_files = already_saved_list(folder, adj_or_sir, 0, my_print= my_print)
        file_name = folder + "_%s_R0_%s_N%s_D%s_p%s_beta%s_mu%s" % (
          "SIR", '{:.3f}'.format(rhu(beta/mu*D,3)),
          N,D, rhu(p,3), rhu(beta,3), rhu(mu,3) ) + ".png"
      if file_name not in saved_files: 
        print("I'm saving", file_name)
        infos_sorted_nodes(G, num_nodes= True)
        if adj_or_sir == "AdjMat": 
          plot_save_net(G = G, folder = folder, p = p, done_iterations = done_iterations)
        if adj_or_sir == "SIR": 
          plot_save_sir(G, folder = folder, beta = beta, mu = mu, p = p, done_iterations = done_iterations)

    
    'save net only if does not exist'
    saved_nets = already_saved_list(folder, "AdjMat", 0, my_print=True)
    name_net = folder + "_AdjMat_N%s_D%s_p%s.png" % (N,rhu(D,1),rhu(p,3)) 
    if name_net not in saved_nets: 
      print("I'm saving", name_net)
      G = nx.watts_strogatz_graph(N, k = D, p = p)
      infos_sorted_nodes(G, num_nodes= True)
      plot_save_net(G = G, folder = folder, p = p, done_iterations = done_iterations)
    
    'save net only if does not exist'
    saved_sir = already_saved_list(folder, "SIR", 0, my_print = True)
    name_sir = folder + "_%s_R0_%s_N%s_D%s_p%s_beta%s_mu%s" % (
          "SIR", '{:.3f}'.format(rhu(beta/mu*D,3)),
          N,D, rhu(p,3), rhu(beta,3), rhu(mu,3) ) + ".png"
    if name_sir not in saved_sir: 
      print("I'm saving", name_sir)
      G = nx.watts_strogatz_graph(N, k = D, p = p)
      infos_sorted_nodes(G, num_nodes= True)
      plot_save_sir(G, folder = folder, beta = beta, mu = mu, p = p, done_iterations = done_iterations)