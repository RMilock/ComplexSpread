import networkx as nx
from definitions import replace_lredges, caveman_defs, NNOverl_pois_net, NN_pois_net, save_net
import matplotlib.pylab as plt
from definitions import NestedDict

'''def plot_ordp_endsir_DR0delta(ordp_pmbD_dic, p, mu, beta, folder, toplot):
  from definitions import my_dir
  _, ax = plt.subplots(figsize = (24,14))

  'WARNING: here suptitle has beta // days but the dict is ordp[p][mu][beta][D] = [std_D, ordp, std_ordp]'
  'since in the article p and days are fixed!'
  
  if folder == "WS_Pruned":
    fix_pmb = ordp_pmbD_dic[p][mu]
  else: fix_pmb = ordp_pmbD_dic[p][mu][beta]
  #print("ordp_pmbD_dic, p0, mu0, beta0, fix_pmb", \
  #  ordp_pmbD_dic, p, mu, beta, fix_pmb)
  x = sorted(fix_pmb.keys())

  if toplot == "OrdPar"
    plt.suptitle("Average SD (Daily New Cases) : "+r"$p:%s,\beta:%s,d:%s$"%(rhu(p,3),rhu(beta,3),rhu(mu**(-1))))
    ax.set_xlabel("Avg Degree D [Indivs]")
    ax.set_ylabel("Avg SD (Cases)")
    xerr = [fix_pmb[i][0] for i in x]
    y = [fix_pmb[i][1] for i in x]
    yerr = [fix_pmb[i][2] for i in x]
    label = "Avg SD(Cases)"

  #print("y", y)
  ax.grid(color='grey', linestyle='--', linewidth = 1)
  ax.errorbar(x,y, xerr = xerr, yerr = yerr, color = "tab:blue", marker = "*", linestyle = "-",
              markersize = 30, mfc = "tab:red", mec = "black", linewidth = 3, label = label)

  if toplot in ["OrdPar","EndSirD"]
    lmbd = beta/mu
    D_cfuse = 1 + 2*mu/(beta*(1+p)) #Rc_net = 1 assumption
    D_cer = 1 + mu/beta
    D_chomo = mu/beta
    ax.axvline(x = D_cfuse, color = "maroon", lw = 4, ls = "--", 
                label = "".join((r"$D_{c-fuse \, model}$",f": {rhu(D_cfuse,3)}")) )
    ax.axvline(x = D_cer, color = "darkblue", lw = 4, ls = "--", 
                label = "".join((r"$D_{c-ER \, model}$",f": {rhu(D_cer,3)}")) )
    ax.axvline(x = D_chomo, color = "darkslategrey", lw = 4, ls = "--", 
                label = "".join((r"$D_{c-homog \, model}: \mu / \beta$",f": {rhu(D_chomo,3)}")) )

  leg = ax.legend(fontsize = 35, loc = "best")
  leg.get_frame().set_linewidth(2.5)

  plt.subplots_adjust(
  top=0.91,
  bottom=0.122,
  left=0.080,
  right=0.99)
  
  if folder == "WS_Pruned":
    ordp_path = f"{my_dir()}{folder}/{toplot}/p{rhu(p,3)}/d{rhu(mu**(-1))}/" 
    if not os.path.exists(ordp_path): os.makedirs(ordp_path)
    plt.savefig("".join((ordp_path,"%s_ordp_p%s_d%s.png" % (folder, rhu(p,3),rhu(mu**(-1))))))
  else: 
    ordp_path = my_dir() + folder + f"/{toplot}/p%s/beta%s/" % (rhu(p,3),rhu(beta,3))
    if not os.path.exists(ordp_path): os.makedirs(ordp_path)
    plt.savefig("".join((ordp_path,"%s_ordp_p%s_beta%s_d%s.png" % (folder, rhu(p,3),rhu(beta,3),rhu(mu**(-1))))))
  plt.close()'''

def save_only_net(G, folder, p = 0, m = 0, N0 = 0, done_iterations = 1, log_dd = False, partition = None, pos = None, numb_onring_links = 0, avg_pl = -1, std_avg_pl = -1, clique_size = "", node_color = "red", width = 1.5, node_size = 50, suptitle = ""):
  import os.path
  from definitions import my_dir, func_file_name, N_D_std_D, rhu, plot_params
  from functools import reduce
  import networkx as nx
  from scipy.stats import poisson
  import matplotlib.pylab as plt
  import numpy as np
  from matplotlib import cm

  'plot G, adj_mat, degree distribution'
  _, ax = plt.subplots(figsize = (22,22), ) #20,20

  #ax = plt.subplot(221)  
  'start with degree distribution'
  'set edges width according to how many "long_range_edges'
  if folder == "WS_Pruned": width = 0.2
  long_range_edges = list(filter( lambda x: x > 30, [np.min((np.abs(i-j),np.abs(N-i+j))) for i,j in G.edges()] )) #list( filter(lambda x: x > 0, )
  length_long_range = len(long_range_edges)
  if length_long_range < 10: print("\nLong_range_edges", long_range_edges, length_long_range)
  else: print("len(long_range_edges)", length_long_range)
  #folders = ["WS_Pruned","BA_Model"]
  #if folder in folders: 
  #width = min(1.5,.5*width*N/max(1,len(long_range_edges)))
  width = width
  if folder== "Caveman_Model":
    nx.draw(G, pos, node_color=list(partition.values()), node_size = 300, width = width, with_labels = False)
  else: nx.draw_circular(G, ax=ax, with_labels=False, font_size=20, node_size=node_size, width = width, node_color = node_color)
  ax.set_title(suptitle, fontsize = 70)
  my_dir = "/home/hal21/MEGAsync/Tour_Physics2.0/Thesis/NetSciThesis/Project/ComplexSpread/LateX/images/Networks/my_nets/"
  my_dir = "/home/hal21/MEGAsync/Tour_Physics2.0/Thesis/NetSciThesis/Project/ComplexSpread/LateX/presentation/Pres_7.9/Plots/NewPlots/Miscellanea/"
  plt.savefig(my_dir+folder, transparent = True)
  plt.close()


partition_layout, comm_caveman_relink = caveman_defs()
N = 100; p = 0.01; numb_onring_links = 1
D = 22
clique_size = D
cliques = int(N/clique_size)
clique_size = int(clique_size) #clique_size is a np.float64!
G = comm_caveman_relink(cliques=cliques, clique_size = clique_size, 
                        p = 0, relink_rnd = clique_size, numb_onring_links = numb_onring_links)
          
for node in range(clique_size*cliques):
  if node == 0: print("node", node, type(node), 
            "int(n/cl_s)", int(cliques/clique_size), 
            "cliques", cliques, type(cliques), "clique_size", clique_size, type(clique_size), 
        )

partition = {node : int(node/clique_size) for node in range(cliques * clique_size)}
pos = partition_layout(G, partition, ratio=clique_size/cliques*0.1)


m, N0 = 0,0
#save_only_net(G, folder = "Caveman_Model", m = m, N0 = N0, pos = pos, partition = partition,
#        p = 0, numb_onring_links = numb_onring_links, clique_size = clique_size, width = 1)

#G = nx.complete_graph(D)
#save_only_net(G, folder = "Complete_Graph", p = 0)

#N, D, p = 22, 6, 0
#G = nx.connected_watts_strogatz_graph(N,D,p)
#save_only_net(G, folder = f"Regular Lattice", node_color = "darkorange", width = 2, node_size = 500)

#G = nx.erdos_renyi_graph(22, 0.1, seed=None,  directed=False)
#save_only_net(G, folder = "Erdoes_Renyi_Graph", node_color = "darkblue", width = 3, node_size = 300)

#G = NNOverl_pois_net(N, D, p = 0)
#save_only_net(G, folder = "Overlapping_PSW", node_color = "maroon", width = 1, suptitle = "Overlapping PSWN", node_size = 300)

#G = NN_pois_net(N, ext_D = D, p = 0, folder = "Sparse_PSW")
#save_net(G, folder = "NN_Conf_Model")

#G = nx.barabasi_albert_graph(1000, 2)
#save_only_net(G, folder = "Barabasi_Albert_Model", node_color = "darkblue", width = 0.3, suptitle = "Barabási-Albert Model")

G = nx.watts_strogatz_graph(n = 10, k = 4, p = 0)
folder = "Fuse_Model1"
G.remove_nodes_from((9,8))
nx.draw_circular(G, with_labels = False, node_color = "red", node_size = 200, width = 2)
my_dir = "/home/hal21/MEGAsync/Tour_Physics2.0/Thesis/NetSciThesis/Project/ComplexSpread/LateX/presentation/Pres_7.9/Plots/NewPlots/Miscellanea/"
plt.savefig(my_dir+folder, transparent = True)
plt.close()
#save_only_net(G, folder = "Fuse_Model", node_color = "darkblue", width = 0.3, suptitle = "Fuse Model")


