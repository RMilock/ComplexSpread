import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm 
import os #to create a folder


def sir(G, beta = 1e-3, mu = 0.05, k = 10, seed = False, D = None):
    'If D == None, the neighbors are not fixed;' 
    'If D = number, will act as MF with fixed numb of neighbors'

    import random
    #here's the modifications of the "test_ver1"
    'Number of nodes in the graph'
    N = G.number_of_nodes()
    
    'Label the individual wrt to the # of the node'
    node_labels = G.nodes()
    
    'Currently infected individuals and the future infected and recovered' 
    inf_list = [] 
    prevalence = []
    recovered = []

    'Initial Conditions'
    current_state = ['S' for i in node_labels] 
    future_state = ['S' for i in node_labels]
    
    if seed == True: random.seed(0)

    'Selects the seed of the disease'
    seeds = random.sample(range(N), k) 
    for seed in seeds:
      current_state[seed] = 'I'
      future_state[seed] = 'I'
      inf_list.append(seed)


    'initilize prevalence and revocered list'
    prevalence = [len(inf_list)/N]
    recovered = [0]
    cum_positives = [k/N]

    'start and continue whenever there s 1 infected'
    while(len(inf_list)>0):        
        
        daily_new_inf = 0
        'Infection Phase: each infected tries to infect all of the neighbors'
        for i in inf_list:
            'Select the neighbors of the infected node'
            if D == None: tests = G.neighbors(i) #only != wrt to the SIS: contact are taken from G.neighbors            
            if D != None: tests = random.choices(node_labels, k = int(D)); beta = beta*D/N
            for j in tests:
                'If the contact is susceptible and not infected by another node in the future_state, try to infect it'
                if current_state[j] == 'S' and future_state[j] == 'S':
                    if random.random() < beta:
                        future_state[j] = 'I'; daily_new_inf += 1
                    else:
                        future_state[j] = 'S'
        
        cum_positives.append(cum_positives[-1]+daily_new_inf/N)  
        #loop +=1;
        #print("loop:", loop, cum_total_inf, cum_total_inf[-1], daily_new_inf/N)

        'Recovery Phase: each infected in the current state recovers with probability mu'        
        for i in inf_list:
            if random.random() < mu:
                future_state[i] = 'R'
            else:
                future_state[i] = 'I'
        
        'Time update: once infections and recovery ended, we move to the next time-step'
        'The future state becomes the current one'
        current_state = future_state.copy() #w/o .copy() it's a mofiable-"view"
       
        'Updates inf_list with the currently fraction of inf/rec' 
        inf_list = [i for i, x in enumerate(current_state) if x == 'I']
        rec_list = [i for i, x in enumerate(current_state) if x == 'R']

        
        'Saves the fraction of infected and recovered in the current time-step'

        prevalence.append(len(inf_list)/float(N))
        recovered.append(len(rec_list)/float(N))
 
    return prevalence, recovered, cum_positives

def itermean_sir(G, D = None, beta = 1e-3, mu = 0.05, k = 10, numb_iter = 200, numb_classes = 3):
    'def a function that iters numb_iter and make an avg of the trajectories'
    'k are the starting infected'
    import itertools
    import numpy as np
    import matplotlib.pyplot as plt
    trajectories = [[] for _ in range(numb_classes)]
    avg = [[] for _ in range(numb_classes)]
    counts = [[],[],[]]
    for i in range(numb_iter):
        tmp_traj = sir(G, beta = beta, mu = mu, k = k, D = D)
        #if i == 1: tmp_traj = [[2,2,1], [3,3,2], [4,4]]
        for idx in range(numb_classes):#zip([0,1,2],[x,y,z],[p,r,c],[cnt_x, cnt_y, cnt_z]):
            trajectories[idx].append(tmp_traj[idx])
            #print("\nit:", i, "idx ", idx, "and traj_idx", trajectories[idx])
            it_sum = [sum(x) for x in itertools.zip_longest(*trajectories[idx], fillvalue=0)]
            for j in range(len(trajectories[idx][i])):
                #print("traj_i", trajectories[idx][i], "lenai", len(trajectories[idx][i]))
                try: counts[idx][j]+=1
                except: counts[idx].append(1)
            avg[idx] = list(np.divide(it_sum,counts[idx]))
            #print("global sum", it_sum)
            #print("counts", counts[idx])
            #print("avg", avg[idx])

    return trajectories, avg

def plot_sir(G, beta = 1e-3, mu = 0.05, k = 10, numb_classes = 3, numb_iter = 100, D = None):
  'D = numb acts only in mf_avg'
  import itertools
  import matplotlib.pyplot as plt
  # MF_SIR: beta = 1e-3, MF_SIR: mu = 0.05
  N = G.number_of_nodes()
  'plot ratio of daily infected and daily cumulative recovered'
  trajectories, avg = itermean_sir(G, beta, mu, k, numb_classes=numb_classes, numb_iter=numb_iter)
  _, mf_avg = itermean_sir(G, mu, k = k, numb_classes=numb_classes, numb_iter=numb_iter)

  'plotting the many realisations'    
  colors = ["paleturquoise","wheat","lightgreen"]
  for i in range(numb_classes):
    for j in range(numb_iter):
        plt.plot(trajectories[i][j], color = colors[i])
  
  plt.plot(avg[0], label="Infected/N", color = "tab:blue") #prevalence
  plt.plot(mf_avg[1], label="SIR_Recovered/N", color = "tab:orange" ) #recovered
  plt.plot(avg[2], label="CD_Inf /N", color = "tab:green") #cum_positives


  'plot horizontal line to highlight the initial infected'
  plt.axhline(k/N, color = "r", ls="dashed", label = "Starting_Inf /N")
  locs, _ = plt.yticks()
  locs_yticks = np.array([])
  for i in range(len(locs)): 
      if locs[i] <= k/N < locs[i+1]:  
          locs_yticks  = np.concatenate((locs[1:i+1], [k/N], locs[i+1:-1])) #omit the 1st and last for better visualisation
  plt.yticks(locs_yticks, np.round(locs_yticks,3))


  'plot labels'
  plt.xlabel('Time', fontsize = 16)
  plt.ylabel('Indivs/N', fontsize = 16)
  plt.yscale("linear")
  plt.legend(loc="best")

def plot_G_degdist_adjmat_sir(G, p = 0, D = None, figsize = (12,12), beta = 1e-3, mu = 0.05, k = 10, log = False):
  import matplotlib.pyplot as plt
  import networkx as nx
  N = G.number_of_nodes()
  def rhu(n, decimals=0): #round_half_up
    import math
    multiplier = 10 ** decimals
    return math.floor(n*multiplier + 0.5) / multiplier

  #plot figimport networkx as nxures in different windows
  fig, axs = plt.subplots(2,2, figsize = figsize)
  nx.draw_circular(G, ax=axs[0,0], with_labels=True, font_size=12, node_size=5, width=.3)
  
  'set xticks to be centered'
  sorted_degree = np.sort([G.degree(n) for n in G.nodes()])

  'degree distribution + possonian distr'
  from scipy.stats import poisson
  bins = np.arange(sorted_degree[0]-1,sorted_degree[-1]+2)
  mean = rhu( np.sum([j for (i,j) in G.degree() ]) / G.number_of_nodes() )
  print("rounded degrees mean", mean)
  y = poisson.pmf(bins, mean)
  n, hist_bins, _ = axs[0,1].hist(sorted_degree, bins = bins, \
                                        log = log, density=0, color="green", ec="black", lw=1, align="left", label = "degrees distr")
  hist_mean = n[np.where(hist_bins == mean)]; pois_mean = poisson.pmf(mean, mean)
  'useful but deletable print'
  #print( "bins = bins", bins, "\nhist_bins", hist_bins, "\ny", y, "\nn", n, \
  #      "\npois mean", pois_mean , "hist_mean", hist_mean,  \
  #      "\nmean", mean, "\nhist_mean", hist_mean, "  poisson.pmf", poisson.pmf(int(mean),mean), )
  axs[0,1].plot(bins, y * hist_mean / pois_mean, "bo--", lw = 2, label = "poissonian distr")
  axs[0,1].set_xlim(bins[0],bins[-1]) 
  axs[0,1].legend(loc = "best")
  

  'plot adjiacency matrix'
  adj_matrix = nx.adjacency_matrix(G).todense()
  axs[1,0].matshow(adj_matrix, cmap=cm.get_cmap("Greens"))
  print("Adj_matrix is symmetric", np.allclose(adj_matrix, adj_matrix.T))

  'plot sir'
  if D == None: np.sum([j for (i,j) in G.degree() ]) / N
  plot_sir(G, beta, mu, k, D = D)
  fig.suptitle("SIR_N%s_D%s_p%s_beta%s_mu%s_R%s"% (N,rhu(D,3),p, rhu(beta,3), rhu(mu,3), rhu(beta/mu*D,3)))

'Net Infos'
def infos_sorted_nodes(G, num_nodes = False):
    import networkx as nx
    'sort nodes by key = degree. printing order: node, adjacent nodes, degree'
    nodes = G.nodes()
    print("Sum_i k_i: ", np.sum([j for (i,j) in G.degree() ]), \
          " <k>: ", np.sum([j for (i,j) in G.degree() ]) / len(nodes), 
          " and <k>/N ", np.sum([j for (i,j) in G.degree() ]) / len(nodes)**2, end="\n\n" )
    
    'put adj_matrix into dic from better visualisation'
    adj_matrix =  nx.adjacency_matrix(G).todense()
    adj_dict = {i: np.nonzero(row)[1].tolist() for i,row in enumerate(adj_matrix)}

    infos = zip([x for x in nodes], [adj_dict[i] for i in range(len(nodes))], [G.degree(x) for x in nodes])
    inner_sorted_nodes = sorted( infos, key = lambda x: x[2])
    
    if num_nodes == True:  num_nodes = len(nodes)
    if num_nodes == False: num_nodes = 0
    for i in range(num_nodes):
      if i == 0: print("Triplets of (nodes, edges, degree) sorted by degree: \n")
      print( inner_sorted_nodes[i] )

def remove_loops_parallel_edges(G, remove_loops = True):
  import networkx as nx
  full_ls = list((G.edges()))
  lpe = []
  for i in full_ls:
    full_ls.remove(i)
    for j in full_ls:
      if i == j: lpe.append(j) #print("i", i, "index", full_ls.index(i)+1, "j", j)
  if remove_loops == True:  
    for x in list(nx.selfloop_edges(G)): lpe.append(x)
    print("Parallel edges and loops removed!")
  return G.remove_edges_from(lpe)

def check_loops_parallel_edges(G):
  ls = list(G.edges())
  print("parallel edges", [i for i in ls for j in ls[ls.index(i)+1:] if i==j])
  print("loops", [(i,j) for (i,j) in set(G.edges()) if i == j])

'number of iterations, i.e. or max power or fixed by user'
def pow_max(N, num_iter = "all"):
  if num_iter == "all":
    'search the 2**pow_max'
    i = 0
    while(N-2**i>0): i+=1
    return i-1
  return int(num_iter)

def rhu(n, decimals=0): #round_half_up
  import math
  multiplier = 10 ** decimals
  return math.floor(n*multiplier + 0.5) / multiplier

def ws_sir(N, k_ws = None, p = 0.1, infos = True, beta = 0.001, mu = 0.16):  
  'in this def: cut_factor = % of links remaining from the full net'
  'round_half_up k_ws for a better approximation of nx.c_w_s_graph+sir'
  
  import networkx as nx
  if k_ws == None: k_ws = N
  k_ws = int(rhu(k_ws))
  cut_factor = k_ws / N #float
  'With p = 1 and <k>/N ~ 0, degree distr is sim to a Poissonian'
  G = nx.connected_watts_strogatz_graph( n = N, k = k_ws, p = p, seed = 1 ) #k is the number of near linked nodes
  #check_loops_parallel_edges(G)
  #if infos == True: infos_sorted_nodes(G, num_nodes = False)
  
  'set spreading parameters'

  beta_eff = beta/cut_factor; mu_eff = mu 
  #Thurner pmts: beta_eff = 0.1, mu = 0.16; k_ws = 3 vel 8
  #MF def: beta_eff, mu_eff = 0.001/cf, 0.05/cf or 0.16/cf ; cf = 1
  print("beta_eff %s ; mu_eff: %s; beta_1.2: %s" % (beta_eff, mu_eff, beta) )
  
  'plot all -- old version: beta = beta_eff'
  plot_G_degdist_adjmat_sir(G, figsize=(15,15), beta = beta, mu = mu_eff, log = False, D = k_ws, p = p)    
  
  'TO SAVE PLOTS'
  try:
    plt.savefig("/home/hal21/MEGAsync/Thesis/NetSci Thesis/Project/WS_plots/SIR_N%s_k%s_p%s_beta%s_mu%s" % (N,k_ws,p, rhu(beta_eff,3), rhu(mu_eff,3)) + ".png")
  except:
    os.mkdir("/home/hal21/MEGAsync/Thesis/NetSci Thesis/Project/WS_plots")
    plt.savefig("/home/hal21/MEGAsync/Thesis/NetSci Thesis/Project/WS_plots/SIR_N%s_k%s_p%s_beta%s_mu%s" % (N,k_ws,p, rhu(beta_eff,3), rhu(mu_eff,3)) + ".png")



'Draw N degrees from a Poissonian sequence with lambda = D and length L'
def pois_pos_degrees(D, N, L = int(2e3)):
    degs = np.random.poisson(lam = D, size = L)
    #print("len(s) in deg", len([x for x in degs if x == 0])) 
    pos_degrees = np.random.choice([x for x in degs if x != 0], N)
    #print("len(s) in pos_degrees", len([x for x in pos_degrees if x == 0]))
    return pos_degrees

def config_pois_model(N, D, p = 0, seed = 123, visual = True):
    '''create a network with the node degrees drawn from a poissonian with even sum of degrees'''
    np.random.seed(seed)
    degrees = pois_pos_degrees(D,N) #for def of poissonian distr degrees are integers

    print("Degree sum:", np.sum(degrees), "with seed:", seed, "numb of 0-degree nodes:", len([x for x in degrees if x == 0]) )
    while(np.sum(degrees)%2 != 0): #i.e. sum is odd --> change seed
        seed+=1
        np.random.seed(seed)
        degrees = pois_pos_degrees(D,N)
        print("Degree sum:", np.sum(degrees), "with seed:", seed, )

    print("\nNetwork Created but w/o standard neighbors wiring!")
    '''create and remove loops since they apppears as neighbors of a node. Check it via print(list(G.neighbors(18))'''
    G = nx.configuration_model(degrees, seed = seed)

    'If D/N !<< 1, by removing loops and parallel edges, we lost degrees. Ex. with N = 50 = D, <k> = 28 != 49.8'
    check_loops_parallel_edges(G)
    remove_loops_parallel_edges(G)
    #check_loops_parallel_edges(G)

    infos_sorted_nodes(G)

    'plot G, degree distribution and the adiaciency matrix'
    cut_factor = 1
    global beta_eff, mu_eff
    beta_eff = 0.2/cut_factor; mu_eff = 0.16
    #Thurner pmts: beta_eff = 0.1, mu = 0.16; k_ws = 3 vel 8
    #MF def: beta_eff, mu_eff = 0.001/cf, 0.05/cf or 0.16/cf ; cf = 1
    #Config_SIR def: D = 8, beta_eff, mu_eff = 0.1, 0.05
    print("beta_eff %s ; mu_eff: %s" % (beta_eff, mu_eff))
    if visual == True: plot_G_degdist_adjmat_sir(G, figsize=(15,15), beta = beta_eff, mu = mu_eff, log = True) 

    try:
        plt.savefig("Config_plots/Conf_SIR_N%s_D%s_p%s_beta%s_mu%s" % (N,D,p, rhu(beta_eff,3), rhu(mu_eff,3)) + ".pdf")
    except:
        os.mkdir("Config_plots")
        plt.savefig("Config_plots/Config_SIR_N%s_D%s_p%s_beta%s_mu%s" % (N,D,p, rhu(beta_eff,3), rhu(mu_eff,3)) + ".pdf")

    return G

'''def:: "replace" existing edges, since built-in method only adds'''
def replace_edges_from(G,list_edges=[]):
    ebunch = [x for x in G.edges()]
    G.remove_edges_from(ebunch)
    if list_edges!=[]: return G.add_edges_from(list_edges)
    return G