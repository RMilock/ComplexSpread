import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm 
import os #to create a folder

#Thurner pmts: beta = 0.1, mu = 0.16; D = 3 vel 8
#MF def: beta, mu = 0.001/cf, 0.05/cf or 0.16/cf ; cf = 1

'plot and save sir'
def sir(G, beta = 1e-3, mu = 0.05, start_inf = 10, seed = False, D = None):
    'If D == None, the neighbors are not fixed;' 
    'If D == number, MF_sir with fixed numb of neighbors'

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
    seeds = random.sample(range(N), start_inf)  #without replacement, i.e. not duplicates
    for seed in seeds:
      current_state[seed] = 'I'
      future_state[seed] = 'I'
      inf_list.append(seed)


    'initilize prevalence and revocered list'
    prevalence = [len(inf_list)/N]
    recovered = [0]
    cum_positives = [start_inf/N]

    'start and continue whenever there s 1 infected'
    while(len(inf_list)>0):        
        
        daily_new_inf = 0
        'Infection Phase: each infected tries to infect all of the neighbors'
        for i in inf_list:
            'Select the neighbors of the infected node'
            if D == None: tests = G.neighbors(i) #only != wrt to the SIS: contact are taken from G.neighbors            
            if D != None: tests = random.choices(range(N), k = int(D)) #spread very fast since multiple infected center
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

def itermean_sir(G,  numb_iter, D = None, beta = 1e-3, mu = 0.05, start_inf = 10,):
    'def a function that iters numb_iter and make an avg of the trajectories'
    'k are the starting infected'
    import itertools
    import numpy as np
    import matplotlib.pyplot as plt
    numb_classes = 3
    trajectories = [[] for _ in range(numb_classes)]
    avg = [[] for _ in range(numb_classes)]
    counts = [[],[],[]]
    for i in range(numb_iter):
        tmp_traj = sir(G, beta = beta, mu = mu, start_inf = start_inf, D = D)
        for idx in range(numb_classes):
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

def plot_sir(G, ax, numb_iter, D = None, beta = 1e-3, mu = 0.05, start_inf = 10):

  'D = numb acts only in mf_avg'
  import itertools
  # MF_SIR: beta = 1e-3, MF_SIR: mu = 0.05
  N = G.number_of_nodes()

  'plot ratio of daily infected and daily cumulative recovered'
  'Inf and Cum_Infected from Net_Sir; Recovered from MF_SIR'
  trajectories, avg = itermean_sir(G, D = None, beta = beta, mu = mu, start_inf  = start_inf, numb_iter=numb_iter)
  mf_trajectories, mf_avg = itermean_sir(G, mu = mu, beta = beta, D = D, start_inf = start_inf, numb_iter = numb_iter)

  'plotting the many realisations'    
  colors = ["paleturquoise","wheat","lightgreen"]
  for j in range(numb_iter):
    y_max = np.max(mf_trajectories[2][j])
    ax.plot(mf_trajectories[2][j], color = colors[1])
    ax.plot(trajectories[2][j], color = colors[2])
    y_max = np.max((y_max, np.max(trajectories[2][j])) )
    ax.plot(trajectories[0][j], color = colors[0])
    y_max = np.max((y_max, np.max(trajectories[0][j])) )

  ax.plot(avg[0], label="Net::Infected/N ", \
    color = "tab:blue") #prevalence
  ax.plot(mf_avg[2], label="MF::CD_Inf/N (%s%%)"% np.round(mf_avg[2][-1]*100,1), \
    color = "tab:orange" ) #recovered
  ax.plot(avg[2], label="Net::CD_Inf /N (%s%%)" % np.round(avg[2][-1]*100,1), \
    color = "tab:green") #cum_positives

  'plot horizontal line to highlight the initial infected'
  ax.axhline(start_inf/N, color = "r", ls="dashed", \
    label = "Start_Inf/N (%s%%) "% np.round(start_inf/N*100,1))

  locs, _ = plt.yticks()
  ax.set_yticks(locs[1:-1])
  ax.set_yticklabels(np.round(locs[1:-1],2))

  '''
  textstr = '\n'.join((
    "Start_Inf/N (%s%%)" % np.round(start_inf/N*100,1),
    "MF::Rec/N (%s%%)"% np.round(mf_avg[1][-1]*100,1),
    "CD_Inf/N (%s%%) "% np.round(avg[2][-1]*100,1)))
  
  
  props = dict(boxstyle='round', facecolor='white', alpha=0.5)

  # place a text box in upper left in axes coords
  ax.text(1.02, .8, textstr, transform=ax.transAxes,
          verticalalignment='top', bbox=props) #fontsize = 10
  '''


  'plot labels'
  ax.set_xlabel('Time[1day]', labelpad = 20 )
  ax.set_ylabel('Indivs/N', labelpad = 20 ) #fontsize = 16
  #ax.set_yscale("linear")

  '''set legend upper right above all'
  ymin, ymax = ax.get_ylim()
  ax.set_ylim(ymin, 1.3*ymax)
  ax.legend(loc="upper right")
  '''
  ax.legend(loc="best")
  #ax.legend(bbox_to_anchor=(0.9, 1), edgecolor="dimgrey", loc='lower right') #add: leg = 

def plot_G_degdist_adjmat_sir(G, numb_iter, p = 0, D = None, beta = 1e-3, mu = 0.05, \
  start_inf = 10, log_dd = False, plot_all = True):
  import matplotlib.pyplot as plt
  import networkx as nx

  'set fontsize for a better visualisation'
  SMALL_SIZE = 30
  MEDIUM_SIZE = 40
  BIGGER_SIZE = 30

  plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
  plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
  plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
  plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the xtick labels
  plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the ytick labels
  plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
  plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

  N = G.number_of_nodes()
  def rhu(n, decimals=0): #round_half_up
    import math
    multiplier = 10 ** decimals
    return math.floor(n*multiplier + 0.5) / multiplier


  if plot_all == True:

    #plot figures in different windows
    #_, axs = plt.subplots(1,3, figsize=(20,20), )
    plt.figure(figsize = (20,20))

    ax = plt.subplot(221)
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin,xmax)
    nx.draw_circular(G, ax=ax, with_labels=False, font_size=12, node_size=10, width=.6)
    

    'set xticks of the degree distribution to be centered'
    sorted_degree = np.sort([G.degree(n) for n in G.nodes()])

    'degree distribution + possonian distr'
    from scipy.stats import poisson
    bins = np.arange(sorted_degree[0]-1,sorted_degree[-1]+2)
    mean = rhu( np.sum([j for (i,j) in G.degree() ]) / G.number_of_nodes() )
    print("rounded degrees mean", mean)
    y = poisson.pmf(bins, mean)
    axs = plt.subplot(212)
    n, hist_bins, _ = axs.hist(sorted_degree, bins = bins, \
                                          log = log_dd, density=0, color="green", ec="black", lw=1, align="left", label = "degrees distr")
    hist_mean = n[np.where(hist_bins == mean)]; pois_mean = poisson.pmf(mean, mean)
    'useful but deletable print'
    axs.plot(bins, y * hist_mean / pois_mean, "bo--", lw = 2, label = "poissonian distr")
    axs.set_xlabel('Degree', )
    axs.set_ylabel('Counts', )
    axs.set_xlim(bins[0],bins[-1]) 
    axs.legend(loc = "best")
      

    'plot adjiacency matrix'
    axs = plt.subplot(222)
    adj_matrix = nx.adjacency_matrix(G).todense()
    axs.matshow(adj_matrix, cmap=cm.get_cmap("Greens"))
    #print("Adj_matrix is symmetric", np.allclose(adj_matrix, adj_matrix.T))
    plt.subplots_adjust(top=0.898,
    bottom=0.088,
    left=0.08,
    right=0.963,
    hspace=0.067,
    wspace=0.164)
    #ax = axs[1,1]

  if plot_all == False: 
    _, ax = plt.subplots()
    'set fontsize for a better visualisation'
    SMALL_SIZE = 30
    MEDIUM_SIZE = 40
    BIGGER_SIZE = 30

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the xtick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the ytick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    plt.rcParams["figure.figsize"] = (20,12)

    'plot always sir'
    if D == None: D = np.sum([j for (i,j) in G.degree()]) / N
    print("The model has N: %s, D: %s, beta: %s, mu: %s" % (N,D,beta,mu))
    plot_sir(G, ax=ax, beta = beta, mu = mu, start_inf = start_inf, D = D, numb_iter = numb_iter)
    plt.subplots_adjust(
    top=0.920,
    bottom=0.151,
    left=0.086,
    right=0.992,
    hspace=0.2,
    wspace=0.2)

  plt.suptitle("SIR_R0_%s_N%s_D%s_p%s_beta%s_mu%s"% (rhu(beta/mu*D,3),N,D,rhu(p,3), rhu(beta,3), rhu(mu,3), ))
  
def plot_save_sir(G, folder, beta, D, mu, p, R0_max = 12,  start_inf = 10, numb_iter = 200, plot_all = True, infos = False):
  intervals = [x for x in np.arange(R0_max)]
  N = G.number_of_nodes()
  R0 = beta * D / mu
  print("R0", R0)    
  for i in range(len(intervals)-1):
    if intervals[i] <= R0 < intervals[i+1]:
      'plot all -- old version: beta = beta'
      plot_G_degdist_adjmat_sir(G, numb_iter = numb_iter, D = D, beta = beta, mu = mu, log_dd = False, p = p, plot_all=plot_all, start_inf = start_inf)    
      #plt.show()
      'TO SAVE PLOTS'
      my_dir = "/home/hal21/MEGAsync/Thesis/NetSciThesis/Project/Trial_Plots/"
      my_dir+=folder+"/"
      r0_folder = "R0_%s-%s/" % (intervals[i], intervals[i+1])
      try:
        os.makedirs(my_dir +  r0_folder)
        plt.savefig(my_dir + r0_folder + "SIR_" + folder + "_R0_%s_N%s_D%s_p%s_beta%s_mu%s"% ( rhu(beta/mu*D,3),N,D,rhu(p,3), rhu(beta,3), rhu(mu,3) ) + ".png")
      except:
        plt.savefig(my_dir + r0_folder + "SIR_" + folder + "_R0_%s_N%s_D%s_p%s_beta%s_mu%s"% ( rhu(beta/mu*D,3),N,D,rhu(p,3), rhu(beta,3), rhu(mu,3) ) + ".png")
  plt.close()

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
      if i == 0: print("Triplets of (nodes, linked_node(s), degree) sorted by degree:")
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
  print("parallel edges", set([i for i in ls for j in ls[ls.index(i)+1:] if i==j]))
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




def ws_sir(G, folder, pruning = False, D = None, p = 0.1, infos = False, beta = 0.001, mu = 0.16, plot_all = False, start_inf = 10):    
  'in this def: cut_factor = % of links remaining from the full net'
  'round_half_up D for a better approximation of nx.c_w_s_graph+sir'
  import networkx as nx
  N = G.number_of_nodes()
  if pruning == True:
    if D == None: D = N
    D = int(rhu(D))
    cut_factor = D / N #float
    'set spreading parameters'
    cut_factor = 1
    beta = beta/cut_factor; mu = mu
  
  if infos == True: check_loops_parallel_edges(G); infos_sorted_nodes(G, num_nodes = False)
  plot_save_sir(G = G, folder = folder, beta = beta, D = D, mu = mu, p = p, start_inf = start_inf, plot_all=plot_all)


'Draw N degrees from a Poissonian sequence with lambda = D and length L'
def pois_pos_degrees(D, N, L = int(2e3)):
    degs = np.random.poisson(lam = D, size = L)
    #print("len(s) in deg", len([x for x in degs if x == 0])) 
    pos_degrees = np.random.choice([x for x in degs if x != 0], N)
    #print("len(s) in pos_degrees", len([x for x in pos_degrees if x == 0]))
    return pos_degrees

def config_pois_model(N, D, beta, mu, p = 0, seed = 123, \
  plot_all = True):
    '''create a network with the node degrees drawn from a poissonian with even sum of degrees'''
    np.random.seed(seed)
    degrees = pois_pos_degrees(D,N) #poiss distr with deg != 0

    print("Degree sum:", np.sum(degrees), "with seed:", seed)
    while(np.sum(degrees)%2 != 0): #i.e. sum is odd --> change seed
        seed+=1
        np.random.seed(seed)
        degrees = pois_pos_degrees(D,N)
        print("Degree sum:", np.sum(degrees), "with seed:", seed, )
    print("numb of 0-degree nodes:", len([x for x in degrees if x == 0]) )

    print("\nNetwork Created but w/o standard neighbors wiring!")
    '''create and remove loops since they apppears as neighbors of a node. Check it via print(list(G.neighbors(18))'''
    G = nx.configuration_model(degrees, seed = seed)

    'If D/N !<< 1, by removing loops and parallel edges, we lost degrees. Ex. with N = 50 = D, <k> = 28 != 49.8'
    #print("Total Edges", G.edges())
    check_loops_parallel_edges(G)
    remove_loops_parallel_edges(G)
    #check_loops_parallel_edges(G)

    'plot G, degree distribution and the adiaciency matrix'
    #Config_SIR def: D = 8, beta, mu = 0.1, 0.05
    #print("beta %s ; mu: %s" % (beta, mu))
    plot_save_sir(G, "Conf_Model", beta = beta, D = D, mu = mu, p = 0, plot_all=plot_all)
    
    return G

def nearest_neighbors_pois_net(G, D, beta, mu, start_inf = 10, p = 0, plot_all = True):
  verbose = False
  def verboseprint(*args):
    if verbose == True:
      print(*args)
    elif verbose == False:
      None

  'for random rewiring with p'
  l_nodes = [x for x in G.nodes()]

  edges = set() #avoid to put same link twice (+ unordered)
  nodes_degree = {}

  'list of the nodes sorted by their degree'
  for node in G.nodes():
    nodes_degree[node] = G.degree(node)
  sorted_nodes_degree = {k: v for k, v in sorted(nodes_degree.items(), key=lambda item: item[1])}
  sorted_nodes = [node for node in sorted_nodes_degree.keys()]
  verboseprint("There are the sorted_nodes", sorted_nodes) #, "\n", sorted_nodes_degree.values())

  'cancel all the edges'
  replace_edges_from(G)

  '------ Start of Rewiring with NNR! ---------'
  'Hint: create edges rewiring from ascending degree'
  print("\nStart fo NN-Rewiring")
  def get_var_name(my_name):
    variables = dict(globals())
    for name in variables:
        if variables[name] is my_name:
            #verboseprint("v[n]", variables[name], "my_n", my_name)
            return name
  def ls_nodes_remove(node): l_nodes.remove(node); sorted_nodes.remove(node)
  def zero_deg_remove(node): 
    if nodes_degree[node] == 0 and node in l_nodes and node in sorted_nodes: ls_nodes_remove(node); verboseprint("\n", get_var_name(node), "=", node, "is removed via if deg == 0")

  verboseprint("\nStart of the wiring:")
  while( len(l_nodes) > 1 ):
    node = sorted_nodes[0]
    verboseprint("---------------------------")
    verboseprint("Wire node", node, " with degree", nodes_degree[node], \
          "\nto be rew with:", l_nodes, "which are in total", len(l_nodes))
    
    
    L = len(l_nodes)
    'define aa_attached'
    aa_attached = l_nodes[(l_nodes.index(node)+1)%L]; verboseprint("the anticlock-nearest is ", aa_attached)
    
    if node in l_nodes:
      'if degreees[node] > 1, forced-oscillation-wiring'
      for j in range(1,nodes_degree[node]//2+1): #neighbors attachment and no self-loops "1"
        L = len(l_nodes)
        if len(l_nodes) == 1: break
        verboseprint("entered for j:",j)
        idx = l_nodes.index(node)
        verboseprint("idx_node:", idx, "errored idx", (idx-j)%L)
        a_attached = l_nodes[(idx+j)%L] #anticlockwise-linked node
        c_attached = l_nodes[(idx-j)%L]
        aa_attached = l_nodes[(idx+nodes_degree[node]//2+1)%L]
        verboseprint(node,a_attached); verboseprint(node,c_attached)
        if node != a_attached: edges.add((node,a_attached)); nodes_degree[a_attached]-=1; \
        verboseprint("deg[%s] = %s" % (a_attached, nodes_degree[a_attached]))
        if node != c_attached: edges.add((node,c_attached)); nodes_degree[c_attached]-=1; \
        verboseprint("deg[%s] = %s"%(c_attached,nodes_degree[c_attached]))

        'remove node whenever its degree is = 0:'
        zero_deg_remove(a_attached)
        zero_deg_remove(c_attached)

      if len(l_nodes) == 1: break       
      'if nodes_degree[i] is odd  and the aa_attached, present in l_nodes, has a stub avaible, then, +1 anticlock-wise'
      if nodes_degree[node] % 2 != 0 and nodes_degree[aa_attached] != 0: 
        edges.add((node, aa_attached)); nodes_degree[aa_attached]-=1
        verboseprint("edge with aa added: (", node, aa_attached, ") and deg_aa_att[%s] = %s"%(aa_attached,nodes_degree[aa_attached]))
      
      'aa_attached == 0 should not raise error since it should be always present in l_n and s_n'
      if nodes_degree[aa_attached] == 0 and aa_attached in l_nodes and aa_attached in sorted_nodes: 
        ls_nodes_remove(aa_attached)
        verboseprint("\naa_attached node", aa_attached, "is removed via if deg == 0")
      if node in l_nodes and node in sorted_nodes: ls_nodes_remove(node);  verboseprint(node, "is removed since it was the selected node")
      if len(l_nodes)==1: verboseprint("I will stop here"); break

  verboseprint("End of wiring")

  replace_edges_from(G, edges)
  check_loops_parallel_edges(G)
  infos_sorted_nodes(G, num_nodes=False)

  plot_save_sir(G, plot_all=plot_all, folder = "NNR_Conf_Model", beta = beta, D = D, mu = mu, p = p, start_inf = start_inf)
  
  return G

'''def:: "replace" existing edges, since built-in method only adds'''
def replace_edges_from(G,list_edges=[]):
    ebunch = [x for x in G.edges()]
    G.remove_edges_from(ebunch)
    if list_edges!=[]: return G.add_edges_from(list_edges)
    return G