import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm 
import os #to create a folder

#Thurner pmts: beta = 0.1, mu = 0.16; D = 3 vel 8
#MF def: beta, mu = 0.001/cf, 0.05/cf or 0.16/cf ; cf = 1


def my_dir():
  #return "/content/drive/MyDrive/Colab_Notebooks/Thesis/Complex_Plots/"
  #return "/content/"
  return "/home/hal21/MEGAsync/Tour_Physics2.0/Thesis/NetSciThesis/Project/Plots/Test/"

def save_log_file(folder, text):
  import os
  from definitions import my_dir
  print(my_dir() + folder)
  try: 
    os.makedirs(my_dir() + folder)
    with open(my_dir() + folder + "/" + folder + "_log.txt", "w") as text_file: #write only 1 time
      text_file.write(text)
  except:
    with open(my_dir() + folder + "/" + folder + "_log.txt", "w") as text_file: #write only 1 time
      text_file.write(text)

'plot and save sir'
def plot_params():
  import matplotlib.pyplot as plt

  'set fontsize for a better visualisation'
  SMALL_SIZE = 40
  MEDIUM_SIZE = 25
  BIGGER_SIZE = 40

  plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
  plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
  plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
  plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the xtick labels
  plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the ytick labels
  plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
  plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
  plt.rc('axes', labelpad = 20)
  plt.rc('xtick.major', pad = 16)
  #plt.rcParams['xtick.major.pad']='16'

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
            if D != None: 
              ls = list(range(N)); ls.remove(i)
              tests = random.choices(ls, k = int(D)) #spread very fast since multiple infected center
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

def itermean_sir(G, numb_iter = 200, D = None, beta = 1e-3, mu = 0.05, start_inf = 10,):
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
            #print("\niteration:", i, "idx ", idx, )#"and traj_idx", trajectories[idx])
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

def plot_sir(G, ax, folder = None, D = None, beta = 1e-3, mu = 0.05, start_inf = 10, numb_iter = 200):

  'D = numb acts only in mf_avg'
  import itertools
  # MF_SIR: beta = 1e-3, MF_SIR: mu = 0.05
  N = G.number_of_nodes(); R0 = beta*D/mu

  'plot ratio of daily infected and daily cumulative recovered'
  'Inf and Cum_Infected from Net_Sir; Recovered from MF_SIR'
  trajectories, avg = itermean_sir(G, D = None, beta = beta, mu = mu, start_inf  = start_inf, numb_iter=numb_iter)
  mf_trajectories, mf_avg = itermean_sir(G, mu = mu, beta = beta, D = D, start_inf = start_inf, numb_iter = numb_iter)

  'plotting the many realisations'    
  colors = ["paleturquoise","wheat","lightgreen", "thistle"]
  
  for j in range(numb_iter):
    ax.plot(mf_trajectories[2][j], color = colors[1])
    ax.plot(trajectories[2][j], color = colors[2])
    ax.plot(trajectories[0][j], color = colors[0])
    ax.plot(mf_trajectories[0][j], color = colors[3])
    
    '''if R0 <= 2 and folder == "NNR_Conf_Model":
      'to set legend above the plot'
      y_max = np.max( np.concatenate((
          np.max(trajectories[2][j]), mf_trajectories[2][j],
          np.max(trajectories[0][j]) ), axis = None ) )'''

  ax.plot(mf_avg[2], label="MF::CD_Inf/N (%s%%)"% np.round(mf_avg[2][-1]*100,1), \
    color = "tab:orange" ) #mf::cd_inf
  ax.plot(avg[2], label="Net::CD_Inf /N (%s%%)" % np.round(avg[2][-1]*100,1), \
    color = "tab:green") #net::cd_inf    
  ax.plot(mf_avg[0], label="MF::Infected/N ", \
    color = "darkviolet") #prevalence
  ax.plot(avg[0], label="Net::Infected/N ", \
    color = "tab:blue") #prevalence

  'plot horizontal line to highlight the initial infected'
  ax.axhline(start_inf/N, color = "r", ls="dashed", \
    label = "Start_Inf/N (%s%%) "% np.round(start_inf/N*100,1))

  locs, _ = plt.yticks()
  ax.set_yticks(locs[1:-1])
  ax.set_yticklabels(np.round(locs[1:-1],2))

  'plot labels'
  ax.set_xlabel('Time[1day]')
  ax.set_ylabel('Indivs/N')
  
  'set legend above the plot if R_0 in [0,2] in the NNR_Config_Model'
  if R0 <= 3 and folder == "NNR_Conf_Model":
    ax_ymin, ax_ymax = ax.get_ylim()
    set_ax_ymax = 1.5*ax_ymax
    ax.set_ylim(ax_ymin, set_ax_ymax)
    ax.legend(bbox_to_anchor=(1, 1), edgecolor="grey", loc='upper right') #add: leg = 
  else: ax.legend(loc="best"); 'set legend in the "best" mat plot lib location'

def rhu(n, decimals=0): #round_half_up
    import math
    multiplier = 10 ** decimals
    return math.floor(n*multiplier + 0.5) / multiplier
  
def plot_save_net(G, folder, D, p, scaled_G, log_dd = False):
  plot_params()

  N = G.number_of_nodes()
  'plot G, adj_mat, degree distribution'
  plt.figure(figsize = (20,20))

  ax = plt.subplot(221)

  '''
  if p == 0 and folder[:3] == "WS_":
    scaled_G = nx.connected_watts_strogatz_graph( n = N, k = 500, p = p, seed = 1 ) 
  elif p >= 0.1 and folder[:3] == "WS_":
    scaled_G = nx.connected_watts_strogatz_graph( n = N, k = 2, p = p, seed = 1 )
  '''
  
  'plot the real G not the scaled one and dont put description of the scaled_G via ax.text'
  width = 0.8
  scaled_G = G
  if folder == "WS_Pruned": width = 0.001
  nx.draw_circular(scaled_G, ax=ax, with_labels=False, font_size=20, node_size=100, width=width)

  '''
  if folder[:3] == "WS_Pruned":
    ax.text(0,1,transform=ax.transAxes,
      s = "D:%s, p:%s" % \
      ( 
        rhu(2*scaled_G.number_of_edges() / float(scaled_G.number_of_nodes()),3), 
        rhu(p,3)) )
  '''
  
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
  left=0.1,
  right=0.963,
  hspace=0.067,
  wspace=0.164)
  plt.suptitle(r"$N:%s, D:%s, p:%s$"% (N,D,rhu(p,3)))

  
  'TO SAVE PLOTS'
  from definitions import my_dir
  my_dir = my_dir() #"/home/hal21/MEGAsync/Thesis/NetSciThesis/Project/Plots/"
  adj_or_sir = "AdjMat"
  my_dir+=folder+"/p%s/"%rhu(p,3)+adj_or_sir+"/"
  try:
    os.makedirs(my_dir)
    plt.savefig(my_dir + \
      "%s_N%s_D%s_p%s"% (
        adj_or_sir,
        N,D,rhu(p,3)) 
      + ".png")
  except:
    plt.savefig(my_dir + \
      "%s_N%s_D%s_p%s"% (
        adj_or_sir,
        N,D,rhu(p,3)) 
      + ".png")
  plt.close()

def plot_save_sir(G, folder, D, p = 0, beta = 0.001, mu = 0.16, R0_max = 16,  start_inf = 10, numb_iter = 200):

  import datetime as dt
  start_time = dt.datetime.now()

  plot_params()

  intervals = [x for x in np.arange(R0_max)]
  N = G.number_of_nodes()
  R0 = beta * D / mu    
  for i in range(len(intervals)-1):
    if intervals[i] <= R0 < intervals[i+1]:

      'plot all'
      N = G.number_of_nodes()
      _, ax = plt.subplots(figsize = (20,12))

      'plot always sir'
      if D == None: D = np.sum([j for (i,j) in G.degree()]) / N
      print("The model has N: %s, D: %s, beta: %s, mu: %s, p: %s, R0: %s" % (N,D,rhu(beta,3),rhu(mu,3),rhu(p,3),rhu(R0,3)) )
      plot_sir(G, ax=ax, folder = folder, beta = beta, mu = mu, start_inf = start_inf, D = D, numb_iter = numb_iter)
      plt.subplots_adjust(
      top=0.920,
      bottom=0.151,
      left=0.086,
      right=0.992,
      hspace=0.2,
      wspace=0.2)
      plt.suptitle(r"$R_0:%s, N:%s, D:%s, p:%s, \beta:%s, \mu:%s$"% (rhu(beta/mu*D,3),N,D,rhu(p,3), rhu(beta,3), rhu(mu,3), ))
      
      
      #plt.show()
      'save plots in different folders'
      adj_or_sir = "SIR"
      from definitions import my_dir
      my_dir = my_dir() #"/home/hal21/MEGAsync/Thesis/NetSciThesis/Project/Plots/"
      my_dir+=folder+"/p%s/"%rhu(p,3)+adj_or_sir+"/" #"/home/hal21/MEGAsync/Thesis/NetSciThesis/Project/Plots/"

      'Intro R0-subfolder only for "Pruning" since epidemic force has to be equal'
      'else: I want to "fix" an epidemic and see how it spreads'
      sub_folders = "R0_%s-%s/" % (intervals[i], intervals[i+1])
      if folder != "WS_Epids": sub_folders += "mu%s/" % (rhu(mu,3))
      #old ver: 
      #if folder == "WS_Pruned"
      #else: sub_folders = "/beta%s/mu%s/" % (rhu(beta,3), rhu(mu,3))
        
      if folder == "WS_Epids": sub_folders += "D%s/" % D 

      plot_name = my_dir + sub_folders + folder + \
          "_%s_R0_%s_N%s_D%s_p%s_beta%s_mu%s"% (
            adj_or_sir, '{:.3f}'.format(rhu(beta/mu*D,3)),
            N,D, rhu(p,3), rhu(beta,3), rhu(mu,3) )
      try:
        os.makedirs(my_dir + sub_folders)
        plt.savefig( plot_name + ".png")
      except:
        plt.savefig( plot_name + ".png")
      print("time 1_plot_save_sir:", dt.datetime.now()-start_time) 
  plt.close()
  print("---")

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


'Watts-Strogatz Model'
'Number of iterations, i.e. or max power or fixed by user'
def pow_max(N, num_iter = "all"):
  if num_iter == "all":
    'search the 2**pow_max'
    i = 0
    while(N-2**i>0): i+=1
    return i-1
  return int(num_iter)

def ws_sir(G, folder, p, saved_nets, pruning = False, D = None, infos = False, beta = 0.001, mu = 0.16, start_inf = 10):    
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

  if "N%s_D%s_p%s"% (N,D,rhu(p,3)) not in saved_nets: 
    plot_save_net(G = G, scaled_G = G, folder = folder, D = D, p = p)
    saved_nets.append("N%s_D%s_p%s" % (N,D,rhu(p,3)))
    print(saved_nets, "\n---")
  #plot_save_sir(G = G, folder = folder, beta = beta, D = D, mu = mu, p = p, start_inf = start_inf)

'Configurational Model'
'Draw N degrees from a Poissonian sequence with lambda = D and length L'
def pois_pos_degrees(D, N, L = int(2e3)):
    degs = np.random.poisson(lam = D, size = L)
    #print("len(s) in deg", len([x for x in degs if x == 0])) 
    pos_degrees = np.random.choice([x for x in degs if x != 0], N)
    #print("len(s) in pos_degrees", len([x for x in pos_degrees if x == 0]))
    return pos_degrees

def config_pois_model(N, D, seed = 123):
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
    #check_loops_parallel_edges(G)
    remove_loops_parallel_edges(G)
    #check_loops_parallel_edges(G)    
    return G

def NN_pois_net(G, D, p = 0):
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

  return G

'''def:: "replace" existing edges, since built-in method only adds'''
def replace_edges_from(G,list_edges=[]):
    ebunch = [x for x in G.edges()]
    G.remove_edges_from(ebunch)
    if list_edges!=[]: return G.add_edges_from(list_edges)
    return G