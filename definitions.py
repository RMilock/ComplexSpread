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

'===Plot and Save SIR + Net'
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

def sir(G, mf = False, beta = 1e-3, mu = 0.05, start_inf = 10, seed = False):
    'If mf == False, the neighbors are not fixed;' 
    'If mf == True, "Quenched" - MF_sir with fixed numb of neighbors'

    import random
    #here's the modifications of the "test_ver1"
    'Number of nodes in the graph'
    N = G.number_of_nodes()
    D = int(rhu(np.sum([j for (i,j) in G.degree() ]) / G.number_of_nodes()))

    
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
            if not mf: tests = G.neighbors(i) #only != wrt to the SIS: contact are taken from G.neighbors            
            if mf: 
              ls = list(range(N)); ls.remove(i)
              tests = random.choices(ls, k = int(D)) #spread very fast since multiple infected center
            tests = [int(x) for x in tests] #convert 35.0 into int
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

def itermean_sir(G, mf = False, numb_iter = 200, beta = 1e-3, mu = 0.05, start_inf = 10,):
  'def a function that iters numb_iter and make an avg of the trajectories'
  from itertools import product
  from itertools import zip_longest
  import numpy as np
  import copy

  numb_idx_cl = 3
  trajectories = [[] for _ in range(numb_idx_cl)]
  avg = [[] for _ in range(numb_idx_cl)]
  counts = [[],[],[]]
  max_len = 0

  import datetime as dt
  
  for i in range(numb_iter):
    start_time = dt.datetime.now()
    if i % 50 == 0: print("time for %s its of max-for-loop %s" % (i, dt.datetime.now()-start_time))
    tmp_traj = sir(G, beta = beta, mu = mu, start_inf = start_inf, mf = mf)
    for idx_cl in range(numb_idx_cl):
        #if idx_cl == 0: print("\ntmp_traj", tmp_traj)
        trajectories[idx_cl].append(tmp_traj[idx_cl])
        tmp_max = len(max(tmp_traj, key = len))
        if tmp_max > max_len: max_len = tmp_max
        #print("\nIteration: %s, tmp_max: %s, len tmp_traj: %s, len tmp_traj %s, len traj[%s] %s" % 
        #  (i, len(max(tmp_traj, key = len)), len(tmp_traj),  \
        #    len(tmp_traj[idx_cl]), idx_cl, len(trajectories[idx_cl]) ))
  #print("\nOverall max_len", max_len)
  #print("All traj", trajectories)
  plot_trajectories = copy.deepcopy(trajectories)

  start_time = dt.datetime.now()

  for i in range(numb_iter):
    if i % 50 == 0: print("time for %s for avg-for-loop %s" % (i, dt.datetime.now()-start_time))
    for idx_cl in range(numb_idx_cl):
        last_el_list = [trajectories[idx_cl][i][-1] for _ in range(max_len-len(trajectories[idx_cl][i]))]
        'traj[classes to be considered, e.g. infected = 0][precise iteration we want, e.g. "-1"]'
        trajectories[idx_cl][i] += last_el_list
        length = len(trajectories[idx_cl][i])
        it_sum = [sum(x) for x in zip_longest(*trajectories[idx_cl], fillvalue=0)]
        for j in range(length):
            try: counts[idx_cl][j]+=1
            except: counts[idx_cl].append(1)
        avg[idx_cl] = list(np.divide(it_sum,counts[idx_cl]))
        
        '''
        print("\niteration(s):", i, "idx_cl ", idx_cl)
        print("last el extension", last_el_list)
        print("(new) trajectories[%s]: %s" % (idx_cl, trajectories[idx_cl]))
        print( "--> trajectories[%s][%s]: %s" % (idx_cl, i, trajectories[idx_cl][i]), 
        "len:", length)
        print("zip_longest same index" , list(zip_longest(*trajectories[idx_cl], fillvalue=0)))#"and traj_idx_cl", trajectories[idx_cl])
        print("global sum indeces", it_sum)
        print("counts of made its", counts[idx_cl])
        print("avg", avg)
        '''
        
        if length != max_len: raise Exception("Error: %s not max_len" % length)
    if i % 50 == 0: print("End of it: %s" % i)
    if i == 199: print("End of 200 scenarios")

  return plot_trajectories, avg

def plot_sir(G, ax, folder = None, beta = 1e-3, mu = 0.05, start_inf = 10, numb_iter = 200):

  'D = numb acts only in mf_avg'
  import itertools
  # MF_SIR: beta = 1e-3, MF_SIR: mu = 0.05
  D = int(rhu(np.sum([j for (i,j) in G.degree() ]) / G.number_of_nodes()))
  N = G.number_of_nodes(); R0 = beta*D/mu

  'plot ratio of daily infected and daily cumulative recovered'
  'Inf and Cum_Infected from Net_Sir; Recovered from MF_SIR'
  trajectories, avg = itermean_sir(G, mf = False, beta = beta, mu = mu, start_inf  = start_inf, numb_iter=numb_iter)
  mf_trajectories, mf_avg = itermean_sir(G, mf = True, mu = mu, beta = beta, start_inf = start_inf, numb_iter = numb_iter)

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
  ax.plot(avg[2], label="Net::CD_Inf/N (%s%%)" % np.round(avg[2][-1]*100,1), \
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
  folders = ["NNR_Conf_Model"] #"WS_Epids"]
  if R0 <= 3 and folder in folders:
    ax_ymin, ax_ymax = ax.get_ylim()
    set_ax_ymax = 1.5*ax_ymax
    ax.set_ylim(ax_ymin, set_ax_ymax)
    ax.legend(bbox_to_anchor=(1, 1), edgecolor="grey", loc='upper right') #add: leg = 
  else: ax.legend(loc="best"); 'set legend in the "best" mat plot lib location'

def rhu(n, decimals=0): #round_half_up
    import math
    multiplier = 10 ** decimals
    return math.floor(n*multiplier + 0.5) / multiplier
  
def plot_save_net(G, folder, p = 0, done_iterations = 1, log_dd = False, partition = None, pos = None):
  import os.path
  from definitions import my_dir
  from functools import reduce
  

  mode = "a"
  #if done_iterations == 1: mode = "w"
  my_dir = my_dir() #"/home/hal21/MEGAsync/Thesis/NetSciThesis/Project/Plots/Tests/"
  N = G.number_of_nodes()
  D = rhu( np.sum([j for (i,j) in G.degree() ]) / G.number_of_nodes() )
  real_D = rhu(D,1)
  print("Real D", real_D)
  D = rhu(D)
  adj_or_sir = "AdjMat"
  log_upper_path = my_dir + folder + "/" #"../Plots/Tests/WS_Epids/"
  my_dir+=folder+"/p%s/"%rhu(p,3)+adj_or_sir+"/" #"../Plots/Tests/WS_Epids/p0.001/AdjMat/"
  file_name = folder + "_%s_N%s_D%s_p%s"% (
        adj_or_sir,
        N,D,rhu(p,3)) + \
      ".png"
  file_path = my_dir + file_name #../AdjMat/AdjMat_N1000_D500_p0.001.png
  log_path = log_upper_path + folder + f"_log_saved_{adj_or_sir}.txt" #"../Plots/Tests/WS_Epids/WS_Epids_log_saved_nets.txt"
      
  plot_params()

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
  long_range_edges = list(filter( lambda x: x > 30, [np.min((np.abs(i-j),np.abs(j-i))) for i,j in G.edges()] )) #list( filter(lambda x: x > 0, )
  length_long_range = len(long_range_edges)
  if length_long_range < 20: print("\nLong_range_edges", long_range_edges, length_long_range)
  else: print("len(long_range_edges", length_long_range)
  folders = ["WS_Pruned"]
  if folder in folders: width = 0.001
  if folder == "Barabasi": width = 0.2*N/len(long_range_edges); print("The edge with is", width)
  if folder== "Caveman_Model":
    nx.draw(G, pos, node_color=list(partition.values()), node_size = 5, width = 0.5, with_labels = False)
  else: nx.draw_circular(G, ax=ax, with_labels=False, font_size=20, node_size=25, width=width)

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
  mean = rhu( D )
  print("Rounded degrees mean", mean)
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
  if not os.path.exists(my_dir): os.makedirs(my_dir)
  plt.savefig(file_path)
  plt.close()

  with open(log_path, mode) as text_file: #write only 1 time
    print("file_name", file_name)
    text_file.write("".join((file_name, "\n")) )    
    #text_file.write("".join(("N: ", file_name, "; D=%s" % real_D, "\n")) )

  'sort line to have the new ones at first'
  sorted_lines = []
  with open(log_path, 'r') as r:
    for line in sorted(r):
      sorted_lines.append(line)
  
  with open(log_path, 'w') as r:
    for line in sorted_lines:
      r.write(line)

def plot_save_sir(G, folder, done_iterations = 1, p = 0, beta = 0.001, mu = 0.16, R0_max = 16,  start_inf = 10, numb_iter = 200):
  import os.path
  from definitions import my_dir

  mode = "a"
  if done_iterations == 1: mode = "w"
  my_dir = my_dir() #"/home/hal21/MEGAsync/Thesis/NetSciThesis/Project/Plots/Tests/"
  N = G.number_of_nodes()
  D = rhu( np.sum([j for (i,j) in G.degree() ]) / G.number_of_nodes() )
  adj_or_sir = "SIR"
  log_upper_path = my_dir + folder + "/" #../Plots/Tests/WS_Epids/
  my_dir+=folder+"/p%s/"%rhu(p,3)+adj_or_sir+"/" #"../Plots/Test/WS_Epids/p0.001/SIR/"
  #file_path depends on a "r0_folder"
  log_path = log_upper_path + "/" + folder + "_log_saved_SIR.txt" #"../Plots/Test/WS_Epids/SIR_log_saved_SIR.txt"
      
  import datetime as dt
  start_time = dt.datetime.now()

  plot_params()
  intervals = [x for x in np.arange(R0_max)]
  N = G.number_of_nodes()
  R0 = beta * D / mu    

  for i in range(len(intervals)-1):
    if intervals[i] <= R0 < intervals[i+1]:

      'Intro R0-subfolder since R0 det epids behaviour on a fixed net'
      r0_folder = "R0_%s-%s/" % (intervals[i], intervals[i+1])
      if folder != "W": r0_folder += "mu%s/" % (rhu(mu,3)) #"R0_1-2/mu0.16/"
      #if folder == "WS_Epids": r0_folder += "D%s/" % D  #"R0_1-2/mu0.16/D6/"
      if not os.path.exists(my_dir + r0_folder): os.makedirs(my_dir + r0_folder)
      file_name = folder + "_%s_R0_%s_N%s_D%s_p%s_beta%s_mu%s"% (
            adj_or_sir, '{:.3f}'.format(rhu(beta/mu*D,3)),
            N,D, rhu(p,3), rhu(beta,3), rhu(mu,3) ) + ".png"
      file_path = my_dir + r0_folder + file_name
      
      'plot all'
      _, ax = plt.subplots(figsize = (20,12))

      'plot always sir'
      print("\nThe model has N: %s, D: %s, beta: %s, mu: %s, p: %s, R0: %s" % (N,D,rhu(beta,3),rhu(mu,3),rhu(p,3),rhu(R0,3)) )
      plot_sir(G, ax=ax, folder = folder, beta = beta, mu = mu, start_inf = start_inf, numb_iter = numb_iter)
      plt.subplots_adjust(
      top=0.920,
      bottom=0.151,
      left=0.086,
      right=0.992,
      hspace=0.2,
      wspace=0.2)
      plt.suptitle(r"$R_0:%s, N:%s, D:%s, p:%s, \beta:%s, \mu:%s$"% (rhu(beta/mu*D,3),N,D,rhu(p,3), rhu(beta,3), rhu(mu,3), ))
      #plt.show()

      plt.savefig( file_path )
      print("time 1_plot_save_sir:", dt.datetime.now()-start_time) 
      
      with open(log_path, mode) as text_file: #write only 1 time
        text_file.write(file_name + "\n")
      
      plt.close()

  'sort line to have the new ones at first'
  sorted_lines = []
  with open(log_path, 'r') as r:
    for line in sorted(r):
      sorted_lines.append(line)
  
  with open(log_path, 'w') as r:
    for line in sorted_lines:
      r.write(line)

def save_log_params(folder, text, done_iterations):
  import os
  from definitions import my_dir
  print("log_params is @:", my_dir() + folder)
  if done_iterations == 1:
    if not os.path.exists(my_dir() + folder): os.makedirs(my_dir() + folder)
    with open(my_dir() +  folder + "/" + folder + "_log_params.txt", "a") as text_file: #write only 1 time
        text_file.write(text)


'===Watts-Strogatz Model'
'Number of iterations, i.e. or max power or fixed by user'
def pow_max(N, num_iter = "all"):
  if num_iter == "all":
    'search the 2**pow_max'
    i = 0
    while(N-2**i>0): i+=1
    return i-1
  return int(num_iter)

def ws_sir(G, folder, p, saved_nets, done_iterations, pruning = False, infos = False, beta = 0.001, mu = 0.16, start_inf = 10):    
  'round_half_up D for a better approximation of nx.c_w_s_graph+sir'
  import networkx as nx
  N = G.number_of_nodes()
  D = int(rhu(np.sum([j for (i,j) in G.degree() ]) / G.number_of_nodes()))
  if infos == True: check_loops_parallel_edges(G); infos_sorted_nodes(G, num_nodes = False)

  if "N%s_D%s_p%s"% (N,D,rhu(p,3)) not in saved_nets: 
    plot_save_net(G = G, folder = folder, p = p, done_iterations = done_iterations)
    saved_nets.append("N%s_D%s_p%s" % (N,D,rhu(p,3)))
    print("saved nets", saved_nets)
  plot_save_sir(G = G, folder = folder, beta = beta, mu = mu, p = p, start_inf = start_inf, done_iterations = done_iterations )

'===Configurational Model'
def pois_pos_degrees(D, N, L = int(2e3)):
  'Draw N degrees from a Poissonian sequence with lambda = D and length L'
  degs = np.random.poisson(lam = D, size = L)
  #print("len(s) in deg", len([x for x in degs if x == 0])) 
  pos_degrees = np.random.choice([x for x in degs if x != 0], N)
  #print("len(s) in pos_degrees", len([x for x in pos_degrees if x == 0]))
  return pos_degrees

def config_pois_model(N, D, seed = 123, folder = None):
  '''create a network with the node degrees drawn from a poissonian with even sum of degrees'''
  np.random.seed(seed)
  degrees = pois_pos_degrees(D,N) #poiss distr with deg != 0

  while(np.sum(degrees)%2 != 0): #i.e. sum is odd --> change seed
      seed+=1
      np.random.seed(seed)
      degrees = pois_pos_degrees(D,N)
  #print("Degree sum:", np.sum(degrees), "with seed:", seed)

  print("\nNetwork Created but w/o standard neighbors wiring!")
  G = nx.configuration_model(degrees, seed = seed)

  if folder != "Overlapping_Rew": folder = "Config_Model" 
  if folder == "Overlapping_Rew":
    dsc_sorted_nodes = {k: v for k,v in sorted( G.degree(), key = lambda x: x[1], reverse=True)}
    edges = set()
    for node in dsc_sorted_nodes.keys():
      k = dsc_sorted_nodes[node]//2
      for i in range(1, k + 1):
          edges.add((node, (node+i)%N))
          edges.add((node, (node-i)%N))
      if dsc_sorted_nodes[node] % 2 == 1: edges.add((node, (node+k+1)%N))
      elif dsc_sorted_nodes[node] % 2 != 0: print("Error of Wiring: dsc[node]%2", dsc_sorted_nodes[node] % 2); break
    replace_edges_from(G, edges)
    remove_loops_parallel_edges(G)
    avg_degree = np.sum([j for i,j in G.degree()])/G.number_of_nodes()
    if avg_degree != D: print("!! avg_deg_Overl_Rew - avg_deg_Conf_Model = ", avg_degree - D)

    return G

  'If D/N !<< 1, by removing loops and parallel edges, we lost degrees. Ex. with N = 50 = D, <k> = 28 != 49.8'
  check_loops_parallel_edges(G)
  remove_loops_parallel_edges(G)
  infos_sorted_nodes(G)
  print("End of %s " % folder)
  
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
  print("Start of NN-Rewiring")
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

  replace_edges_from(G, edges)
  check_loops_parallel_edges(G)
  infos_sorted_nodes(G, num_nodes=False)

  print("End of wiring")

  return G

def NN_Overl_pois_net(N, D, p, add_edges_only = False):
    from itertools import chain
    import datetime as dt
    import random

    
    def edges_node(x):
        return [(i,j) for i,j in all_edges if i == x]

    G = config_pois_model(N,D)
    adeg_CM = sum([j for i,j in G.degree()])/G.number_of_nodes()

    dsc_sorted_nodes = {k: v for k,v in sorted( G.degree(), key = lambda x: x[1], reverse=True)}
    edges = set()
    for node in dsc_sorted_nodes.keys():
        k = dsc_sorted_nodes[node]//2
        for i in range(1, k + 1):
            edges.add((node, (node+i)%N))
            edges.add((node, (node-i)%N))
        if dsc_sorted_nodes[node] % 2 == 1: edges.add((node, (node+k+1)%N))
        elif dsc_sorted_nodes[node] % 2 != 0: print("Error of Wiring: dsc[node]%2", dsc_sorted_nodes[node] % 2); break
    replace_edges_from(G, edges)
    #print("bef rmv loops and //", G.edges(), G.degree(), sum([j for i,j in G.degree()])/G.number_of_nodes())

    remove_loops_parallel_edges(G)
    #print("after rmv loops and //", G.edges(), G.degree(), sum([j for i,j in G.degree()])/G.number_of_nodes())
    #print([(i,npr.choice(G.nodes().pop(i))) for i in G.edges() if i == 0])

    all_edges = [list(G.edges(node)) for node in G.nodes()]
    all_edges = list(chain.from_iterable(all_edges))
    initial_length = len(all_edges)
    #print( "inital edges ", all_edges )

    start_time = dt.datetime.now()

    if add_edges_only:
        for node in G.nodes():
            left_nodes = list(G.nodes())
            left_nodes.remove(node) 
            re_link = random.choice( left_nodes )
            if random.random() < p:
                all_edges.append((node,re_link))

    else:
        for node in G.nodes():
            left_nodes = list(G.nodes())
            left_nodes.remove(node) 
            #print("edges of the node %s: %s" % (node, edges_node(node)) ) 
            i_rmv, j_rmv = random.choice(edges_node(node))  
            if random.random() < p and len(edges_node(j_rmv)) > 1: #if, with prob p, the "node" is not the only friend of "j_rmv" 
                all_edges.remove((i_rmv,j_rmv)); all_edges.remove((j_rmv,i_rmv))
                re_link = random.choice(left_nodes)
                #print("rmv_choice: (%s,%s)" % (i_rmv, j_rmv), "relink with node: ", re_link)
                all_edges.append((node, re_link)); all_edges.append((re_link, node))
                all_edges = [(node,j) for node in range(len(all_edges)) for j in [j for i,j in all_edges if i == node] ]
                #print("all_edges", all_edges)


    print("len(all_edges)_final", len(all_edges), "is? equal to start", initial_length )
    replace_edges_from(G, all_edges)
    remove_loops_parallel_edges(G, False)
    print(f"Time for add_edges = {add_edges_only}:", dt.datetime.now()-start_time)

    adeg_OR = sum([j for i,j in G.degree()])/G.number_of_nodes()
    print("Rel Error wrt to D %s %%" % (rhu((adeg_OR - D)/D,1 )*100))
    print("Rel Error wrt to adeg_CM %s %%" % (rhu( (adeg_OR - adeg_CM)/adeg_CM,1 )*100))
    
    return G

'Net Infos'
def check_loops_parallel_edges(G):
  ls = list(G.edges())
  print("parallel edges", set([i for i in ls for j in ls[ls.index(i)+1:] if i==j]),
        "; loops", [(i,j) for (i,j) in set(G.edges()) if i == j])

def infos_sorted_nodes(G, num_nodes = False):
    import networkx as nx
    'sort nodes by key = degree. printing order: node, adjacent nodes, degree'
    nodes = G.nodes()
    print("<k>: ", np.sum([j for (i,j) in G.degree() ]) / len(nodes), 
          " and <k>/N ", np.sum([j for (i,j) in G.degree() ]) / len(nodes)**2, end="\n" )
    
    'put adj_matrix into dic for better visualisation'
    adj_matrix =  nx.adjacency_matrix(G).todense()
    adj_dict = {i: np.nonzero(row)[1].tolist() for i,row in enumerate(adj_matrix)}

    infos = zip([x for x in nodes], [len(adj_dict[i]) for i in range(len(nodes))], [G.degree(x) for x in nodes])
    dsc_sorted_nodes = sorted( infos, key = lambda x: x[2], reverse=True)

    cut_off = 4
    if num_nodes == True:  
      num_nodes = len(nodes) 
      for i in range(cut_off):
        if i == 0: print("Triplets of (nodes, neighbors(%s), degree) sorted by descending degree:" % i)
        print( dsc_sorted_nodes[i] )

    if num_nodes == False: num_nodes = 0
    
def remove_loops_parallel_edges(G, remove_loops = True):
  print("\n")
  import networkx as nx

  'create a list of what we want to remove'
  full_ls = list((G.edges()))
  lpe = []
  for i in full_ls:
    full_ls.remove(i)
    for j in full_ls:
      if i == j: lpe.append(j) #print("i", i, "index", full_ls.index(i), "j", j)
  if remove_loops == True:  
    for x in list(nx.selfloop_edges(G)): lpe.append(x)
    print("Parallel edges and loops removed!")
  return G.remove_edges_from(lpe)

def replace_edges_from(G,list_edges=[]):
  '''def:: "replace" existing edges, since built-in method only adds'''
  present_edges = [x for x in G.edges()]
  G.remove_edges_from(present_edges)
  if list_edges!=[]: return G.add_edges_from(list_edges)
  return G


'===Caveman Defs'
def caveman_defs():
  import numpy as np
  import matplotlib.pyplot as plt
  import networkx as nx
  NODE_LAYOUT = nx.circular_layout
  COMMUNITY_LAYOUT = nx.circular_layout
  def partition_layout(g, partition, ratio=0.3):
      """
      Compute the layout for a modular graph.

      Arguments:
      ----------
      g -- networkx.Graph or networkx.DiGraph instance
          network to plot

      partition -- dict mapping node -> community or None
          Network partition, i.e. a mapping from node ID to a group ID.

      ratio: 0 < float < 1.
          Controls how tightly the nodes are clustered around their partition centroid.
          If 0, all nodes of a partition are at the centroid position.
          if 1, nodes are positioned independently of their partition centroid.

      Returns:
      --------
      pos -- dict mapping int node -> (float x, float y)
          node positions

      """

      pos_communities = _position_communities(g, partition)

      pos_nodes = _position_nodes(g, partition)
      pos_nodes = {k : ratio * v for k, v in pos_nodes.items()}

      # combine positions
      pos = dict()
      for node in g.nodes():
          pos[node] = pos_communities[node] + pos_nodes[node]

      return pos

  def _position_communities(g, partition, **kwargs):

      # create a weighted graph, in which each node corresponds to a community,
      # and each edge weight to the number of edges between communities
      between_community_edges = _find_between_community_edges(g, partition)

      communities = set(partition.values())
      hypergraph = nx.DiGraph()
      hypergraph.add_nodes_from(communities)
      for (ci, cj), edges in between_community_edges.items():
          hypergraph.add_edge(ci, cj, weight=len(edges))

      # find layout for communities
      pos_communities = COMMUNITY_LAYOUT(hypergraph, **kwargs)

      # set node positions to position of community
      pos = dict()
      for node, community in partition.items():
          pos[node] = pos_communities[community]

      return pos

  def _find_between_community_edges(g, partition):

      edges = dict()

      for (ni, nj) in g.edges():
          ci = partition[ni]
          cj = partition[nj]

          if ci != cj:
              try:
                  edges[(ci, cj)] += [(ni, nj)]
              except KeyError:
                  edges[(ci, cj)] = [(ni, nj)]

      return edges

  def _position_nodes(g, partition, **kwargs):
      """
      Positions nodes within communities.
      """
      communities = dict()
      for node, community in partition.items():
          if community in communities:
              communities[community] += [node]
          else:
              communities[community] = [node]

      pos = dict()
      for community, nodes in communities.items():
          subgraph = g.subgraph(nodes)
          pos_subgraph = NODE_LAYOUT(subgraph, **kwargs)
          pos.update(pos_subgraph)

      return pos

  def _layout(networkx_graph):
      edge_list = [edge for edge in networkx_graph.edges]
      node_list = [node for node in networkx_graph.nodes]

      pos = nx.circular_layout(edge_list)

      # NB: some nodes might not be connected and hence will not be in the edge list.
      # Assuming a [0, 0, 1, 1] canvas, we assign random positions on the periphery
      # of the existing node positions.
      # We define the periphery as the region outside the circle that covers all
      # existing node positions.
      xy = list(pos.values())
      centroid = np.mean(xy, axis=0)
      delta = xy - centroid[np.newaxis, :]
      distance = np.sqrt(np.sum(delta**2, axis=1))
      radius = np.max(distance)

      connected_nodes = set(_flatten(edge_list))
      for node in node_list:
          if not (node in connected_nodes):
              pos[node] = _get_random_point_on_a_circle(centroid, radius)

      return pos

  def _flatten(nested_list):
      return [item for sublist in nested_list for item in sublist]

  def _get_random_point_on_a_circle(origin, radius):
      x0, y0 = origin
      random_angle = 2 * np.pi * np.random.random_sample()
      x = x0 + radius * np.cos(random_angle)
      y = y0 + radius * np.sin(random_angle)
      return np.array([x, y])

  def comm_caveman_relink(cliques = 8, clique_size = 7, p = 0,  relink_rnd = 0, numb_rel_inring = 0):
      import numpy as np
      import numpy.random as npr
      'caveman_graph'
      G = nx.caveman_graph(l = cliques, k = clique_size)

      'relink nodes to neighbor "cave"'
      total_nodes = clique_size*cliques
      #if numb_rel_inring != 0: 
      for clique in range(cliques):
          nodes_inclique = np.arange(clique_size*(clique), clique_size*(1+clique))
          if numb_rel_inring != 0:
              attached_nodes = npr.choice( np.arange(clique_size*(1+clique), 
                                          clique_size*(2+clique)), 
                                          size = len(nodes_inclique) )
              attached_nodes = attached_nodes % np.max((total_nodes,1))
              for test, att_node in zip(nodes_inclique, attached_nodes):
                  #print("NN - clique add:", (test,att_node))
                  G.add_edge(test,att_node)
          
          'here I add a new edge but as for the Overl_Rew, I relink one of the existing node'
          'In the last way, avg_degree is preserved'
          if p != 0:
              attached_nodes = npr.choice([x for x in G.nodes() if x not in nodes_inclique], 
                                          size = len(nodes_inclique))
              for test, att_node in zip(nodes_inclique, attached_nodes):
                  #print("relink", (test,att_node))
                  if npr.uniform() < p: G.add_edge(test,att_node)

      #check_loops_parallel_edges(G)
      remove_loops_parallel_edges(G)
      #check_loops_parallel_edges(G)

      print("size/cliq: %s, cliq/size: %s" % (clique_size/cliques, cliques/clique_size) )

      
      return G
  
  return partition_layout, comm_caveman_relink


'===Saving Strategies'
def already_saved_list(folder, adj_or_sir, chr_min, chr_max = None, my_print = True, done_iterations = 1):
  log_upper_path = "".join((my_dir(),folder,"/")) #../Plots/Test/Overlapping.../
  log_path = "".join((log_upper_path, folder, f"_log_saved_{adj_or_sir}.txt"))

  '''good idea but not in the goal of succint
  if done_iterations == 1 and os.path.exists(log_path):
    'set to O_ld all the already saved nets'
    lines = []
    with open(log_path, 'r') as r:
      for line in r:
        tmp_list = list(line)
        tmp_list[0] = "O"
        lines.append("".join(tmp_list)) 
    with open(log_path, 'w') as r:
      for line in lines:
        r.write(line)
    '''
  saved_list = []
  if os.path.exists(log_path):
    with open(log_path, "r") as file:
        saved_list = [l.rstrip("\n")[chr_min:] for l in file]
  if my_print: print(f"\nThe already saved {adj_or_sir} are", saved_list)
  return saved_list

def plot_save_nes(G, p, folder, adj_or_sir, beta = 0.3, \
  mu = 0.3, my_print = True, done_iterations = 1, chr_min = 0): #save new_entrys
  'save net only if does not exist in the .txt. So, to overwrite all just delete .txt'
  from definitions import already_saved_list
  D = mean = rhu( np.sum([j for (i,j) in G.degree() ]) / G.number_of_nodes() )
  N = G.number_of_nodes()
  if adj_or_sir == "AdjMat": 
    saved_files = already_saved_list(folder, adj_or_sir, chr_min = chr_min, my_print= my_print, done_iterations= done_iterations)
    file_name = folder + "_AdjMat_N%s_D%s_p%s.png" % (N,rhu(D,1),rhu(p,3)) 
  if adj_or_sir == "SIR": 
    saved_files = already_saved_list(folder, adj_or_sir, chr_min = chr_min, my_print= my_print, done_iterations=done_iterations)
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