import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm 
import os #to create a folder

#Thurner pmts: beta = 0.1, mu = 0.16; D = 3 vel 8
#MF def: beta, mu = 0.001/cf, 0.05/cf or 0.16/cf ; cf = 1

#this has been saved in morning of the 3.6.2021

def pos_deg_nodes(G): # G "real" nodes
  return [i for i,j in G.degree() if j > 0]

def my_dir():
  #return "/content/drive/MyDrive/Colab_Notebooks/Thesis/Complex_Plots/"
  #return "/content/"
  return "/home/hal21/MEGAsync/Tour_Physics2.0/Thesis/NetSciThesis/Project/Plots/Test/"

def parameters_net_and_sir(folder = None, p_max = 0.1):
  'progression of net-parameters'

  'WARNING: put SAME beta, mu, D and p to compare at the end the different topologies'
  #B-A_Model parameters
  k_prog = np.concatenate(([0.3,1,2],np.arange(3,40,2)))
  #In B-A model, these are the fully connected initial cliques
  p_prog = [rhu(x,1) for x in np.linspace(0,p_max,int(p_max*10)+1)]
  beta_prog = [0.05, 0.1, 0.2, 0.25]; mu_prog = beta_prog
  R0_min = 0; R0_max = 30   

  'this should be deleted to have same params and make comparison more straight-forward'
  if folder == "WS_Epids": 
    beta_prog = np.linspace(0.01,1,7); mu_prog = beta_prog
  if folder == "B-A_Model": 
    beta_prog = np.linspace(0.01,1,14); mu_prog = beta_prog
    p_prog = [0]; R0_min = 0; R0_max = 6  
  if folder == "NN_Conf_Model": 
    beta_prog = [0.05, 0.1, 0.2, 0.25]; mu_prog = beta_prog
    # past parameters: beta_prog = np.linspace(0.01,1,8); mu_prog = beta_prog
    #k_prog = np.arange(2,34,2)    
  if folder == "Caveman_Model": 
    k_prog = np.arange(1,11,2) #https://www.prb.org/about/ -> Europe householdsize = 3
    beta_prog = np.linspace(0.001,1,6); mu_prog = beta_prog
  if folder[:5] == "Overl": 
    beta_prog = [0.05, 0.1, 0.2, 0.25]; mu_prog = beta_prog #np.linspace(0.01,1,4)

  return k_prog, p_prog, beta_prog, mu_prog, R0_min, R0_max 

'for now the difference in name are only for the net'
def func_file_name(folder, adj_or_sir, N, D, p, R0 = -1, m = 0, N0 = 0, beta = 0.111, mu = 1.111):
  from definitions import rhu
  if adj_or_sir == "AdjMat":
    if folder == "B-A_Model": 
      name = folder + "_%s_N%s_D%s_p%s_m%s_N0_%s" % (
      adj_or_sir, N, D, rhu(p,3), m, N0) + \
        ".png"  
      return name
    else: return folder + "_%s_N%s_D%s_p%s.png" % (adj_or_sir, N,D,rhu(p,3)) 

  if adj_or_sir == "SIR":
    return folder + "_%s_R0_%s_N%s_D%s_p%s_beta%s_mu%s"% (
            adj_or_sir, '{:.3f}'.format(R0),
            N,D, rhu(p,3), rhu(beta,3), rhu(mu,3) ) + ".png"

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
  'If mf == True, std mf by choosing @ rnd the num of neighbors'

  import random
  #here's the modifications of the "test_ver1"
  'Number of nodes in the graph'
  N = G.number_of_nodes()
  mean = int(rhu(np.sum([j for (i,j) in G.degree() ]) / G.number_of_nodes()))

  'Label the individual wrt to the # of the node'
  node_labels = G.nodes()
  
  'Currently infected individuals and the future infected and recovered' 
  inf_list = [] #infected node list @ each t
  prevalence = [] # = len(inf_list)/N, i.e. frac of daily infected for every t
  recovered = [] #recovered nodes for a fixed t
  arr_daily_new_inf = [] #arr to computer SD(daily_new_inf(t)) for daily_new_inf(t)!=0

  'Initial Conditions'
  current_state = ['S' for i in node_labels] 
  future_state = ['S' for i in node_labels]
  
  if seed == True: random.seed(0)

  'Selects the seed of the disease'
  seeds = random.sample(node_labels, start_inf)  #without replacement, i.e. not duplicates
  for seed in seeds:
    current_state[seed] = 'I'
    future_state[seed] = 'I'
    inf_list.append(seed)

  'initilize prevalence and revocered list'
  'trying set prevalence as the daily_new_inf as in the article'
  prevalence = [0] #[len(inf_list)/N] 
  recovered = [0]
  cum_prevalence = [start_inf/N]
  num_susc = [N-start_inf]

  'start and continue whenever there s 1 infected'
  while(len(inf_list)>0):        
    daily_new_inf = 0
    'Infection Phase: inf_list = prev_time infecteds'
    'each infected tries to infect all of the neighbors'
    for i in inf_list:
        'Select the neighbors of the infected node'
        if not mf: tests = G.neighbors(i) #only != wrt to the SIS: contact are taken from G.neighbors            
        if mf: 
          ls = list(range(N)); ls.remove(i)
          tests = random.choices(ls, k = int(mean)) #spread very fast since multiple infected center
        tests = [int(x) for x in tests] #convert 35.0 into int
        for j in tests:
            'If the contact is susceptible and not infected by another node in the future_state, try to infect it'
            if current_state[j] == 'S' and future_state[j] == 'S':
                if random.random() < beta:
                    future_state[j] = 'I'; daily_new_inf += 1     
                else:
                    future_state[j] = 'S'
    
    #print("dail_n_i", daily_new_inf)
    if daily_new_inf != 0: arr_daily_new_inf.append(daily_new_inf)
    #print("arr_ni", arr_daily_new_inf)

    'Recovery Phase: only the prev inf nodes (=inf_list) recovers with probability mu'
    'not the new infected'        
    for i in inf_list:
        if random.random() < mu:
            future_state[i] = 'R'
        else:
            future_state[i] = 'I'
    
    'Time update: once infections and recovery ended, we move to the next time-step'
    'The future state becomes the current one'
    current_state = future_state.copy() #w/o .copy() it's a mofiable-"view"
    
    'Updates inf_list with the currently fraction of inf/rec and save lenS to avg_R' 
    inf_list = [i for i, x in enumerate(current_state) if x == 'I']
    rec_list = [i for i, x in enumerate(current_state) if x == 'R']

    'Saves the fraction of infected and recovered in the current time-step'
    prevalence.append(daily_new_inf/float(N))
    #prevalence.append(len(inf_list)/float(N))
    #recovered.append(len(rec_list)/float(N))
    cum_prevalence.append(cum_prevalence[-1]+daily_new_inf/N)  
    #loop +=1;
    #print("loop:", loop, cum_total_inf, cum_total_inf[-1], daily_new_inf/N)

    num_susc.append(N*(1 - prevalence[-1] - recovered[-1]))
    #print("\nnum_susc, prevalence, recovered",num_susc, prevalence, recovered, 
    #len(num_susc), len(prevalence), len(recovered))
  
  if not mf:
    ddof = 0
    if len(arr_daily_new_inf) > 1: ddof = 1
    if arr_daily_new_inf == []: arr_daily_new_inf = 0
    avg_dn_inf = np.mean(arr_daily_new_inf)
    std_dn_inf = np.std( arr_daily_new_inf, ddof = ddof )
    #print("dni, arr_daily, std", daily_new_inf, arr_daily_new_inf, std_dn_inf)

    degrees = [j for i,j in G.degree()]
    sum_degrees = np.sum(degrees)
    #print("sum_degrees", sum_degrees)
    D = sum_degrees / N

    #print("R0, b,m,D", beta*D/mu, beta, mu, D)
    avg_R = beta*D/(mu*num_susc[0])*(np.sum(num_susc))/len(prevalence)
    #print("num_su[0], np.sum(num_susc), len(prev), avg_R2", \
    #  num_susc[0],np.sum(num_susc), len(prevalence), avg_R)

    return avg_R, avg_dn_inf, std_dn_inf, prevalence, recovered, cum_prevalence
  
  return prevalence, recovered, cum_prevalence
  
  
  #return avg_R, prevalence, recovered, cum_prevalence

def itermean_sir(G, mf = False, numb_iter = 200, beta = 1e-3, mu = 0.05, start_inf = 10,verbose = False):
  'def a function that iters numb_iter and make an avg of the trajectories'
  from itertools import zip_longest
  import numpy as np
  import datetime as dt
  import copy

  numb_idx_cl = 3
  trajectories = [[] for _ in range(numb_idx_cl)]
  std_traj = [[] for _ in range(numb_idx_cl)]
  avg_std_traj = [[] for _ in range(numb_idx_cl)]
  
  avg = [[] for _ in range(numb_idx_cl)]
  itermean_R = 0; itermean_std_dn_inf = 0
  counts = [[],[],[]]
  max_len = 0
  start_time = dt.datetime.now()

  'find the maximum time of 1 scenario among numb_iter ones'
  for i in range(numb_iter):
    sir_start_time = dt.datetime.now()
    if not mf:
      avg_R, avg_dn_inf, std_dn_inf, prev, rec, cum_prev = sir(G, beta = beta, mu = mu, start_inf = start_inf, mf = mf)
      #avg_R, avg_dn_inf, std_dn_inf, prev, rec, cum_prev \
      #  = 1,3,5,[1,2],[3,4,5],[6,7,8,9]
      itermean_R += avg_R / max(1,numb_iter)
      itermean_std_dn_inf += std_dn_inf / max(1,numb_iter)
    else: prev, rec, cum_prev = sir(G, beta = beta, mu = mu, start_inf = start_inf, mf = mf)
    
    tmp_traj = prev, rec, cum_prev
    if (i+1) % 50 == 0: 
      time_1sir = dt.datetime.now()-sir_start_time
      #print("The time for 1 sir is", time_1sir)
      time_50sir = dt.datetime.now()-start_time
      print("Total time for %s its of max-for-loop %s. Time for 1 sir %s" % (i+1, time_50sir, time_1sir))
      if not mf:
        print("After %s its: avg_dn_inf, std_dn_inf, std_dn_inf / max(1,numb_iter), itermean_std:\n %s" %
          (i+1, (avg_dn_inf, std_dn_inf, std_dn_inf / max(1,numb_iter), itermean_std_dn_inf)) 
        )
      start_time = dt.datetime.now()
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
    if (i+1) % 50 == 0: 
        time_50sir = dt.datetime.now()-start_time
        print("Total time for %s its of avg-for-loop %s. Time for 1 sir %s" % (i+1, time_50sir, time_1sir))
        start_time = dt.datetime.now()
    for idx_cl in range(numb_idx_cl):
      'create a list repeating the last element to reach a len of max_len'
      last_el_list = [trajectories[idx_cl][i][-1] for _ in range(max_len-len(trajectories[idx_cl][i]))]
      'traj[classes to be considered, e.g. infected = 0][precise iteration we want, e.g. "-1"]'
      'add the last_el_list to the starting one'
      trajectories[idx_cl][i] += last_el_list
      length = len(trajectories[idx_cl][i]) #should be max_len
      'find the avg = mean of the trajectories for each class'
      'sum all the value at the same position, e.g. all 0th'
      #it_sum = [sum(x) for x in zip_longest(*trajectories[idx_cl], fillvalue=0)]
      #'create counts'
      #counts = [[numb_iter],[numb_iter],[numb_iter]]
      #avg[idx_cl] = list(np.divide(it_sum,counts[idx_cl]))
      
      if verbose:
        print("\niteration(s):", i, "idx_cl ", idx_cl)
        print("last el extension", last_el_list)
        print("(new) trajectories[%s]: %s" % (idx_cl, trajectories[idx_cl]))
        print( "--> trajectories[%s][%s]: %s" % (idx_cl, i, trajectories[idx_cl][i]), 
        "len:", length)
        print("zip_longest same index" , list(zip_longest(*trajectories[idx_cl], fillvalue=0)))#"and traj_idx_cl", trajectories[idx_cl])
        print("global sum indeces", it_sum)
        print("counts of made its", counts[idx_cl])
        print("avg", avg)
      if length != max_len: raise Exception("Error: %s not max_len of %s-th it" % (length,i))
    if i == 199: print("End of avg on 200 scenarios")

  print("\nNow compute the std of the avg")
  for idx_cl in range(numb_idx_cl):    
    avg[idx_cl] = np.mean(trajectories[idx_cl], axis = 0) 
    std_traj[idx_cl] = np.std(trajectories[idx_cl], axis = 0)
    avg_std_traj[idx_cl] = np.mean(std_traj[idx_cl])
    #print("idx_cl %s, avg %s, std %s and avg_std %s" \
    #  % (idx_cl, avg[idx_cl], std_traj[idx_cl], std_traj[idx_cl][-1]) )
  print("End idx_cl %s round"%idx_cl)
  
  if not mf: return itermean_R, itermean_std_dn_inf, plot_trajectories, avg, std_traj
  return plot_trajectories, avg, std_traj

def plot_sir(G, ax, folder = None, beta = 1e-3, mu = 0.05, start_inf = 10, numb_iter = 200):

  'D = numb acts only in mf_avg'
  import itertools
  # MF_SIR: beta = 1e-3, MF_SIR: mu = 0.05
  D = int(rhu(np.sum([j for (i,j) in G.degree() ]) / G.number_of_nodes()))
  N = G.number_of_nodes(); R0 = beta*D/mu

  'plot ratio of daily infected and daily cumulative recovered'
  'Inf and Cum_Infected from Net_Sir; Recovered from MF_SIR'
  print("\nNetwork-SIR loading...")
  itermean_R_net, itermean_std_inf_net, trajectories, avg, std_traj = itermean_sir(G, mf = False, beta = beta, mu = mu, start_inf  = start_inf, numb_iter=numb_iter)
  print("Final itermean_std_inf_net", itermean_std_inf_net)
  print("\nMF-SIR loading...")
  mf_trajectories, mf_avg, mf_std_traj = itermean_sir(G, mf = True, mu = mu, beta = beta, start_inf = start_inf, numb_iter = numb_iter)
  'plotting the many realisations'    
  colors = ["paleturquoise","wheat","lightgreen", "thistle"]
  
  for j in range(numb_iter):
    ax.plot(trajectories[0][j], color = colors[0])
    ax.plot(mf_trajectories[0][j], color = colors[3])
    ax.plot(mf_trajectories[2][j], color = colors[1])
    ax.plot(trajectories[2][j], color = colors[2])
    
    '''if R0 <= 2 and folder == "NNR_Conf_Model":
      'to set legend above the plot'
      y_max = np.max( np.concatenate((
          np.max(trajectories[2][j]), mf_trajectories[2][j],
          np.max(trajectories[0][j]) ), axis = None ) )'''

  ax.plot(mf_avg[0], label="MF::NDaily_Inf/N ", \
    color = "darkviolet") #prevalence
  ax.plot(avg[0], label="Net::NDayly_Inf/N ", \
    color = "tab:blue") #prevalence
  ax.plot(mf_avg[2], label="MF::Sum_NDI/N (%s%%\pm%s%%)"\
    % (np.round(mf_avg[2][-1]*100,1), np.round(mf_std_traj[2][-1]*100,1) ), \
    color = "tab:orange" ) #mf::cd_inf
  ax.plot(avg[2], label="Net::Sum_NDI/N (%s%%\pm%s%%)" %
    (np.round(avg[2][-1]*100,1), np.round(std_traj[2][-1]*100,1) ), \
    color = "tab:green") #net::cd_inf

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

    handles, labels = plt.gca().get_legend_handles_labels()
    order = [2,3,0,1,4]
    ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order],
              bbox_to_anchor=(1, 1), edgecolor="grey", loc='upper right') #add: leg = 
  else: 
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [2,3,0,1,4]
    ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order],loc="best"); 'set legend in the "best" mat plot lib location'

  return itermean_R_net, itermean_std_inf_net

def rhu(n, decimals=0): #round_half_up
    import math
    multiplier = 10 ** decimals
    return math.floor(n*multiplier + 0.5) / multiplier
  
def plot_save_net(G, folder, p = 0, m = 0, N0 = 0, done_iterations = 1, log_dd = False, partition = None, pos = None):
  import os.path
  from definitions import my_dir, func_file_name
  from functools import reduce
  import networkx as nx
  from scipy.stats import poisson
  
  mode = "a"
  #if done_iterations == 1: mode = "w"
  my_dir = my_dir() #"/home/hal21/MEGAsync/Thesis/NetSciThesis/Project/Plots/Tests/"
  N = G.number_of_nodes()
  D = rhu(np.sum([j for (i,j) in G.degree() ]) / G.number_of_nodes() , 2)
  rhuD = rhu(D)
  print("D%s VS rhuD %s" % (D,rhuD))
  #D = rhu( D)
  adj_or_sir = "AdjMat"
  if nx.is_connected(G): avg_l = nx.average_shortest_path_length(G)
  else: avg_l = 0

  'find the major hub and the "ousiders", i.e. highly connected nodes'
  infos = G.degree()
  dsc_sorted_nodes = sorted( infos, key = lambda x: x[1], reverse=True)
  _, max_degree = dsc_sorted_nodes[0]
  i,count_outsiders, threshold = 0,0,3*D
  while( list(map(lambda x: x[1], dsc_sorted_nodes))[i] >  threshold):
    count_outsiders += 1
    i+=1
  print("Outsiders", count_outsiders)

  log_upper_path = my_dir + folder + "/" #"../Plots/Tests/WS_Epids/"
  my_dir+=folder+"/p%s/"%rhu(p,3)+adj_or_sir+"/" #"../Plots/Tests/WS_Epids/p0.001/AdjMat/"
  file_name = func_file_name(folder = folder, adj_or_sir = adj_or_sir, \
    N = N, D = rhuD, p = p, m = m, N0 = N0)
  '''
  folder + "_N%s_D%s_k_{max}%s_m%s_N0_%s_p%s" % (
    N, D, max_degree, m, N0, rhu(p,3) ) + \
      ".png"
  '''
  file_path = my_dir + file_name #../Plot/Test/AdjMat/AdjMat_N1000_D500_p0.001.png
  log_path = log_upper_path + folder + f"_log_saved_{adj_or_sir}.txt" #"../Plots/Tests/WS_Epids/WS_Epids_log_saved_nets.txt"
  nc_path = log_upper_path + folder + "_not_connected_nets.txt"

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
  
  'start with degree distribution'
  'set edges width according to how many "long_range_edges'
  width = 0.8
  long_range_edges = list(filter( lambda x: x > 30, [np.min((np.abs(i-j),np.abs(j-i))) for i,j in G.edges()] )) #list( filter(lambda x: x > 0, )
  length_long_range = len(long_range_edges)
  if length_long_range < 20: print("\nLong_range_edges", long_range_edges, length_long_range)
  else: print("len(long_range_edges)", length_long_range)
  folders = ["WS_Pruned"]
  if folder in folders: width = 0.2*N/len(long_range_edges)
  if folder == "B-A_Model": width = 0.2*N/len(long_range_edges); print("The edge width is", int(width*10)/10)
  if folder== "Caveman_Model":
    nx.draw(G, pos, node_color=list(partition.values()), node_size = 5, width = 0.5, with_labels = False)
  else: nx.draw_circular(G, ax=ax, with_labels=False, font_size=20, node_size=25, width=width)

  #ax.text(0,1,transform=ax.transAxes, s = "D:%s" % D)

  'set xticks of the degree distribution to be centered'
  sorted_degree = np.sort([G.degree(n) for n in G.nodes()])

  'degree distribution + possonian distr (check mean <-> D)'
  bins = np.arange(sorted_degree[0]-1,sorted_degree[-1]+2)
  mean = rhuD #rhu (2.6 -> 3) is right, since axs.hist exclude the highest ( [0,1) ) 
              #in the count of "n"
  y = poisson.pmf(bins, D)

  axs = plt.subplot(212)

  'count how many sorted_degree there are in the bins. Then, align them on the left,'
  'so they are centered in the int'
  n, hist_bins, _ = axs.hist(sorted_degree, bins = bins, \
                                        log = log_dd, density=0, color="green", ec="black", 
                                        lw=1, align="left", label = "Degrees Distr")
  hist_mean = n[np.where(hist_bins == mean)]; pois_mean = poisson.pmf(rhuD, D)
  'useful but deletable print'
  axs.plot(bins, y * hist_mean / pois_mean, "bo--", lw = 2, label = "Poissonian Distr")
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

  '''
  if you need to automatize the procedure. Try with,
  string = string.strip("".join((folder,"_")))
  string = "".join(("r\"$", string,"$\""))
  '''

  if folder == "B-A_Model": 
    plt.suptitle(r"$N:%s, D:%s, k_{max}: %s, N_{3-out}: %s, m: %s, N_0: %s, p:%s$" % (
    N, D, max_degree, count_outsiders, m, N0, rhu(p,3),  ))
  else: plt.suptitle(r"$N:%s, D:%s, N_{3-out}: %s, p:%s, SW_{coeff}:%s$"
                    % (N,D, count_outsiders, rhu(p,3), rhu(avg_l / np.log(N)) ))

  'TO SAVE PLOTS'
  if not os.path.exists(my_dir): os.makedirs(my_dir)
  plt.savefig(file_path)
  plt.close()

  'save not_connected_nets'
  if not nx.is_connected(G):
    sorted_disc_components = sorted(nx.connected_components(G), key=len, reverse=True)
    with open(nc_path, mode) as nc_file:
      nc_file.write("".join((file_name, " #disc_comp: %s" % len(sorted_disc_components), "\n")))

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

def plot_save_sir(G, folder, std_pmbD_dic, done_iterations = 1, p = 0, beta = 0.001, mu = 0.16, R0_max = 16,  start_inf = 10, numb_iter = 200):
  import os.path
  from definitions import my_dir, func_file_name
  import datetime as dt
  import matplotlib.pylab as plt
  start_time = dt.datetime.now()

  mode = "a"
  #if done_iterations == 1: mode = "w"
  my_dir = my_dir() #"/home/hal21/MEGAsync/Thesis/NetSciThesis/Project/Plots/Tests/"
  N = G.number_of_nodes()
  D = np.sum([j for (i,j) in G.degree()]) / G.number_of_nodes()
  adj_or_sir = "SIR"
  'find the major hub and the "ousiders", i.e. highly connected nodes'
  infos = G.degree()
  dsc_sorted_nodes = sorted( infos, key = lambda x: x[1], reverse=True)
  i,count_outsiders, threshold = 0,0,3*D
  while( list(map(lambda x: x[1], dsc_sorted_nodes))[i] >  threshold):
    count_outsiders += 1
    i+=1
  print("Outsiders", count_outsiders)

  'directiories'
  log_upper_path = my_dir + folder + "/" #../Plots/Tests/WS_Epids/
  my_dir+=folder+"/p%s/"%rhu(p,3)+adj_or_sir+"/" #"../Plots/Test/WS_Epids/p0.001/SIR/"
  #file_path depends on a "r0_folder"
  log_path = log_upper_path + folder + "_log_saved_SIR.txt" #"../Plots/Test/WS_Epids/SIR_log_saved_SIR.txt"

  plot_params()
  intervals = [x for x in np.arange(R0_max+1)]
  N = G.number_of_nodes()
  R0 = beta * D / mu
  dict_std_inf = {}

  for i in range(len(intervals)-1):
    if intervals[i] <= R0 < intervals[i+1]:
      'Intro R0-subfolder since R0 det epids behaviour on a fixed net'
      r0_folder = "beta_%s/mu%s/" % (rhu(beta,3),rhu(mu,3)) #"R0_%s-%s/" % (intervals[i], intervals[i+1])
      if folder == "WS_Pruned": r0_folder += "mu%s/" % (rhu(mu,3)) #"R0_1-2/mu0.16/"
      #if folder == "WS_Epids": r0_folder += "D%s/" % rhuD  #"R0_1-2/mu0.16/D6/"
      if not os.path.exists(my_dir + r0_folder): os.makedirs(my_dir + r0_folder)
      if not os.path.exists(my_dir + r0_folder + "/Sel_R0/"): os.makedirs(my_dir + r0_folder + "/Sel_R0/")
      file_name = func_file_name(folder = folder, adj_or_sir = adj_or_sir, \
        N = N, D = rhu(D,1), R0 = rhu(R0,3), p = p, beta = beta, mu = mu)
      '''
      file_name = folder + "_%s_R0_%s_N%s_rhuD%s_p%s_beta%s_mu%s"% (
            adj_or_sir, '{:.3f}'.format(rhu(beta/mu*rhuD,3)),
            N,rhuD, rhu(p,3), rhu(beta,3), rhu(mu,3) ) + ".png"
      '''
      file_path = my_dir + r0_folder + file_name
      
      'plot all'
      _, ax = plt.subplots(figsize = (20,12))

      'plot always sir'
      rhuD2 = rhu(D,2)
      print("\nThe model has N: %s, D: %s, beta: %s, mu: %s, p: %s, R0: %s" % (N,rhuD2,rhu(beta,3),rhu(mu,3),rhu(p,3),rhu(R0,3)) )
      itermean_R_net, itermean_std_inf_net = \
        plot_sir(G, ax=ax, folder = folder, beta = beta, mu = mu, start_inf = start_inf, numb_iter = numb_iter)
      plt.subplots_adjust(
      top=0.920,
      bottom=0.151,
      left=0.086,
      right=0.992,
      hspace=0.2,
      wspace=0.2)
      plt.suptitle(r"$R_0:%s, \bar{R}_{net}:%s, D_{%s}:%s, N_{3-out}: %s, p:%s, \beta:%s, \mu:%s$"
      % (rhu(R0,3),rhu(itermean_R_net,3), N, rhuD2, count_outsiders, rhu(p,3), rhu(beta,3), rhu(mu,3), ))
      #plt.show()
      plt.savefig( file_path )
      print("time 1_plot_save_sir:", dt.datetime.now()-start_time) 

      with open(log_path, mode) as text_file: #write only 1 time
              text_file.write(file_name + "\n")
            
      plt.close()


      'overwrite overy update in std_inf to look @ it in run-time'
      del my_dir
      from definitions import my_dir; import json; from definitions import NestedDict
      std_pmbD_dic = NestedDict(std_pmbD_dic)
      value = itermean_std_inf_net
      print("\nTo be added pmbD itermean:", p, mu, beta, D, value)
      
      d = std_pmbD_dic #rename std_pmbD_dic to have compact wrinting
      if p in d.keys():
        if mu in d[p].keys():
            if beta in d[p][mu].keys():
                d[p][mu][beta][D] = value
            else: d[p][mu] = { **d[p][mu], **{beta: {D:value}} }
        else: d[p] = {**d[p], **{mu:{beta:{D:value}}} }
      else:
        d[p][mu][beta][D] = value

      pp_std_pmbD_dic = json.dumps(std_pmbD_dic, sort_keys=False, indent=4)
      print(pp_std_pmbD_dic)

      fixed_std = std_pmbD_dic[p][mu][beta]
      #print("std_pmbD_dic, p0, mu0, beta0, fixed_std", \
      #  std_pmbD_dic, p, mu, beta, fixed_std)
      x = sorted(fixed_std.keys())
      #print("x", x)
      y = [fixed_std[i] for i in x]
      #print("y", y)
      plt.plot(x,y,'-*')
      #plt.show()
      std_path = my_dir() + folder + "/Std/"
      if not os.path.exists(std_path): os.makedirs(std_path)
      plt.savefig("".join((std_path,"std_p%s_mu%s_beta%s.png" % (p,rhu(mu,3),rhu(beta,3)))))

      std_file = "".join((std_path,"saved_std_dicts.txt"))
      with open(std_file, 'w') as file:
        file.write(pp_std_pmbD_dic) # use `json.loads` to do the reverse
      
      plt.close()


      


  if os.path.exists(log_path):
    'sort line to have the new ones at first'
    sorted_lines = []
    with open(log_path, 'r') as r:
      for line in sorted(r):
        sorted_lines.append(line)
    
    with open(log_path, 'w') as r:
      for line in sorted_lines:
        r.write(line)

def already_saved_list(folder, adj_or_sir, chr_min, my_print = True, done_iterations = 1):
  from definitions import my_dir
  log_upper_path = "".join((my_dir(),folder,"/")) #../Plots/Test/Overlapping.../
  log_path = "".join((log_upper_path, folder, f"_log_saved_{adj_or_sir}.txt"))

  saved_list = []
  if os.path.exists(log_path):
    with open(log_path, "r") as file:
      saved_list = [l.rstrip("\n")[chr_min:] for l in file]
  if my_print: print(f"\nThe already saved {adj_or_sir} are", saved_list)
  return saved_list

def plot_save_nes(
  G, p, folder, adj_or_sir, R0_max = 12, m = 0, N0 = 0, 
  beta = 0.3, mu = 0.3, my_print = True, pos = None, 
  partition = None, dsc_sorted_nodes = False, done_iterations = 1, 
  chr_min = 0, std_pmbD_dic = 0): #save new_entrys
  'save net only if does not exist in the .txt. So, to overwrite all just delete .txt'
  from definitions import already_saved_list, func_file_name
  D =  np.sum([j for (i,j) in G.degree() ]) / G.number_of_nodes()
  N = G.number_of_nodes()
  R0 = rhu( beta*D/mu, 3)
  if adj_or_sir == "AdjMat": 
    saved_files = already_saved_list(folder, adj_or_sir, chr_min = chr_min, my_print= my_print, done_iterations= done_iterations)
    file_name = func_file_name(folder = folder, adj_or_sir = adj_or_sir, \
    N = N, D = rhu(D), R0 = R0, p = p, m = m, N0 = N0)
  if adj_or_sir == "SIR": 
    saved_files = already_saved_list(folder, adj_or_sir, chr_min = chr_min, my_print= my_print, done_iterations=done_iterations)
    file_name = func_file_name(folder = folder, adj_or_sir = adj_or_sir, \
    N = N, D = rhu(D,1), R0 = R0, p = p, m = m, N0 = N0, beta = beta, mu = mu)
  if file_name not in saved_files: 
    print("I'm saving", file_name)
    infos_sorted_nodes(G, num_sorted_nodes = True)
    if adj_or_sir == "AdjMat": 
      plot_save_net(G = G, pos = pos, partition = partition, m = m, N0 = N0, 
      folder = folder, p = p, done_iterations = done_iterations)
      infos_sorted_nodes(G, num_sorted_nodes = 0)
    if adj_or_sir == "SIR": 
      plot_save_sir(G, folder = folder, beta = beta, mu = mu, p = p, R0_max = R0_max, 
      done_iterations = done_iterations, std_pmbD_dic = std_pmbD_dic)

def save_log_params(folder, text, done_iterations = 1):
  import os
  from definitions import my_dir
  print("log_params is @:", my_dir() + folder)
  #if done_iterations == 1:
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

'no needed anymore -- delete it'
def ws_sir(G, folder, p, saved_nets, done_iterations, pruning = False, infos = False, beta = 0.001, mu = 0.16, start_inf = 10):    
  'round_half_up D for a better approximation of nx.c_w_s_graph+sir'
  N = G.number_of_nodes()
  rhuD = rhu(np.sum([j for (i,j) in G.degree() ]) / G.number_of_nodes())
  if infos == True: check_loops_parallel_edges(G); infos_sorted_nodes(G, num_sorted_nodes = False)

  if "N%s_D%s_p%s"% (N,rhuD,rhu(p,3)) not in saved_nets: 
    plot_save_net(G = G, folder = folder, p = p, done_iterations = done_iterations)
    saved_nets.append("N%s_D%s_p%s" % (N,rhuD,rhu(p,3)))
    print("saved nets", saved_nets)
  plot_save_sir(G = G, folder = folder, beta = beta, mu = mu, p = p, start_inf = start_inf, done_iterations = done_iterations )

'===Configurational Model'
def pois_pos_degrees(D, N, L = int(2e3)):
  'Draw N degrees from a Poissonian sequence with lambda = D and length L'
  degs = np.random.poisson(lam = D, size = L)
  print(D)
  if D < 1: 
    print(degs)
    custom_degs = list(map(lambda x: 0 if x == -1 else x, degs))
    print("\n custom", custom_degs)
    custom_degs = [x for x in degs if x >= 0]
  else: custom_degs = [x for x in degs if x >= 0]
  pos_degrees = np.random.choice(degs, N)

  #print("len(s) in deg", len([x for x in degs if x == 0]))
  #print("len(s) in pos_degrees", len([x for x in pos_degrees if x == 0]))
  return pos_degrees

def config_pois_model(N, D, seed = 123, folder = None):
  '''create a network with the node degrees drawn from a poissonian with even sum of degrees'''
  #seed = int(np.random.random_sample()*100)
  np.random.seed(seed)
  degrees = pois_pos_degrees(D,N) #poiss distr with deg != 0
  print("\nseed1 %s, sum(degrees) %s" % (seed, np.sum(degrees)%2))
  'change seed to have a even sum of the degrees'
  while(np.sum(degrees)%2 != 0):
    seed+=1# int(np.random.random_sample()*100) #1
    np.random.seed(seed)
    print("\nseed2 %s" % (seed))
    degrees = pois_pos_degrees(D,N)
  #print("Degree sum:", np.sum(degrees), "with seed:", seed)

  print("\nNetwork Created but w/o standard neighbors wiring!")
  G = nx.configuration_model(degrees, seed = seed)

  'If D/N !<< 1, by removing loops and parallel edges, we lost degrees. Ex. with N = 50 = D, <k> = 28 != 49.8'
  check_loops_parallel_edges(G)
  remove_loops_parallel_edges(G)
  infos_sorted_nodes(G)
  print("End of %s " % folder)
  return G

'===def of net functions'
def long_range_edge_add(G, p = 0, time_int = False):
  from itertools import chain
  import random
  import datetime as dt
  from definitions import replace_edges_from, remove_loops_parallel_edges
  'add an long-range edge over the existing ones'
  
  if time_int: start_time = dt.datetime.now()
  
  all_edges = [list(G.edges(node)) for node in pos_deg_nodes(G)]
  all_edges = list(chain.from_iterable(all_edges))
  initial_length = len(all_edges)
  if p != 0:
    for node in pos_deg_nodes(G):
      left_nodes = list(pos_deg_nodes(G))
      left_nodes.remove(node) 
      re_link = random.choice( left_nodes )
      if random.random() < p:
          all_edges.append((node,re_link))
  print("len(all_edges)_final", len(all_edges), "is? equal to start", initial_length )
  replace_edges_from(G, all_edges)
  remove_loops_parallel_edges(G, False)
  if time_int: print(f"Time for add edges over ddistr:", dt.datetime.now()-start_time)

def NN_pois_net(N, ext_D, p = 0):
  'p is not used right now'
  from definitions import config_pois_model
  import numpy as np
 
  G = config_pois_model(N, ext_D, seed = 123, folder = None)

  verbose = False
  def verboseprint(*args):
    if verbose == True:
      print(*args)
    elif verbose == False:
      None

  'for random rewiring with p'
  l_nodes = [x for x in pos_deg_nodes(G)]

  edges = set() #avoid to put same link twice (+ unordered)
  nodes_degree = {}

  'list of the nodes sorted by their degree'
  for node in pos_deg_nodes(G):
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
    if nodes_degree[node] == 0 and node in l_nodes and node in sorted_nodes: 
      ls_nodes_remove(node)
      verboseprint("\n", get_var_name(node), "=", node, "is removed via if deg == 0")

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
  infos_sorted_nodes(G, num_sorted_nodes=False)
  
  long_range_edge_add(G, p = p)

  '''
  CUT OFF THE DISC COMPONENT SINCE WE WANT TO PRESERVE SOLELY NODES

  'there is only a node with 2 degree left. So, if rewired correctly only a +1 in the ddistr'
  'So, connect all the disconnected components'
  its = 0
  sorted_disc_components = sorted(nx.connected_components(G), key=len, reverse=True)
  for c in sorted_disc_components: #set of conn_comp
    if its == 0: 
      base_node = np.random.choice(([x for x in c])); 
    else: 
      linking_node = np.random.choice(([x for x in c]))
      G.add_edge(linking_node,base_node)
      base_node = linking_node
    its += 1
  #print("Total links to have", len(list(nx.connected_components(G))),"connected component are", its)
  if len(list(nx.connected_components(G)))>1: print("Disconnected net!")
  '''
  print(f"There are {len([j for i,j in G.degree() if j == 0])} 0 degree node as")
  print("End of wiring")
  

  return G

def NN_Overl_pois_net(N, ext_D, p, add_edges_only = False):
  from itertools import chain
  from definitions import replace_edges_from, remove_loops_parallel_edges
  import datetime as dt
  import random

  def edges_node(x):
      return [(i,j) for i,j in all_edges if i == x]

  G = config_pois_model(N,ext_D)
  adeg_ConfM = sum([j for i,j in G.degree()])/G.number_of_nodes()

  'rewire left and right for the max even degree'
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

  D = np.sum([j for i,j in G.degree()])/G.number_of_nodes()
  if D != adeg_ConfM: print("!! avg_deg_Overl_Rew - avg_deg_Conf_Model = ", D - adeg_ConfM)
  
  if add_edges_only and p != 0: #add on top of the distr, a new long-range edge
    long_range_edge_add(G, p = p)

  else: #remove local edge and add long-range one
    start_time = dt.datetime.now()
    all_edges = [list(G.edges(node)) for node in pos_deg_nodes(G)]
    all_edges = list(chain.from_iterable(all_edges))
    initial_length = len(all_edges)
    for node in pos_deg_nodes(G):
      left_nodes = list(pos_deg_nodes(G))
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
  print("Rel Error wrt to ext_D %s %%" % (rhu((adeg_OR - ext_D)/ext_D,1 )*100))
  print("Rel Error wrt to adeg_ConfM %s %%" % (rhu( (adeg_OR - adeg_ConfM)/adeg_ConfM,1 )*100))
  
  # from definitions import NN_pois_net
  # while(not nx.is_connected(G)): G = NN_pois_net(N, ext_D = D)
  
  return G

'Net Infos'
def check_loops_parallel_edges(G):
  ls = list(G.edges())
  print("parallel edges", set([i for i in ls for j in ls[ls.index(i)+1:] if i==j]),
        "; loops", [(i,j) for (i,j) in set(G.edges()) if i == j])

def infos_sorted_nodes(G, num_sorted_nodes = False):
    import networkx as nx
    'sort nodes by key = degree. printing order: node, adjacent nodes, degree'
    nodes = pos_deg_nodes(G)
    print("<k>: ", np.sum([j for (i,j) in G.degree() ]) / len(nodes), 
          " and <k>/N ", np.sum([j for (i,j) in G.degree() ]) / len(nodes)**2, end="\n" )
    
    'put adj_matrix into dic for better visualisation'
    adj_matrix =  nx.adjacency_matrix(G).todense()
    adj_dict = {i: np.nonzero(row)[1].tolist() for i,row in enumerate(adj_matrix)}

    infos = zip([x for x in nodes], [len(adj_dict[i]) for i in range(len(nodes))], [G.degree(x) for x in nodes])
    dsc_sorted_nodes = sorted( infos, key = lambda x: x[2], reverse=True)

    cut_off = 0
    if len(dsc_sorted_nodes) != 0: min(len(dsc_sorted_nodes),4)
    if num_sorted_nodes == True:  
      num_sorted_nodes = len(nodes) 
      for i in range(cut_off):
        if i == 0: print("Triplets of (nodes, neighbors(%s), degree) sorted by descending degree:" % i)
        print( dsc_sorted_nodes[i] )

    if num_sorted_nodes == False: num_sorted_nodes = 0
    
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
                #print("ci,cj", ci,cj)
                edges[(ci, cj)] = [(ni, nj)] #SHOULD BE += [(NI,NJ)] HERE
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

  def comm_caveman_relink(cliques = 8, clique_size = 7, p = 0,  relink_rnd = 0, numb_link_inring = 0):
      import numpy as np
      import numpy.random as npr
      'caveman_graph'
      G = nx.caveman_graph(l = cliques, k = clique_size)

      'decide how many nodes are going to relink to the neighbor "cave" (cfr numb_link_inring'
      total_nodes = clique_size*cliques
      #if numb_link_inring != 0: 
      for clique in range(cliques):
        if numb_link_inring != 0:
          first_cl_node = clique_size*clique
          nodes_inclique = np.arange(first_cl_node, first_cl_node+numb_link_inring)
          attached_nodes = npr.choice( np.arange(clique_size*(1+clique), 
                                      clique_size*(2+clique)), 
                                      size = len(nodes_inclique) )
          attached_nodes = attached_nodes % np.max((total_nodes,1))
          for test, att_node in zip(nodes_inclique, attached_nodes):
              #print("NN - clique add:", (test,att_node))
              G.add_edge(test,att_node)
          
        'add a new edge by relinking one of the existing node'
        'decide how many nodes in the clique would go into rnd relink via relink_rnd'
        'In the last way, avg_degree is preserved'
        if p != 0:
          relink_rnd = clique_size
          first_cl_node = clique_size*clique
          nodes_inclique = np.arange(first_cl_node, first_cl_node + relink_rnd)
          attached_nodes = npr.choice([x for x in pos_deg_nodes(G) if x not in nodes_inclique], 
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

'===Barabasi-Albert Model'
def bam(N,m,N0):
  import random
  '''    
  Arguments:
  1) N: number of nodes in the graph   
  2) m: number of links to be added at each time
  3) N0: starting fully connected clique
  '''
  
  'Creates an empty graph'
  G = nx.Graph()
  
  'adds the N0 initial nodes'
  G.add_nodes_from(range(N0))
  edges = []
  
  'creates the initial clique connecting all the N0 nodes'
  edges = [(i,j) for i in range(N0) for j in range(i,N0) if i!=j]
  
  'adds the initial clique to the network'
  G.add_edges_from(edges)

  'list to store the nodes to be selected for the preferential attachment.'
  'instead of calculating the probability of being selected a trick is used: if a node has degree k, it will appear'
  'k times in the list. This is equivalent to select them according to their probability.'
  prob = []
  
  'runs over all the reamining nodes'
  for i in range(N0,N):
      G.add_node(i)
      'for each new node, creates m new links'
      for j in range(m):
          'creates the list of nodes'
          for k in list(G.nodes):
              'add to prob a node as many time as its degree'
              for _ in range(G.degree(k)):
                  prob.append(k)
          'picks up a random node, so nodes will be selected proportionally to their degree'
          node = random.choice(prob)
          
          G.add_edge(node,i)
      
          'the list must be created from 0 for every link since with every new link probabilities change'
          prob.clear()
  'returns the graph'

  return G

'===STD_Infected'
class NestedDict(dict):
    def __missing__(self, key):
        self[key] = NestedDict()
        return self[key]

'===main, i.e. automatize common part for different nets'
