from datetime import datetime
import networkx as nx
from networkx.algorithms.components.connected import number_connected_components
from networkx.generators.community import caveman_graph
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import matplotlib.cm as cm 
import os #to create a folder

##THESE ARE THE DEFS BEFORE ENCOUTERING THE MASTER OF CS

#config.THREADING_LAYER = "default"

#Thurner pmts: beta = 0.1, mu = 0.16; D = 3 vel 8
#MF def: beta, mu = 0.001/cf, 0.05/cf or 0.16/cf ; cf = 1

#this has been saved in morning of the 3.6.2021

def rmv_folder(folder, rmv_flag = False):
  import os
  from definitions import my_dir
  import sys
  import shutil
  # Get directory name
  if rmv_flag:
    my_dir = my_dir()+folder
    if os.path.exists(my_dir):
      #rmv_flag = input("Really need to delete it (1) or no(0)?")
      if bool(int(rmv_flag)):
        print(f"I'm deleting since b(rmv_flag) is {bool(int(rmv_flag))}")
        try:
          shutil.rmtree(my_dir)
        except OSError as e:
          print("Error: %s - %s." % (e.filename, e.strerror))

def my_dir():
  import datetime as dt
  datetime = str(dt.datetime.now()).replace(" ", "_").replace(":", "")[:-7]+"/"
  #return "/content/drive/MyDrive/Colab_Notebooks/Thesis/Complex_Plots/"
  #return "/content/drive/MyDrive/Colab_Notebooks/Thesis/Complex_Plots/reduced_ver/"
  #return "/content/"
  return "/home/hal21/MEGAsync/Tour_Physics2.0/Thesis/NetSciThesis/Project/Plots/Test/"

def pos_deg_nodes(G): # G "real" nodes
  return np.asarray([i for i,j in G.degree() if j > 0])

def N_D_std_D(G):
  degrees = np.asarray([j for i,j in G.degree()])
  return G.number_of_nodes(), np.mean(degrees), np.std(degrees, ddof = 1)

def fstd_Rc_net(Rc_net, D, avgk2, std_D, std_avgk2):
  from math import sqrt
  coeff0 = (Rc_net / D)**4
  coeff1 = (2*avgk2/D - 1)**2
  return sqrt(coeff0 * (coeff1*std_D**2+std_avgk2**2))

def jsonKeys2int(x):
    if isinstance(x, dict):
        return {float(k):v for k,v in x.items()}
    return x

'for now the difference in name are only for the net'
def func_file_name(folder, adj_or_sir, N, D, p, R0 = -1, m = 0, N0 = 0, beta = 0.111, mu = 1.111):
  from definitions import rhu
  if adj_or_sir == "AdjMat":
    if folder == "BA_Model":
      name = f'{folder}_{adj_or_sir}_{N}_{rhu(D)}_{rhu(p,3)}_{m}_{N0}.png'
      #name = "".join(folder, "_%s_N%s_D%s_p%s_m%s_N0_%s" % (
      #adj_or_sir, N, rhu(D), rhu(p,3), m, N0),".png")  
      return name
    elif folder == "Caveman_Model":
      return f'{folder}_{adj_or_sir}_{N}_{rhu(D,1)}_{rhu(p,3)}.png'
    else: return f'{folder}_{adj_or_sir}_{N}_{rhu(D)}_{rhu(p,3)}.png' #folder + "_%s_N%s_D%s_p%s.png" % (adj_or_sir, N,rhu(D),rhu(p,3)) 
    

  if adj_or_sir == "SIR":
    #if folder == "Caveman_Model":
    #  return f'{folder}_{adj_or_sir}_{N}_{rhu(D,3)}_{rhu(p,3)}.png'
    return folder + "_%s_R0_%s_N%s_D%s_p%s_beta%s_d%s"% (
            adj_or_sir, int(rhu(R0)),
            N,rhu(D), rhu(p,3), rhu(beta,3), rhu(mu**(-1)) ) + ".png"

'===Plot and Save SIR + Net'
def plot_params():
  import matplotlib.pyplot as plt

  'set fontsize for a better visualisation'
  SMALL_SIZE = 40
  MEDIUM_SIZE = 25
  BIGGER_SIZE = 40

  plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
  plt.rc('axes', titlesize=BIGGER_SIZE, labelsize=BIGGER_SIZE, labelpad = 20)     # fontsize of the axes title
  #plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
  plt.rc('xtick', labelsize=MEDIUM_SIZE, direction="in")    # fontsize of the xtick labels
  plt.rc('ytick', labelsize=MEDIUM_SIZE, direction="in")    # fontsize of the ytick labels
  plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
  plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
  #plt.rc('axes', labelpad = 20)
  plt.rc('xtick.major', pad = 16)
  plt.rc('ytick.major', pad = 16)
  plt.rcParams["figure.figsize"] = [32,14]
  plt.rc('axes', edgecolor='black', lw = 1.2)
  #plt.rc("patch", linewidth = 15)

def sir(G, mf = False, beta = 1e-3, mu = 0.05, seed = False, start_inf = 10):
  'If mf == False, the neighbors are not fixed;' 
  'If mf == True, std mf by choosing @ rnd the num of neighbors'

  import random
  #here's the modifications of the "test_ver1"
  'Number of nodes in the graph'
  N, D, _ = N_D_std_D(G)

  'Label the idnividual wrt to the # of the node'
  node_labels = G.nodes()  
  fr_stinf = start_inf/N

  'Currently infected idnividuals and the future infected and recovered' 
  susceptible = [1-fr_stinf]
  prevalence = [fr_stinf]
  recovered = [0]
  #totcases = [fr_stinf]

  inf_list = [] #infected node list @ each t

  dni_cases = [0] #it was dni_cases = [] # = len(inf_list)/N, i.e. frac of daily infected for every t
  dni_totcases = [fr_stinf] #  dni_totcases = [fr_stinf]
  dni_susceptible = [1-fr_stinf] #  dni_susceptible = [1-fr_stinf] 


  pos_dni_cases = [] #arr to computer Std(daily_new_inf(t)) for daily_new_inf(t)!=0

  'Initial Codnitions'
  current_state = ['S' for i in node_labels] 
  future_state = ['S' for i in node_labels]
  
  if seed: 
    random.seed(0)

  'Selects the seed of the disease'
  inf_list = random.sample(node_labels, start_inf)  #without replacement, i.e. not duplicates
  if rhu(D,0)-1 < 0 and mf: #too slow for D = 1
    inf_list = []
  for seed in inf_list:
    current_state[seed] = 'I'
    future_state[seed] = 'I'

  'initilize dni (new daily infected) and recovered list'
  'we dont track infected'

  'start and continue whenever there s 1 infected'
  while(len(inf_list)>0):        
    daily_new_inf = 0
    'Infection Phase: inf_list = prev_time infecteds'
    'each infected tries to infect all of the neighbors'
    for i in inf_list:
        'Select the neighbors of the infected node'
        if not mf: tests = G.neighbors(i) #only != wrt to the SIS: contact are taken from G.neighbors            
        if mf: 
          #tests = G.neighbors(i) -- Watts-Strogatz(p = 1)??
          ls = list(range(N)); ls.remove(i)
          tests = random.sample(ls, k = int(rhu(D))) #spread very fast since multiple infected center
        tests = [int(x) for x in tests] #convert 35.0 into int
        for j in tests:
          if current_state[j] == 'S' and future_state[j] == 'S' and random.random() < beta:
            future_state[j] = 'I'; daily_new_inf += 1
    
    #"+" to join lists
    if daily_new_inf != 0: pos_dni_cases = pos_dni_cases+[daily_new_inf]

    'Recovery Phase: only the prev inf nodes (=inf_list) recovers with probability mu'
    'not the new infected'    
    'This part is important in the OrderPar since diminishes the inf_list # that is in the "while-loop"'    
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

    'Saves the fraction of new daily infected (dni) and recovered in the current time-step'
    prevalence.append(len(inf_list)/float(N))
    #recovered.append(len(rec_list)/float(N))
    #susceptible.append(1-prevalence[-1]-recovered[-1])

    dni_cases.append(daily_new_inf/float(N))        
    dni_totcases.append(dni_totcases[-1]+daily_new_inf/float(N))            
    #totcases.append(totcases[-1]+prevalence[-1]-prevalence[-2]+recovered[-1]-recovered[-2])
    #dni_susceptible.append(1-dni_totcases[-1])

  'Order Parameter (op) = Std(Avg_dni(t)) s.t. dni(t)!=0 as a func of D'
  if not mf:
    ddof = 0
    if len(pos_dni_cases) > 1: ddof = 1
    if pos_dni_cases == []: pos_dni_cases = [0]
    'op = std_dni_cases'
    op = np.std( pos_dni_cases, ddof = ddof )
    if len(pos_dni_cases) > 1:
      if len([x for x in pos_dni_cases if x == 0]) > 0: 
        raise Exception("Error There's 0 dni_cases: dni_cases, arr_daily, std", daily_new_inf, pos_dni_cases, op)

    
    return op, prevalence, dni_cases, dni_totcases

  return prevalence, dni_cases, dni_totcases

def itermean_sir(G, p = 0, mf = False, numb_iter = 200, beta = 1e-2, mu = 0.05, start_inf = 10,verbose = False,):
  'def a function that iters numb_iter and make an avg of all the trajectories'
  from itertools import zip_longest
  from itertools import chain
  from definitions import sir
  import numpy as np
  import datetime as dt
  import copy

  #mi servono solo gli infetti...quidni numb_idx_cl = 2, prev and dni_totcases
  numb_idx_cl = 3; counts = [[],[],[]]; max_len = 0 

  'here avg are "means" among all the iterations'
  'So, better avg <-> itavg'
  trajectories = [[] for _ in range(numb_idx_cl)]
  avg_traj= [[] for _ in range(numb_idx_cl)]
  std_avg_traj = [[] for _ in range(numb_idx_cl)]
  avg_ordp = 0; list_ordp = [] #ordp = std(dni_cases s.t. dni_cases!=0)
  start_time = dt.datetime.now()

  'find the maximum time of 1 scenario among numb_iter ones'
  for i in range(numb_iter):
    'generate 1 sir'
    if not (i+1) % 50: onesir_start_time = dt.datetime.now()
    
    'save in a list std_dni to compute the ordp'
    if not mf:
      std_dni, infected, dni_cases, dni_totcases \
        = sir(G, beta = beta, mu = mu, start_inf = start_inf, mf = mf)
      list_ordp.append(std_dni)
    else: infected, dni_cases, dni_totcases = sir(G, beta = beta, mu = mu, start_inf = start_inf, mf = mf)
    
    def rescale_wrt_ymax(ls):
      #print("ls", ls)
      max_y = max(ls)
      if not max_y: return max_y, ls
      rescaled_ls = [x/max_y for x in ls]
      #print("\nmax_y", max_y, rescaled_ls)
      return max_y, rescaled_ls
  
    'save the 1 sir generated'
    tmp_traj = dni_cases, dni_totcases, infected
    if not (i+1) % 50: 
      'time stamp'
      time_1sir = dt.datetime.now()-onesir_start_time
      time_50sir = dt.datetime.now()-start_time
      print("Total time for %s its of max-for-loop %s. Time for 1 sir %s" % (i+1, time_50sir, time_1sir))
      if not mf:
        print("NET: After %s its: std_dni %s" % (i+1, (std_dni)) 
        )
      start_time = dt.datetime.now()

    'append tmp_traj and search for the max time of one epidemic'
    for idx_cl in range(numb_idx_cl):     #save only infected right? In fact, idx_cl = 0
      trajectories[idx_cl].append(tmp_traj[idx_cl])
  
  max_len = len(max(trajectories[0], key=len))
    
  'Only for nets: compute mean and std of avg_R and std_dni'
  if not mf:
    'order parameter with std'
    avg_ordp = sum(list_ordp) / float(len(list_ordp))
    std_avg_ordp = np.std(list_ordp, ddof = 1)

  plot_trajectories = copy.deepcopy(trajectories)

  start_time = dt.datetime.now()
  for i in range(numb_iter):
    if (i+1) % 50 == 0: 
        time_50sir = dt.datetime.now()-start_time
        print("Total time for %s its of avg-for-loop %s. Time for 1 sir %s" % (i+1, time_50sir, time_1sir))
        start_time = dt.datetime.now()
    for idx_cl in range(numb_idx_cl):
      'create a max-len list repeating the last element'
      'Varying idx_cl, traj[idx][i] has the != len wrt trj[0][i] since its increasing, fel, with +=last_el_list'
      'traj[classes to be considered, e.g. infected = 0][precise iteration we want, e.g. "-1"]'
      old = trajectories[idx_cl][i].copy()
      last_el_list = [trajectories[idx_cl][i][-1]]*(max_len-len(trajectories[idx_cl][i]))
      trajectories[idx_cl][i] = list(chain(old,last_el_list))
      if verbose:
        print("\niteration(s):", i, "idx_cl ", idx_cl)
        print("last el extension", last_el_list)
        print("(new) trajectories[%s]: %s" % (idx_cl, trajectories[idx_cl]))
        print( "--> trajectories[%s][%s]: %s" % (idx_cl, i, trajectories[idx_cl][i]), 
        "len:", length)
        print("zip_longest same index" , list(zip_longest(*trajectories[idx_cl], fillvalue=0)))#"and traj_idx_cl", trajectories[idx_cl])
        print("counts of made its", counts[idx_cl])
        print("avg", avg_traj)

      length = len(trajectories[idx_cl][i]) #should be max_len
      if length != max_len: raise Exception("Error: %s not max_len %s of %s-th it" % (length,max_len,i))
    if i == 199: print("End of avg_traj on 200 scenarios")

  'std_avg_traj != order_param (std(arr_dni)) since prevalence(dni/N) and dni = 0 are allowed'  
  print("\nNow compute the Std of the prevalence")
  for idx_cl in range(numb_idx_cl):    
    avg_traj[idx_cl] = np.mean(trajectories[idx_cl], axis = 0) #avg wrt index, e.g. all 0-indexes
    std_avg_traj[idx_cl] = np.std(trajectories[idx_cl], axis = 0, ddof = 1)
    'compute p(t) s.t. D(1-p(t))lambda = Rc_net'
    N, D, _ = N_D_std_D(G)
    if idx_cl == 1: #idx_cl(dni_totcases) = 1
      R0 = D*beta/mu
      if not mf:
        k2 = [j**2 for _,j in G.degree()]
        avgk2 = np.mean(k2)        
        std_avgk2 = np.std(k2, ddof = 1)
        #print(f'G.degree(): {list(G.degree())}',)
        #print(f'k2: {k2}',)
        print(f'avgk2: {avgk2}',)
        print(f'std_avgk2: {std_avgk2}',)
        Rc_net = D**2/(avgk2-D); lmd = beta/mu
        RcR0 = Rc_net / (R0 - lmd)
        p_c = (1 - RcR0) / (1-p)
      if mf and int(D)>1:
        D = int(D)
        Rc_net = 1/(1-D**(-1))
        RcR0 = Rc_net / R0
        p_c = 1 - RcR0
      elif mf and int(D)<=1: p_c = 0

      
      dni_totcases = np.array(avg_traj[idx_cl])#[1-x for x in avg_traj[idx_cl]]
      t_c = 0
      for i in np.arange(len(dni_totcases)-1):
        if dni_totcases[i] <= p_c < dni_totcases[i+1]:
          print("limit", RcR0, p_c)
          y1 = dni_totcases[i+1]; x1 = np.where(dni_totcases == dni_totcases[i+1])[0][0]
          dy = y1 - dni_totcases[i]; dt = 1
          m = dy/dt
          t_c =  1/m * (p_c - y1) + x1
          #t_c = t_c[0][0]
          print(
            "\n dni_totalcases[i], p_c, dni[i+1],x1, np.where, tc, mu, beta",
            dni_totcases[i],p_c, dni_totcases[i+1], x1, np.where(dni_totcases == dni_totcases[i])[0][0], t_c, mu, beta)
          break
          print("End of t_c")
        #else: t_c = 0

  if not mf: 
    return avg_ordp, std_avg_ordp, plot_trajectories, avg_traj, std_avg_traj, \
                    Rc_net, avgk2, std_avgk2, t_c, p_c,  
  #else: return avg_ordp, std_avg_ordp, plot_trajectories, avg_traj, std_avg_traj,
  return plot_trajectories, avg_traj, std_avg_traj, t_c, p_c,  

def plot_sir(G, ax1, p = 0, folder = None, beta = 1e-3, mu = 0.05, start_inf = 10, numb_iter = 200):

  'D = numb acts only in mf_avg_traj'
  from definitions import rhu
  # MF_SIR: beta = 1300e-3, MF_SIR: mu = 0.05
  N, D, _ = N_D_std_D(G)

  'plot ratio of daily infected and daily cumulative recovered'
  'Inf and Cum_Infected from Net_Sir; Recovered from MF_SIR'
  print("\nNetwork-SIR loading...")
  #old ver: add avg_R_net, std_avg_R_net, 
  avg_ordp_net, std_avg_ordp_net, trajectories, avg_traj, std_avg_traj, Rc_net, avgk2, std_avgk2, t_c, p_c,   = \
    itermean_sir(G, p, mf = False, beta = beta, mu = mu, start_inf  = start_inf, numb_iter=numb_iter,)
  
  print("Final avg_ordp_net", avg_ordp_net)
  print("\nMF-SIR loading...")
  plot_mf = False
  if plot_mf:
    mf_trajectories, mf_avg_traj, mf_std_avg_traj, mf_t_c, mf_p_c = \
    itermean_sir(G, mf = True, mu = mu, beta = beta, start_inf = start_inf, numb_iter = numb_iter,)

  'plotting the many realisations'  
  #from matplotlib import cm
  #green = cmap.get_cmap("Greens")(avg_traj[1][-1] / std_avg_traj[1][-1])
  #orange = cmap.get_cmap("Oranges")(mf_avg_traj[1][-1] / mf_std_avg_traj[1][-1])
  colors = ["paleturquoise", "#BDA27E", "#57D657", "thistle"] #limegreen
  if beta*D/mu <= 1: colors = ["paleturquoise","coral","seagreen", "thistle"]
  ax1.grid(color = "darkgrey", ls = "--", lw = 1)
  ax2 = ax1.twinx()
  ax3 = ax1.twiny()

  lw_traj = 1.3
  for j in range(numb_iter):
    if plot_mf: ax1.plot(mf_trajectories[1][j], color = colors[1], lw = lw_traj) #MF_dni_totcases
    ax1.plot(trajectories[1][j], color = colors[2], lw = lw_traj) #NET_dni_totcases
    ax2.plot(trajectories[0][j], color = colors[0], lw = 0) #NET_dni_cases
    if plot_mf: ax2.plot(mf_trajectories[0][j], color = colors[3], lw = 0) #MF_dni_cases
    #ax2.plot(trajectories[2][j], color = colors[0], lw = 0) #NET_infected
    #ax2.plot(mf_trajectories[2][j], color = colors[3], lw = 0) #MF_infected'''
  
  'define a string_format to choose the best way to format the std of the mean'
  if plot_mf:
    value = mf_std_avg_traj[1][-1]
    string_format = str(np.round(value*100,1))[:3]
    if string_format == "0.0" and np.isclose(value, 0):
      string_format = format(value, ".1e")

  lw_totc, ls_totc = 5, "solid"
  if plot_mf:
    ax3.plot(mf_avg_traj[1], label=r"MF:TotalCases (%s%%$\pm$%s%%)"\
      % (np.round(mf_avg_traj[1][-1]*100,1), string_format ), \
      color = "darkorange", lw = lw_totc, ls = ls_totc)
  ax3.plot(avg_traj[1], label=r"Net:TotalCases (%s%%$\pm$%s%%)" %
    (np.round(avg_traj[1][-1]*100,1), np.round(std_avg_traj[1][-1]*100,1) ), \
    color = "green", lw = lw_totc, ls = ls_totc) #net:cd_inf

  'plot horizontal line to highlight the initial infected'  
  ax1.axhline(start_inf/N, color = "r", ls="dashed", \
            label = "Start_Inf/N (%s%%) "% np.round(start_inf/N*100,1))
  ax2.axhline(start_inf/N, color = "r", ls="dashed")

  def list_replace(lst, old, new):
    """replace list elements (inplace)"""
    lst = list(lst)
    print(lst)
    i = -1
    try:
      while 1:
        i = lst.index(old, i + 1)
        lst[i] = new
    except:
        pass
    return lst

  if plot_mf:
    ax2.plot(mf_avg_traj[0], label="MF:DailyNewInf", \
    color = "darkviolet", lw = lw_totc) #prevalence
  ax2.plot(avg_traj[0], label="Net:DailyNewInf", \
    color = "tab:blue", lw = lw_totc) #prevalence
  
  ax2.plot(avg_traj[2], label="Net:Infected", \
    color = "cadetblue", ms = 8, marker = "o", lw = 0) #prevalence
  
  label = r"Net:$t_c,p_c$" + f" = ({rhu(t_c)}d,{rhu(p_c,2)})"
  ms = 50
  gcolor = "#003312"
  if t_c > 0: 
    ax3.plot(t_c, p_c, color = gcolor, marker = "*", markersize = ms - 10, mec = "k",
            label = label)
  else: ax3.plot([], marker = "*", mfc = gcolor, mec = "k", ms = ms - 10, label = label)
  
  if plot_mf:
    label = r"MF:$t_c,p_c$" + f" = ({rhu(mf_t_c)}d,{rhu(mf_p_c,2)})"
    mfmc = "orange"
    if mf_t_c > 0: ax3.plot(mf_t_c, mf_p_c, color = mfmc, marker = "*", markersize = ms - 10, mec = "black",
              label = label)
    else: ax3.plot([], marker = "*", mfc = mfmc, mec = "k", ms = ms - 10, label = label)

  'exclude first and last yticks'
  
  import matplotlib.pylab as plt
  locs = ax1.get_yticks()
  ax1.set_yticks(locs[1:-1])
  ax1.set_yticklabels(np.round(locs[1:-1],2), color='#003312')

  locs = ax2.get_yticks()
  ax2.set_yticks(locs[1:-1])
  ax2.set_yticklabels(np.round(locs[1:-1],3), color='darkblue')
  
  ax3.set_xticklabels([])

  #'plot labels'
  ax1.set_xlabel('Time-steps')
  ax1.set_ylabel('Indivs/N')
  #for ax in [ax1,ax2,ax3]:
  #  ax.set_yscale("linear")

  'plotting figsize depending on legend'
  R0 = beta*D/mu
  
  'set legend above the plot if R_0 in [0,2] in the NNR_Config_Model'
  ax2.set_zorder(0.5)
  ax3.set_zorder(ax2.get_zorder()+1)
  #ax1.patch.set_visible(False)

  handles, labels = ax1.get_legend_handles_labels()
  handles2, labels2 = ax2.get_legend_handles_labels()
  handles3, labels3 = ax3.get_legend_handles_labels()
  handles = handles + handles2 + handles3
  labels = labels + labels2 + labels3


  if plot_mf: order = [4,5,2,3,1,6,7,0]
  else: order = [3,2,1,0]
  #print("handles, labels", handles, labels)
  loc = "best"
  folders = ["WS_Pruned"]
  #if "WS_Pruned" in folders: loc = "center right"
  leg = ax3.legend([handles[idx] for idx in order],[labels[idx] for idx in order],
            bbox_to_anchor=(1.07, 1), edgecolor="black", shadow = False, framealpha = 0.6, loc='upper left')
  leg.get_frame().set_linewidth(2.5)
  
  leg.set_zorder(ax3.get_zorder()+1)

  return avg_ordp_net, std_avg_ordp_net, Rc_net, avgk2, std_avgk2,

def rhu(n, decimals=0, integer = False): #round_half_up
  import math
  multiplier = 10 ** decimals
  res = math.floor(n*multiplier + 0.5) / multiplier
  if integer: return int(res)
  return res

def mean_std_avg_pl(G, N = 0):
  import datetime as dt
  import networkx as nx
  from math import sqrt
  if not N:
    N = G.number_of_nodes()

  import datetime as dt
  start = dt.datetime.now()
  avg_pl = max([  
                nx.average_shortest_path_length(C) for C in (G.subgraph(c).copy() for c in \
                nx.connected_components(G))
                ])
  print(f'avg_pl in mean_std: {avg_pl}',)
  #print(f'The total time of avg_pl is: {dt.datetime.now()-start}')
  
  'calculate the std_avg_pl'
  #start = dt.datetime.now()
  std_avg_pl = 0
  for node in G:
      path_length=nx.single_source_shortest_path_length(G, node)
      #print(f'path_length.values(): {path_length.values()}',)
      sq_deviation = [(x-avg_pl)**2/2 for x in path_length.values()]
      #print(f'sq_deviation: {sq_deviation}',)
      std_avg_pl += sum(sq_deviation) / (N/2*(N-1)-1)
      #print(f'std_avg_pl: {std_avg_pl}',)
  std_avg_pl = sqrt(std_avg_pl)
  print(f'std_avg_pl in mean_std: {std_avg_pl}',)
  return avg_pl, std_avg_pl

def save_sir(G, folder, ordp_pmbD_dic, done_iterations = 1, p = 0, beta = 0.001, mu = 0.16, R0_max = 16,  start_inf = 10, numb_iter = 50, numb_onring_links = 0, avg_pl = -1, std_avg_pl = -1):
  import os.path
  from definitions import my_dir, func_file_name, N_D_std_D, mean_std_avg_pl
  import datetime as dt
  import matplotlib.pylab as plt
  start_time = dt.datetime.now()

  print("numb_iter", numb_iter)  
  mode = "a"
  #if done_iterations == 1: mode = "w"
  my_dir = my_dir() #"/home/hal21/MEGAsync/Thesis/NetSciThesis/Project/Plots/Tests/"
  N, D, std_D = N_D_std_D(G) #D used in ordp_pmbD, outsiders, R0
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
  intervals = np.arange(R0_max+1)
  N = G.number_of_nodes()
  rhuD2 = rhu(D,2) #use in suptitle, to print "The model...", 
  #rhuD1 in file_name_save
  R0 = beta * D / mu  
  std_R0 = beta/mu*std_D

  print(f'beta,mu: {beta},{mu},{std_D}, {std_R0}',)
  
  for i in intervals:
    if i <= R0 < i+1:
      'Intro R0-subfolder since R0 det epids behaviour on a fixed net'
      if folder == "WS_Pruned": r0_folder = "R0_%s-%s/R0_%s/" % (i, i+1, rhu(beta*D/mu,3)) #"R0_1-2/mu0.16/"
      else: r0_folder = "beta%s/d%s/" % (rhu(beta,3),rhu(mu**(-1))) #"R0_%s-%s/" % (intervals[i], intervals[i+1])
      #if folder == "WS_Epids": r0_folder += "D%s/" % rhuD  #"R0_1-2/mu0.16/D6/"
      if not os.path.exists(my_dir + r0_folder): os.makedirs(my_dir + r0_folder)
      if not os.path.exists(my_dir + r0_folder + "/Sel_R0/"): os.makedirs(my_dir + r0_folder + "/Sel_R0/")
      file_name = func_file_name(folder = folder, adj_or_sir = adj_or_sir, \
        N = N, D = D, R0 = R0, p = p, beta = beta, mu = mu)
      
      file_path = my_dir + r0_folder + file_name

      'plot all'
      _, ax = plt.subplots()
      #if vertical: _, ax = plt.subplots(figsize = (24,20)) #init = (24,14); BA_Model = (20,20) should change also the subplots_adjust!!!
      
      'plot sir'
      
      print("\nThe model has N: %s, D: %s(%s), beta: %s, d: %s, p: %s, R0: %s" % 
      (N,rhuD2,rhu(std_D,2),rhu(beta,5),rhu(mu**(-1)),rhu(p,3), beta*D/mu ))
      avg_ordp_net, std_avg_ordp_net, Rc_net, avgk2, std_avgk2, = \
        plot_sir(G, p = p, ax1=ax, folder = folder, beta = beta, mu = mu, start_inf = start_inf, 
                numb_iter = numb_iter)


      right = 0.95
      #folders = ["WS_Pruned"]
      if True: right = 0.72 #if R0 <= 1 #0.73
      plt.subplots_adjust(
      top=0.920,
      bottom=0.12, #if vertical, bottom=0.10,
      left=0.07, # if vertical, left=0.10, #0.06
      right=right, #0.99
      hspace=0.2,
      wspace=0.2)

      string_format = str(np.round(std_avg_ordp_net,3))[:5]
      if string_format == "0.000":
        string_format = format(std_avg_ordp_net, ".1e")

      #if not nx.is_connected(G): connect_net(G, True)
      delta = r"\delta_C"
      #comps = [nx.average_shortest_path_length(s) for c in nx.connected_components(G) 
      #for s in G.subgraph(c)]
      #print(comps)

      if std_avg_pl == -1 or avg_pl == -1:
        'calculate now the avg_pl and its std_avg_pl'
        print("I'm calculating in save_sir the avg_pl and std_avg_pl")
        avg_pl, std_avg_pl = mean_std_avg_pl(G)
      if std_avg_pl == -1 or avg_pl == -1: raise ValueError("avg_pl = -1 or std_avg_pl = -1")
      
      'if not connected also show the len of the maxcc component'
      if not nx.is_connected(G): 
        ls_cc = nx.connected_components(G)
        #print("ls_cc", ls_cc)
        max_cc = max(ls_cc, key = len)
        len_max_cc = len(max_cc)
        str_len = f"{len_max_cc}"
        delta = r"\delta_{"+str_len+r"}"

      'find good way to put signs'
      R0pl = R0/avg_pl 
      R0_sign = "<"; R0pl_sign = "<"
      if R0 > Rc_net: R0_sign = ">"
      elif R0 == Rc_net: R0_sign = "="
      if R0pl > Rc_net/avg_pl: R0pl_sign = ">"
      elif R0pl == Rc_net/avg_pl: R0pl_sign = "="

      'estimate the StDs'
      std_Rc_net = fstd_Rc_net(Rc_net, D, avgk2, std_D, std_avgk2)
      std_DeltaR0Rc = np.sqrt(std_R0**2+std_Rc_net**2)

      if folder == "Caveman_Model":
          ax.set_title(r"$R_0:%s, D_{%s}:%s(%s), OnRing:%s, p:%s, \beta:%s, d:%s$"
        % (
          f"{rhu(R0,3)}({rhu(std_R0,3)})"+R0_sign+f"{rhu(Rc_net,3)}({rhu(std_Rc_net,3)})",
          N, rhuD2, rhu(std_D,2), 
          numb_onring_links,
          rhu(p,3), rhu(beta,3), rhu(mu**(-1)),
          ), pad = 30
        )
      else:
        ax.set_title(r"$R_0:%s, D_{%s}:%s(%s), p:%s, \beta:%s, d:%s$"
        % (
          f"{rhu(R0,3)}({rhu(std_R0,3)})"+R0_sign+f"{rhu(Rc_net,3)}({rhu(std_Rc_net,3)})",
          N, rhuD2, rhu(std_D,2), rhu(p,3), rhu(beta,3), rhu(mu**(-1)),
          ), pad = 30
        )
      
      textstr = '\n'.join((
        "Other Measures:",
        r"$\Delta R_0: %s(%s)$"%(rhu(R0 - Rc_net,3), rhu(std_DeltaR0Rc,3)),
        r"$< k^{2} >: %s(%s)$"%(rhu(avgk2,3),rhu(std_avgk2,3)),
        r"$%s: %s(%s)$"%(delta, rhu(avg_pl,2), rhu(std_avg_pl,2)),
        r"$R_0 / \delta = $"+f"{rhu(R0/avg_pl,3)} "+R0pl_sign+f" {rhu(Rc_net/avg_pl,3)}",
        r"$\Delta R_0 / \delta: %s$"%(rhu((R0 - Rc_net)/avg_pl,3)),
        #r"$R_{c-IBM}: %s$"%(rhu(D/np.max(nx.adjacency_spectrum(G).real),3)),
        r"OrdPar: %s(%s)"%(rhu(avg_ordp_net,3), string_format),
        ))

      # these are matplotlib.patch.Patch properties
      props = dict(boxstyle='round', facecolor='white', alpha=0.5, lw = 2.5)

      # place a text box in upper left in axes coords
      ax.text(1.085, 0.3, textstr, transform=ax.transAxes, fontsize=25, #x = 1.07
              bbox=props,) #fontfamily = "serif", ha = "center" with x = 1.188

      plt.savefig( file_path )
      print("time 1_save_sir:", dt.datetime.now()-start_time) 

      with open(log_path, mode) as text_file: #write only 1 time
        text_file.write(file_name + "\n")            
      plt.close()

      del my_dir
      from definitions import my_dir; import json; from definitions import NestedDict
      ordp_pmbD_dic = NestedDict(ordp_pmbD_dic)
      value = avg_ordp_net
      std = std_avg_ordp_net

      #pp_ordp_pmbD_dic = json.dumps(ordp_pmbD_dic, sort_keys=False, indent=4)
      #print("Start dic", pp_ordp_pmbD_dic)

      print("\nTo be added pmbD itermean:", p, mu, beta, D, value)        
      
      if folder == "WS_Pruned":
        d = ordp_pmbD_dic #rename ordp_pmbD_dic to have compact wrinting
        if p in d.keys():
          if mu in d[p].keys():
              d[p][mu] = { **d[p][mu], **{D:[std_D, value, std]} }
          else: d[p] = {**d[p], **{mu:{D:[std_D,value,std]}}}
        else:
          d[p][mu][D] = [std_D,value,std]

      else:
        d = ordp_pmbD_dic #rename ordp_pmbD_dic to have compact wrinting
        if p in d.keys():
          if mu in d[p].keys():
              if beta in d[p][mu].keys():
                  d[p][mu][beta][D] = [std_D,value,std]
              else: d[p][mu] = { **d[p][mu], **{beta: {D:[std_D, value, std]}} }
          else: d[p] = {**d[p], **{mu:{beta:{D:[std_D,value,std]}}} }
        else:
          d[p][mu][beta][D] = [std_D,value,std]
      '''   
          NEED TO TEST IT
            def update(d, D, std_D, p, beta, mu, value, std):
          def update_ordp_dict(d, D, std_D, p, beta, mu, value, std):
              if p in d.keys():
                if mu in d[p].keys():
                    if beta in d[p][mu].keys():
                        d[p][mu][beta][D] = [std_D,value,std]
                    else: d[p][mu] = { **d[p][mu], **{beta: {D:[std_D, value, std]}} }
                else: d[p] = {**d[p], **{mu:{beta:{D:[std_D,value,std]}}} }
              else:
                d[p][mu][beta][D] = [std_D,value,std]

          #print(f'starting d: {d}',)
          try: 
              #print(f'Entered in try',)        
              Dkeys = list(d[p][mu][beta].keys())
              #print(f'Dkeys: {Dkeys}',)
              for Dkey in Dkeys:
                  if Dkey-1 < D < Dkey+1: 
                      #print("Inside the if")
                      #print(f'Not saving dic, D similar: {Dkey}')
                  else:
                      #print(f'Check updating dict',)
                      update_ordp_dict(d, D, std_D, p, beta, mu, value, std)
                      #print(f'd: {d}',)         
          except:
              #print(f'turned into except',)
              update_ordp_dict(d, D, std_D, p, beta, mu, value, std)
              #print(f'd: {d}',)
      '''
          
      print(f'\nAfter Dic Insertion:',)
      
      _, ax = plt.subplots(figsize = (24,14))

      'WARNING: here suptitle has beta // mu but the dict is ordp[p][mu][beta][D] = [std_D, ordp, std_ordp]'
      'since in the article p and mu are fixed!'
      plt.suptitle("Average SD(Daily New Cases) : "+r"$p:%s,\beta:%s,d:%s$"%(rhu(p,3),rhu(beta,3),rhu(mu**(-1))))
      ax.set_xlabel("Avg Degree D [Indivs]")
      ax.set_ylabel("Avg SD(Cases)")
      
      if folder == "WS_Pruned":
        fix_pmb = ordp_pmbD_dic[p][mu]
      else: fix_pmb = ordp_pmbD_dic[p][mu][beta]
      #print("ordp_pmbD_dic, p0, mu0, beta0, fix_pmb", \
      #  ordp_pmbD_dic, p, mu, beta, fix_pmb)
      x = sorted(fix_pmb.keys())
      xerr = [fix_pmb[i][0] for i in x]
      y = [fix_pmb[i][1] for i in x]
      yerr = [fix_pmb[i][2] for i in x]

      #print("y", y)
      ax.grid(color='grey', linestyle='--', linewidth = 1)
      ax.errorbar(x,y, xerr = xerr, yerr = yerr, color = "tab:blue", marker = "*", linestyle = "-",
         markersize = 30, mfc = "tab:red", mec = "black", linewidth = 3, label = "Avg SD(Cases)")
      
      def D_cRc(lmbd, avgk2):
        from math import sqrt
        delta = avgk2**2 * lmbd - 2*avgk2*lmbd - 4*avgk2 + lmbd
        if delta >= 0:
          print(f'delta: {delta}',)
          add1 = sqrt(delta*lmbd)        
          num_less = - add1 + avgk2 * lmbd + lmbd
          num_great = + add1 + avgk2 * lmbd + lmbd
          den = 2*(lmbd+1)
          D_great = num_great /den
          D_less = num_less /den
          return D_great, D_less
        else: return 0,0

      lmbd = beta/mu
      D_cfuse = 1 + 2*mu/(beta*(1+p)) #Rc_net = 1 assumption
      D_cer = 1 + mu/beta
      D_chomo = mu/beta
      D_cRcfuse, _ = D_cRc(lmbd = lmbd*(1+p)/2, avgk2 = avgk2)
      D_cRcer, _ = D_cRc(lmbd = lmbd, avgk2 = avgk2)

      ax.axvline(x = D_cfuse, color = "maroon", lw = 4, ls = "--", 
                 label = "".join((r"$D_{c-fuse \, model}$",f": {rhu(D_cfuse,3)}")) )
      ax.axvline(x = D_cer, color = "darkblue", lw = 4, ls = "--", 
                 label = "".join((r"$D_{c-ER \, model}$",f": {rhu(D_cer,3)}")) )
      ax.axvline(x = D_chomo, color = "darkslategrey", lw = 4, ls = "--", 
                 label = "".join((r"$D_{c-homog \, model}: \mu / \beta$",f": {rhu(D_chomo,3)}")) )


      '''print(f'D_cRcfuse: {D_cRcfuse}',)
      print(f'D_cRcer: {D_cRcer}',)
      
      if D_cRcfuse:
        ax.axvline(x = D_cRcfuse, color = "maroon", lw = 4, marker='o', ls='dashed', ms = 12,
                  label = "".join((r"$D_{Rc-fuse \, model}$",f": {rhu(D_cRcfuse,3)}")) )
      if D_cRcer:
        ax.axvline(x = D_cRcer, color = "darkblue", lw = 4, ls = "-.", ms = 12,
                  label = "".join((r"$D_{Rc-ER \, model}$",f": {rhu(D_cRcer,3)}")) )'''
      leg = ax.legend(fontsize = 35, loc = "best")
      leg.get_frame().set_linewidth(2.5)

      plt.subplots_adjust(
      top=0.91,
      bottom=0.122,
      left=0.080,
      right=0.99)
      
      if folder == "WS_Pruned":
        ordp_path = f"{my_dir()}{folder}/OrdParam/p{rhu(p,3)}/d{rhu(mu**(-1))}/" 
        if not os.path.exists(ordp_path): os.makedirs(ordp_path)

        plt.savefig("".join((ordp_path,"%s_ordp_p%s_d%s.png" % (folder, rhu(p,3),rhu(mu**(-1))))))
      else: 
        ordp_path = my_dir() + folder + "/OrdParam/p%s/beta%s/" % (rhu(p,3),rhu(beta,3))
        if not os.path.exists(ordp_path): os.makedirs(ordp_path)
        plt.savefig("".join((ordp_path,"%s_ordp_p%s_beta%s_d%s.png" % (folder, rhu(p,3),rhu(beta,3),rhu(mu**(-1))))))
      plt.close()

      'pretty print the dictionary of the ordp'
      pp_ordp_pmbD_dic = json.dumps(ordp_pmbD_dic, sort_keys=False, indent=4)
      #print("Final dic to be saved", pp_ordp_pmbD_dic)
      ordp_file = "".join((ordp_path,"saved_ordp_dict.txt"))
      with open(ordp_file, 'w') as file:
        file.write(pp_ordp_pmbD_dic) # use `json.loads` to do the reverse
      
  if os.path.exists(log_path):
    'sort line to have the new ones at first'
    sorted_lines = []
    with open(log_path, 'r') as r:
      for line in sorted(r):
        sorted_lines.append(line)
    
    with open(log_path, 'w') as r:
      for line in sorted_lines:
        r.write(line)

def suptitle_AdjMat(G, folder, D, std_D, p, numb_onring_links, avg_pl, std_avg_pl, clique_size = ""):
    import networkx as nx
    import numpy as np

    string_onlinks = ""; string_kmax = "" 
    N = G.number_of_nodes()
    if nx.is_connected(G): 
      str_sw = r"SW_{C}"
      #avg_pl, std_avg_pl = mean_std_avg_pl(G)
      len_max_cc = G.number_of_nodes()
        
    else:  
      ls_cc = nx.connected_components(G)
      print("ls_cc", ls_cc)
      max_cc = max(ls_cc, key = len)
      len_max_cc = len(max_cc)
      #avg_pl, std_avg_pl = mean_std_avg_pl(G.subgraph(max_cc))
      str_sw = r"SW_{c-max:%s-%s}"%(number_connected_components(G), len_max_cc)

    if std_avg_pl == -1 or avg_pl == -1: raise ValueError("avg_pl = -1 or std_avg_pl = -1")
      
    'find the major hub and the "ousiders", i.e. highly connected nodes'
    infos = G.degree()
    dsc_sorted_nodes = sorted( infos, key = lambda x: x[1], reverse=True) # List w/ elements (node, deg)
    _, max_degree = dsc_sorted_nodes[0]
    count_outsiders, threshold = 0,3*D

    deg_dsc_nodes = list(map(lambda x: x[1], dsc_sorted_nodes) )
    for i in range(len(deg_dsc_nodes)):
        tmp_count = 0
        if deg_dsc_nodes[i] >=  threshold:  tmp_count += 1
        if -deg_dsc_nodes[-i] >=  threshold: tmp_count += 1
        if tmp_count == 0: break
        count_outsiders += tmp_count

    if folder == "BA_Model": 
        sw_coeff = rhu(avg_pl / np.log(np.log(len_max_cc)),3)
        std_sw_coeff = np.sqrt(1/(np.log(np.log(len_max_cc)))) \
                    *std_avg_pl
        print(f'std_sw_coeff: {std_sw_coeff}',)
        str_sw = "".join((r"U",str_sw))
        string_kmax = r", k_{max}: %s" % max_degree

    else: 
        sw_coeff = rhu(avg_pl / np.log(len_max_cc),3)
        std_sw_coeff = np.sqrt(1/np.log(len_max_cc)) \
                    *std_avg_pl
        print(f'std_sw_coeff: {std_sw_coeff}',)
        

    string_format = str(sw_coeff)[:5]
    print(string_format)
    if string_format == "0.000":
        string_format = format(sw_coeff, ".1e")
    if folder == "Caveman_Model":
        string_onlinks = f"\, Inlinks: {numb_onring_links},"

        #plt.suptitle(r"$N:%s, D:%s(%s), p:%s, N_{3-out}: %s, %s: %s,  k_{max}: %s, $" % (
        #N, D, std_D, rhu(p,3), count_outsiders, str_sw, rhu(sw_coeff,1),  max_degree, )) 

    str_suptitle = r"$N_{%s}:%s, D:%s(%s), p:%s,%s N_{3-out}: %s, %s:%s(%s) %s$" % \
                    (clique_size, N, D, rhu(std_D,3), rhu(p,3), string_onlinks, count_outsiders, 
                    str_sw, sw_coeff, rhu(std_sw_coeff,3), string_kmax)

    return str_suptitle

def save_net(G, folder, p = 0, m = 0, N0 = 0, done_iterations = 1, log_dd = False, partition = None, pos = None, numb_onring_links = 0, avg_pl = -1, std_avg_pl = -1, clique_size = ""):
  import os.path
  from definitions import my_dir, func_file_name, N_D_std_D, rhu, plot_params
  from functools import reduce
  import networkx as nx
  from scipy.stats import poisson
  import matplotlib.pylab as plt
  import numpy as np
  from matplotlib import cm
  
  mode = "a"
  #if done_iterations == 1: mode = "w"
  my_dir = my_dir() #"/home/hal21/MEGAsync/Thesis/NetSciThesis/Project/Plots/Tests/"
  N, D, std_D = N_D_std_D(G)
  D = rhu( D, 2)  #D used for suptitle, outsider, hist
  std_D = rhu( std_D, 2 )
  print("D and std_D in Adj Mat", D, std_D)

  rhuD = rhu(D) #used for func_file_name, mean in hist, 
  #D = rhu( D)
  adj_or_sir = "AdjMat"

  log_upper_path = my_dir + folder + "/" #"../Plots/Tests/WS_Epids/"
  my_dir+=folder+"/p%s/"%rhu(p,3)+adj_or_sir+"/" #"../Plots/Tests/WS_Epids/p0.001/AdjMat/"
  file_name = func_file_name(folder = folder, adj_or_sir = adj_or_sir, \
                             N = N, D = D, p = p, m = m, N0 = N0)

  file_path = my_dir + file_name #../Plot/Test/AdjMat/AdjMat_N1000_D500_p0.001.png
  log_path = log_upper_path + folder + f"_log_saved_{adj_or_sir}.txt" #"../Plots/Tests/WS_Epids/WS_Epids_log_saved_nets.txt"
  nc_path = log_upper_path + folder + "_not_connected_nets.txt"

  plot_params()

  'plot G, adj_mat, degree distribution'
  plt.figure(figsize = (22,22), ) #20,20

  ax = plt.subplot(221)  
  'start with degree distribution'
  'set edges width according to how many "long_range_edges'
  width = 0.8
  if folder == "WS_Pruned": width = 0.2
  long_range_edges = list(filter( lambda x: x > 30, [np.min((np.abs(i-j),np.abs(N-i+j))) for i,j in G.edges()] )) #list( filter(lambda x: x > 0, )
  length_long_range = len(long_range_edges)
  if length_long_range < 10: print("\nLong_range_edges", long_range_edges, length_long_range)
  else: print("len(long_range_edges)", length_long_range)
  #folders = ["WS_Pruned","BA_Model"]
  #if folder in folders: 
  width = min(3,width*N/max(1,len(long_range_edges)))
  if folder== "Caveman_Model":
    nx.draw(G, pos, node_color=list(partition.values()), node_size = 5, width = 0.5, with_labels = False)
  else: nx.draw_circular(G, ax=ax, with_labels=False, font_size=20, node_size=25, width=width)

  
  #ax.text(0,1,transform=ax.transAxes, s = "D:%s" % D)

  'plot adjiacency matrix'
  axs = plt.subplot(222)
  adj_matrix = nx.adjacency_matrix(G).todense()
  axs.matshow(adj_matrix, cmap=cm.get_cmap("Greens"))
  #print("Adj_matrix is symmetric", np.allclose(adj_matrix, adj_matrix.T))

  'set xticks of the degree distribution to be centered'
  sorted_degree = np.sort([G.degree(n) for n in G.nodes()])

  sorted_nd = sorted(list(G.degree()),key = lambda x: x[1])
  #print("\n sorted degrees in save_net", sorted_nd)

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
  #hist_mean = n[np.where(hist_bins == mean)]; pois_mean = poisson.pmf(rhuD, D)
  hist_mean = max(n); pois_mean = max(poisson.pmf(bins,D))
  axs.plot(bins, y * hist_mean / pois_mean, "bo--", lw = 2, label = "Poissonian Distr")
  axs.set_xlabel('Degree', )
  axs.set_ylabel('Counts', )
  axs.grid(color = "grey", ls = "--", lw = 1)
  axs.set_xlim(bins[0],bins[-1]) 

  if folder == "BA_Model": 
    print("")
    #axs.set_xscale("log")
    #axs.set_yscale("log")

  leg = axs.legend(loc = "best")
  leg.get_frame().set_linewidth(2.5)

    
  plt.subplots_adjust(top=0.85,
  bottom=0.09,  #0.088
  left=0.1,
  right=0.963,
  hspace=0.067, #0.067
  wspace=0.164) #0.164
    
  if std_avg_pl == -1 or avg_pl == -1: raise ValueError("avg_pl = -1 or std_avg_pl = -1")
  plt.suptitle(suptitle_AdjMat(G, folder, D, std_D, p, numb_onring_links, avg_pl = avg_pl, std_avg_pl = std_avg_pl, clique_size = clique_size))

  'TO SAVE PLOTS'
  if not os.path.exists(my_dir): os.makedirs(my_dir)
  plt.savefig(file_path)
  plt.close()

  'report in a .txt the disconnected_nets'
  if not nx.is_connected(G):
    sorted_disc_components = sorted(nx.connected_components(G), key=len, reverse=True)
    with open(nc_path, mode) as nc_file:
      nc_file.write("".join((file_name, " #disc_comp: %s" % len(sorted_disc_components), "\n")))

  with open(log_path, mode) as text_file: #write only 1 time
    #print("file_name", file_name)
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

def already_saved_list(folder, adj_or_sir, chr_min, my_print = True, done_iterations = 1):
  from definitions import my_dir
  log_upper_path = "".join((my_dir(),folder,"/")) #../Plots/Test/NNO_Conf_Model.../
  log_path = "".join((log_upper_path, folder, f"_log_saved_{adj_or_sir}.txt"))

  saved_list = []
  if os.path.exists(log_path):
    with open(log_path, "r") as file:
      saved_list = [l.rstrip("\n")[chr_min:] for l in file]
  if my_print: print(f"\nThe already saved {adj_or_sir} are", saved_list)
  return saved_list

def save_nes(
  G, p, folder, adj_or_sir, R0_max = 12, m = 0, N0 = 0, 
  beta = 0.3, mu = 0.3, my_print = True, pos = None, 
  partition = None, dsc_sorted_nodes = False, done_iterations = 1, 
  chr_min = 0, ordp_pmbD_dic = 0, numb_onring_links = 1, avg_pl = -1, std_avg_pl = -1,
  start_inf = 10, clique_size = ""
  ): #save new_entrys

  'save net only if does not exist in the .txt. So, to overwrite all just delete .txt'
  from definitions import already_saved_list, func_file_name, N_D_std_D
  N,D,_ = N_D_std_D(G)
  R0 = rhu( beta*D/mu, 3)
  if adj_or_sir == "AdjMat": 
    saved_files = already_saved_list(folder, adj_or_sir, chr_min = chr_min, my_print= my_print, done_iterations= done_iterations)
    file_name = func_file_name(folder = folder, adj_or_sir = adj_or_sir, \
                               N = N, D = D, R0 = R0, p = p, m = m, N0 = N0)
  if adj_or_sir == "SIR": 
    saved_files = already_saved_list(folder, adj_or_sir, chr_min = chr_min, my_print= my_print, done_iterations=done_iterations)
    file_name = func_file_name(folder = folder, adj_or_sir = adj_or_sir, \
                               N = N, D = D, R0 = R0, p = p, m = m, N0 = N0, beta = beta, mu = mu)

  if adj_or_sir == "AdjMat" and file_name in saved_files:
    return -1,-1

  if file_name not in saved_files: 
    print("I'm saving", file_name)
    #infos_sorted_nodes(G, num_sorted_nodes = True)
    if adj_or_sir == "AdjMat": 
      'pass the average path length to save_sir onyl if calculated here'
      'calculate once for all the avg_pl and its std_avg_pl'
      print("Printing avg_pl from save_nes AdjMat")
      avg_pl, std_avg_pl = mean_std_avg_pl(G)
      save_net(G = G, pos = pos, partition = partition, m = m, N0 = N0, 
               folder = folder, p = p, done_iterations = done_iterations, 
               numb_onring_links = numb_onring_links, avg_pl = avg_pl, std_avg_pl = std_avg_pl, clique_size = clique_size)
      infos_sorted_nodes(G, num_sorted_nodes = 0)
      return avg_pl, std_avg_pl

    if adj_or_sir == "SIR": 
      save_sir(G, folder = folder, beta = beta, mu = mu, p = p, R0_max = R0_max, 
              done_iterations = done_iterations, ordp_pmbD_dic = ordp_pmbD_dic, 
              numb_onring_links = numb_onring_links, avg_pl = avg_pl, std_avg_pl = std_avg_pl,
              start_inf = start_inf)


def save_log_params(folder, text):
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

'''
'===Configurational Model'
def pois_pos_degrees(N, D, L = int(2e3)):
  'Draw N degrees from a Poissonian sequence with lambda = D and length L'
  degs = np.random.poisson(lam = D, size = L)
  print("NN_Starting D", D)
  pos_degrees = np.random.choice(degs, N)

  print("len(deg==0) in deg", len([x for x in degs if x == 0]))
  #print("len(s) in pos_degrees", len([x for x in pos_degrees if x == 0]))
  return pos_degrees

def config_pois_model(N, D, seed = 123, folder = None):
  'create a network with the node degrees drawn from a poissonian with even sum of degrees'
  
  np.random.seed(seed)
  degrees = pois_pos_degrees(N,D) #poiss distr with deg != 0
  print("\nseed1 %s, sum(degrees) %s" % (seed, np.sum(degrees)%2))
  'change seed to have a even sum of the degrees'
  while(np.sum(degrees)%2 != 0):
    seed+=1
    np.random.seed(seed)
    print("\nseed2 %s" % (seed))
    degrees = pois_pos_degrees(N,D)

  print("\nNetwork Created but w/o standard neighbors wiring!")
  G = nx.configuration_model(degrees, seed = seed)

  'If D/N !<< 1, by removing loops and parallel edges, we lost degrees. Ex. with N = 50 = D, <k> = 28 != 49.8'
  check_loops_parallel_edges(G)
  remove_loops_parallel_edges(G)
  infos_sorted_nodes(G)
  print("End of %s " % folder)
  return G

'''
'Net Infos'
def check_loops_parallel_edges(G):
  ls = list(G.edges())
  print("parallel edges", set([i for i in ls for j in ls[ls.index(i)+1:] if i==j]),
        "; loops", [(i,j) for (i,j) in set(G.edges()) if i == j])

def infos_sorted_nodes(G, num_sorted_nodes = False):
    import networkx as nx
    'sort nodes by key = degree. printing order: node, adjacent nodes, degree'
    nodes = G.nodes()
    #print("<k>: ", np.sum([j for (i,j) in G.degree() ]) / len(nodes), 
    #      " and <k>/N ", np.sum([j for (i,j) in G.degree() ]) / len(nodes)**2, end="\n" )
    
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
    
'Net Actions'
'append lists of // and loops and remove them'
def remove_loops_parallel_edges(G, remove_loops = True):
  print("\n")
  import networkx as nx
  'append // edges'
  full_ls = list((G.edges()))
  lpe = []
  for i in full_ls:
    full_ls.remove(i)
    for j in full_ls:
      if i == j: lpe.append(j) #print("i", i, "index", full_ls.index(i), "j", j)
  
  'append loops'
  if remove_loops == True:  
    for x in list(nx.selfloop_edges(G)): lpe.append(x)
    print("Parallel edges and loops removed!")
  return G.remove_edges_from(lpe)

'replace edges from list or delete all edges'
def replace_edges_from(G,list_edges=[]):
  '''def: "replace" existing edges, since built-in method only adds'''
  present_edges = [x for x in G.edges()]
  G.remove_edges_from(present_edges)
  if list_edges!=[]: return G.add_edges_from(list_edges)
  return G

'addition of distant nodes'
def add_lredges(G, p = 0, time_int = False):
  from itertools import chain
  import random
  import datetime as dt
  from definitions import replace_edges_from, remove_loops_parallel_edges
  'add an long-range edge over the existing ones'
  
  if time_int: start_time = dt.datetime.now()
  
  'list of tuple of edges for non-zero-degree nodes'
  all_edges = [list(G.edges(node)) for node in pos_deg_nodes(G)]
  all_edges = list(chain.from_iterable(all_edges))
  #??? why the 2 prev lines?? all_edges = G.edges()
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
  remove_loops_parallel_edges(G, True)
  if time_int: print(f"Time for add edges over ddistr:", dt.datetime.now()-start_time)

'replace lr-edges with local ones'
def replace_lredges(G, p = 0): 
    'replace long range edges'
    import random
    from itertools import chain
    from definitions import pos_deg_nodes

    def fnode_edges(G, node):
        return G.edges(node)

    #random.seed(2)
    len_start = len(list(G.edges()))
    for node in G.nodes():
      if random.random() < p and len(G.edges(node)):
        #print(f"\nI'm looking for a new neighbors for {node}")
        node_edges = fnode_edges(G, node)
        #print(f'Inside: {node_edges}',)
        neighbors = list(G.neighbors(node))
        #print(f'neighbors: {neighbors}',)

        'rm only if node as deg > 0'
        rm_neigh = random.choice(neighbors)
        add_lrnode = random.choice(list(G.nodes()))
        G.remove_edge(node, rm_neigh)
        'dont select any present node (no // edges and loop)'
        while(add_lrnode in list(chain.from_iterable([[rm_neigh], [node], neighbors]))):
            add_lrnode = random.choice(list(G.nodes()))
        #print(f'(rm_neigh: {rm_neigh} --> add_lrnode): {add_lrnode}',)
        G.add_edge(node, add_lrnode)
        node_edges = fnode_edges(G, node)
        #print(f'final G.edges(node): {node_edges}',)
    if  len(list(G.edges())) != len_start: 
        raise ValueError(f"Error value: {len(list(G.edges()))} not {len_start}") 

'forcing connection of a net'
def connect_net(G, conn_flag): #set solo_nodes = False to have D < 1 nets
  if conn_flag:
    import networkx as nx
    import numpy as np
    from definitions import N_D_std_D
    print("I'm here to connect")
    'there is only a node with 2 degree left. So, if rewired correctly only a +1 in the ddistr'
    'So, connect all the disconnected components'
    its = 0
    sorted_disc_components = sorted(nx.connected_components(G), key=len, reverse=True)
    max_cc = sorted_disc_components[0]
    for c in sorted_disc_components: #set of conn_comp
      if its == 0: 
        base_node = np.random.choice((list(c))); 
      else: 
        linking_node = np.random.choice((list(c)))
        G.add_edge(linking_node,base_node)
        base_node = linking_node
      its += 1
    #print("Total links to have", len(list(nx.connected_components(G))),"connected component are", its)
    if len(list(nx.connected_components(G)))>1: print("Disconnected net!"); raise Exception("G is not connected")
    else: print("The network is connected!")
    return sorted_disc_components

'''
def NN_pois_net(N, folder, ext_D, p = 0, conn_flag = True):
  'p is not used right now'
  from definitions import config_pois_model, connect_net
  import numpy as np
 
  G = config_pois_model(N, ext_D, seed = 123, folder = folder)

  verbose = False
  def verboseprint(*args):
    if verbose == True:
      print(*args)
    elif verbose == False:
      None

  'for random rewiring with p -- select pos_deg nodes since need to rewiring to them'
  l_nodes = list(pos_deg_nodes(G))

  edges = set() #avoid to put same link twice (+ unordered)
  nodes_degree = {}

  'list of the nodes sorted by their degree'
  for node in pos_deg_nodes(G):
    nodes_degree[node] = G.degree(node)
  sorted_nodes_degree = {k: v for k, v in sorted(nodes_degree.items(), key=lambda item: item[1])}
  sorted_nodes = list(sorted_nodes_degree.keys())
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
  connect_net(G, conn_flag = conn_flag)

  print(f"There are {len([j for i,j in G.degree() if j == 0])} 0 degree node as")
  _,D,_ = N_D_std_D(G)
  print(f"End of wiring with average degree {D} vs {ext_D}")
  

  return G

'This kind of network is similar to a WS'
def NNOverl_pois_net(N, ext_D, p, add_edges_only = False):
  #Nearest Neighbors Overlapping
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
'''

'New Cluster of defs for Poissonian Small World Network'
def pois_pos_degrees(N, D):
  import numpy as np
  #np.random.seed(0)

  'Draw N degrees from a Poissonian sequence with lambda = D and length L'
  def remove_zeros(array):
    #print("array", array, "degarray", np.sum(array))
    its = 0
    while True:
      its += 1
      mask = np.where(array == 0)
      if not mask[0].size: 
        #print(f"Replacing non-0 degrees in {its} iterations")
        return array
      
      'the sum of the degrees must be even'
      psum = np.sum(array)
      #print("psum", psum)
      if not psum % 2: #psum is even return even cover
        while True:
          its += 1
          cover = np.random.poisson(lam = D, size = len(array[mask]))
          #print("even cover?", cover)
          if not np.sum(cover) % 2: break
      else:
        while True: #psum is odd return odd cover
          its += 1
          cover = np.random.poisson(lam = D, size = len(array[mask]))
          if np.sum(cover) % 2: break
      #print("cover final", cover)
      array[mask] = cover

  pos_degrees = np.random.poisson(lam = D, size = N)
  pos_degrees = remove_zeros(pos_degrees)
  #print(pos_degrees)
  return pos_degrees

def dic_nodes_degrees(N, degrees):
    #np.random.seed(1)
    #print("degrees", degrees, sum(degrees))
    N = len(degrees)
    nodes = np.arange(N)
    dic_nodes = nodes.copy()
    np.random.shuffle(dic_nodes)
    dic_nodes = {k:v for k in dic_nodes for v in np.sort(degrees)[np.where(dic_nodes == k)]}
    sorted_nodes = np.array([x for x in dic_nodes.keys()])
    #print(f'nodes: {nodes}, sorted_nodes: {sorted_nodes}', "dic_nodes", dic_nodes, np.sort(degrees))
    return dic_nodes

def delete_node_from_both(avl_node, nodes, sorted_nodes, b_bool):
    nodes = np.delete(nodes, np.where(nodes == avl_node))
    sorted_nodes = np.delete(sorted_nodes, np.where(sorted_nodes == avl_node))
    #print(f'len(nodes): {len(nodes)}',)
    if len(nodes) == 1: 
        #print(f'\nEnd By len(nodes):{len(nodes)},{len(sorted_nodes)}')
        b_bool = True
    return nodes, sorted_nodes, b_bool

def add_edge(snode, i, b_bool, edges, sorted_nodes, nodes, dic_nodes):
    D = len(nodes)
    snode_idx = np.where(nodes == snode)[0]
    avl_node = nodes[(snode_idx+i)%D] #nearest available node
    ##print(f'Inside i={i} with D = {D} add_edge: nodes: {nodes}, snode_idx[{snode}]:{snode_idx},' )   
    ##print(f"Before edge.add: selected node: {avl_node}, deg[{avl_node}]: {dic_nodes[int(avl_node)]}")
    edges.add((int(snode),int(avl_node)))
    dic_nodes[int(avl_node)] -= 1
    ##print(f'after edge.add: edges, dic_nodes[{avl_node}]', edges, dic_nodes[int(avl_node)])
    if dic_nodes[int(avl_node)]==1:
        nodes, sorted_nodes, b_bool = delete_node_from_both(avl_node,  nodes, sorted_nodes, b_bool)
        ##print(f'Deleted for low degree {avl_node}: nodes, sorted_nodes', nodes, sorted_nodes)

def edges_nearest_node(dic_nodes):
    from copy import deepcopy
    dc_dic_nodes = deepcopy(dic_nodes)
    sorted_nodes = np.array([x for x in dc_dic_nodes.keys()])
    nodes = np.arange(len(sorted_nodes))
    ##print(f'nodes: {nodes}',f'sorted_nodes: {sorted_nodes}',)
    ##print(f'id(nodes): {id(nodes)}',f"id(dc_dic_nodes.keys)", id(dic_nodes.keys()), np.array(dic_nodes.keys()))
    edges = set()
    b_bool = False #breakingbool
    for snode in sorted_nodes:
        ##print(f'\nRecap nodes: nodes: {nodes}, sorted_nodes & degree', sorted_nodes, [dc_dic_nodes[k] for k in sorted_nodes])
        ##print(f'Choosen snode: {snode} with degree: {dc_dic_nodes[snode]}')
        snode_idx = np.where(nodes == snode)[0]
        for i in np.arange(1, dc_dic_nodes[snode]//2+1):
            no_avl_nodes = [a for (a,b) in edges if b == snode]
            ##print(f'check already takes edges & avl_nodes: {edges}, {no_avl_nodes}', )
            #if nodes[(snode_idx+i)%D] not in no_avl_nodes:
            add_edge(snode, i, b_bool, edges = edges, sorted_nodes = sorted_nodes, \
                nodes = nodes, dic_nodes = dc_dic_nodes)
            if b_bool: 
                ##print(f'i bbool break: ',)
                break
            add_edge(snode, -i, b_bool, edges = edges, sorted_nodes = sorted_nodes, \
                nodes = nodes, dic_nodes = dc_dic_nodes)
            if b_bool: 
                ##print(f'-i bbool break: ',)
                break

        if dc_dic_nodes[snode]%2: #and nodes[(snode_idx+1)%D] not in [b for (a,b) in edges if a == snode]:
            ##print("Odd last attachment")
            add_edge(snode = snode, i = 1, b_bool = b_bool, edges = edges, sorted_nodes = sorted_nodes, \
                nodes = nodes, dic_nodes = dc_dic_nodes)
            if b_bool: 
                ##print(f'Odd last bbool break: ',)
                break
        nodes, sorted_nodes, b_bool = delete_node_from_both(snode, nodes = nodes, sorted_nodes = sorted_nodes, b_bool = b_bool)
        ##print(f'End of 1 cycle deleted snode: {snode}')
        
        if b_bool: 
            ##print(f'End of all',)
            break
        ##print('After all the rewiring, left nodes', nodes, "sorted_nodes", sorted_nodes, "edges", edges)
    return edges

def NN_pois_net(N, folder, ext_D, p = 0):
    from definitions import check_loops_parallel_edges, infos_sorted_nodes, \
        replace_lredges, connect_net, N_D_std_D

    D = ext_D
    degrees = pois_pos_degrees(N, D)
    dic_nodes = dic_nodes_degrees(N, degrees)

    edges = edges_nearest_node(dic_nodes)
    G = nx.Graph()
    G.add_nodes_from(np.arange(N))
    G.add_edges_from(edges)

    check_loops_parallel_edges(G)
    infos_sorted_nodes(G, num_sorted_nodes=False)

    replace_lredges(G, p = p)
    #if not nx.is_connected(G):
    #  connect_net(G, conn_flag = False)

    ##print(f"There are {len([j for i,j in G.degree() if j == 0])} 0 degree node as")
    _,D,_ = N_D_std_D(G)
    ##print(f"End of wiring with average degree {D} vs {ext_D}")
    ##print(f'G.is_connected(): {nx.is_connected(G)}',)
    
    return G
'new_Overl_PSW'
def NNOverl_pois_net(N, ext_D, p, conn_flag = False, return_dic_nodes = False):
    #Nearest Neighbors Overlapping
    from itertools import chain
    from definitions import replace_edges_from, check_loops_parallel_edges,\
        remove_loops_parallel_edges, N_D_std_D, infos_sorted_nodes, connect_net
    import datetime as dt
    import numpy as np
    import random
    import networkx as nx


    def edges_node(x):
        return [(i,j) for i,j in all_edges if i == x]

    D = ext_D
    G = nx.Graph()
    G.add_nodes_from(np.arange(N))
    degrees = pois_pos_degrees(N,D)
    dic_nodes = dic_nodes_degrees(N, degrees)

    deg_mean = np.mean(degrees)   
    

    'rewire left and right for the max even degree'
    dsc_sorted_nodes = dic_nodes_degrees(N, degrees)
    #print(f'dsc_sorted_nodes: {dsc_sorted_nodes}',)
    
    #dsc_sorted_nodes = {k: v for k,v in sorted( G.degree(), key = lambda x: x[1], reverse=True)}
    edges = set()
    for node in dsc_sorted_nodes.keys():
        k = dsc_sorted_nodes[node]//2
        for i in range(1, k + 1):
            edges.add((node, (node+i)%N))
            edges.add((node, (node-i)%N))
        if dsc_sorted_nodes[node] % 2 == 1: 
            edges.add((node, (node+k+1)%N))
        elif dsc_sorted_nodes[node] % 2 != 0: 
            print("Error of Wiring: dsc[node]%2", dsc_sorted_nodes[node] % 2); break
    G.add_edges_from(edges)
    remove_loops_parallel_edges(G)

    _, D, _ = N_D_std_D(G)
    if D != deg_mean: print("!! avg_deg_Overl_Rew - avg_deg_Conf_Model = ", D - deg_mean)

    if p != 0: #add on top of the distr, a new long-range edge
        replace_lredges(G, p = p)
    
    adeg_OR = sum([j for i,j in G.degree()])/G.number_of_nodes()
    print("Rel Error wrt to ext_D %s %%" % (rhu((adeg_OR - ext_D)/ext_D,1 )*100))
    print("Rel Error wrt to deg_mean %s %%" % (rhu( (adeg_OR - deg_mean)/deg_mean,1 )*100))

    '''
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
    '''

    check_loops_parallel_edges(G)
    infos_sorted_nodes(G, num_sorted_nodes=False)
    connect_net(G, conn_flag = conn_flag)

    _,D,_ = N_D_std_D(G)
    print(f"End of wiring with average degree {D} vs {ext_D}")
    print(f'G.is_connected(): {nx.is_connected(G)}',)
    #print(f'G.edges(): {G.edges()}',)

    if return_dic_nodes: return G, dic_nodes
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
      #print(f'edges: {g.edges()}', len(g.edges()))
      for (ni, nj) in g.edges():
          ci = partition[ni]
          cj = partition[nj]
          if ci != cj:
            if (ci,cj) in edges.keys():
              edges[(ci, cj)] = [*edges[(ci, cj)], (ni, nj)] #SHOULD BE += [(NI,NJ)] HERE
              #print("ni, nj", ni, nj)
              #print("ci,cj", ci, cj); print("edgesij", edges[(ci, cj)])
            else:
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

  def comm_caveman_relink(cliques = 8, clique_size = 7, p = 0,  relink_rnd = 0, numb_onring_links = 1, p_clique = 0):
    import numpy as np
    import numpy.random as npr
    from definitions import my_dir

    'caveman_graph'
    G = nx.caveman_graph(l = cliques, k = clique_size)
    print("G.nodes are", len(G.nodes()))
    print("size/cliq: %s, cliq/size: %s" % (clique_size/cliques, cliques/clique_size) )

    'decide how many nodes are going to relink to the neighbor "cave" (cfr numb_onring_links'
    total_nodes = clique_size*cliques
    
    'clique size = D. So, if D < 1 dont relink since we need a disconnected net'
    for clique in range(cliques):
      if numb_onring_links != 0:
        first_cl_node = clique_size*clique
        #nodes_inclique = np.arange(first_cl_node, first_cl_node+numb_onring_links)
        nodes_inclique = np.arange(first_cl_node, first_cl_node+clique_size)
        next_clique_nodes = np.arange(clique_size*(1+clique), clique_size*(2+clique)) % total_nodes

        #attached_nodes = npr.choice( np.arange(clique_size*(1+clique), 
        #                            clique_size*(2+clique)), 
        #                            size = len(nodes_inclique) )
        #attached_nodes = attached_nodes % np.max((total_nodes,1))
        from itertools import product
        import random
        prod = random.sample(list(product(nodes_inclique, next_clique_nodes)), k = numb_onring_links)
        G.add_edges_from(prod)
        #for test, att_node in prod:
            #print("NN - clique add:", (test,att_node))
        #    G.add_edge(test,att_node)
        
      'add a new edge by relinking one of the existing node to a new link in the sequent clique'
      'decide how many nodes in the clique would go into rnd relink via relink_rnd'
      '--comment from the fu'
      if p:
        #relink_rnd = clique_size #all nodes in the clique are tried to be relinked
        first_cl_node = clique_size*clique
        nodes_inclique = np.arange(first_cl_node, first_cl_node + clique_size)
        lr_target_nodes = npr.choice(
                                    [x for x in pos_deg_nodes(G) if x not in nodes_inclique], 
                                    size = len(nodes_inclique))
        for test, lr_node in zip(nodes_inclique, lr_target_nodes):
          if npr.uniform() < p: 
            #print(f"with probability {p}, added relink {(test,lr_node)}")
            G.add_edge(test,lr_node)
      
      'rnd rewiring clique by clique -- it produces disc net => go w/ the first numb_onring_links.'
      if p_clique: 
        first_cl_node = clique_size*clique
        nodes_inclique = np.arange(first_cl_node, first_cl_node + clique_size)
        next_clique_nodes = np.arange(clique_size*(1+clique), clique_size*(2+clique)) % total_nodes
        from itertools import product
        for source, next_clique_node in product(nodes_inclique, next_clique_nodes):
          if npr.uniform() < p_clique:
            #print(f"with probability {p_clique}, added relink {(source,next_clique_node)}")
            G.add_edge(source,next_clique_node)
      
    remove_loops_parallel_edges(G)
    

    '''
    if deg_for_ordp != False:
      i = 1
      print("cliquesize VS wanna-have average", clique_size, deg_for_ordp)
      while(deg_for_ordp <= np.mean([j for _,j in G.degree()])):
        chosen_nodes = np.random.choice(G.nodes(), 50, replace = False)
        for node in chosen_nodes:
          G.remove_edges_from([(i,j) for i,j in G.edges() if i == node or j == node])
          #new_edges = list(filter(lambda x: x[0]!=node or x[1]!=node, G.edges()))
        #print("Removed nodes", node, np.mean([j for _,j in G.degree()]))
        i+=1
      print("The Caveman_Model has %s left_nodes with avg %s" 
      % (i,np.mean([j for _,j in G.degree()])))
    '''
    return G
  
  return partition_layout, comm_caveman_relink

'===Barabasi-Albert Model -- this is too slow for N = int(1e4)'
'Anyway, I am using nx.barabasi_albert_graph'
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
  G.add_nodes_from(range(N0)) #[0],[0,1],[0,1,2]
  edges = []
  
  'creates the initial clique connecting all the N0 nodes'
  edges = [(i,j) for i in range(N0) for j in range(i,N0+1) if i!=j]
  print("Initial clique edges", m, edges)

  'adds the initial clique to the network'
  G.add_edges_from(edges)

  #import matplotlib.pylab as plt
  #nx.draw_circular(G)
  #plt.show()

  'list to store the nodes to be selected for the preferential attachment.'
  'instead of calculating the probability of being selected a trick is used: if a node has degree k, it will appear'
  'k times in the list. This is equivalent to select them according to their probability.'
  prob = []
  
  'runs over all the reamining nodes'
  print("Creating a B-A Model")
  for i in range(N0+1,N):
    if i % 300 == 0: print("I am in the %s-th it" % i)
    G.add_node(i)
    'for each new node, creates m new links'
    for j in range(m):
      'creates the list of nodes'
      for k in np.delete(G.nodes(), np.where(G.nodes() == i)):
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
def main(folder, N, k_prog, p_prog, beta_prog, mu_prog, 
  R0_min, R0_max, start_inf, epruning = False):
  from definitions import save_log_params, save_nes, \
    NestedDict, jsonKeys2int, my_dir, mean_std_avg_pl
  from itertools import product
  import networkx as nx
  import json

  print("Datetime of this log is:", datetime.now())
  'load a dic to save D-order parameter'
  ordp_pmbD_dic = NestedDict()
  'these are the links among near communities'
  numb_onring_links = [1]
  if folder == "Caveman_Model": numb_onring_links = [1,2]
  
  'unique try of saving both, but generalize to all other nets'
  'try only with p = 0.1'
  total_iterations, done_iterations = 0,0
  for D,mu,beta in product(k_prog, mu_prog, beta_prog): 
      if R0_min <= beta*D/mu <= R0_max:
        total_iterations+=1*len(p_prog)*len(numb_onring_links)
  print("Total Iterations:", total_iterations)
  
  text = "N %s;\nk_prog %s, len: %s;\np_prog %s, len: %s;\nbeta_prog %s, len: %s;\nmu_prog %s, \
        len: %s;\nR0_min %s, R0_max %s; \nstart_inf: %s; \nTotal Iterations: %s;\n---\n" \
        % (N, k_prog, len(k_prog), p_prog, len(p_prog), beta_prog, len(beta_prog), \
        mu_prog, len(mu_prog),  R0_min, R0_max, start_inf, total_iterations)
  print(text)
  save_log_params(folder = folder, text = text)

  for D,numb_onring_links,p,beta,mu in product(k_prog, numb_onring_links, p_prog, beta_prog, mu_prog): #long product
      'since D_real ~ 2*D (D here is fixing only the m and N0), R0_max-folder ~ 2*R0_max'
      if R0_min <= beta*D/mu <= R0_max:
        done_iterations+=1

        print("\nIterations left: %s" % ( total_iterations - done_iterations ) )
        
        ordp_path = "".join((my_dir(), folder, "/OrdParam/p%s/beta%s/" % (rhu(p,3),rhu(beta,3)) ))
        #ordp_path = "".join((my_dir(), folder, "/OrdParam/"))#p%s/beta_%s/" % (rhu(p,3),rhu(beta,3)) ))
        ordp_path = "".join( (ordp_path, "saved_ordp_dict.txt"))
        if os.path.exists(ordp_path): 
          with open(ordp_path,"r") as f:
            ordp_pmbD_dic = json.loads(f.read(), object_hook=jsonKeys2int)

        pp_ordp_pmbD_dic = json.dumps(ordp_pmbD_dic, sort_keys=False, indent=4)
        if done_iterations == 1: print("Previous ordp %s" % (pp_ordp_pmbD_dic))
        
        m, N0 = 0,0
        pos, partition = None, None
        clique_size = ""

        'snippet to save a D < 1, make it int and create a network'
        'Then, solo nodes for the desired mean, e.g. 0.2'
        #print("2.1nd: D %s, deg_ordp %s" % (D, deg_for_ordp))     
        #print("2nd: D %s, deg_ordp %s" % (D, deg_for_ordp))
        
        'intro regD has the "regularized D" even for D < 1'
        if D <= 1: 
          if folder == ["vvvCaveman_Model"]: regD = 2
          else: regD = 1
        else: regD = D
        regD = int(regD)

        if folder == "WS_Epids":
          if regD % 2: regD += 1
          G = nx.connected_watts_strogatz_graph( n = N, k = regD, p = p, seed = 1 )

        if folder == "BA_Model":
          m, N0 = regD,regD; 
          G = nx.barabasi_albert_graph(N, m = regD) #bam(N, m = int(m), N0 = int(N0))
          
        if folder == "Complete":
          G = nx.connected_watts_strogatz_graph(100, regD, p)

        if folder == "NN_Conf_Model":
          from definitions import NN_pois_net
          '''
          if np.any([x<1. for x in k_prog]):
            conn_flag = False
          else: conn_flag = True'''
          conn_flag = False
          G = NN_pois_net(N = N, folder = folder, ext_D = regD, p = p)
          print("connected components", len(list(nx.connected_components(G))))
          if len(list(nx.connected_components(G))) != 1 and conn_flag:
            raise Exception("Error: it should be connected")
        
        #this is a model really similar to Watts-Strogatz. So, decide wheter to insert it
        if folder == f"NNO_Conf_Model": #add edges instead of rew
          from definitions import NNOverl_pois_net
          G = NNOverl_pois_net(N, regD, p = p)
        
        if folder == "Caveman_Model":
          from definitions import caveman_defs
          partition_layout, comm_caveman_relink = caveman_defs()
          clique_size = regD
          cliques = int(N/clique_size)
          clique_size = int(clique_size) #clique_size is a np.float64!
          G = comm_caveman_relink(cliques=cliques, clique_size = clique_size, 
                                  p = p, relink_rnd = clique_size, p_clique = 0, numb_onring_links = numb_onring_links)
          
          for node in range(clique_size*cliques):
            if node == 0: print("node", node, type(node), 
                      "int(n/cl_s)", int(cliques/clique_size), 
                      "cliques", cliques, type(cliques), "clique_size", clique_size, type(clique_size), 
                  )

          partition = {node : int(node/clique_size) for node in range(cliques * clique_size)}
          '''print("partition", partition, type(partition), "node", partition.keys(), type(partition.keys()), 
          "int(n/cl_s)", partition.values(), type(partition.values()), "cliques", cliques, type(cliques),
          "clique_size", clique_size, type(clique_size), 
           )'''
          pos = partition_layout(G, partition, ratio=clique_size/cliques*0.1)

        'generate solo nodes to reduce <k>'
        
        if k_prog[0] < 1: epruning = True
        if epruning:
          i = 1
          print("\nCreating solo-nodes to match the wanted D...")
          print("regD VS wanna-have D and num of edges", regD, D, len(G.edges()))
          while(D < np.mean([j for _,j in G.degree()])):
            num_rm = 25
            if folder == "NN_Conf_Model": 
              edges = np.array([x for x in G.edges()])
            else: edges = np.array(G.edges())
            idxs = npr.choice(len(edges), num_rm, replace = False)
            chosen_edges = edges[idxs] 
            G.remove_edges_from(chosen_edges)
            #new_edges = list(filter(lambda x: x[0]!=node or x[1]!=node, G.edges()))
            #print("Removed nodes", node, np.mean([j for _,j in G.degree()]))
            i+=1
          print("At the end of pruning: %s has %s left_edges with avg %s" 
          % (folder, len(G.edges()), np.mean([j for _,j in G.degree()])))

        #print("\nIterations left: %s" % ( total_iterations - done_iterations ) )

        avg_pl, std_avg_pl = -1, -1

        'if obtain avg_pl, std_avg_pl pass it to the sir'
        import datetime as dt
        start_time = dt.datetime.now()       
        avg_pl, std_avg_pl = save_nes(G, m = m, N0 = N0, pos = pos, partition = partition,
                 p = p, folder = folder, adj_or_sir="AdjMat", done_iterations=done_iterations, numb_onring_links = numb_onring_links, avg_pl = avg_pl, std_avg_pl = std_avg_pl,
                 start_inf = start_inf, clique_size = clique_size)
        print("\nThe end-time of 1 generation of one AdjMat plot is", dt.datetime.now()-start_time)

        start_time = dt.datetime.now()       
        save_nes(G, m = m, N0 = N0,
                 p = p, folder = folder, adj_or_sir="SIR", R0_max = R0_max, beta = beta, mu = mu, 
                 ordp_pmbD_dic = ordp_pmbD_dic, done_iterations=done_iterations, 
                 numb_onring_links = numb_onring_links, avg_pl = avg_pl, std_avg_pl = std_avg_pl, start_inf = start_inf)
        print("\nThe end-time of the generation of one SIR plot is", dt.datetime.now()-start_time)

def parameters_net_and_sir(folder = None):
  'progression of net-parameters'
  import numpy as np
  'WARNING: put SAME beta, mu, D and p to compare at the end the different topologies'
  #k_prog = np.concatenate(([1.0],np.arange(2,20,2)))
  start_inf = 1
  p_prog = [0, 0.3] #0.2 misses
  beta_prog = [0.015, 0.05, 0.1, 0.2, 0.4]; 
  days_prog = [14,9,6,4,1] 
  days_prog = [14,6,4]
  mu_prog = [1/x for x in days_prog] #mu_prog = [0.07, 0.11, 0.167, 0.25, 1]
  # old @ 5.9.2021: mu_prog = [0.14, 0.16, 0.2, 0.25, 0.8]#, 0.33,0.5]
  k_prog = np.hstack((np.arange(1,13,2),[14,19,24,40,50]))
  R0_min = 0; R0_max = 100

  'this should be deleted to have same params and make comparison more straight-forward'
  if folder == "WS_Epids": 
    'beta_prog = np.linspace(0.01,1,7); mu_prog = beta_prog'
  if folder == "BA_Model": 
    'beta_prog = np.linspace(0.01,1,14); mu_prog = beta_prog'
    #k_prog = np.arange(1,11,1)
    p_prog = [0]; R0_min = 0; R0_max = 60  
  if folder == "NN_Conf_Model": 
    'beta_prog = [0.05, 0.1, 0.2, 0.25]; mu_prog = beta_prog'
    # past parameters: beta_prog = np.linspace(0.01,1,8); mu_prog = beta_prog
    #k_prog = np.hstack((np.arange(1,13,2),np.arange(14,34,5)))
    R0_max = 300     
  if folder == "Caveman_Model": 
    k_prog = [3, 5, 8, 9, 11, 14, 24]
    'k_prog = np.arange(1,11,2)' #https://www.prb.org/about/ -> Europe householdsize = 3
    #beta_prog = np.linspace(0.001,1,6); mu_prog = beta_prog
  if folder[:5] == "NNO_C": 
    'beta_prog = [0.05, 0.1, 0.2, 0.25]; mu_prog = beta_prog #np.linspace(0.01,1,4)'

  return k_prog, p_prog, beta_prog, mu_prog, R0_min, R0_max, start_inf