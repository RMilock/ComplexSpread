import networkx as nx
from networkx.algorithms import clique
from networkx.generators.community import caveman_graph
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import matplotlib.cm as cm 
import os #to create a folder
from numba import jit, config
from numba import jit_module
config.THREADING_LAYER = "default"

@jit(nopython = True, parallel = True)
def filter_out_k(arr, k = 0):
  
  filtered = np.array([np.float64(x) for x in np.arange(0)])
  for i in np.arange(arr.size):
      if arr[i] == k:
          filtered = np.append(filtered, arr[i])
  return filtered

from numba import njit, prange
@njit(parallel = True)
def pinff(tests,daily_new_inf,current_state,future_state,beta):
    'If the contact is susceptible and not infected by another node in the future_state, try to infect it'
    print("I'm in pinff")
    for j in prange(len(tests)):
        if current_state[tests[j]] == 'S' and future_state[tests[j]] == 'S':  
            if npr.random_sample() < beta:
                future_state[tests[j]] = 'I'; daily_new_inf += 1   
                print("Ive infected", tests[j], future_state[tests[j]], daily_new_inf)
            else:
                future_state[tests[j]] = 'S'


def nsir(G, mf = False, beta = 1e-3, mu = 0.05, start_inf = 10, seed = False):
  'If mf == False, the neighbors are not fixed;' 
  'If mf == True, std mf by choosing @ rnd the num of neighbors'

  #here's the modifications of the "test_ver1"
  'Number of nodes in the graph'
  N, mean, _ = N_D_std_D(G)

  'Label the individual wrt to the # of the node'
  node_labels = G.nodes()
  
  'Currently infected individuals and the future infected and recovered' 
  inf_list = np.array([], dtype = int) #infected node list @ each t
  rec_list = np.asarray([]) #recovered nodes for a fixed t
  arr_ndi = np.asarray([]) #arr to computer Std(daily_new_inf(t)) for daily_new_inf(t)!=0

  'Initial Conditions'
  current_state = np.asarray(['S' for i in node_labels])
  future_state = np.asarray(['S' for i in node_labels])
  
  npr.seed(0)

  'Selects the seed of the disease'
  seeds = npr.choice(node_labels, start_inf, replace = False)  #without replacement, i.e. not duplicates
  for seed in seeds:
    current_state[seed] = 'I'
    future_state[seed] = 'I'
    inf_list = np.append(inf_list,seed)

  #print("inf_lsit", inf_list, type(inf_list[0]))

  'initilize prevalence (new daily infected) and recovered list'
  'we dont track infected'
  prevalence = np.asarray([start_inf/N]) #[len(inf_list)/N] 
  rec_list = np.asarray([0])
  cum_prevalence = np.asarray([start_inf/N])
  num_susc = np.asarray([N-start_inf])

  import ray
  ray.init()
  
  '''
  import sys
  module_name = globals()['__name__']
  current_module = sys.modules[module_name]
  sys.modules[module_name] = current_module
  '''

  'start and continue whenever there s 1 infected'
  while(len(inf_list)>0):  
    daily_new_inf = 0

    '''
    @njit()
    def pinff(j,daily_new_inf,current_state,future_state,beta):
        print("I'm in pinff")
        if current_state[j] == 'S' and future_state[j] == 'S':  
          if npr.random_sample() < beta:
            future_state[j] = 'I'; daily_new_inf += 1   
            print("Ive infected", j, future_state[j], daily_new_inf)
          else:
            future_state[j] = 'S'
    '''

    
        
    #def ptests_prep(G,i,daily_new_inf,current_state,future_state,beta):

    @ray.remote
    def ptests_prep(i):
        'Select the neighbors of the infected node'
        if not mf: tests = np.asarray(list(G.neighbors(i))) #only != wrt to the SIS: contact are taken from G.neighbors            
        if mf: 
            ls = np.concatenate((np.arange(i), np.arange(i+1,N)))
            tests = npr.choice(ls, size = int(mean)) #spread very fast since multiple infected center
            tests = np.asarray([int(x) for x in tests]) #convert 35.0 into int
        beta = 1
        print("Inf node", i, "tests", tests, daily_new_inf,current_state,future_state,beta)
        print("\nbefore pinff")
        pinff(tests,daily_new_inf,current_state,future_state,beta)
        print("jump all over pinff")


        '''
        'If the contact is susceptible and not infected by another node in the future_state, try to infect it'
        print("I'm entered in pinff")
        if current_state[j] == 'S' and future_state[j] == 'S':
          if npr.random_sample() < beta:
            future_state[j] = 'I'; daily_new_inf += 1   
            print("Ive infected", j, future_state[j], daily_new_inf)
            print("break")  
          else:
            future_state[j] = 'S'
        '''

    print("inf_list", inf_list, "\ncurr", current_state, "\nfut", future_state, "\ndni", daily_new_inf)
    
    'use ray see line'

    res = []
    for i in inf_list:
      res.append(ptests_prep.remote(i))
      #ptests_prep(i)

    ray.get(res)
    
    
    #print("I'l before Pool")
    #with Pool(2) as p:
    #  print([p.apply(ptests_prep, \
    #    args = product(G,i,daily_new_inf,current_state,future_state,beta)) for i in inf_list])

    #print("dail_n_i", daily_new_inf)
    if daily_new_inf != 0: arr_ndi = np.append(arr_ndi,daily_new_inf)
    #print("arr_ni", arr_ndi)

    'Recovery Phase: only the prev inf nodes (=inf_list) recovers with probability mu'
    'not the new infected'    
    'This part is important in the OrderPar since diminishes the inf_list # that is in the "while-loop"'    

    #@jit(nopython = True)
    def precf(inf_list,future_state,mu, ):
        for i in inf_list:
            if npr.random_sample() < mu:
                future_state[i] = 'R'
            else:
                future_state[i] = 'I'

    precf(inf_list,future_state,1)

    print("future state", future_state)
    
    'Time update: once infections and recovery ended, we move to the next time-step'
    'The future state becomes the current one'
    current_state = future_state.copy() #w/o .copy() it's a mofiable-"view"
    
    'Updates inf_list with the currently fraction of inf/rec and save lenS to avg_R' 
    inf_list = np.asarray([i for i, x in enumerate(current_state) if x == 'I'])
    rec_list = np.asarray([i for i, x in enumerate(current_state) if x == 'R'])

    'Saves the fraction of new daily infected (ndi) and recovered in the current time-step'
    prevalence = np.append(prevalence,daily_new_inf/float(N))
    #prevalence.append(len(inf_list)/float(N))
    #recovered.append(len(rec_list)/float(N))
    cum_prevalence = np.append(cum_prevalence, cum_prevalence[-1]+daily_new_inf/N)  
    #loop +=1;
    #print("loop:", loop, cum_total_inf, cum_total_inf[-1], daily_new_inf/N)

    np.append(num_susc, N*(1 - cum_prevalence[-1]))
    #print("\nnum_susc, prevalence, recovered",num_susc, prevalence, recovered, 
    #len(num_susc), len(prevalence), len(recovered))

  #print("inf, rec", inf_list, rec_list)
  'Order Parameter (op) = Std(Avg_ndi(t)) s.t. ndi(t)!=0 as a func of D'
  if not mf:
    ddof = 0
    if len(arr_ndi) > 1: ddof = 1
    if not arr_ndi.size: arr_ndi = np.asarray([0])
    oneit_avg_dni = np.mean(arr_ndi)
    print(ddof)
    'op = std_dni'
    
    op = np.std( arr_ndi, ddof = ddof )
    if len(arr_ndi) > 1:
      #filtered = np.array([])
      if len(filter_out_k(arr_ndi, k=0)) > 0: #[x for x in arr_ndi if x == 0])
        raise Exception("Error There's 0 ndi: dni, arr_daily, std", daily_new_inf, arr_ndi, op)

    'oneit_avg_R is the mean over the time of 1 sir. Then, avg over-all iterations'
    'Then, compute std_avg_R'
    degrees = np.asarray([j for i,j in G.degree()])
    D = np.mean(degrees)
    #print("R0, b,m,D", beta*D/mu, beta, mu, D)
    c = beta*D/(mu*num_susc[0])
    oneit_avg_R = c*np.mean(num_susc)
    ddof = 0
    if len(num_susc) > 1: ddof = 1
    std_oneit_avg_R = c*np.std(num_susc, ddof = ddof)
    #print("num_su[0], np.sum(num_susc), len(prev), oneit_avg_R2", \
    #  num_susc[0],np.sum(num_susc), len(prevalence), oneit_avg_R)

    return oneit_avg_R, std_oneit_avg_R, oneit_avg_dni, op, prevalence, cum_prevalence
  
  return prevalence, cum_prevalence

def N_D_std_D(G):
  degrees = np.asarray([j for i,j in G.degree()])
  return G.number_of_nodes(), np.mean(degrees), np.std(degrees, ddof = 1)

if __name__ == "__main__":
    from playgr import nsir
    G = nx.complete_graph(20)
    nsir(G)

    #import pandas as pd
    #df = pd.DataFrame([[1,2,3],[3,5]]).fillna(0)
    #print(df)
