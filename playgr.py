#parallelisation
from definitions import sir, N_D_std_D
import multiprocessing as mp
import networkx as nx
print("Number of processors: ", mp.cpu_count())
import numpy as np
import datetime as dt
from numba import jit
import numpy.random as npr

def N_D_std_D(G):
  degrees = [j for i,j in G.degree()]
  return G.number_of_nodes(), np.mean(degrees), np.std(degrees, ddof = 1)

def sirjit(G, mf = False, beta = 1e-1, mu = 0.05, start_inf = 10, seed = False):
  'If mf == False, the neighbors are not fixed;' 
  'If mf == True, std mf by choosing @ rnd the num of neighbors'
  import numpy.random as npr
  from definitions import N_D_std_D
  
  #here's the modifications of the "test_ver1"
  'Number of nodes in the graph'
  N, mean, _ = N_D_std_D(G)
  'daily_new_inf'
  'Label the individual wrt to the # of the node'
  node_labels = G.nodes()
  
  'Currently infected individuals and the future infected and recovered' 
  inf_list = [] #infected node list @ each t
  prevalence = [] # = len(inf_list)/N, i.e. frac of daily infected for every t
  recovered = [] #recovered nodes for a fixed t
  arr_ndi = [] #arr to computer Std(daily_new_inf(t)) for daily_new_inf(t)!=0

  'Initial Conditions'
  current_state = ['S' for i in node_labels] 
  future_state = ['S' for i in node_labels]
  
  if seed == True: npr.seed(0)

  'Selects the seed of the disease'
  seeds = npr.choice(a = node_labels, size = start_inf, replace = False)  #without replacement, i.e. not duplicates
  for seed in seeds:
    current_state[seed] = 'I'
    future_state[seed] = 'I'
    inf_list.append(seed)

  'initilize prevalence (new daily infected) and recovered list'
  'we dont track infected'
  prevalence = [0] #[len(inf_list)/N] 
  recovered = [0]
  cum_prevalence = [start_inf/N]
  num_susc = [N-start_inf]

  @jit(nopython = True, parallel = True)
  def whilefunc(G, N, arr_ndi, inf_list, prevalence, cum_prevalence):
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
            tests = npr.choice(ls, size = int(mean)) #spread very fast since multiple infected center
        tests = [int(x) for x in tests] #convert 35.0 into int
        for j in tests:
          'If the contact is susceptible and not infected by another node in the future_state, try to infect it'
          if current_state[j] == 'S' and future_state[j] == 'S':
            if npr.random_sample() < beta:
                future_state[j] = 'I'; daily_new_inf += 1     
            else:
                future_state[j] = 'S'
  
      #print("dail_n_i", daily_new_inf)
      if daily_new_inf != 0: arr_ndi.append(daily_new_inf)
      #print("arr_ni", arr_ndi)

      'Recovery Phase: only the prev inf nodes (=inf_list) recovers with probability mu'
      'not the new infected'    
      'This part is important in the OrderPar since diminishes the inf_list # that is in the "while-loop"'    
      for i in inf_list:
        if npr.random_sample() < mu:
            future_state[i] = 'R'
        else:
            future_state[i] = 'I'
      
      'Time update: once infections and recovery ended, we move to the next time-step'
      'The future state becomes the current one'
      current_state = future_state.copy() #w/o .copy() it's a mofiable-"view"
      
      'Updates inf_list with the currently fraction of inf/rec and save lenS to avg_R' 
      inf_list = [i for i, x in enumerate(current_state) if x == 'I']
      rec_list = [i for i, x in enumerate(current_state) if x == 'R']

      'Saves the fraction of new daily infected (ndi) and recovered in the current time-step'
      prevalence.append(daily_new_inf/float(N))
      #prevalence.append(len(inf_list)/float(N))
      #recovered.append(len(rec_list)/float(N))
      cum_prevalence.append(cum_prevalence[-1]+daily_new_inf/N)  
      #loop +=1;
      #print("loop:", loop, cum_total_inf, cum_total_inf[-1], daily_new_inf/N)

      num_susc.append(N*(1 - cum_prevalence[-1]))
      #print("\nnum_susc, prevalence, recovered",num_susc, prevalence, recovered, 
      #len(num_susc), len(prevalence), len(recovered))
  
    'Order Parameter (op) = Std(Avg_ndi(t)) s.t. ndi(t)!=0 as a func of D'
    if not mf:
      ddof = 0
      if len(arr_ndi) > 1: ddof = 1
      if arr_ndi == []: arr_ndi = [0]
      oneit_avg_dni = np.mean(arr_ndi)
      'op = std_dni'
      op = np.std( arr_ndi, ddof = ddof )
      if len(arr_ndi) > 1:
          if len([x for x in arr_ndi if x == 0]) > 0: 
              raise Exception("Error There's 0 ndi: dni, arr_daily, std", daily_new_inf, arr_ndi, op)

      'oneit_avg_R is the mean over the time of 1 sir. Then, avg over-all iterations'
      'Then, compute std_avg_R'
      degrees = [j for i,j in G.degree()]
      D = np.mean(degrees)
      #print("R0, b,m,D", beta*D/mu, beta, mu, D)
      c = beta*D/(mu*num_susc[0])
      oneit_avg_R = c*np.mean(num_susc)
      ddof = 0
      if len(num_susc) > 1: ddof = 1
      std_oneit_avg_R = c*np.std(num_susc, ddof = 1)
      #print("num_su[0], np.sum(num_susc), len(prev), oneit_avg_R2", \
      #  num_susc[0],np.sum(num_susc), len(prevalence), oneit_avg_R)

      return oneit_avg_R, std_oneit_avg_R, oneit_avg_dni, op, prevalence, cum_prevalence

    oneit_avg_R, std_oneit_avg_R, oneit_avg_dni, op, prevalence, cum_prevalence = \
        whilefunc(G, N, ls, arr_ndi,inf_list, prevalence, cum_prevalence)

    return oneit_avg_R, std_oneit_avg_R, oneit_avg_dni, op, prevalence, cum_prevalence
  
  return prevalence, cum_prevalence


def sir(G, mf = False, beta = 1e-3, mu = 0.05, start_inf = 10, seed = False):
  'If mf == False, the neighbors are not fixed;' 
  'If mf == True, std mf by choosing @ rnd the num of neighbors'

  import random
  #here's the modifications of the "test_ver1"
  'Number of nodes in the graph'
  N, mean, _ = N_D_std_D(G)

  'Label the individual wrt to the # of the node'
  node_labels = G.nodes()
  
  'Currently infected individuals and the future infected and recovered' 
  inf_list = [] #infected node list @ each t
  prevalence = [] # = len(inf_list)/N, i.e. frac of daily infected for every t
  recovered = [] #recovered nodes for a fixed t
  arr_ndi = [] #arr to computer Std(daily_new_inf(t)) for daily_new_inf(t)!=0

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

  'initilize prevalence (new daily infected) and recovered list'
  'we dont track infected'
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
    if daily_new_inf != 0: arr_ndi.append(daily_new_inf)
    #print("arr_ni", arr_ndi)

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

    'Saves the fraction of new daily infected (ndi) and recovered in the current time-step'
    prevalence.append(daily_new_inf/float(N))
    #prevalence.append(len(inf_list)/float(N))
    #recovered.append(len(rec_list)/float(N))
    cum_prevalence.append(cum_prevalence[-1]+daily_new_inf/N)  
    #loop +=1;
    #print("loop:", loop, cum_total_inf, cum_total_inf[-1], daily_new_inf/N)

    num_susc.append(N*(1 - cum_prevalence[-1]))
    #print("\nnum_susc, prevalence, recovered",num_susc, prevalence, recovered, 
    #len(num_susc), len(prevalence), len(recovered))
  
  'Order Parameter (op) = Std(Avg_ndi(t)) s.t. ndi(t)!=0 as a func of D'
  if not mf:
    ddof = 0
    if len(arr_ndi) > 1: ddof = 1
    if arr_ndi == []: arr_ndi = [0]
    oneit_avg_dni = np.mean(arr_ndi)
    'op = std_dni'
    op = np.std( arr_ndi, ddof = ddof )
    if len(arr_ndi) > 1:
      if len([x for x in arr_ndi if x == 0]) > 0: 
        raise Exception("Error There's 0 ndi: dni, arr_daily, std", daily_new_inf, arr_ndi, op)

    'oneit_avg_R is the mean over the time of 1 sir. Then, avg over-all iterations'
    'Then, compute std_avg_R'
    degrees = [j for i,j in G.degree()]
    D = np.mean(degrees)
    #print("R0, b,m,D", beta*D/mu, beta, mu, D)
    c = beta*D/(mu*num_susc[0])
    oneit_avg_R = c*np.mean(num_susc)
    ddof = 0
    if len(num_susc) > 1: ddof = 1
    std_oneit_avg_R = c*np.std(num_susc, ddof = 1)
    #print("num_su[0], np.sum(num_susc), len(prev), oneit_avg_R2", \
    #  num_susc[0],np.sum(num_susc), len(prevalence), oneit_avg_R)

    return oneit_avg_R, std_oneit_avg_R, oneit_avg_dni, op, prevalence, cum_prevalence
  
  return prevalence, cum_prevalence

N = int(1e3)

G = nx.watts_strogatz_graph(N, k = 400, p = 0.1)

start = dt.datetime.now()

#itermean_sir(G, mf = False, numb_iter = 200, beta = 1e-3, mu = 0.05, start_inf = 10,verbose = False)

print("no // time", dt.datetime.now()-start)

start = dt.datetime.now()

# Step 1: Init multiprocessing.Pool()
#pool = mp.Pool(mp.cpu_count())

# Step 2: `pool.apply` the `howmany_within_range()`
#results = [pool.apply(howmany_within_range, args=(row, 4, 8)) for row in data]

for i in range(30):
  start = dt.datetime.now()
  #sir(G, False, 1e-3, 0.05, 10, False)
  print("no // time", i, dt.datetime.now()-start)
  start = dt.datetime.now()
  #pool = mp.Pool(mp.cpu_count())
  #pool.apply(sir, args = (G, False, 1e-3, 0.05, 10, False))
  print(sirjit(G, False, 1e-3, 0.05, 10, False))
  print("// time", i, dt.datetime.now()-start)
  #pool.close()  

# Step 3: Don't forget to close
#pool.close()    

#print(results[:10])

print("// time", dt.datetime.now()-start)