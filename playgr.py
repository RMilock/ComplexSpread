import random
import networkx as nx
import numpy as np
import json
from definitions import N_D_std_D
N = 15
start_inf = 10

G = nx.complete_graph(N)

'''#sir = {"S":{np.arange(node_labels)}}
#sir["I"] = random.choices(sir["S"],)

#I left lists since the double items are already remove via random.choice
inn_sir = {"S":[x for x in node_labels if x not in inf_list], "I":inf_list, "R":[]}
#sir = {"S":{np.arange(node_labels).pop(inf_list)}, "I":{*inf_list}, "R":{}}
dsir = {0: inn_sir, 1: inn_sir.copy()} 
#e.g. dic[1]["S"]'''


def sir(G, mf = False, beta = 1, mu = 1, start_inf = 10, seed = True):
  'If mf == False, the neighbors are not fixed;' 
  'If mf == True, std mf by choosing @ rnd the num of neighbors'

  import random
  import pprint
  from functools import partial 
  from itertools import product, chain
  import copy

  #here's the modifications of the "test_ver1"
  'Number of nodes in the graph'
  N, mean, _ = N_D_std_D(G)

  'Label the idnividual wrt to the # of the node'
  node_labels = G.nodes()
  
  'Currently infected idnividuals and the future infected and recovered'   
  if seed: 
    random.seed(0)

  'Selects the seed of the disease'
  inf_list = random.sample(node_labels, start_inf) #inf nodes @ time t
  print("inf_list", inf_list)

  #no set since random.choice not double
  # current = fut.copy() not possible since set() have fixed leng --> use list
  inn_sir = {"S":[x for x in node_labels if x not in inf_list], "I":inf_list, "R":[]}
  #inn_sir1 since if not for future_state["S"].add(j) changes also current_state
  inn_sir1 = copy.deepcopy(inn_sir)
  #inn_sir1 = {"S":[x for x in node_labels if x not in inf_list], "I":inf_list.copy(), "R":[]}
  dsir = {0: inn_sir, 
          1: inn_sir1} 

  #or NEW IDEA: 
  # sir = {"S":{np.arange(node_labels)}}
  #sir["I"] = random.choices(sir["S"],)
  
  pp = pprint.PrettyPrinter(indent=4, sort_dicts=False)
  pp.pprint(dsir)

  'initilize prevalence (new daily infected) and recovered list'
  'we dont track infected'
  current_state = dsir[0] #['S' for i in node_labels] #sir[0]
  future_state = dsir[1] #['S' for i in node_labels] #sir[1]
  prevalence = [0] #[daily_new_inf/N] since dni ~ 0 in many cases. Start with 0 to have smaller std
  cum_prevalence = [start_inf/N]
  arr_dni = [] # arr to computer Std(daily_new_inf(t)) for daily_new_inf(t)!=0

  'start and continue whenever there s 1 infected'
  while(len(inf_list)>0):        
    daily_new_inf = 0
    
    def Gneigbors(i, mean, mf):
      if not mf: tests = G.neighbors(i) #only != wrt to the SIS: contact are taken from G.neighbors            
      if mf: 
          ls = list(range(N)); ls.remove(i)
          tests = random.sample(ls, k = int(mean)) #spread very fast since multiple infected center
      tests = list(map(lambda x : int(x), tests)) #convert 35.0 into int
      return tests

    fixed_G = partial(Gneigbors,mean = mean, mf = mf)
    tests = list(map(fixed_G, inf_list))

    #print("inf, tests", inf_list, "\ntests", tests)

    'Infection Phase: inf_list = prev_time infecteds'
    'each infected tries to infect all of the neighbors'

    #make [12] x [0,..,11,13,...19]
    prod = [product([i], tests[inf_list.index(i)]) for i in inf_list]
    prod = chain.from_iterable(prod)

    for i,j in prod:
        ''''Select the neighbors of the infected node'
        if not mf: tests = G.neighbors(i) #only != wrt to the SIS: contact are taken from G.neighbors            
        if mf: 
          ls = list(range(N)); ls.remove(i)
          tests = random.sample(ls, k = int(mean)) #spread very fast since multiple infected center
        tests = list(map(lambda x : int(x), tests)) #convert 35.0 into int
        for j in tests[inf_list.index(i)]:
          'If the contact is susceptible and not infected by another node in the future_state, try to infect it'
          #NEW IDEA: create a dict for current and future i.e.
          
          #if j in dic[0]["S"] and j in dic[1]["S"] and random.random() < beta:
          #  dic[1]["I"].add(j)
          '''
        if j in current_state["S"] and j in future_state["S"] and random.random() < beta:
          #print("f", future_state, "\nc", current_state, "\j", j)
          future_state["I"].append(j); future_state["S"].remove(j); daily_new_inf += 1
          #print("INFECTED f", future_state, "\nc", current_state, "\j", j)

    
    #New IDEA: this is more velox
    #if daily_new_inf: print("dail_n_i", daily_new_inf)
    if daily_new_inf: arr_dni = arr_dni+[daily_new_inf]
    #print("arr_ni", arr_dni)

    'Recovery Phase: only the prev inf nodes (=inf_list) recovers with probability mu'
    'not the new infected'    
    'This part is important in the OrderPar since diminishes the inf_list # that is in the "while-loop"'    

    def future_rec(i):
      if random.random() < mu:
        future_state["I"].remove(i)
        future_state["R"].append(i)
        return i


    print("new cicle current_state before recovering")
    pp.pprint(current_state)
    
    #for i in inf_list
    print("recovered nodes", list(map(lambda x: future_rec(x), current_state["I"])) )

    pp.pprint(current_state)
    pp.pprint(future_state)
    
    'Time update: once infections and recovery ended, we move to the next time-step'
    'The future state becomes the current one'
    current_state = copy.deepcopy(future_state) #w/o .copy() it's a mofiable-"view"
    
    'Updates inf_list with the currently fraction of inf/rec and save lenS to avg_R' 

    #IDEA: this is = to take the index on "current_state"
    inf_list = current_state["I"]
    #rec_list = current_state["R"]

    'Saves the fraction of new daily infected (dni) and recovered in the current time-step'
    prevalence.append(daily_new_inf/float(N))
    #prevalence.append(len(inf_list)/float(N))
    #recovered.append(len(rec_list)/float(N))
    cum_prevalence.append(cum_prevalence[-1]+daily_new_inf/N)  
    #loop +=1;
    #print("loop:", loop, cum_total_inf, cum_total_inf[-1], daily_new_inf/N)
  
  'Order Parameter (op) = Std(Avg_dni(t)) s.t. dni(t)!=0 as a func of D'
  if not mf:
    ddof = 1
    if not len(arr_dni)-1: ddof = 0
    if arr_dni == []: arr_dni = [0]
    oneit_avg_dni = np.mean(arr_dni)
    'op = std_dni'
    op = np.std( arr_dni, ddof = ddof )
    #if len(arr_dni) > 1:
    #  if len([x for x in arr_dni if not x]) > 0: 
    #    raise Exception("Error There's 0 dni: dni, arr_daily, std", daily_new_inf, arr_dni, op)
    return oneit_avg_dni, op, prevalence, cum_prevalence
  
  return prevalence, cum_prevalence

sir(G = G)