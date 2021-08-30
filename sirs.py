#@njit(parallel = True)
def filter_out_k(arr, k = 0):
  filtered = np.array([np.float64(x) for x in np.arange(0)])
  for i in np.arange(arr.size):
      if arr[i] == k:
        filtered = np.append(filtered, arr[i])
  return filtered

#@njit(parallel = True)
def pinff(j,daily_new_inf,current_state,future_state,beta):
  'If the contact is susceptible and not infected by another node in the future_state, try to infect it'
  if current_state[j] == 'S' and future_state[j] == 'S':  
    if npr.random_sample() < beta:
      future_state[j] = 'I'; daily_new_inf += 1   
    else:
        future_state[j] = 'S'
  return current_state, future_state, daily_new_inf

#@njit(parallel = True)
def preclist(i,future_state,mu): 
  #for i in np.range(len(inf_list)):
  if npr.random_sample() < mu:
    future_state[i] = 'R'
  else:
    future_state[i] = 'I'
  return future_state

def sirnumpy(G, mf = False, beta = 1e-3, mu = 0.05, start_inf = 10, seed = False):
  'If mf == False, the neighbors are not fixed;' 
  'If mf == True, std mf by choosing @ rnd the num of neighbors'

  #here's the modifications of the "test_ver1"
  'Number of nodes in the graph'
  N, mean, _ = N_D_std_D(G)

  beta, mu = 1,1

  'Label the idnividual wrt to the # of the node'
  node_labels = G.nodes()
  
  'Currently infected idnividuals and the future infected and recovered' 
  inf_list = np.array([], dtype = int) #infected node list @ each t
  rec_list = np.asarray([]) #recovered nodes for a fixed t
  arr_dni = np.asarray([]) #arr to computer Std(daily_new_inf(t)) for daily_new_inf(t)!=0

  'Initial Codnitions'
  #global current_state
  current_state = np.asarray(['S' for i in node_labels])

  #global future_state
  future_state = np.asarray(['S' for i in node_labels])
  
  if seed: npr.seed(0)

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

  'start and continue whenever there s 1 infected'
  while(len(inf_list)>0):    
    #global daily_new_inf 
    daily_new_inf = 0
    'Infection Phase: inf_list = current_time infected'
    'each infected tries to infect all of the neighbors'
    for i in inf_list:
      'Select the neighbors from the net of the infected node'
      if not mf: tests = np.asarray(list(G.neighbors(i))) 
      'Select the neighbors at random as in a mean-field theory'
      if mf: 
        ls = np.concatenate((np.arange(i), np.arange(i+1,N)))
        tests = npr.choice(ls, size = rhu(mean, integer = True)) #spread very fast since multiple infected center
      tests = tests.astype(int) #convert 35.0 into int
      #print(tests, type(tests))

      #print("I'm infecting with the node %s of %s" % (i, inf_list))
      for i in tests:
        #print("daily new infected", daily_new_inf)
        _,_,daily_new_inf = pinff(i, daily_new_inf, current_state, future_state, beta)
        #if daily_new_inf: print("dni!=0 current-future out", daily_new_inf, "\n", \
                            #current_state, "\n", future_state)
      '''
      with Pool(processes=4) as f:
        f.starmap(pinff, [(i, daily_new_inf, current_state, future_state,1) for i in tests])
      '''
    
    
    #print("dail_n_i", daily_new_inf)
    if daily_new_inf != 0: arr_dni = np.append(arr_dni,daily_new_inf)
    #print("arr_ni", arr_dni)

    'Recovery Phase: only the prev inf nodes (=inf_list) recovers with probability mu'
    'not the new infected'    
    'This part is important in the OrderPar since diminishes the inf_list # that is in the "while-loop"'    


    for i in inf_list:
      preclist(i, future_state,mu)
    
    '''
    with Pool(processes=4) as f:
      f.starmap(preclist, [(i, future_state, mu) for i in range(len(inf_list))])
    '''

    'Time update: once infections and recovery ended, we move to the next time-step'
    'The future state becomes the current one'
    current_state = future_state.copy() #w/o .copy() it's a mofiable-"view"
    
    #print("end curr-futu", current_state, future_state)
    'Updates inf_list with the currently fraction of inf/rec and save lenS to avg_R' 
    inf_list = np.asarray([i for i, x in enumerate(current_state) if x == 'I'])
    rec_list = np.asarray([i for i, x in enumerate(current_state) if x == 'R'])

    'Saves the fraction of new daily infected (dni) and recovered in the current time-step'
    prevalence = np.append(prevalence, daily_new_inf/float(N))
    #prevalence.append(len(inf_list)/float(N))
    #recovered.append(len(rec_list)/float(N))
    cum_prevalence = np.append(cum_prevalence, cum_prevalence[-1]+daily_new_inf/N)  
    #loop +=1;
    #print("loop:", loop, cum_total_inf, cum_total_inf[-1], daily_new_inf/N)

    np.append(num_susc, N*(1 - cum_prevalence[-1]))
    #print("\nnum_susc, prevalence, recovered",num_susc, prevalence, recovered, 
    #len(num_susc), len(prevalence), len(recovered))

  #print("inf, rec", inf_list, rec_list)
  'Order Parameter (op) = Std(Avg_dni(t)) s.t. dni(t)!=0 as a func of D'
  if not mf:
    ddof = 0
    if len(arr_dni) > 1: ddof = 1
    if not arr_dni.size: arr_dni = np.asarray([0])
    oneit_avg_dni = np.mean(arr_dni)
    #print(ddof)
    'op = std_dni'
    op = np.std( arr_dni, ddof = ddof )
    if len(arr_dni) > 1:
      #filtered = np.array([])
      if len(filter_out_k(arr_dni, k=0)) > 0: #[x for x in arr_dni if x == 0])
        raise Exception("Error There's 0 dni: dni, arr_daily, std", daily_new_inf, arr_dni, op)

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

def sir(G, mf = False, beta = 1e-3, mu = 0.05, start_inf = 10, seed = False):
  #this is the regular sir
  'If mf == False, the neighbors are not fixed;' 
  'If mf == True, std mf by choosing @ rnd the num of neighbors'

  import random
  #here's the modifications of the "test_ver1"
  'Number of nodes in the graph'
  N, mean, _ = N_D_std_D(G)

  'Label the idnividual wrt to the # of the node'
  node_labels = G.nodes()
  
  'Currently infected idnividuals and the future infected and recovered' 
  inf_list = [] #infected node list @ each t
  prevalence = [] # = len(inf_list)/N, i.e. frac of daily infected for every t
  recovered = [] #recovered nodes for a fixed t
  arr_dni = [] #arr to computer Std(daily_new_inf(t)) for daily_new_inf(t)!=0

  'Initial Codnitions'
  current_state = ['S' for i in node_labels] 
  future_state = ['S' for i in node_labels]
  
  if seed: 
    random.seed(0)

  'Selects the seed of the disease'
  inf_list = random.sample(node_labels, start_inf)

  inf_list = random.sample(node_labels, start_inf)  #without replacement, i.e. not duplicates
  if rhu(mean,0)-1 <= 0 and mf: #too slow for D = 1
    inf_list = []
  for seed in inf_list:
    current_state[seed] = 'I'
    future_state[seed] = 'I'

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
          #if rhu(mean,0)-1 <= 0: #too slow for D = 1
          #  mean = 0
          ls = list(range(N)); ls.remove(i)
          #print(rhu(mean), type(rhu(mean)))
          tests = random.sample(ls, k = int(rhu(mean))) #spread very fast since multiple infected center
        tests = [int(x) for x in tests] #convert 35.0 into int
        for j in tests:
          'If the contact is susceptible and not infected by another node in the future_state, try to infect it'
          #NEW IDEA: create a dict for current and future i.e.
          
          #if j in dic[0]["S"] and j in dic[1]["S"] and random.random() < beta:
          #  dic[1]["I"].add(j)

          if current_state[j] == 'S' and future_state[j] == 'S' and random.random() < beta:
            future_state[j] = 'I'; daily_new_inf += 1
    
    #New IDEA: this is more velox
    #print("dail_n_i", daily_new_inf)
    if daily_new_inf != 0: arr_dni = arr_dni+[daily_new_inf]
    #print("arr_ni", arr_dni)

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

    #IDEA: this is = to take the index on "current_state"
    inf_list = [i for i, x in enumerate(current_state) if x == 'I']
    rec_list = [i for i, x in enumerate(current_state) if x == 'R']

    'Saves the fraction of new daily infected (dni) and recovered in the current time-step'
    prevalence.append(daily_new_inf/float(N))
    #infected.append(len(inf_list)/float(N))
    #recovered.append(len(rec_list)/float(N))
    cum_prevalence.append(cum_prevalence[-1]+daily_new_inf/N)  
    #loop +=1;
    #print("loop:", loop, cum_total_inf, cum_total_inf[-1], daily_new_inf/N)

    num_susc.append(N*(1 - cum_prevalence[-1]))
    #print("\nnum_susc, prevalence, recovered",num_susc, prevalence, recovered, 
    #len(num_susc), len(prevalence), len(recovered))
  
  'Order Parameter (op) = Std(Avg_dni(t)) s.t. dni(t)!=0 as a func of D'
  if not mf:
    ddof = 0
    if len(arr_dni) > 1: ddof = 1
    if arr_dni == []: arr_dni = [0]
    oneit_avg_dni = np.mean(arr_dni)
    'op = std_dni'
    op = np.std( arr_dni, ddof = ddof )
    if len(arr_dni) > 1:
      if len([x for x in arr_dni if x == 0]) > 0: 
        raise Exception("Error There's 0 dni: dni, arr_daily, std", daily_new_inf, arr_dni, op)

    'oneit_avg_R is the mean over the time of 1 sir. Then, avg over-all iterations'
    'Then, compute std_avg_R'
    degrees = [j for _,j in G.degree()]
    D = np.mean(degrees)
    #print("R0, b,m,D", beta*D/mu, beta, mu, D)
    c = beta*D/(mu*num_susc[0])
    oneit_avg_R = c*np.mean(num_susc)
    ddof = 0
    if len(num_susc) > 1: ddof = 1
    std_oneit_avg_R = c*np.std(num_susc, ddof = 1)
    #print("num_su[0], np.sum(num_susc), len(prev), oneit_avg_R2", \
    #  num_susc[0],np.sum(num_susc), len(prevalence), oneit_avg_R)

    #return oneit_avg_R, std_oneit_avg_R, oneit_avg_dni, op, prevalence, cum_prevalence
    return oneit_avg_dni, op, prevalence, cum_prevalence
  
  return prevalence, cum_prevalence

def sirvec(G, mf = False, beta = 1e-3, mu = 0.05, start_inf = 10, seed = False):
  'If mf == False, the neighbors are not fixed;' 
  'If mf == True, std mf by choosing @ rnd the num of neighbors'

  import random
  import numpy.random as npr
  import pprint
  from functools import partial 
  from itertools import product, chain
  import copy

  #here's the modifications of the "test_ver1"
  'Number of nodes in the graph'
  N, mean, _ = N_D_std_D(G)

  'Label the idnividual wrt to the # of the node'
  node_labels = list(G.nodes())
  
  'Currently infected idnividuals and the future infected and recovered'   
  if seed: 
    random.seed(0)

  'Selects the seed of the disease'
  inf_list = random.sample(node_labels, start_inf) #inf nodes @ time t
  #print("inf_list", inf_list)

  #no set since random.choice not double
  # current = fut.copy() not possible since set() have fixed leng --> use list
  inn_sir = {"S":list(set(node_labels) - set(inf_list)), "I":inf_list, "R":[]}
  #inn_sir1 since if not for future_state["S"].add(j) changes also current_state
  inn_sir1 = copy.deepcopy(inn_sir)
  #inn_sir1 = {"S":[x for x in node_labels if x not in inf_list], "I":inf_list.copy(), "R":[]}
  dsir = {0: inn_sir, 1: inn_sir1} 

  #or NEW IDEA: 
  # sir = {"S":{np.arange(node_labels)}}
  #sir["I"] = random.choices(sir["S"],)
  
  pp = pprint.PrettyPrinter(indent=4, sort_dicts=False)
  #pp.pprint(dsir)

  'initilize prevalence (new daily infected) and recovered list'
  'we dont track infected'
  current_state = dsir[0] #['S' for i in node_labels] #sir[0]
  future_state = dsir[1] #['S' for i in node_labels] #sir[1]
  prevalence = [start_inf/N] #[daily_new_inf/N] since dni ~ 0 in many cases. Start with 0 to have smaller std
  cum_prevalence = [start_inf/N]
  arr_dni = [] # arr to computer Std(daily_new_inf(t)) for daily_new_inf(t)!=0

  'start and continue whenever there s 1 infected'
  while(len(inf_list)>0):        
    daily_new_inf = 0
    
    def Gneigbors(i, mean, mf):
      if not mf: 
        tests = G.neighbors(i) #only != wrt to the SIS: contact are taken from G.neighbors            
      if mf: 
          ls = np.arange(N); ls = ls[ls!=i]
          tests = npr.choice(ls, size =  int(mean), replace = False) #spread very fast since multiple infected center
      #tests = np.array(tests).astype(int) #convert 35.0 into int
      return tests

    #import time 
    #start = time.time()
    fixed_G = partial(Gneigbors, mean = mean, mf = mf)
    tests = list(map(fixed_G, inf_list))
    print( "tests", tests )
    #print("time in secs, ", time.time()-start)
    #print("inf, tests", inf_list, "\ntests", tests)

    'Infection Phase: inf_list = prev_time infecteds'
    'each infected tries to infect all of the neighbors'

    #make [12] x [0,..,11,13,...19]
    
    #prod = [product([i], tests[inf_list.index(i)]) for i in inf_list]
    prod = chain.from_iterable(tests) #this is an iterable

    #print("prod", list(prod))

    for j in prod:
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
          future_state["I"].append(j); future_state["S"].remove(j); daily_new_inf += 1
          #print("Infection node", j); pp.pprint(current_state)
          #pp.pprint(future_state);  print("end of inf\n")      
    
    #New IDEA: this is more velox
    #if daily_new_inf: print("dail_n_i", daily_new_inf)
    if daily_new_inf: arr_dni = arr_dni+[daily_new_inf]
    #print("arr_ni", arr_dni)

    'Recovery Phase: only the prev inf nodes (=inf_list) recovers with probability mu'
    'not the new infected'    
    'This part is important in the OrderPar since diminishes the inf_list # that is in the "while-loop"'    

    def future_rec(i):
      if random.random() < mu:
        #future_state["I"].remove(i)
        #future_state["R"].append(i)
        #print("Recovering node", i); pp.pprint(current_state); 
        #pp.pprint(future_state); print("End of Rec\n")
        return i


    #print("new cicle current_state before recovering")
    #pp.pprint(current_state)
    
    #questo funziona
    future_state["I"] = random.sample(future_state["I"], 
                        k = rhu(mu*len(future_state["I"]), integer = True)
    )
    #rec_list = list(map(lambda x: future_rec(x), current_state["I"]))
    #if np.any(rec_list): print("recovered nodes", rec_list )
      
    
    'Time update: once infections and recovery ended, we move to the next time-step'
    'The future state becomes the current one'
    current_state = copy.deepcopy(future_state) #w/o .copy() it's a mofiable-"view"
    
    'Updates inf_list with the currently fraction of inf/rec and save lenS to avg_R' 

    #IDEA: this is = to take the index on "current_state"
    inf_list = current_state["I"]
    #rec_list = current_state["R"]

    'Saves the fraction of new daily infected (dni) and recovered in the current time-step'
    prevalence.append(daily_new_inf/float(N))
    cum_prevalence.append(cum_prevalence[-1]+daily_new_inf/N)  

  'Order Parameter (op) = Std(Avg_dni(t)) s.t. dni(t)!=0 as a func of D'
  if not mf:
    ddof = 1
    if len(arr_dni) <= 1: ddof = 0
    if arr_dni == []: arr_dni = [0]
    oneit_avg_dni = np.mean(arr_dni)
    'op = std_dni'
    op = np.std( arr_dni, ddof = ddof )
    #if len(arr_dni) > 1:
    #  if len([x for x in arr_dni if not x]) > 0: 
    #    raise Exception("Error There's 0 dni: dni, arr_daily, std", daily_new_inf, arr_dni, op)
    return oneit_avg_dni, op, prevalence, cum_prevalence
  
  return prevalence, cum_prevalence
