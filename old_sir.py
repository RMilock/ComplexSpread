def sir(G, mf = False, beta = 1e-3, mu = 0.05, start_inf = 10, seed = False):
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
  #sir = {"S":{np.arange(node_labels)}}
  #sir["I"] = random.choices(sir["S"],)

  #if .pop(inf_list) exists. It's the right way. 
  #sir = {"S":{np.arange(node_labels).pop(inf_list)}, "I":{*inf_list}, "R":{}}
  #dic = {0: sir, 1: sir.copy()} 
  #e.g. dic[1]["S"]

  inf_list = random.sample(node_labels, start_inf)  #without replacement, i.e. not duplicates
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
          ls = list(range(N)); ls.remove(i)
          tests = random.sample(ls, k = int(mean)) #spread very fast since multiple infected center
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
    #prevalence.append(len(inf_list)/float(N))
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
