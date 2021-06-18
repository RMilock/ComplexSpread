import multiprocessing

from numba.np.ufunc import parallel

from definitions import N_D_std_D
import numpy as np
import numpy.random as npr
import numba as nb
import networkx as nx
from numba import jit
import timeit

@jit(nopython=True)
def f(x):
    tmp = np.array([np.float64(x) for x in np.arange(0)]) # defined empty
    for i in np.arange(x):
        tmp = np.append(tmp,i+1) # list type can be inferred from the type of `i`
    return tmp

#print(f(5))
N = int(1e3)
G = nx.complete_graph(N)

N, mean, _ = N_D_std_D(G)

'Label the individual wrt to the # of the node'
node_labels = G.nodes()

'Currently infected individuals and the future infected and recovered' 
start_inf = 10
inf_list = np.array([], dtype = int) #infected node list @ each t
prevalence = np.asarray([start_inf/N]) # = len(inf_list)/N, i.e. frac of daily infected for every t
rec_list = np.asarray([0]) #recovered nodes for a fixed t
arr_ndi = np.array([np.int64(x) for x in np.arange(0)]) #arr to computer Std(daily_new_inf(t)) for daily_new_inf(t)!=0

'Initial Conditions'
current_state = np.asarray(['S' for i in node_labels])
future_state = np.asarray(['S' for i in node_labels])
mf = False; beta = 1e-1; mu = 0.6; 
#daily_new_inf = 0

'Selects the seed of the disease'
seeds = npr.choice(node_labels, start_inf, replace = False)  #without replacement, i.e. not duplicates
for seed in seeds:
    current_state[seed] = 'I'
    future_state[seed] = 'I'
    inf_list = np.append(inf_list,seed)

print("inf_list",inf_list)

#all this could be put in the for seed right?
test_G = np.array([], dtype= "i8")
ls = []
for i in inf_list:
    ls.append([x for x in G.neighbors(i)])
    test_G = np.asarray(ls)

print(test_G)
from numba import prange

@jit(nopython = True, parallel = True)
def nbinfection(N,test_G,mf,mean,beta,inf_list,current_state,future_state,arr_ndi):
    count = 0
    while(len(inf_list)>0):
        daily_new_inf = np.int64(0)
        for i in prange(len(inf_list)): #if inf_list throws error
            'Select the neighbors of the infected node'

            if not mf: 
                tests = test_G[i] #these are G.neighbors[i] -- numb non riconosce la class nx            
                #print("inside nbinf", test_G[i])
            if mf:
                ls = np.concatenate((np.arange(inf_list[i]), np.arange(inf_list[i]+1,N)))
                tests = npr.choice(ls, size = int(mean)) #spread very fast since multiple infected center
            #print("inf_list", inf_list)
            #print("tests %s for node %s" % (tests,inf_list[i]))
            #print("current state", current_state)
            for j in prange(len(tests)):
                'If the contact is susceptible and not infected by another node in the future_state, try to infect it'
                if current_state[tests[j]] == 'S' and future_state[tests[j]] == 'S':
                    if npr.random_sample() < beta:
                        future_state[tests[j]] = 'I'; daily_new_inf += 1   
                        #print("node", tests[j], "is infected")  
                    else:
                        future_state[tests[j]] = 'S'
            #print("future state", future_state)    
                
        if not daily_new_inf: 
            arr_ndi = np.append(arr_ndi,daily_new_inf)
        #print("arr_ni", arr_ndi)

        'Recovery Phase: only the prev inf nodes (=inf_list) recovers with probability mu'
        'not the new infected'    
        'This part is important in the OrderPar since diminishes the inf_list # that is in the "while-loop"'    
        
        for i in prange(len(inf_list)):
            if npr.random_sample() < mu:
                future_state[inf_list[i]] = 'R'
                count += 1
                #print("Recovered node and count", (inf_list[i], count))
            else:
                future_state[inf_list[i]] = 'I'  
        'Time update: once infections and recovery ended, we move to the next time-step'
        'The future state becomes the current one'
        #print("future state", future_state)
        current_state = future_state.copy() #w/o .copy() it's a mofiable-"view"
        
        'Updates inf_list with the currently fraction of inf/rec and save lenS to avg_R' 
        inf_list = np.asarray([i for i, x in enumerate(current_state) if x == 'I'])
        #print("numb_inf", inf_list)
        rec_list = np.asarray([i for i, x in enumerate(current_state) if x == 'R'])
    
        return current_state


start = timeit.default_timer()
nbinfection(N,test_G,mf,mean,beta,inf_list,current_state,future_state,arr_ndi)
print(timeit.default_timer()-start)

print("\nStart Of a New Infection")
start = timeit.default_timer()
print(nbinfection(N,test_G,mf,mean,beta,inf_list,current_state,future_state, arr_ndi),
timeit.default_timer()-start
)

