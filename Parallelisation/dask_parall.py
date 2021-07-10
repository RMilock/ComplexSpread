from dask.distributed import Client, progress
client = Client(processes=False, threads_per_worker=4,
                n_workers=1, memory_limit='4GB')


import dask.array as da
'numba problems'
from numba import jit, prange
import numpy.random as npr
import numpy as np
import dask.array as da
import time
from dask import delayed

inf_list = [2,0]
N = 4
current_state = np.array(["I","S","I","S","S","S","S"]); future_state = np.array(["I","S","I","S","S","S","S"])
#future_state = future_state.persist()
#tests = da.array([0,1,2])
daily_new_inf = 0

@jit(nopython = True, parallel = True)
def pinff(tests,daily_new_inf,current_state,future_state,beta):
    'If the contact is susceptible and not infected by another node in the future_state, try to infect it'
    print("I'm in pinff")
    for j in prange(len(tests)):
        if current_state[tests[j]] == 'S' and future_state[tests[j]] == 'S':  
            if npr.random_sample() < beta:
                future_state[tests[j]] = 'I'; daily_new_inf += 1   
                print("Ive infected", tests[j], " ", future_state[tests[j]], " ", daily_new_inf)
            else:
                future_state[tests[j]] = 'S'
    print("future_state", future_state)
    return current_state, future_state, daily_new_inf


def ptests_prep(i,daily_new_inf,current_state,future_state):
    mean = 3
    ls = np.concatenate((np.arange(i), np.arange(i+1,7)))
    tests = npr.choice(ls, size = int(mean), replace = False) #spread very fast since multiple infected center
    tests = np.array([int(x) for x in tests]) #convert 35.0 into int
    beta = 1
    print("\nInf node", i, "tests", tests, daily_new_inf)
    print("before pinff")
    #def f(*args):
    #    return [np.array(arg) for arg in args]
    #args = f(current_state,future_state)
    #tests = args[0] 
    #current_state = args[0]
    #future_state = args[1]
    pinff(tests,daily_new_inf,current_state,future_state,beta)
    print("jump all over pinff")
    return future_state

start = time.time()
print(pinff(np.array([0]),0,np.array(["S"]),np.array(["R"]),1))
print("time of 1st compilation", time.time()-start)


start = time.time()
print("\nSecond inflist", inf_list)
for i in inf_list:
    ptests_prep(i,daily_new_inf,current_state,future_state)    

print(current_state, future_state, time.time()-start )


client.close()
