import numpy as np
import ray
ray.init()

@ray.remote
def f(i):
  y[i] = "I"
  return y

#x = ray.get(f.remote(y = [3]))

#y = np.copy(x)
#y[0] = 1
#y += np.ones(100)
res = []
y = ["R","S"]
for i in [0,1]:
    res.append(f.remote(i))
print(ray.get(res))

'numba problems'
from numba import njit, prange
import numba as nb
import numpy.random as npr
inf_list = [2]
N = 4
current_state = np.array(["S","R","I","S","I"]); future_state = np.array(["S","R","I","S","I"])
#tests = np.array([0,1,2])
daily_new_inf = 0
@njit(parallel = True)
def pinff(tests,daily_new_inf,current_state,future_state, beta):
    'If the contact is susceptible and not infected by another node in the future_state, try to infect it'
    print("I'm in pinff")
    for j in prange(len(tests)):
        if current_state[tests[j]] == 'S' and future_state[tests[j]] == 'S':  
            if npr.random_sample() < beta:
                future_state[tests[j]] = 'I'; daily_new_inf += 1   
                print("Ive infected", tests[j], " ", future_state[tests[j]], " ", daily_new_inf)
            else:
                future_state[tests[j]] = 'S'

ray.init(num_cpus=2, ignore_reinit_error=True)
@ray.remote
def ptests_prep(i):
    mean = 3
    ls = np.concatenate((np.arange(i), np.arange(i+1,N)))
    tests = npr.choice(ls, size = int(mean), replace = False) #spread very fast since multiple infected center
    tests = np.array([int(x) for x in tests]) #convert 35.0 into int
    beta = 1
    print("\nInf node", i, "tests", tests, daily_new_inf)
    print("before pinff")
    pinff(tests,daily_new_inf,current_state,future_state,beta)
    print("jump all over pinff")
    return 1

res = []
print("inflist", inf_list)
for i in range(len(inf_list)):
    res.append(ptests_prep.remote(inf_list[i]))
res = ray.get(res)
print("ray.get(res)", res)

print(current_state, future_state)


'old stable ver'

print("old stable version")

current_state = np.array(["S","R","I"]); future_state = np.array(["S","R","I"])
tests = [0]
daily_new_inf = 0
@njit(parallel = True)
def pinff(daily_new_inf,current_state,future_state,beta):
    'If the contact is susceptible and not infected by another node in the future_state, try to infect it'
    print("I'm in pinff")
    for j in prange(len(current_state)):
        if current_state[j] == 'S' and future_state[j] == 'S':  
            if npr.random_sample() < beta:
                future_state[j] = 'I'; daily_new_inf += 1   
                print("Ive infected", j, future_state[j], daily_new_inf)
            else:
                future_state[j] = 'S'
    return daily_new_inf,current_state,future_state 

ray.init(num_cpus=2, ignore_reinit_error=True)
@ray.remote
def ptests_prep(i):
    pinff(daily_new_inf,current_state,future_state,1)
    print("i", i)

res = []
for i in range(10):
    res.append(ptests_prep.remote(i))

import time 
time.sleep(4)    
print(ray.get(res))

print(current_state, future_state)