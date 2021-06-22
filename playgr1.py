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
current_state = np.array(["S", "R"]); future_state = np.array(["S","S"])
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
ray.get(res)



print(current_state, future_state)