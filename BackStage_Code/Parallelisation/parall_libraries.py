from multiprocessing import Pool
import numpy as np
from functools import partial

def inner_loop(ls,N,i):
    for j in np.arange(N):
        ls.append(i+j)
    return ls

def outer_loop(ls,N):
    for i in np.arange(N):
        print("Entered in out_loop")
        inner_loop(i,ls,N)
    return ls

N = 2; ls = []

for i in np.arange(N):
    print(inner_loop(ls,N,i))

ls = []
pinner_l = partial(inner_loop,ls,N)
#print( pouter_l(np.arange(2)) )
with Pool(1) as p:
    result = p.map(pinner_l, np.arange(N))

print(result)

'parallelization with numba'
from numba import jit

@jit(nopython=True)
def f(x):
    tmp = np.array([np.float64(x) for x in np.arange(0)]) # defined empty
    for i in np.arange(x):
        tmp = np.append(tmp,i+1) # list type can be inferred from the type of `i`
    return tmp

print(f(3))

'parallelize with multiprocesses'
ls = []
def psquare(x, a, b):
    ls = x**2*a+b
    return ls

if __name__ == '__main__':
    ls = [33]
    print(psquare(1,1,1))
    with Pool(2) as p:
        ls = [p.apply(psquare, args = (x, 1, 1)) for x in np.arange(N)]
    print("ls_pool", ls)

'// with ray'

'problem with import ray since I installed it using pip not conda'
import ray
import pandas as pd

ray.init()
@ray.remote
def rsquare(x, a, b):
    return a*x**2+b

@ray.remote
def rsquare2(x, a, b):
    return a*x**2+b

res =  []

for j in np.arange(N):
    res.append(rsquare.remote(j,1,1))

print( ray.get(res) )
res[0] = 3
print(res)

from joblib import Parallel, delayed
from itertools import product
jres = Parallel(n_jobs = 2)(delayed(psquare)(x,a,b) for x,a,b in product(np.arange(N),[2],[1]))
print( jres )


'fill an array with multiprocess'

import numpy as np
from functools import partial
from multiprocessing import Process, Pool
import multiprocessing.managers

x = 3;
y = 3;
z = 10; 

class MyManager(multiprocessing.managers.BaseManager):
    pass

MyManager.register('np_zeros', np.zeros, multiprocessing.managers.ArrayProxy)


para = []
process = []
def local_func(ar, section):
    print("section %s" % str(section))
    ar[section] = section 
    print("value set %d" % ar[section])

if __name__ == "__main__":
    m = MyManager()
    m.start()
    ar = m.np_zeros((z))

    pool = Pool(1)

    run_list = range(0,10)
    func = partial(local_func, ar)
    list_of_results = pool.map(func, run_list)

    print(ar)



