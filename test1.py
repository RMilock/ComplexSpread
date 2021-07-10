import random
import networkx as nx
import numpy as np
import json
from definitions import N_D_std_D
from functools import partial

def Gneigbors(i, mean, mf):
    if not mf: tests = G.neighbors(i) #only != wrt to the SIS: contact are taken from G.neighbors            
    if mf: 
        ls = list(range(N)); ls.remove(i)
        tests = random.sample(ls, k = int(mean)) #spread very fast since multiple infected center
    tests = list(map(lambda x : int(x), tests)) #convert 35.0 into int
    return tests


N = 20
start_inf = 10

G = nx.complete_graph(N)
mf = True
_,mean,_= N_D_std_D(G)

fixed_G = partial(Gneigbors,mean = mean, mf = mf)
print( list(map(fixed_G, G.nodes())) )

from itertools import product, chain
inf_list = [1,2,3]; tests = [[1,2,3],[3,4,5],[6,7,8]]
prod = [product([i], tests[inf_list.index(i)]) for i in inf_list]
print(chain.from_iterable(prod))


for i,j in chain.from_iterable(prod):
    print(i,j)

import random
import networkx as nx
import numpy as np
import json
from functools import partial
from itertools import chain


L = int(2)

a = list(range(1,L)) #1
b = list(range(L, 2*L)) #2,3
import time
start = time.time()
a[len(a):len(a)] = b
print("1",time.time()-start)
print(len(a))

a = list(range(1,L)) #1
b = list(range(L, 2*L)) #2,3
start = time.time()
a.append(b)
print("2",time.time()-start)
print(len(a))


a = list(range(1,L)) #1
b = list(range(L, 2*L)) #2,3
start = time.time()
a = a+b
print("3", time.time()-start)
print(len(a))

a = list(range(1,L)) #1
b = list(range(L, 2*L)) #2,3
start = time.time()
a+=b
print("4",time.time()-start)
print(len(a))

a = list(range(1,L)) #1
b = list(range(L, 2*L)) #2,3
start = time.time()
a = chain(a,b)
print("5", time.time()-start)
print(len(list(a)))


a = list(range(1,L)) #1
b = list(range(L, 2*L)) #2,3
start = time.time()
max_len = len(np.max(a, axis = 0))
print("6: max", time.time()-start)
print(max_len)

a = list(range(1,L)) #1
b = list(range(L, 2*L)) #2,3
start = time.time()
max_len = max(a, key=len)
print("6: max", time.time()-start)
print(max_len)