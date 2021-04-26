import numpy as np
from itertools import product


D = np.arange(2,16,2)
p_prog = np.concatenate((np.array([0.001]), np.linspace(0.012,1,9)))

print( list([(D[i],p_prog*D[i]) for i in range(len(D))]))

count = 0
for d, p in product(D,p_prog):
    D_pruned = D[D!=d]
    print(d,p,D_pruned)
    p_pruned = p_prog[p_prog!=p]
    for d2, p2 in product(D_pruned, p_pruned):
        if d*p == d2*p2: count+=1; print("This is equal", d*p)

print("There're %s equals values" % count)