import numpy as np
from functools import reduce
import time
import random

N = int(1e3)
l = random.sample(range(0, N), 50)

def test1():
    return np.mean(np.array(l))

def test2():
    return sum(l) / float(len(l))

def test3():
    return reduce(lambda x, y: x + y, l) / len(l)

def timed():
    start = time.time()
    test1()
    print('{} seconds'.format(time.time() - start))
    start = time.time()
    test2()
    print('{} seconds'.format(time.time() - start))
    start = time.time()
    test3()
    print('{} seconds'.format(time.time() - start))

timed()