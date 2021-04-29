import numpy as np

b = list( range(30) )
total = len(b)

f = lambda x: x/total
quotient_b = list(map(f, b))

print(quotient_b)