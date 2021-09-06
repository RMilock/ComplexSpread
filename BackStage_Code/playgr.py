import matplotlib.pylab as plt
import numpy as np
def rhu(n, decimals=0, integer = False): #round_half_up
    import math
    multiplier = 10 ** decimals
    res = math.floor(n*multiplier + 0.5) / multiplier
    if integer: return int(res)
    return res



def poisson(k, mu):
    import math
    return math.exp(-mu)*(mu**k) / math.factorial(k)

def bernoulli(N,k,p):
    import math
    import scipy.special as sp
    return sp.binom(N-1, k)*p**k*(1-p)**(N-1-k)

mu, N = 30, 1e2
p = mu / (N-1)
k = [int(x) for x in np.arange(mu-30,mu+30)]
k = list(filter(lambda x: x > 0, k))
print(f'mu/N-1: {mu/(N-1)}',)
print(f'mu: {mu}',)


yp = [poisson(x,mu) for x in k]
yb = [bernoulli(N,x,p) for x in k]

plt.plot(k,yp, label = "Poissonian")
plt.plot(k,yb, label = "Bernoullian")
plt.title(rf"Bernoullian VS Poissonian: D: {rhu(mu,3)}, D / N-1 = p = {rhu(mu/(N-1),3)}")
plt.legend()
plt.grid(color = "grey", lw = 0.4, ls = "--")
plt.show()

