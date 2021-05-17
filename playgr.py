import numpy as np

beta_prog = np.linspace(0.001,1,15)
print(beta_prog)

a = np.linspace(0.001,0.01,8)
b = np.linspace(0.01,1,7)[1:]

c = np.concatenate((a,b))
#c = np.delete( c, np.where(c == 0.01)[0][0])

print(c)

bar_beta_prog = np.linspace(0.01,1,14)
beta_prog = np.concatenate(([0.001],bar_beta_prog))

print(beta_prog)