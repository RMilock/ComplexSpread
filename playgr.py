from networkx.algorithms.components.connected import number_connected_components
import numpy.random as npr
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from math import log, exp

x1 = np.linspace(0.1,10,100)
x = np.array([log(i) for i in x1])

y2 = np.array([exp(i) for i in x1])
y1 = np.array([i**(-2.1) for i in x1])
y = np.array([log(i**(-2.1)) for i in x1])
plt.plot(x1,y2, "g^", )
#plt.plot(x,y, "b--" )
plt.yscale("log")
#plt.xscale("log")
plt.show()