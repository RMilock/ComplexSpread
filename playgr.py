import matplotlib.pyplot as plt
import numpy as np
import itertools
import random
from mfVSnetdefinitions import sir, itermean_sir, plot_sir,plot_G_degdist_adjmat_sir
import networkx as nx

#numb_classes = 3; numb_iter = 30
beta = 0.1; mu = 0.16; N = int(100); D = N; numb_iter = 500
G = nx.connected_watts_strogatz_graph(N, D, p = 0, seed = 1) #k is the number of near linked nodes

plot_sir(G, D = D, beta = beta, mu = mu, numb_iter=numb_iter) #mf! expected

#Net Traj
traj, avg = itermean_sir(G, beta = beta, mu = mu, D = None, numb_iter=numb_iter)
'plotting the many realisations'  

plt.plot(avg[0], marker = "o", markersize = 3, label="Net_Infected/N", color = "darkblue") #prevalence
plt.plot(avg[1], marker = "*", markersize = 5, linestyle = "None",  label= "Net_Recovered/N", color = "darkorange" ) #recovered
plt.plot(avg[2], marker = "o", markersize = 3, linestyle = "None", label="Net_CD_Inf /N", color = "darkgreen") #cum_positives
plt.legend()

##MF traj
traj, avg = itermean_sir(G, beta = beta, mu = mu, D = D, numb_iter=numb_iter)

plt.plot(avg[0], marker = "*", markersize = 5, linestyle = "None", label="mf_Infected/N", color = "darkblue") #prevalence
plt.plot(avg[1], marker = "o", markersize = 3, linestyle = "None", label="mf_Recovered/N", color = "darkorange" ) #recovered
plt.plot(avg[2], marker = "*", markersize = 5, linestyle = "None", label="mf_Inf /N", color = "darkgreen") #cum_positives

plt.legend()
plt.show()