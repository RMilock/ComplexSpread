'def a function that iters numb_iter and make an avg of the trajectories'
from itertools import zip_longest
import numpy as np
import copy

numb_idx_cl = 2
trajectories = [[] for _ in range(numb_idx_cl)]
avg = [[] for _ in range(numb_idx_cl)]
counts = [[],[],[]]
max_len = 0
numb_iter = 2

tank_of_values = [[3,4,5], [2,4,6,8,10]],[[1,2,3,4,4],[4,5,6]]

import datetime as dt
start_time = dt.datetime.now()

for i in range(numb_iter):
  sir_start_time = dt.datetime.now()
  tmp_traj = tank_of_values[i]
  print(tmp_traj)
  for idx_cl in range(numb_idx_cl):
      #if idx_cl == 0: print("\ntmp_traj", tmp_traj)
      trajectories[idx_cl].append(tmp_traj[idx_cl])
      tmp_max = len(max(tmp_traj, key = len))
      if tmp_max > max_len: max_len = tmp_max
      #print("\nIteration: %s, tmp_max: %s, len tmp_traj: %s, len tmp_traj %s, len traj[%s] %s" % 
      #  (i, len(max(tmp_traj, key = len)), len(tmp_traj),  \
      #    len(tmp_traj[idx_cl]), idx_cl, len(trajectories[idx_cl]) ))
#print("\nOverall max_len", max_len)
#print("All traj", trajectories)
plot_trajectories = copy.deepcopy(trajectories)

start_time = dt.datetime.now()

for i in range(numb_iter):
  if i % 50 == 0: print("time for %s for avg-for-loop %s" % (i, dt.datetime.now()-start_time))
  for idx_cl in range(numb_idx_cl):
      last_el_list = [trajectories[idx_cl][i][-1] for _ in range(max_len-len(trajectories[idx_cl][i]))]
      'traj[classes to be considered, e.g. infected = 0][precise iteration we want, e.g. "-1"]'
      trajectories[idx_cl][i] += last_el_list
      length = len(trajectories[idx_cl][i])
      it_sum = [sum(x) for x in zip_longest(*trajectories[idx_cl], fillvalue=0)]
      for j in range(length):
          try: counts[idx_cl][j]+=1
          except: counts[idx_cl].append(1)
      avg[idx_cl] = list(np.divide(it_sum,counts[idx_cl]))
      
      
      print("\niteration(s):", i, "idx_cl ", idx_cl)
      print("last el extension", last_el_list)
      print("(new) trajectories[%s]: %s" % (idx_cl, trajectories[idx_cl]))
      print( "--> trajectories[%s][%s]: %s" % (idx_cl, i, trajectories[idx_cl][i]), 
      "len:", length)
      print("zip_longest same index" , list(zip_longest(*trajectories[idx_cl], fillvalue=0)))#"and traj_idx_cl", trajectories[idx_cl])
      print("global sum same indeces", it_sum)
      print("counts of made its", counts[idx_cl])
      print("avg", avg)

import matplotlib.pyplot as plt
x = np.arange(max_len)
plt.plot(trajectories[0][0], "r--", label = "first bunch")
plt.plot(trajectories[0][1], "b--", label = "second bunch")
plt.plot(avg[0], "o", label = "bunch averaged")
#plt.plot(avg[1], "*", label = "second bunch averaged")
plt.title("Trial of the avg Algorithm")
plt.legend()
plt.show()
      