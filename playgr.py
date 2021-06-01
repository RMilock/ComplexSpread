from scipy.stats import poisson
import numpy as np
import matplotlib.pyplot as plt

bins = np.arange(-40,40)
D = 3.5643

rv = poisson(D)
y = rv.pmf(bins)
mean1 = rv.mean()
mean2 = np.sum(y*bins)/np.sum(y)

plt.plot(bins, y    )
print(mean1-mean2, mean1 == mean2 )
plt.show()