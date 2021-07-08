import matplotlib.pyplot as plt
import numpy as np
import nagano_cy as nag

e = np.arange(0.0, 1.0, 0.15).reshape(-1, 1)
nu = np.linspace(0, 4*np.pi, 50000)
x =  np.tan(nu/2.)
M2evals = np.arctan2(1,1/x)
#M2evals = np.arctan(x)
#M2evals[M2evals<0] += np.pi
#need to somehow shift each subsequent period up by 2pi to get this to be a continuous line
#M2evals[nu%(np.pi) == 0] += np.pi*

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.plot(nu, M2evals)
#plt.legend(loc='upper left')
plt.show()


