import matplotlib.pyplot as plt
import numpy as np
from math import log2, floor

n=np.array([1e3, 1e4, 1e5, 1e6, 1e7,1e8], dtype=int)
i=[np.linspace(1,floor(log2(ni)),floor(log2(ni))) for ni in n]
x=[]
for idx,row in enumerate(i):
    x.append(np.divide(n[idx],row))

fig, axs = plt.subplots(3, 2, figsize=(10, 10))

axs[0, 0].plot(i[0],x[0],label='n=1e3')
axs[0, 0].set_title('n=1e3')
axs[0, 1].plot(i[1],x[1],label='n=1e4')
axs[0, 1].set_title('n=1e4')
axs[1, 0].plot(i[2],x[2],label='n=1e5')
axs[1, 0].set_title('n=1e5')
axs[1, 1].plot(i[3],x[3],label='n=1e6')
axs[1, 1].set_title('n=1e6')
axs[2, 0].plot(i[4],x[4],label='n=1e7')
axs[2, 0].set_title('n=1e7')
axs[2, 1].plot(i[5],x[5],label='n=1e8')
axs[2, 1].set_title('n=1e8')

for ax in axs.flat:
    ax.set(xlabel='i', ylabel='x')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()

plt.tight_layout()
plt.legend()
plt.suptitle('Infograph')
plt.savefig('infograph.png')


