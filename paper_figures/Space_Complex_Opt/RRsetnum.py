import math
import matplotlib.pyplot as plt
import numpy as np


n=np.array([200,2000,20000])
k=np.array([2,20,200])
epsilon=np.array([0.5,0.2,0.13])
epsilon_p= math.sqrt(2)*epsilon
l=1

def calc_logcomb(n,k):
    logcomb=np.empty(len(n))
    for i in range(len(n)):
        temp_1=0
        temp_2=0
        for j in range(n[i]-k[i]):
            temp_1+=math.log(k[i]+j+1)
            temp_2+=math.log(j+1)
        logcomb[i]=temp_1-temp_2
    return logcomb

lc=calc_logcomb(n,k)

lambda_p=(2+2*epsilon_p/3)*(lc+l*np.log(n)+np.log(np.log2(n)))*n/(epsilon_p**2)

best_case_theta= (2/n)*lambda_p 
worst_case_theta= ((2**(np.log2(n)-1))/n)*lambda_p
print(best_case_theta)
print(worst_case_theta)

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(epsilon, n, best_case_theta, cmap='viridis')
ax.set_xlabel(r'$\epsilon$')
ax.set_ylabel(r'$n$')
ax.set_zlabel(r'$\theta$')
ax.set_title(r'$\theta$ vs $\epsilon$ and $n$, best case')
plt.savefig('best_case_theta.pdf')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(epsilon, n, worst_case_theta, cmap='viridis')
ax.set_xlabel(r'$\epsilon$')
ax.set_ylabel(r'$n$')
ax.set_zlabel(r'$\theta$')
ax.set_zlim(worst_case_theta.min(), worst_case_theta.max())  # Set z-axis limits
ax.set_title(r'$\theta$ vs $\epsilon$ and $n$, worst case')
plt.savefig('worst_case_theta.pdf')