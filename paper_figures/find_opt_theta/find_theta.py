import matplotlib.pyplot as plt
import numpy as np
from math import exp, comb, sqrt,log
from mpl_toolkits.mplot3d import Axes3D



def theta_1(n,k,l,OPT,epsilon):
    return -2*n*np.log(1/(2*n**l))/(OPT*epsilon**2)

def theta_2(n,k,l,OPT,epsilon, epsilon_1):
    # if epsilon_1<epsilon:
    #     raise ValueError("$epsilon_1$ must be greater than epsilon")
    return (2+2*exp(-1))*n*np.log(comb(n,k)/(2*n**l))/(OPT*(epsilon-(1-exp(-1))*epsilon_1)**2)

n=np.array([10,1e2,1e3])
k=np.array([1,5,50])
l=1
OPT=n
epsilon=np.array([0.1,0.01,0.001])
a= np.sqrt(l*np.log(n)+log(2))
from scipy.special import comb

b = []
for ni, ki in zip(n, k):
    const=(1-exp(-1))
    arr_val=np.log(comb(ni,ki))+l*np.log(ni)+log(2)
    print((comb(ni,ki)))
    print(arr_val)
    b_fin=np.sqrt(const*arr_val)
    b.append(b_fin)
b = np.array(b)
epsilon_1=epsilon*a/((1-exp(-1))*a+b)
print(a)
print(b)    

theta1=theta_1(n,k,l,OPT,epsilon)
theta2=theta_2(n,k,l,OPT,epsilon,epsilon_1)

if np.isnan(epsilon).any() or np.isnan(epsilon_1).any() or np.isnan(theta1).any():
    print("NaN values found")
if np.isinf(epsilon).any() or np.isinf(epsilon_1).any() or np.isinf(theta1).any():
    print("inf values found")

# Handle NaN or inf values (e.g., replace them with a specific value)
epsilon = np.nan_to_num(epsilon, nan=0.0, posinf=0.0, neginf=0.0)
epsilon_1 = np.nan_to_num(epsilon_1, nan=0.0, posinf=0.0, neginf=0.0)
theta1 = np.nan_to_num(theta1, nan=0.0, posinf=0.0, neginf=0.0)

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(epsilon, n, theta1, cmap='viridis')
ax.set_xlabel(r'$\epsilon$')
ax.set_ylabel('n')
ax.set_ylim(10,1e3)
ax.set_zlabel(r'$\theta_{1}$')
ax.text2D(0.0, 0.95, r"k=[1,5,50]", transform=ax.transAxes)
ax.text2D(0.00, 0.90, r"$\epsilon$=[0.1,0.01,0.001]", transform=ax.transAxes)
plt.savefig("theta1.pdf")
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(epsilon, n, theta2, cmap='viridis')
ax.set_xlabel(r'$\epsilon$')
ax.set_ylabel('n')
ax.set_ylim(10,1e3)
ax.set_zlabel(r"$\theta_{2}$")
ax.text2D(0.0, 0.95, r"k=[1,5,50]", transform=ax.transAxes)
ax.text2D(0.00, 0.90, r"$\epsilon$=[0.1,0.01,0.001]", transform=ax.transAxes)
plt.savefig("theta2.pdf")
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(epsilon, n, theta2, cmap='viridis')
ax.set_xlabel(r'$\epsilon_{1}$')
ax.set_ylabel('n')
ax.set_ylim(10,1e3)
ax.set_zlabel(r"$\theta_{2}$")
ax.text2D(0.0, 1, r"k=[1,5,50]", transform=ax.transAxes)
ax.text2D(0.00, 0.95, fr"$\epsilon_1$={epsilon_1}", transform=ax.transAxes)
plt.savefig("theta2_ep1.pdf")
plt.show()



