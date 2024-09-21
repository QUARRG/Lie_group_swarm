from embedding import Embedding
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from scipy.spatial.transform import Rotation as R

N = 5000
r = 0.5
k_phi = 5
kx = 10
kv = 2.5*np.sqrt(2)
n_agents = 3
phi_dot = 0.5
dt = 0.001

target_r = np.zeros((3,n_agents,N))
target_v = np.zeros((3,n_agents,N))
accels = np.zeros((3,n_agents,N))
phi_cur = np.zeros((n_agents, N))

agents_r = np.zeros((3,n_agents, N))
agents_v = np.zeros((3,n_agents, N))
n_diff = int(math.factorial(n_agents) / (math.factorial(2) * math.factorial(n_agents-2)))
phi_diff =  np.zeros((n_diff,N))
distances = np.zeros((n_diff,N))
agents_r[:, 0, 0] = np.array([0.15,-0.15,0]).T
agents_r[:, 1, 0] = np.array([-0.5,-0.05,0]).T
agents_r[:, 2, 0] = np.array([0.36,0.17 ,0]).T


embedding = Embedding(r, phi_dot,k_phi, 'dumbbell',n_agents)

for i in range(N-1):
    phi_new, target_r_new, target_v_new, _, phi_diff_new, distances_new = embedding.targets(agents_r[:,:,i], phi_cur[:,i])
    phi_cur[:,i+1] = phi_new
    target_r[:,:,i+1] = target_r_new
    target_v[:,:,i+1] = target_v_new
    phi_diff[:,i] = phi_diff_new
    distances[:,i] = distances_new

    accels[:,:,i] = kx*(target_r[:,:,i] - agents_r[:,:,i]) + kv*(target_v[:,:,i] - agents_v[:,:,i])
    agents_v[:,:,i+1] = agents_v[:,:,i] + accels[:,:,i]*dt
    agents_r[:,:,i+1] = agents_r[:,:,i] + agents_v[:,:,i]*dt + 0.5*accels[:,:,i]*dt**2

    #accels[:,:,i] = kx*(target_r[:,:,i] - agents_r[:,:,i]) 
    #agents_r[:,:,i+1] = agents_r[:,:,i] + accels[:,:,i]*dt
    #agents_r[:,:,i+1] = agents_r[:,:,i] + agents_v[:,:,i]*dt + 0.5*accels[:,:,i]*dt**2

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for agent in range(n_agents):
    ax.plot3D(target_r[0,agent,0:-1], target_r[1,agent,0:-1], target_r[2,agent,0:-1])
plt.show()

for i in range(n_diff):
    plt.plot(distances[i,0:-1])
plt.show()

for i in range(n_diff):
    plt.plot(phi_diff[i,0:-1])
plt.show()






