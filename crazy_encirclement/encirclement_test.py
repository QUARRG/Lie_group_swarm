from embedding_new import Embedding
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from scipy.spatial.transform import Rotation as R
from utils import generate_reference
from icecream import ic

N = 2*2500
r = 1
k_phi = 5
kx = 15
kv = 2.5*np.sqrt(2)
n_agents = 3
phi_dot = 0.5
dt = 0.01

ra_r = np.zeros((3,n_agents,N))
va_r = np.zeros((3,n_agents,N))
accels = np.zeros((3,n_agents,N))
phi_cur = np.zeros((n_agents, N))
phi_dot_cur = np.zeros((n_agents, N))

agents_r = np.zeros((3,n_agents, N))
agents_v = np.zeros((3,n_agents, N))
if n_agents >1:
    n_diff = int(math.factorial(n_agents) / (math.factorial(2) * math.factorial(n_agents-2)))
else:
    n_diff = 1
phi_diff =  np.zeros((n_diff,N))
distances = np.zeros((n_diff,N))
va_r_dot = np.zeros((3,n_agents,N))
Ca_r = np.zeros((3,3,n_agents,N))
identity_matrix = np.eye(3)
Ca_b = np.tile(identity_matrix[:, :, np.newaxis, np.newaxis], (1, 1, n_agents, N))
va_r = np.zeros((3,n_agents,N))
f_T_r = np.zeros((n_agents,N))
angles = np.zeros((3,n_agents,N))
Wr_r = np.zeros((3,n_agents,N))

agents_r[:, 0, 0] = np.array([r*np.cos(np.pi/4),r*np.sin(np.pi/4),0]).T
agents_r[:, 1, 0] = np.array([r*np.cos(np.pi/5),r*np.sin(np.pi/5),0]).T
agents_r[:, 2, 0] = np.array([r*np.cos(2*np.pi/3),r*np.sin(2*np.pi/3) ,0]).T
# for i in range(n_agents):
#     phi_cur[i,0] = np.arctan2(agents_r[1,i,0],agents_r[0,i,0])

embedding = Embedding(r, phi_dot,k_phi, 'dumbbell',n_agents,dt)


# for i in range(10):
#     phi_new, target_r_new, target_v_new, _,phi_diff_new, distances_new = embedding.targets(agents_r[:,:,0],agents_v[:,:,0], phi_cur[:,0])

    # Wr_r_new, f_T_r_new, angles_new,_, Ca_r_new = generate_reference(va_r_dot[:,:,0],Ca_r[:,:,:,0],Ca_b[:,:,:,0],va_r[:,:,0],dt)
    # Ca_r[:,:,:,0] = Ca_r_new
    # if i >0:
    #     va_r_dot[:,:,i] = (va_r[:,:,i] - va_r[:,:,i-1])/dt

# Ca_r[:,:,:,0] = Ca_r_new

# ra_r[:,:,0] = target_r_new

# va_r[:,:,0] = target_v_new
# phi_cur[:,0] = phi_new

for i in range(N-1):
    print("percentage: ", float(i/N))
    # Wr_r_new, f_T_r_new, angles_new,_, Ca_r_new = generate_reference(va_r_dot[:,:,i],Ca_r[:,:,:,i],Ca_b[:,:,:,i],va_r[:,:,i],dt)
    # Ca_r[:,:,:,i+1] = Ca_r_new
    # f_T_r[:,i] = f_T_r_new
    # angles[:,:,i] = angles_new
    # Wr_r[:,:,i] = Wr_r_new



    phi_new, target_r_new, target_v_new, phi_diff_new, distances_new = embedding.targets(agents_r[:,:,i], agents_v[:,:,i],phi_cur[:,i])
    phi_cur[:,i+1] = phi_new
    phi_dot_cur[:,i] = (phi_cur[:,i+1] - phi_cur[:,i])/dt
    ra_r[:,:,i+1] = target_r_new
    if i >0:
        va_r[:,:,i] = (ra_r[:,:,i+1] - ra_r[:,:,i])/dt
    va_r[:,:,i+1] = target_v_new
    va_r_dot[:,:,i+1] = (va_r[:,:,i+1] - va_r[:,:,i])/dt
    
    phi_diff[:,i] = phi_diff_new
    distances[:,i] = distances_new
    accels[:,:,i] = kx*(ra_r[:,:,i+1] - agents_r[:,:,i]) + kv*(va_r[:,:,i+1] - agents_v[:,:,i])
    agents_v[:,:,i+1] = agents_v[:,:,i] + accels[:,:,i]*dt
    agents_r[:,:,i+1] = agents_r[:,:,i] + agents_v[:,:,i]*dt + 0.5*accels[:,:,i]*dt**2

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for agent in range(n_agents):
    ax.plot3D(ra_r[0,agent,1:-1], ra_r[1,agent,1:-1], ra_r[2,agent,1:-1])

ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')

plt.show()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for agent in range(n_agents):
    ax.plot3D(agents_r[0,agent,1:-1], agents_r[1,agent,1:-1], agents_r[2,agent,1:-1])

ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')

plt.show()

for i in range(n_diff):
    plt.plot(distances[i,0:-1])
plt.show()

for i in range(n_diff):
    plt.plot(np.rad2deg(phi_diff[i,0:-1]))
plt.show()

for agent in range(n_agents):
    plt.subplot(3,1,1)
    plt.title(f"X {agent}")
    plt.plot(ra_r[0,agent,0:-1]-agents_r[0,agent,0:-1])
    plt.subplot(3,1,2)
    plt.title(f"Y {agent}")
    plt.plot(ra_r[1,agent,0:-1]-agents_r[1,agent,0:-1])
    plt.subplot(3,1,3)
    plt.title(f"Z {agent}")
    plt.plot(ra_r[2,agent,0:-1]-agents_r[2,agent,0:-1])

    plt.show()





