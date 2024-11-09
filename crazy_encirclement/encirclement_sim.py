from embedding_SO3_sim import Embedding
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
# plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
from mpl_toolkits.mplot3d import Axes3D
import math
from scipy.spatial.transform import Rotation as R
from utils import R3_so3, so3_R3
from icecream import ic
import pandas as pd
import os

N = 5000
r = 1
k_phi = 5
kx = 5
kv = 2.5*np.sqrt(2)
n_agents = 3
phi_dot = 0.5
dt = 0.01
save = True

mb = 0.04
g = 9.81
I3 = np.array([0,0,1]).T.reshape(3)
w_r = 0 #reference yaw
ca_1 = np.array([np.cos(w_r),np.sin(w_r),0]).T #auxiliar vector 
Ca_r = np.zeros((3,3,n_agents,N))
Ca_r[:,:,0,0] = np.eye(3)
quat = np.zeros((4,n_agents,N))

ra_r = np.zeros((3,n_agents,N))
va_r = np.zeros((3,n_agents,N))
va_r_dot = np.zeros((3,n_agents,N))
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

agents_r[:, 0, 0] = 1.1*np.array([r*np.cos(0),r*np.sin(0),0.6]).T
agents_r[:, 1, 0] = 1.1*np.array([r*np.cos(np.deg2rad(10)),r*np.sin(np.deg2rad(10)),0.6]).T
agents_r[:, 2, 0] = 1.1*np.array([r*np.cos(np.deg2rad(260)),r*np.sin(np.deg2rad(260)) ,0.6]).T

ra_r[:,:,0] = agents_r[:,:,0]
for i in range(n_agents):
    phi_cur[i,0] = np.mod(np.arctan2(agents_r[1,i,0],agents_r[0,i,0]),2*np.pi)

embedding = Embedding(r, phi_dot,k_phi, 'dumbbell',n_agents,agents_r[:,:,0],dt)

for i in range(N-1):
    #print("percentage: ", float(i/N))

    phi_new, target_r_new, target_v_new, phi_diff_new, distances_new,debug = embedding.targets(agents_r[:,:,i],phi_cur[:,i])

    #ic(target_r_new)
    phi_cur[:,i+1] = phi_new
    phi_dot_cur[:,i] = (phi_cur[:,i+1] - phi_cur[:,i])/dt
    ra_r[:,:,i+1] = target_r_new#*np.random.uniform(0.99,1.01)
    va_r[:,:,i+1] = target_v_new#*np.random.uniform(0.99,1.01)

    va_r[:,:,i+1] = ((ra_r[:,:,i+1] - ra_r[:,:,i])/(dt))#*np.random.uniform(0.8,1.2)
    va_r_dot[:,:,i] = (va_r[:,:,i+1] - va_r[:,:,i])/dt
    phi_diff[:,i] = phi_diff_new
    distances[:,i] = distances_new
    #ic(va_r[:,:,i+1])


    accels[:,:,i] =  kx*(ra_r[:,:,i+1] - agents_r[:,:,i]) + kv*(va_r[:,:,i+1] - agents_v[:,:,i]) # +
    # if debug:
    #     ic(ra_r[:,1,i+1] - agents_r[:,1,i])
    #     ic(va_r[:,1,i+1] - agents_v[:,1,i])
    #     input()
    agents_v[:,:,i+1] = agents_v[:,:,i] + accels[:,:,i]*dt# *np.random.uniform(0.2,1.2)
    agents_r[:,:,i+1] = agents_r[:,:,i] + agents_v[:,:,i]*dt + 0.5*accels[:,:,i]*dt**2#*np.random.uniform(0.2,1.2)

figures_dir = "figures/"
os.makedirs(figures_dir, exist_ok=True)
t = np.linspace(0, N*dt, N)
colors = plt.cm.viridis(np.linspace(0, 1, n_agents))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
legends = []
for agent in range(n_agents):
    color = colors[agent]
    ax.plot3D(ra_r[0,agent,1:-1], ra_r[1,agent,1:-1], ra_r[2,agent,1:-1],color=color)
    ax.plot3D(agents_r[0,agent,1:-1], agents_r[1,agent,1:-1], agents_r[2,agent,1:-1],color=color, linestyle='dashed')
    legends.append(f"Desired trajectory agent {agent+1}")
    legends.append(f"Real trajectory agent {agent+1}")
ax.legend(legends)#, bbox_to_anchor=(1.05, 0.5), loc='center left', borderaxespad=0.)
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')

if save:
# # List of perspectives to save (elev, azim)
    angles = [
        (20, 30),   # Low angle, front-left
        (60, 45),   # High angle, front-left
        (45, 90),   # Side view
        (90, 0),    # Directly above
        (10, 180),  # Low angle, back view
        (75, 210),  # High angle, back-right
        (30, 270),  # Side view, right
        (60, 315)   # High angle, front-right
    ]

    # Save the plot from each perspective
    for j, (elev, azim) in enumerate(angles):
        ax.view_init(elev=elev, azim=azim)  # Set the view
        plt.savefig(f"{figures_dir}/3_agents_SO3_{i+1}_view_{j+1}.png", bbox_inches='tight', pad_inches=0.1)  # Save with unique filename

    # Show the plot in the last perspective if needed
    #plt.show()
    plt.close()
else:
    plt.show()


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# for agent in range(n_agents):
#     ax.plot3D(agents_r[0,agent,1:-1], agents_r[1,agent,1:-1], agents_r[2,agent,1:-1])

# ax.set_xlabel('X Axis')
# ax.set_ylabel('Y Axis')
# ax.set_zlabel('Z Axis')
# plt.savefig()
for i in range(n_agents):
    plt.subplot(3,1,1)
    plt.title(f"Velocities agent {i+1}")
    plt.plot(t[0:-1],va_r[0,i,0:-1])
    plt.ylabel("$v_x$ (m/s)")
    plt.subplot(3,1,2)
    plt.plot(t[0:-1],va_r[1,i,0:-1])
    plt.ylabel("$v_y$ (m/s)")
    plt.subplot(3,1,3)
    plt.plot(t[0:-1],va_r[2,i,0:-1])
    plt.ylabel("$v_z$ (m/s)")
    plt.xlabel("Time (s)")
    
    #plt.show()
    if save:
        plt.savefig(f"{figures_dir}/velocities_agent_{i+1}.png")
        plt.close()
    else:
        plt.show()

for i in range(n_agents):
    plt.subplot(3,1,1)
    plt.title(f"Positions agent {i+1}")
    plt.plot(t[0:-1],ra_r[0,i,0:-1])
    plt.plot(t[0:-1],agents_r[0,i,0:-1],linestyle='dashed')
    plt.legend(["Desired","Real"])
    plt.ylabel("x (m)")
    plt.subplot(3,1,2)
    plt.plot(t[0:-1],ra_r[1,i,0:-1])
    plt.plot(t[0:-1],agents_r[1,i,0:-1],linestyle='dashed')
    plt.legend(["Desired","Real"])
    plt.ylabel("y (m)")
    plt.subplot(3,1,3)
    plt.plot(t[0:-1],ra_r[2,i,0:-1])
    plt.plot(t[0:-1],agents_r[2,i,0:-1],linestyle='dashed')
    plt.legend(["Desired","Real"])
    plt.ylabel("z (m)")
    plt.xlabel("Time (s)")
    if save:
        plt.savefig(f"{figures_dir}/positions_agent_{i+1}.png")
        plt.close()
    else:
        plt.show()

for i in range(n_diff):
    plt.plot(t[0:-1],distances[i,0:-1])
plt.ylabel("Distances (m)")
plt.xlabel("Time (s)")
plt.title("Distances between agents")
if save:
    plt.savefig(f"{figures_dir}/distances.png")
    plt.close()
else:
    plt.show()

for i in range(n_diff):
    plt.plot(t[0:-1],np.rad2deg(phi_diff[i,0:-1]))
plt.title("Phase differences between agents")
plt.ylabel("$\phi$ (degrees)")
plt.xlabel("Time (s)")
if save:
    plt.savefig(f"{figures_dir}/phase_diff.png")
    plt.close()
else:
    plt.show()

for agent in range(n_agents):
    plt.subplot(3,1,1)
    plt.title(f"Positions x, y, and z errors of all the agents")
    plt.ylabel(f"$e_x$ (m)")
    plt.plot(t[0:-1],ra_r[0,agent,0:-1]-agents_r[0,agent,0:-1],label=f"agent {agent+1}")
    plt.subplot(3,1,2)
    plt.ylabel(f"$e_y$ (m)")
    plt.plot(t[0:-1],ra_r[1,agent,0:-1]-agents_r[1,agent,0:-1],label=f"agent {agent+1}")
    plt.subplot(3,1,3)
    plt.ylabel(f"$e_z$ (m)")
    plt.plot(t[0:-1],ra_r[2,agent,0:-1]-agents_r[2,agent,0:-1],label=f"agent {agent+1}")
    plt.xlabel("Time (s)")
    
if save:
    plt.savefig(f"{figures_dir}/error_agent.png")
    plt.close()
else:
    plt.show()





