import numpy as np
from scipy.spatial.transform import Rotation as R
import quaternion
import math
from crazy_encirclement.utils2 import generate_reference

class Embedding():
    def __init__(self,r,phi_dot,k_phi,tactic,n_agents,dt,circle_height=0.3):
        self.phi_dot = phi_dot
        self.r = r
        self.k_phi = k_phi
        self.tactic = tactic
        self.n = 3
        self.dt = dt
        self.circle_height = circle_height
        self.target_r = np.zeros((3))
        self.target_v = np.zeros((3))
        self.target_a = np.zeros((3))
        self.phi_cur = np.zeros(self.n)
        self.unit = np.zeros((self.n))
        self.n_diff = int(np.math.factorial(self.n) / (math.factorial(2) * math.factorial(self.n-2)))
        self.phi_diff = np.zeros(self.n_diff)
        self.distances = np.zeros(self.n_diff)
        self.unit = np.zeros((self.n, 3))
        self.Ca_r = np.zeros((3,3))

       
    def targets(self,agent_r, agent_v,phi_prev,Ca_b):


            # Circle position
        pos = np.quaternion(0, agent_r[0], agent_r[1], agent_r[2])
        quat = self.tactic_parameters(phi_prev)
        pos_rot = self.rotate(pos, quat.conjugate())
        phi_i, _ = self.cart2pol(pos_rot)
        #pos_x, pos_y, _ = pos_rot.parts[1:]  # Ignoring the scalar part

    # for i in range(self.n):
    #     phi_i = self.phi_cur[i]
    #     if i == 0:
    #         phi_k = self.phi_cur[i+1]
    #         phi_j = self.phi_cur[self.n-1]
    #     elif i == self.n-1:
    #         phi_k = self.phi_cur[0]
    #         phi_j = self.phi_cur[i-1]
    #     else:
    #         phi_k = self.phi_cur[i+1]
    #         phi_j = self.phi_cur[i-1]
        phi_k = phi_prev[0] #the first phase is the heading agent's
        phi_j = phi_prev[2] #the last phase is the lagging agent's
        wd = self.phi_dot_desired(phi_i, phi_j, phi_k, self.phi_dot, self.k_phi)
        v_d_hat = np.quaternion(0, 0, 0, -wd)
        quat = self.tactic_parameters(phi_i)
        v_d = self.rotate(v_d_hat, quat)

        #v_x, v_y, v_z = v_d.parts[1:]
        v = np.cross([v_d.x, v_d.y, v_d.z], agent_r)

        self.target_v[0] = v[0]
        self.target_v[1] = v[1]
        self.target_v[2] = v[2]

        x = self.r * np.cos(phi_i)
        y = self.r * np.sin(phi_i)
        pos_d_hat = np.quaternion(0, x, y, 0)
        pos_d = self.rotate(pos_d_hat, quat)
        #pos_x, pos_y, pos_z = pos_d.parts[1:]

        self.target_r[0] = pos_d.x
        self.target_r[1] = pos_d.y
        if self.tactic == 'circle':
            self.target_r[2] = self.circle_height
        else:
            self.target_r[2] = pos_d.z
        # self.unit[i, :] = [np.cos(phi_i), np.sin(phi_i), 0]
        
        self.target_v = np.clip(self.target_v, -0.3, 0.3)
        self.target_a = (self.target_v - agent_v)/self.dt
        self.target_a = np.clip(self.target_a, -0.3, 0.3)

        # k = 0
        # for i in range(self.n):
        #     for j in range(i+1, self.n):
        #         self.distances[k] = np.linalg.norm(self.target_r - self.target_r[:, j])
        #         self.phi_diff[k] = np.arccos(np.dot(self.unit[i,:],self.unit[j,:]))
        #         k += 1
        Wr_r, _, _, quaternion, self.Ca_r = generate_reference(self.target_v,self.target_a, self.Ca_r, Ca_b, self.dt)   
        return phi_i, self.target_r, self.target_v, self.target_a, quaternion, Wr_r

    def phi_dot_desired(self,phi_i, phi_j, phi_k, phi_dot_des, k):
        phi_ki = np.mod(phi_i - phi_k, 2*np.pi)
        phi_ij = np.mod(phi_j - phi_i, 2*np.pi)
        return (3 * phi_dot_des + k * (phi_ki - phi_ij)) / 3

    def rotate(self,pos, quat):
        return quat * pos * quat.conjugate()

    def cart2pol(self,pos_rot):
        #pos_x, pos_y, _ = pos_rot.parts[1:]
        phi = np.arctan2(pos_rot.y, pos_rot.x)
        phi = np.mod(phi, 2*np.pi)
        r = np.linalg.norm([pos_rot.x, pos_rot.y])
        return phi, r

    def tactic_parameters(self,phi):
        if self.tactic == 'dumbbell':
            a = -np.sqrt(2) * np.sqrt(np.cos(phi)**2 + 1) / 2
            b = -np.sqrt(2) * np.sqrt(-(np.cos(phi)-1)*(np.cos(phi)+1)) / 2
            norm = np.sqrt(a**2 + b**2)
            a = a / norm
            b = b / norm
        elif self.tactic == 'circle':
            a = 1
            b = 0
        elif self.tactic == 'bernoulli':
            a = -(np.sqrt(2)*np.sqrt(np.cos(phi) + 1))/(2*np.sqrt(np.sin(phi)**2 + 1))
            b = -(np.sqrt(2)*np.sqrt(1-np.cos(phi)))/(2*np.sqrt(np.sin(phi)**2 + 1))
            norm = np.sqrt(a**2 + b**2)
            a = a / norm
            b = b / norm
        return np.quaternion(a, b, 0, 0)

    # Quaternion multiplication function (can be skipped if using numpy.quaternion)
    def quat_mult(self,q1, q2):
        return np.quaternion(
            q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z,
            q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y,
            q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x,
            q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w
        )
