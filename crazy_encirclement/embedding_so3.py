import numpy as np
from scipy.spatial.transform import Rotation as R
import quaternion
import math
from utils2 import R3_so3
from scipy.linalg import expm

class Embedding():
    def __init__(self,r,phi_dot,k_phi,tactic,n_agents,dt):
        self.phi_dot = phi_dot
        self.r = r
        self.k_phi = k_phi
        self.tactic = tactic
        self.n = n_agents
        self.dt = dt

       
    def targets(self,agent_r, agent_v,phi_prev):

        target_r = np.zeros((3, self.n))
        target_v = np.zeros((3, self.n))
        phi_cur = np.zeros(self.n)
        unit = np.zeros((self.n, 3))
        n_diff = int(np.math.factorial(self.n) / (math.factorial(2) * math.factorial(self.n-2)))
        phi_diff = np.zeros(n_diff)
        distances = np.zeros(n_diff)
        unit = np.zeros((self.n, 3))

        pos_circle = np.zeros((3, self.n))

        for i in range(self.n):
            # Circle position
            pos = np.array([agent_r[0, i], agent_r[1, i], agent_r[2, i]])
            Rot = self.tactic_parameters(phi_prev[i])
            pos_rot = np.linalg.inv(Rot)@pos.T
            phi, _ = self.cart2pol(pos_rot)
            pos_x = pos_rot[0]
            pos_y = pos_rot[1]
            #pos_x, pos_y, _ = pos_rot.parts[1:]  # Ignoring the scalar part
            phi_cur[i] = phi
            pos_circle[0, i] = pos_x
            pos_circle[1, i] = pos_y
            unit[i, :] = [np.cos(phi), np.sin(phi), 0]

        for i in range(self.n):
            phi_i = phi_cur[i]
            if i == 0:
                phi_k = phi_cur[i+1]
                phi_j = phi_cur[self.n-1]
            elif i == self.n-1:
                phi_k = phi_cur[0]
                phi_j = phi_cur[i-1]
            else:
                phi_k = phi_cur[i+1]
                phi_j = phi_cur[i-1]

            wd = self.phi_dot_desired(phi_i, phi_j, phi_k, self.phi_dot, self.k_phi)
            v_d_hat = np.array([0, 0, -wd])
            Rot = self.tactic_parameters(phi_i)
            #Rot = Rot@expm(R3_so3(v_d_hat.reshape(-1,1))*self.dt)
            v_d = Rot@v_d_hat.T
            #v_x, v_y, v_z = v_d.parts[1:]
            v = np.cross(v_d.T, agent_r[:, i])
            target_v[0, i] = v[0]
            target_v[1, i] = v[1]
            target_v[2, i] = v[2]

            x = self.r * np.cos(phi_i)
            y = self.r * np.sin(phi_i)
            pos_d_hat = np.array([x, y, 0])
            pos_d = Rot@pos_d_hat.T
            #pos_x, pos_y, pos_z = pos_d.parts[1:]

            target_r[0, i] = pos_d[0]
            target_r[1, i] = pos_d[1]
            target_r[2, i] = pos_d[2]
            if self.tactic == 'circle':
                target_r[2,i] = 0.5
            unit[i, :] = [np.cos(phi_i), np.sin(phi_i), 0]
        k = 0
        for i in range(self.n):
            for j in range(i+1, self.n):
                distances[k] = np.linalg.norm(target_r[:, i] - target_r[:, j])
                phi_diff[k] = np.arccos(np.dot(unit[i,:],unit[j,:]))
                k += 1
            
        return phi_cur, target_r, target_v, pos_circle, phi_diff, distances
    

    def phi_dot_desired(self,phi_i, phi_j, phi_k, phi_dot_des, k):
        phi_ki = np.mod(phi_i - phi_k, 2*np.pi)
        phi_ij = np.mod(phi_j - phi_i, 2*np.pi)
        return (3 * phi_dot_des + k * (phi_ki - phi_ij)) / 3

    def rotate(self,pos, Rot):
        return Rot*pos

    def cart2pol(self,pos_rot):
        pos_x = pos_rot[0]
        pos_y = pos_rot[1]
        #pos_x, pos_y, _ = pos_rot.parts[1:]
        phi = np.arctan2(pos_y, pos_x)
        phi = np.mod(phi, 2*np.pi)
        r = np.linalg.norm([pos_x, pos_y])
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
        
        elif self.tactic == 'gerono':
            a = -(np.sqrt(2)*np.sqrt(1-np.sin(phi))/2)
            b = -(np.sqrt(2)*np.sqrt(1+np.sin(phi))/2)
        r = R.from_quat([b, 0, 0,a])

        #print('rotation',r.as_matrix())
        mat = np.array([[1, 0, 0],
                        [0, np.cos(phi), -np.sin(phi)],
                        [0, np.sin(phi), np.cos(phi)]])
        #print('mat',mat)
        if np.linalg.det(mat-r.as_matrix()) > 10e-7:
            print('det',np.linalg.det(mat-r.as_matrix()))
            input()
        return mat#r.as_matrix()

    # Quaternion multiplication function (can be skipped if using numpy.quaternion)
    def quat_mult(self,q1, q2):
        return np.quaternion(
            q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z,
            q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y,
            q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x,
            q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w
        )
