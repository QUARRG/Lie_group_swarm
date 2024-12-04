import numpy as np
from scipy.spatial.transform import Rotation as R
import warnings
import math
from utils2 import R3_so3
from scipy.linalg import expm
from icecream import ic
#the rotation is counterclockwise #################################
class Embedding():
    def __init__(self,r,phi_dot,k_phi,tactic,n_agents,initial_pos,dt):
        self.phi_dot = phi_dot
        self.r = r
        self.k_phi = k_phi
        self.tactic = tactic
        self.n = n_agents
        self.dt = dt
        self.initial_phase = np.zeros(self.n)
        self.Rot_des = np.zeros((3,3,self.n))
        self.Rot_act = np.zeros((3,3,self.n))
        self.scale = self.phi_dot #scale the distortion around the x axis
        self.pass_zero = np.zeros(self.n)
        self.pass_ref = np.zeros(self.n)
        self.count = 0
        for i in range(self.n):
            self.Rot_des[:,:,i] = np.eye(3)
            self.Rot_act[:,:,i] = np.eye(3)
            self.initial_phase[i] = np.mod(np.arctan2(initial_pos[1,i],initial_pos[0,i]),2*np.pi)
            self.pass_ref[i] = False
            self.pass_zero[i] = False
            # if self.initial_phase[i] == 0:
            #     self.initial_phase[i] = 2*np.pi
        self.wd = np.zeros(self.n)
        self.T = 24
        self.t = np.arange(0,self.T, self.dt)
        self.timer = 1
        self.phi_des = np.zeros(self.n)
        self.phi_cur = np.zeros(self.n)
        self.phi_dot_actual = np.zeros(self.n)

       
    def targets(self,agent_r,counter):
        debug = False
        target_r = np.zeros((3, self.n))
        target_v = np.zeros((3, self.n))
        
        
        unit = np.zeros((self.n, 3))

        unit = np.zeros((self.n, 3))
        if self.n >1:
            n_diff = int(np.math.factorial(self.n) / (math.factorial(2) * math.factorial(self.n-2)))
        else:
            n_diff = 1
        phi_diff = np.zeros(n_diff)
        distances = np.zeros(n_diff)
        unit = np.zeros((self.n, 3))

        pos_circle = np.zeros((3, self.n))

        for i in range(self.n):
            # Circle position
            # pos = np.array([agent_r[0, i] - self.phi_dot*np.cos(phi_prev[i])*np.sin(phi_prev[i]), agent_r[1, i] - self.r*np.cos(phi_prev[i])**2, agent_r[2, i]-0.6])
            pos = np.array([agent_r[0, i] , agent_r[1, i] , agent_r[2, i]-0.6])

            #Rot = self.tactic_parameters(phi_i)
            #self.Rot[:,:,i] = self.Rot[:,:,i]@expm(R3_so3(v_d_hat.reshape(-1,1))*self.dt)
            
            pos_rot = self.Rot_des[:,:,i].T@pos.T
            phi, _ = self.cart2pol(pos_rot)
            phi_dot_x = self.calc_wx(phi)

            v_d_hat_x = np.array([-phi_dot_x, 0, 0])
            Rot_x = expm(R3_so3(v_d_hat_x.reshape(-1,1))*self.dt)
            self.Rot_act[:,:,i] =Rot_x@self.Rot_act[:,:,i]# self.Rot_des[:,:,i].copy()#Rot_x@

            pos_x = pos_rot[0]
            pos_y = pos_rot[1]
            #pos_x, pos_y, _ = pos_rot.parts[1:]  # Ignoring the scalar part
            self.phi_dot_actual[i] = (phi - self.phi_cur[i])/self.dt
            self.phi_cur[i] = phi
            pos_circle[0, i] = pos_x
            pos_circle[1, i] = pos_y
            unit[i, :] = [np.cos(phi), np.sin(phi), 0]

            
        for i in range(self.n):
            if self.n > 1:
                phi_i = self.phi_cur[i]
                if i == 0:
                    phi_k = self.phi_cur[self.n-1] #ahead
                    phi_j = self.phi_cur[i+1] #behind
                elif i == self.n-1:
                    phi_k = self.phi_cur[i-1]
                    phi_j = self.phi_cur[0]
                else:
                    phi_k = self.phi_cur[i-1]
                    phi_j = self.phi_cur[i+1]


                wd = self.phi_dot_desired(phi_i, phi_j, phi_k, self.phi_dot, self.k_phi,i)
            else:
                wd = self.phi_dot
                phi_i = self.phi_cur[0]
            #wd = self.phi_dot
            
            #first evolve the agent in phase
            v_d_hat_z = np.array([0, 0, -wd])
            x = self.r * np.cos(phi_i)
            y = self.r * np.sin(phi_i)
            Rot_z = expm(R3_so3(v_d_hat_z.reshape(-1,1))*self.dt)
            pos_d_hat = np.array([x, y, 0])
            pos_d_hat = Rot_z@pos_d_hat
            phi_d, _ = self.cart2pol(pos_d_hat)

            phi_dot_x = self.calc_wx(phi_d)#*(phi_i-self.phi_new[i])
            phi_dot_y = 0*self.scale*np.cos(phi_d)**2*np.sin(phi_d) #phi_i-phi_prev[i]*
            v_d_hat_x = np.array([-phi_dot_x, 0, 0])
            
            Rot_x = expm(R3_so3(v_d_hat_x.reshape(-1,1))*self.dt)
            
            v_d_hat_y = np.array([0, -phi_dot_y, 0])
            Rot_y = expm(R3_so3(v_d_hat_y.reshape(-1,1))*self.dt)


            #self.Rot_des[:,:,i] = Rot_x@Rot_y@self.Rot_act[:,:,i].copy()
            self.Rot_des[:,:,i] = Rot_x@self.Rot_act[:,:,i].copy()
            Rot = self.Rot_des[:,:,i]#@Rot_y@Rot_z

            if (((np.pi)-self.phi_des[i]) <0) and (((np.pi)-phi_d) >0):# and (wd > 0):
                self.Rot_act[:,:,i] = np.eye(3)
                self.Rot_des[:,:,i] = np.eye(3)
                #ic(self.phi_des[i],phi_d)
            if ((self.phi_des[i]) <0.01) and (np.abs((2*np.pi)-phi_d) <0.01) and (wd > 0):
                self.Rot_act[:,:,i] = np.eye(3)
                self.Rot_des[:,:,i] = np.eye(3)
                ic(self.phi_des[i],phi_d)
            pos_d = Rot@pos_d_hat
            # if i == 1 and not self.pass_ref[i]:

            #pos_d = (self.Rot_des[:,:,0].T@pos_d)
            target_r[0, i] = pos_d[0] #+ self.phi_dot*np.cos(phi_i)*np.sin(phi_i)
            target_r[1, i] = pos_d[1] #+ self.r*np.cos(phi_i)**2
            target_r[2, i] = pos_d[2] + 0.6
            # if self.tactic == 'circle':
            #     target_r[2,i] = 0.5
            unit[i, :] = [np.cos(phi_d), np.sin(phi_d), 0]
            self.timer += 1
            self.phi_des[i] = phi_d
        # with warnings.catch_warnings(record=True) as w:
        #     warnings.simplefilter("always")  # Ensure all warnings are captured
        k = 0
        for i in range(self.n):
            for j in range(i+1, self.n):
                distances[k] = np.linalg.norm(target_r[:, i] - target_r[:, j])
                phi_diff[k] = np.arccos(np.dot(unit[i,:],unit[j,:]))
                k += 1
        # if len(w) > 0:
        #     ic(unit[i,:],unit[j,:],np.dot(unit[i,:],unit[j,:]))
        
            
        return  self.phi_cur,target_r, target_v, phi_diff, distances, debug
    

    def calc_wx(self,phi):
        return self.scale*(np.sin(phi)*np.cos(phi)-np.sin(phi)**3)
    
    def calc_wy(self,phi):
        return self.scale*np.cos(phi)

    def phi_dot_desired(self,phi_i, phi_j, phi_k, phi_dot_des, k,i):

        phi_ki = np.mod(phi_i - phi_k, 2*np.pi)
        phi_ij = np.mod(phi_j - phi_i, 2*np.pi)
        return (3 * phi_dot_des + k * (phi_ki - phi_ij)) / 3


    def cart2pol(self,pos_rot):
        pos_x = pos_rot[0]
        pos_y = pos_rot[1]
        #pos_x, pos_y, _ = pos_rot.parts[1:]
        phi = np.arctan2(pos_y, pos_x)
        phi = np.mod(phi, 2*np.pi)
        r = np.linalg.norm([pos_x, pos_y])
        return phi, r


    # Quaternion multiplication function (can be skipped if using numpy.quaternion)
    def quat_mult(self,q1, q2):
        return np.quaternion(
            q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z,
            q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y,
            q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x,
            q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w
        )
    
    
        
