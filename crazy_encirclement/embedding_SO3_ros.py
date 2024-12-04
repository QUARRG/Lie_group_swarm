import numpy as np
from scipy.spatial.transform import Rotation as R
import quaternion
import math
from crazy_encirclement.utils2 import R3_so3
from scipy.linalg import expm
from icecream import ic
class Embedding():
    def __init__(self,r,phi_dot,k_phi,tactic,n_agents,initial_pos,hover_height,dt,multiplier):
        self.phi_dot = phi_dot
        self.r = r
        self.k_phi = k_phi
        self.tactic = tactic
        self.hover_height = hover_height
        self.n = n_agents
        self.dt = dt
        self.scale = self.phi_dot #scale the distortion around the x axis
        self.Rot = np.zeros((3,3))
        self.pass_zero = False
        self.pass_ref = False

        self.Rot_des = np.eye(3)
        self.Rot_act = np.eye(3)
        self.initial_phase = np.arctan2(initial_pos[1],initial_pos[0])

        # if self.initial_phase[i] == 0:
        #     self.initial_phase[i] = 2*np.pi
        self.wd = np.zeros(self.n)
        self.T = 24
        self.t = np.arange(0,self.T, self.dt)
        self.target_r = initial_pos
        self.target_v = np.zeros(3)
       
    def targets(self,agent_r,phi_prev):
            # Circle position
        pos = np.array([agent_r[0], agent_r[1], agent_r[2]-self.hover_height])
        #Rot = self.tactic_parameters(phi_i)
        #self.Rot[:,:,i] = self.Rot[:,:,i]@expm(R3_so3(v_d_hat.reshape(-1,1))*self.dt)
        
        pos_rot = np.linalg.inv(self.Rot_des)@pos.T
        phi_i, _ = self.cart2pol(pos_rot)

        #pos_x, pos_y, _ = pos_rot.parts[1:]  # Ignoring the scalar part

            
        wd = self.phi_dot
            #first evolve the agent in phase
        v_d_hat_z = np.array([0, 0, -wd])
        x = self.r * np.cos(phi_i)
        y = self.r * np.sin(phi_i)
        Rot_z = expm(R3_so3(v_d_hat_z.reshape(-1,1))*self.dt)
        pos_d_hat = np.array([x, y, 0])
        pos_d_hat = Rot_z@pos_d_hat
        phi_d, _ = self.cart2pol(pos_d_hat)

 
        phi_dot_x = self.calc_wx(phi_d)#*(phi_d-self.phi_des[i])
        phi_dot_y = self.calc_wy(phi_d) #phi_i-phi_prev[i]*
        v_d_hat_x_y = np.array([-phi_dot_x, -phi_dot_y, 0])
        self.Rot_des = expm(R3_so3(v_d_hat_x_y.reshape(-1,1)))



        pos_d = self.Rot_des@pos_d_hat.T
        # pos_d = Rot@pos_d_hat.T
        target_r_old = self.target_r.copy()
        self.target_r[0] = pos_d[0]
        self.target_r[1] = pos_d[1]
        self.target_r[2] = pos_d[2] + self.hover_height
        self.target_r[2] = np.clip(self.target_r[2],0.15,1.5)
        self.target_v = (self.target_r - target_r_old)/(self.dt)

            
        return phi_i, self.target_r, self.target_v, phi_i
    
    def calc_wx(self,phi):
        return self.scale*(np.sin(phi)*np.cos(phi)-np.sin(phi)**3)
        #return self.scale*np.cos(phi)*np.sin(phi)
    
    def calc_wy(self,phi):
        return self.scale*np.sin(-phi)*np.cos(phi)**2
    def phi_dot_desired(self,phi_i, phi_j, phi_k, phi_dot_des, k):
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
        
