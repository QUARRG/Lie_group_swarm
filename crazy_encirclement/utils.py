import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.linalg import logm
import time
import rclpy
import rclpy.logging
D = np.zeros((3,3))
mb= 0.042
g = 9.81
I3 = np.array([0,0,1]).T
w_r = 0 #reference yaw
ca_1 = np.array([np.cos(w_r),np.sin(w_r),0]).T #auxiliar vector 
info = rclpy.logging.get_logger("rclpy").info

def generate_reference(va_r_dot,Ca_r,Ca_b,va_r,dt):

    try:
        n_agents = va_r.shape[1]
    except:
        n_agents = 1
    Ca_r_new = np.zeros((3,3,n_agents))
    f_T_r = np.zeros((n_agents))
    Wr_r = np.zeros((3,n_agents))
    angles = np.zeros((3,n_agents))
    quaternion = np.zeros((4,n_agents))

    for i in range(n_agents):
        fa_r = mb*va_r_dot[:,i] +mb*g*I3 + Ca_r[:,:,i]@D@Ca_r[:,:,i].T@va_r[:,i]
        f_T_r[i] = I3.T@Ca_r[:,:,i].T@fa_r.T
        if np.linalg.norm(fa_r) != 0:
            r3 = fa_r.reshape(3,1)/np.linalg.norm(fa_r)
        else:
            r3 = np.zeros((3,1))
        aux = R3_so3(r3)@ca_1;
        if np.linalg.norm(aux) != 0:
            r2 = aux.reshape(3,1)/np.linalg.norm(aux);
        else:
            r2 = np.zeros((3,1))
        r1 = (R3_so3(r2)@r3).reshape(3,1);
        Ca_r_new[:,:,i] = np.hstack((r1, r2, r3))
        if np.linalg.norm(Ca_r[:,0,i]) != 0:
            Wr_r[:,i] = so3_R3(np.linalg.inv(Ca_r[:,:,i])@Ca_r_new[:,:,i])/dt

        angles[:,i] = R.from_matrix(Ca_r_new[:,:,i]).as_euler('zyx', degrees=False)
        quaternion[:,i] = R.from_matrix(Ca_r_new[:,:,i]).as_quat()
        Wr_r[:,i] = Ca_b[:,:,i].T@Ca_r[:,:,i]@Wr_r[:,i]
    return Wr_r, f_T_r, angles, quaternion, Ca_r_new

def R3_so3(w):
    v3 = w[2,0]
    v2 = w[1,0]
    v1 = w[0,0]
    so3 = np.array([[ 0 , -v3,  v2],
          [v3,   0, -v1],
          [-v2,  v1,   0]])

    return so3

def so3_R3(Rot):

    log_R = logm(Rot)
    w1 = log_R[2,1]
    w2 = log_R[0,2]
    w3 = log_R[1,0]
    w = np.array([w1,w2,w3]).T
    return np.mod(w,2*np.pi)

