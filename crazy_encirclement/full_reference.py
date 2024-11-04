import rclpy
from rclpy.node import Node
from crazy_encirclement.utils2 import  trajectory, R3_so3, so3_R3
from crazyflie_interfaces.msg import FullState, StringArray
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import Pose, Twist

import numpy as np

class FullReference(Node):
    def __init__(self):
        super().__init__('full_reference_node')
        self.declare_parameter('robot_prefix', 'C20')
        self.robot = self.get_parameter('robot_prefix').value
        self.info = self.get_logger().info
        self.mb = 0.04
        self.g = 9.81
        self.I3 = np.array([0,0,1]).T.reshape(3)
        w_r = 0 #reference yaw
        self.ca_1 = np.array([np.cos(w_r),np.sin(w_r),0]).T #auxiliar vector 
        self.Ca_r = np.eye(3)
        self.Ca_b = np.eye(3)
        self.timer_period = 0.002

        self.create_subscription(FullState,'/'+ self.robot +'/full_state', self.full_state_callback, 10)
        self.full_state_pub = self.create_publisher(FullState,'/'+ self.robot + '/cmd_full_state', 10)
        self.attitude_thrust_pub = self.create_publisher(Twist,'/'+ self.robot + '/cmd_vel_legacy', 10)
        self.target_r = None
        self.target_v = None
        self.target_a = None

        self.timer = self.create_timer(self.timer_period, self.timer_callback)

    def timer_callback(self):
        if self.target_a is not None:
            self.target_a = np.clip(self.target_a, -1, 1)
            # self.target_r[:2] = np.clip(self.target_r[:,2], -1.5, 1.5)
            # self.target_r[2] = np.clip(self.target_r[2], 0.2, 1.5)
            f_T_r, roll, pitch, yawrate = self.generate_reference()
            #Wr_r = np.clip(np.rad2deg(Wr_r), -5, 5)
            self.next_point_attitude_thrust(pitch, roll,np.rad2deg(yawrate), f_T_r)
            #self.next_point_full_state(self.target_r,self.target_v,self.target_a,Wr_r,quat)

    def generate_reference(self):

        fa_r = self.mb*self.target_a +self.mb*self.g*self.I3 #+ Ca_r@D@Ca_r.T@va_r
        f_T_r = self.I3.T@self.Ca_r.T@fa_r
        if np.linalg.norm(fa_r) != 0:
            r3 = fa_r.reshape(3,1)/np.linalg.norm(fa_r)
        else:
            r3 = np.zeros((3,1))

        aux = R3_so3(r3)@self.ca_1
        if np.linalg.norm(aux) != 0:
            r2 = aux.reshape(3,1)/np.linalg.norm(aux)
        else:
            r2 = np.zeros((3,1))

        r1 = (R3_so3(r2)@r3).reshape(3,1);
        Ca_r_new = np.hstack((r1, r2, r3))
        if np.linalg.norm(r3) != 0:
            Wr_r = so3_R3(np.linalg.inv(self.Ca_r)@Ca_r_new)/self.timer_period
        else:
            Wr_r = np.zeros((3,1))
        self.Ca_r = Ca_r_new

        angles = R.from_matrix(Ca_r_new).as_euler('zyx', degrees=True)
        #quat = R.from_matrix(Ca_r_new).as_quat()
        roll = angles[2]
        pitch = angles[1]
        Wr_r = self.Ca_b.T@self.Ca_r@Wr_r
        return f_T_r, roll, pitch, Wr_r[2]
        
    def full_state_callback(self, msg):
        self.target_r = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        self.target_v = np.array([msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z])
        self.target_a = np.array([msg.acc.x, msg.acc.y, msg.acc.z])
        self.Ca_b = R.from_quat([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]).as_matrix()
    
    def clip_value(self,value, min_val=-32768, max_val=32767):
        return max(min_val, min(value, max_val))
    
    def next_point_full_state(self,r,v,v_dot=np.zeros(3),Wr_r_new=np.zeros(3),quat_new=np.array([0,0,0,1])):
        #self.info(f"debug before {r}, {v}, {v_dot}, {Wr_r_new}, {quat_new}")
        r = [self.clip_value(i) for i in r]
        v = [self.clip_value(i) for i in v]
        v_dot = [self.clip_value(i) for i in v_dot]
        Wr_r_new = [self.clip_value(i) for i in Wr_r_new]
        quat_new = [self.clip_value(i) for i in quat_new]

        #self.info(f"debug after {r}, {v}, {v_dot}, {Wr_r_new}, {quat_new}")
        msg = FullState()
        msg.pose.position.x = float(r[0])
        msg.pose.position.y = float(r[1])
        msg.pose.position.z = float(r[2])
        msg.acc.x = float(v_dot[0])
        msg.acc.y = float(v_dot[1])
        msg.acc.z = float(v_dot[2])
        msg.pose.orientation.x = float(quat_new[0])
        msg.pose.orientation.y = float(quat_new[1])
        msg.pose.orientation.z = float(quat_new[2])
        msg.pose.orientation.w = float(quat_new[3])
        msg.twist.linear.x = float(v[0])
        msg.twist.linear.y = float(v[1])
        msg.twist.linear.z = float(v[2])
        msg.twist.angular.x = float(Wr_r_new[0])
        msg.twist.angular.y = float(Wr_r_new[1])
        msg.twist.angular.z = float(Wr_r_new[2])
        self.full_state_pub.publish(msg)

    def next_point_attitude_thrust(self,pitch, roll,yawrate, f_T_r):
        msg = Twist()
        msg.angular.x = np.rad2deg(float(pitch))
        msg.angular.y = np.rad2deg(float(roll))
        msg.angular.z = np.rad2deg(float(yawrate))
        msg.linear.z = float(f_T_r)
        self.attitude_thrust_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    full_reference = FullReference()
    rclpy.spin(full_reference)
    full_reference.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()