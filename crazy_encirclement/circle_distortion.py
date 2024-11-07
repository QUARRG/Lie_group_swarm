import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy,QoSDurabilityPolicy
from motion_capture_tracking_interfaces.msg import NamedPoseArray
from crazyflie_interfaces.msg import FullState, StringArray
from std_msgs.msg import Bool
from rclpy.duration import Duration
from crazy_encirclement.embedding_SO3_ros import Embedding
from std_srvs.srv import Empty
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray, Float32
from geometry_msgs.msg import Pose, Twist, PoseStamped
from crazy_encirclement.utils2 import  trajectory, R3_so3, so3_R3

import time
import numpy as np
from scipy.spatial.transform import Rotation as R

class Circle_distortion(Node):
    def __init__(self):
        super().__init__('circle_distortion')
        self.info = self.get_logger().info
        self.info('Circle distortion node has been started.')
        self.declare_parameter('r', '1')
        self.declare_parameter('robot', 'C20')
        self.declare_parameter('number_of_agents', '1')
        self.declare_parameter('phi_dot', '0.5')
        self.declare_parameter('tactic', 'circle')    
        self.declare_parameter('initial_pase', '0')

        self.robot = self.get_parameter('robot').value
        self.n_agents  = int(self.get_parameter('number_of_agents').value)
        self.r  = float(self.get_parameter('r').value)
        self.k_phi  = 0#float(self.get_parameter('k_phi').value)
        self.phi_dot  = float(self.get_parameter('phi_dot').value)
        self.initial_phase = float(self.get_parameter('initial_pase').value)
        self.tactic  = self.get_parameter('tactic').value
        self.reboot_client = self.create_client(Empty,  '/'+'all'+'/reboot')

        self.mb = 0.04
        self.g = 9.81
        self.I3 = np.array([0,0,1]).T.reshape(3)
        w_r = 0 #reference yaw
        self.ca_1 = np.array([np.cos(w_r),np.sin(w_r),0]).T #auxiliar vector 
        self.Ca_r = np.eye(3)
        self.Ca_b = np.eye(3)

        self.order = []
        self.has_initial_pose = False
        self.has_final = False
        self.has_taken_off = False
        self.has_hovered = False
        self.has_landed = False
        self.land_flag = False
        self.encircle_flag = False
        self.has_order = False
        self.final_pose = np.zeros(3)
        self.agents_r = np.zeros(3)
        self.initial_pose = np.zeros(3)
        self.Ca_r = np.eye(3)
        self.quat = [0,0,0,1]
        self.agents_v = np.zeros(3)
        self.hover_height = 0.6
        self.target_a = np.zeros(3)
        #change back #############################################################
        # self.agents_r[0] = self.r*np.cos(self.initial_phase)
        # self.agents_r[1] = self.r*np.sin(self.initial_phase)
        # self.agents_r[2] = self.hover_height
        # self.initial_pose = self.agents_r.copy()
        self.i_landing = 0
        self.i_takeoff = 0
        if self.n_agents > 1:
            self.phases = None
        else:
            self.phases = 0
        self.phi_cur = Float32()
        
        self.create_subscription(
            Bool,
            '/landing',
            self._landing_callback,
            10)
        self.create_subscription(
            Bool,
            '/encircle',
            self._encircle_callback,
            10)
        # qos_profile = QoSProfile(reliability =QoSReliabilityPolicy.BEST_EFFORT,
        #     history=QoSHistoryPolicy.KEEP_LAST,
        #     depth=1,
        #     deadline=Duration(seconds=0, nanoseconds=0))

        # self.create_subscription(
        #     NamedPoseArray, "/poses",
        #     self._poses_changed, qos_profile
        # )
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,  # Keep the last 10 messages in the buffer
            durability=QoSDurabilityPolicy.VOLATILE,
            # deadline=rclpy.duration.Duration(seconds=0.05),  # 50 ms deadline
            # lifespan=rclpy.duration.Duration(seconds=1.0)    # Message lifespan of 1 second
        )
        self.create_subscription(
            Odometry, "/"+self.robot+"/odom",
            self._poses_changed, qos_profile
        )
        

        if self.n_agents > 1:
            self.create_subscription(
                Float32MultiArray,
                '/'+self.robot+'/phases',
                self._phase_callback,
                10)
            while not self.phases:
                rclpy.spin_once(self, timeout_sec=0.1)
                
        while (not self.has_initial_pose):
            rclpy.spin_once(self, timeout_sec=0.1)

        self.info(f"Initial pose: {self.initial_pose}")
        self.info("First pose received. Moving on...")


        self.position_pub = self.create_publisher(Pose,'/'+ self.robot + '/cmd_position', 10)
        self.phase_pub = self.create_publisher(Float32,'/'+ self.robot + '/phase', 10)
        self.full_state_pub = self.create_publisher(FullState,'/'+ self.robot + '/full_state', 10)
        self.attitude_thrust_pub = self.create_publisher(Twist,'/'+ self.robot + '/vel_legacy', 100)

        #initiating some variables
        self.target_v = np.zeros(3)
        self.kx = 5
        self.kv = 2.5*np.sqrt(2)

        self.timer_period = 0.01
        self.embedding = Embedding(self.r, self.phi_dot,self.k_phi, self.tactic,self.n_agents,self.initial_pose,self.hover_height,self.timer_period,4)
        # while (not self.has_initial_pose):
        #     rclpy.spin_once(self, timeout_sec=0.1)
        self.landing_traj(4)
        self.takeoff_traj(1)

        input("Press Enter to takeoff")
        #time.sleep(2.0)

        self.timer = self.create_timer(self.timer_period, self.timer_callback)

    def timer_callback(self):

        try:
            if self.land_flag:
                self.landing()

                if self.i_landing < len(self.t_landing)-1:
                    self.i_landing += 1
                else:
                    self.has_landed = True
                    self.has_taken_off = False
                    self.has_hovered = False
                    self.reboot()
                    self.destroy_node()
                    self.info('Exiting circle node')

            elif not self.has_taken_off and not self.has_landed:
                self.takeoff()

            elif (not self.has_hovered) and (self.has_taken_off) and not self.has_landed:
                self.hover()   
                if self.encircle_flag:
                    self.has_hovered = True
                    self.info('Hovering finished')
            
            elif not self.has_landed and self.has_hovered:# and self.pose.position.z > 0.10:#self.ra_r[:,0]:
                
                phi, target_r, target_v, phi_dot_x= self.embedding.targets(self.agents_r,self.phases)
                #self.info(f"phi_dot_x: {phi_dot_x}")
                # self.info(f"target_r: {target_r},{target_v}")

                
                self.phi_cur.data = float(phi)
                self.phase_pub.publish(self.phi_cur)
                if self.n_agents > 1:
                    self.phases[1] = phi
                else:
                    self.phases = phi
                #self.info(f"agents_r: {target_r}, {self.agents_r}, agents_v: {target_v}, {self.agents_v}")
                #accels = self.kx*(target_r - self.agents_r) + self.kv*(target_v - self.agents_v)
                #accels = (target_v - self.agents_v)/self.timer_period
                #accels = np.clip(accels, -1, 1)
                # agents_v = self.agents_v + accels*self.timer_period
                #self.info(f"agents_r: {self.agents_r}, agents_v*dt: {self.agents_v*self.timer_period }, accel*dt^2: {0.5*accels*self.timer_period**2}")
                # self.agents_r = self.agents_r + self.agents_v*self.timer_period + 0.5*accels*self.timer_period**2
                # self.agents_v = agents_v               
                
                #self.target_a = np.clip(self.target_a, -1, 1)
                # self.target_r[:2] = np.clip(self.target_r[:,2], -1.5, 1.5)
                # self.target_r[2] = np.clip(self.target_r[2], 0.2, 1.5)
                #f_T_r, roll, pitch, yawrate = self.generate_reference(self.target_a)
                #Wr_r = np.clip(np.rad2deg(Wr_r), -5, 5)
                self.next_point(target_r)
                #self.next_point_full_state(target_r,target_v,self.target_a,np.zeros(3),self.quat)
                #self.next_point_attitude_thrust(pitch, roll,yawrate, f_T_r)
                #self.landing()
                

                

        except KeyboardInterrupt:
            self.landing()
            self.info('Exiting open loop command node')

            #self.publishers[i].publish(msg)
    
    def _poses_changed(self, msg):
        """
        Topic update callback to the motion capture lib's
           poses topic to send through the external position
           to the crazyflie 
        """
        self.quat = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]

        if not self.has_initial_pose:      
            self.initial_pose[0] = msg.pose.pose.position.x
            self.initial_pose[1] = msg.pose.pose.position.y
            self.initial_pose[2] = msg.pose.pose.position.z      
            self.has_initial_pose = True    
            
        elif not self.land_flag :

            self.agents_r[0] = msg.pose.pose.position.x
            self.agents_r[1] = msg.pose.pose.position.y
            self.agents_r[2] = msg.pose.pose.position.z
            self.agents_v[0] = msg.twist.twist.linear.x
            self.agents_v[1] = msg.twist.twist.linear.y
            self.agents_v[2] = msg.twist.twist.linear.z
        elif self.has_final == False and self.land_flag == True:
            self.has_final = True
            self.final_pose = np.zeros(3)
            self.info("Landing...")


            self.final_pose[0] = msg.pose.pose.position.x
            self.final_pose[1] = msg.pose.pose.position.y
            self.final_pose[2] = msg.pose.pose.position.z
            self.r_landing[0,:] += self.final_pose[0]*np.ones(len(self.t_landing))
            self.r_landing[1,:] += self.final_pose[1]*np.ones(len(self.t_landing))


        

    def _phase_callback(self, msg):
        self.phases = msg.data

    def takeoff(self):
        self.next_point(self.r_takeoff[:,self.i_takeoff])
        #self.info(f"Publishing to {msg.pose.position.x}, {msg.pose.position.y}, {msg.pose.position.z}")
        if self.i_takeoff < len(self.t_takeoff)-1:
            self.i_takeoff += 1
        else:
            self.has_taken_off = True
        self.t_init = time.time()

    def takeoff_traj(self,t_max):
        #takeoff trajectory
        self.t_takeoff = np.arange(0,t_max,self.timer_period)
        #self.t_takeoff = np.tile(t_takeoff[:,np.newaxis],(1,self.n_agents))
        self.r_takeoff = np.zeros((3,len(self.t_takeoff))) 
        self.r_takeoff[0,:] += self.initial_pose[0]*np.ones(len(self.t_takeoff))
        self.r_takeoff[1,:] += self.initial_pose[1]*np.ones(len(self.t_takeoff))
        self.r_takeoff[2,:] = self.hover_height*(self.t_takeoff/t_max)
        v,_ = trajectory(self.r_takeoff,self.timer_period)
        self.r_dot_takeoff = v

    def landing_traj(self,t_max):
        #landing trajectory
        self.t_landing = np.arange(t_max,1,-self.timer_period)
        self.i_landing = 0
        self.r_landing = np.zeros((3,len(self.t_landing)))
        self.r_landing[2,:] = self.hover_height*(self.t_landing/t_max)
        v,_ = trajectory(self.r_landing,self.timer_period)
        self.r_dot_landing = v
    
    def _landing_callback(self, msg):
        self.land_flag = msg.data

    def _encircle_callback(self, msg):
        self.encircle_flag = msg.data

    def hover(self):

        msg = Pose()
        msg.position.x = self.r*np.cos(self.initial_phase)
        msg.position.y = self.r*np.sin(self.initial_phase)
        msg.position.z = self.hover_height
        self.position_pub.publish(msg)


    def landing(self):
        self.next_point(self.r_landing[:,self.i_landing])

    def reboot(self):
        req = Empty.Request()
        self.reboot_client.call_async(req)
        time.sleep(2.0)

    def next_point(self,r,quat_new=np.array([0,0,0,1])):
        msg = Pose()
        msg.position.x = float(r[0])
        msg.position.y = float(r[1])
        msg.position.z = float(r[2])
        msg.orientation.x = float(quat_new[0])
        msg.orientation.y = float(quat_new[1])
        msg.orientation.z = float(quat_new[2])
        msg.orientation.w = float(quat_new[3])   
        self.position_pub.publish(msg)

    def clip_value(self,value, min_val=-32768, max_val=32767):
        return max(min_val, min(value, max_val))
    def next_point_full_state(self,r,v,v_dot=np.zeros(3),Wr_r_new=np.zeros(3),quat_new=np.array([0,0,0,1])):
        # self.info(f"debug before {r}, {v}, {v_dot}, {Wr_r_new}, {quat_new}")
        # r = [self.clip_value(i) for i in r]
        # v = [self.clip_value(i) for i in v]
        # v_dot = [self.clip_value(i) for i in v_dot]
        # Wr_r_new = [self.clip_value(i) for i in Wr_r_new]

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
        msg.twist.angular.x = np.rad2deg(float(Wr_r_new[0]))
        msg.twist.angular.y = np.rad2deg(float(Wr_r_new[1]))
        msg.twist.angular.z = np.rad2deg(float(Wr_r_new[2]))
        self.full_state_pub.publish(msg)

    def next_point_attitude_thrust(self,pitch, roll,yawrate, f_T_r):
        msg = Twist()
        msg.angular.x = float(pitch)
        msg.angular.y = float(roll)
        msg.angular.z = float(yawrate)
        msg.linear.z = float(f_T_r)
        self.attitude_thrust_pub.publish(msg)

    def generate_reference(self, va_r_dot=np.zeros(3)):

        fa_r = self.mb*va_r_dot +self.mb*self.g*self.I3 #+ Ca_r@D@Ca_r.T@va_r
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
        return f_T_r, roll, pitch, np.rad2deg(Wr_r[2])

 
def main():
    rclpy.init()
    encirclement = Circle_distortion()
    rclpy.spin(encirclement)
    encirclement.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
