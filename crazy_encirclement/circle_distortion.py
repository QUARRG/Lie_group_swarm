import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy,QoSDurabilityPolicy
from motion_capture_tracking_interfaces.msg import NamedPoseArray
from crazyflie_interfaces.msg import FullState, StringArray, Position
from std_msgs.msg import Bool
from rclpy.duration import Duration
from crazy_encirclement.embedding_SO3_ros import Embedding
from std_srvs.srv import Empty
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray, Float32
from geometry_msgs.msg import Pose, Twist, PoseStamped
from crazy_encirclement.utils2 import  trajectory, R3_so3, so3_R3
from scipy.linalg import expm, logm
import time
import numpy as np
from scipy.spatial.transform import Rotation as R

class Circle_distortion(Node):
    def __init__(self):
        super().__init__('circle_distortion')
        self.info = self.get_logger().info
        self.info('Circle distortion node has been started.')
        self.declare_parameter('r', '1.2')
        self.declare_parameter('robot', 'C20')
        self.declare_parameter('number_of_agents', '3')
        self.declare_parameter('phi_dot', '0.6')
  
        self.robot = self.get_parameter('robot').value
        self.n_agents  = int(self.get_parameter('number_of_agents').value)
        self.r  = float(self.get_parameter('r').value)
        self.k_phi  = 10#float(self.get_parameter('k_phi').value)
        self.phi_dot  = float(self.get_parameter('phi_dot').value)
        self.initial_phase = 0
        self.reboot_client = self.create_client(Empty, self.robot + '/reboot')


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
        self.hover_height = 0.8
        self.target_a = np.zeros(3)
        self.leader = None
        self.follower = None
        self.Rot_des = np.eye(3)
        self.target_r = np.zeros(3)
        self.timer_period = 0.1

        self.i_landing = 0
        self.i_takeoff = 0
        if self.n_agents > 1:
            self.phases = np.zeros(self.n_agents)
        else:
            self.phases = 0
        self.phi_cur = Float32()
        self.phase_pub = self.create_publisher(Float32,'/'+ self.robot + '/phase', 1)

        
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
        )
        self.create_subscription(
            PoseStamped, "/"+self.robot+"/pose",
            self._poses_changed, 10
        )
                
        while (not self.has_initial_pose):
            rclpy.spin_once(self, timeout_sec=0.1)

        if self.n_agents > 1:
            self.phi_cur.data = self.initial_phase
            self.phase_pub.publish(self.phi_cur)
            self.create_subscription(
                StringArray, '/agents_order',
                self._order_callback,
                10)
            while not self.has_order:
                self.phase_pub.publish(self.phi_cur)
                rclpy.spin_once(self, timeout_sec=0.1)

        self.info(f"Initial pose: {self.initial_pose}")

        self.create_subscription(Float32, '/'+ self.leader + '/phase', self._phase_callback_leader, 1)
        self.create_subscription(Float32, '/'+ self.follower + '/phase', self._phase_callback_follower, 1)
        self.phi_cur.data = self.initial_phase
        while (self.phases[0] == 0):
            self.phase_pub.publish(self.phi_cur)
            rclpy.spin_once(self, timeout_sec=0.1)

        while (self.phases[2] == 0):
            self.phase_pub.publish(self.phi_cur)
            rclpy.spin_once(self, timeout_sec=0.1)

        self.info(f"agents phases: {self.phases}")
        self.wd = Float32()
        
        self.position_pub = self.create_publisher(Position,'/'+ self.robot + '/cmd_position', 10)
        # self.full_state_pub = self.create_publisher(FullState,'/'+ self.robot + '/full_state', 10)
        # self.attitude_thrust_pub = self.create_publisher(Twist,'/'+ self.robot + '/vel_legacy', 100)
        self.publisher_w = self.create_publisher(Float32,'/'+ self.robot + '/omega_d', 10)
        self.publisher_w.publish(self.wd)
        #initiating some variables

        self.embedding = Embedding(self.r, self.phi_dot,self.k_phi, self.n_agents,self.initial_pose,self.hover_height,self.timer_period,self.phase_pub)

        self.landing_traj(4)
        self.takeoff_traj(1)

        input("Press Enter to takeoff")

        self.timer = self.create_timer(self.timer_period, self.timer_callback)

    def timer_callback(self):

        try:
            if self.land_flag and self.has_final:
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
                
                # phi, target_r, target_v, phi_dot_x, wd = self.embedding.targets(self.agents_r,self.phases)
                pos = np.array([self.agents_r[0], self.agents_r[1], self.agents_r[2]-self.hover_height])
                
                pos_rot = np.linalg.inv(self.Rot_des)@pos.T
                phi_i, _ = self.embedding.cart2pol(pos_rot)
                self.phi_cur.data = float(phi_i)
                self.phase_pub.publish(self.phi_cur)
                if self.n_agents > 1:
                    self.phases[1] = phi_i
                else:
                    self.phases = phi_i

                phi_k = self.phases[0]
                phi_j = self.phases[2]
                #wd = self.phi_dot
                wd = self.embedding.phi_dot_desired(phi_i, phi_j, phi_k, self.phi_dot, self.k_phi)
                    #first evolve the agent in phase
                v_d_hat_z = np.array([0, 0, wd])
                x = self.r * np.cos(phi_i)
                y = self.r * np.sin(phi_i)
                Rot_z = expm(R3_so3(v_d_hat_z.reshape(-1,1))*self.timer_period)
                pos_d_hat = np.array([x, y, 0])
                pos_d_hat = Rot_z@pos_d_hat
                phi_d, _ = self.embedding.cart2pol(pos_d_hat)

        
                phi_dot_x = self.embedding.calc_wx(phi_d)#*(phi_d-self.phi_des[i])
                phi_dot_y = self.embedding.calc_wy(phi_d) #phi_i-phi_prev[i]*
                v_d_hat_x_y = np.array([phi_dot_x, phi_dot_y, 0])
                self.Rot_des = expm(R3_so3(v_d_hat_x_y.reshape(-1,1)))


                pos_d = self.Rot_des@pos_d_hat.T
                # pos_d = Rot@pos_d_hat.T

                self.target_r[0] = pos_d[0]
                self.target_r[1] = pos_d[1]
                self.target_r[2] = pos_d[2] + self.hover_height
                self.target_r[2] = np.clip(self.target_r[2],0.15,1.5)

                self.wd.data = wd
                self.publisher_w.publish(self.wd)
                



                self.next_point(self.target_r)


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

        if not self.has_initial_pose:      
            self.initial_pose[0] = msg.pose.position.x
            self.initial_pose[1] = msg.pose.position.y
            self.initial_pose[2] = msg.pose.position.z   
            self.initial_phase = np.mod(np.arctan2(self.initial_pose[1], self.initial_pose[0]),2*np.pi)   
            self.has_initial_pose = True    
            
        elif not self.land_flag :

            self.agents_r[0] = msg.pose.position.x
            self.agents_r[1] = msg.pose.position.y
            self.agents_r[2] = msg.pose.position.z

        elif self.has_final == False and self.land_flag == True:
            
            self.final_pose = np.zeros(3)
            self.info("Landing...")


            self.final_pose[0] = msg.pose.position.x
            self.final_pose[1] = msg.pose.position.y
            self.final_pose[2] = msg.pose.position.z
            self.r_landing[0,:] += self.final_pose[0]*np.ones(len(self.t_landing))
            self.r_landing[1,:] += self.final_pose[1]*np.ones(len(self.t_landing))
            self.has_final = True

    def _phase_callback_leader(self, msg):
        if msg.data:
            self.phases[0] = msg.data

    def _phase_callback_follower(self, msg):
        if msg.data:
            self.phases[2] = msg.data
        

    def _order_callback(self, msg):
        if not self.has_order:
            self.get_logger().info(f"Phase received: {msg.data}")
            order = msg.data
            for robot in order:
                if robot == self.robot:
                    i = order.index(robot)
                    if i == 0:
                        self.leader = order[self.n_agents-1]
                        self.follower = order[i+1]
                    elif i == (self.n_agents-1):
                        self.leader = order[i-1]
                        self.follower = order[0]
                    else:
                        self.leader = order[i-1]
                        self.follower = order[i+1]
            self.has_order = True
            self.get_logger().info(f"Leader: {self.leader}, Follower: {self.follower}")

    def takeoff(self):
        self.next_point(self.r_takeoff[:,self.i_takeoff])
        self.phase_pub.publish(self.phi_cur)
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

        msg = Position()
        msg.x = self.r*np.cos(self.initial_phase)
        msg.y = self.r*np.sin(self.initial_phase)
        msg.z = self.hover_height
        self.position_pub.publish(msg)


    def landing(self):
        self.next_point(self.r_landing[:,self.i_landing])

    def reboot(self):
        req = Empty.Request()
        self.reboot_client.call_async(req)
        time.sleep(1.0)    

    def next_point(self,r):
        msg = Position()
        msg.x = float(r[0])
        msg.y = float(r[1])
        msg.z = float(r[2])

        self.position_pub.publish(msg)

    def next_point_full_state(self,r,v,v_dot=np.zeros(3),Wr_r_new=np.zeros(3),quat_new=np.array([0,0,0,1])):

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



 
def main():
    rclpy.init()
    encirclement = Circle_distortion()
    rclpy.spin(encirclement)
    encirclement.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
