import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from motion_capture_tracking_interfaces.msg import NamedPoseArray
from crazyflie_interfaces.msg import FullState, StringArray
from std_msgs.msg import Bool
from rclpy.duration import Duration
from crazy_encirclement.embedding_SO3_ros import Embedding
from std_srvs.srv import Empty
from std_msgs.msg import Float32MultiArray, Float32
from geometry_msgs.msg import Pose
from crazy_encirclement.utils2 import generate_reference, trajectory

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

        self.order = []
        self.has_initial_pose = False
        self.has_final = False
        self.has_taken_off = False
        self.has_hovered = False
        self.has_landed = False
        self.land_flag = False
        self.encircle_flag = False
        self.has_order = False
        self.final_pose = np.zeros((3,1))
        self.agents_r = np.zeros((3,1))
        self.initial_pose = np.zeros((3,1))
        
        self.agents_v = np.zeros((3,1))
        self.hover_height = 0.6
        #change back #############################################################
        # self.agents_r[0,0] = self.r*np.cos(self.initial_phase)
        # self.agents_r[1,0] = self.r*np.sin(self.initial_phase)
        # self.agents_r[2,0] = self.hover_height
        # self.initial_pose = self.agents_r.copy()
        self.i_landing = 0
        self.i_takeoff = 0
        self.phases = np.zeros(self.n_agents)
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
        qos_profile = QoSProfile(reliability =QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            deadline=Duration(seconds=0, nanoseconds=0))

        self.create_subscription(
            NamedPoseArray, "/poses",
            self._poses_changed, qos_profile
        )
        
        # self.create_subscription(
        #     Float32MultiArray,
        #     '/'+self.robot+'/phases',
        #     self._phase_callback,
        #     10)
        
        # while not self.has_order:
        #     rclpy.spin_once(self, timeout_sec=0.1)
        while (not self.has_initial_pose):
            rclpy.spin_once(self, timeout_sec=0.1)

        self.info(f"Initial pose: {self.initial_pose}")
        self.info("First pose received. Moving on...")


        self.position_pub = self.create_publisher(Pose,'/'+ self.robot + '/cmd_position', 10)
        self.phase_pub = self.create_publisher(Float32,'/'+ self.robot + '/phase', 10)

        #initiating some variables
        self.target_v = np.zeros(3)
        self.kx = 15
        self.kv = 2.5*np.sqrt(2)

        self.timer_period = 0.010
        self.embedding = Embedding(self.r, self.phi_dot,self.k_phi, self.tactic,self.n_agents,self.initial_pose,self.hover_height,self.timer_period)
        # while (not self.has_initial_pose):
        #     rclpy.spin_once(self, timeout_sec=0.1)
        self.landing_traj(4)
        self.takeoff_traj(3)

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
                beginning = time.time()

                self.info(f"agents: {self.agents_r}")
                phi, target_r, target_v= self.embedding.targets(self.agents_r,self.phases)
                self.next_point(target_r[:,0])
                self.phi_cur.data = float(phi)
                self.phase_pub.publish(self.phi_cur)
                if self.n_agents > 1:
                    self.phases[1] = phi
                else:
                    self.phases = phi

                # self.info(f"target_r_new: {target_r_new}")
                # self.info(f"target_v_new: {target_v_new}")
                # self.info(f"agents_r: {self.agents_r}")
 

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
        if not self.has_initial_pose:      
            for pose in msg.poses:
                if pose.name == self.robot:
                    self.initial_pose[0] = pose.pose.position.x
                    self.initial_pose[1] = pose.pose.position.y
                    self.initial_pose[2] = pose.pose.position.z      
            self.has_initial_pose = True    
            
        elif not self.land_flag :
            for pose in msg.poses:
                if pose.name == self.robot:
                    self.agents_r[0] = pose.pose.position.x
                    self.agents_r[1] = pose.pose.position.y
                    self.agents_r[2] = pose.pose.position.z
                    # self.agents_v[0] = pose.velocity.linear.x
                    # self.agents_v[1] = pose.velocity.linear.y
                    # self.agents_v[2] = pose.velocity.linear.z
        elif self.has_final == False and self.land_flag == True:
            self.has_final = True
            self.final_pose = np.zeros(3)
            self.info("Landing...")
            for pose in msg.poses:
                if pose.name == self.robot:
                    self.final_pose[0] = pose.pose.position.x
                    self.final_pose[1] = pose.pose.position.y
                    self.final_pose[2] = pose.pose.position.z
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
        self.r_takeoff[0,:] += self.initial_pose[0,0]*np.ones(len(self.t_takeoff))
        self.r_takeoff[1,:] += self.initial_pose[1,0]*np.ones(len(self.t_takeoff))
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

def main():
    rclpy.init()
    encirclement = Circle_distortion()
    rclpy.spin(encirclement)
    encirclement.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
