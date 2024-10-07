import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from motion_capture_tracking_interfaces.msg import NamedPoseArray
from crazyflie_interfaces.msg import FullState
from std_msgs.msg import Bool
from rclpy.duration import Duration
from crazy_encirclement.embedding import Embedding
from std_srvs.srv import Empty

from crazy_encirclement.utils import generate_reference

import time
import numpy as np
from scipy.spatial.transform import Rotation as R

class Encirclement(Node):
    def __init__(self):
        super().__init__('encirclement')
        self.get_logger().info('Encirclement node has been started.')
        self.declare_parameter('r', '1')
        self.declare_parameter('k_phi', '0.5')
        self.declare_parameter('robot_data', ['C103', 'C120', 'C112'])
        self.declare_parameter('phi_dot', '0.005')
        self.declare_parameter('tactic', 'circle')    

        self.robots = self.get_parameter('robot_data').value
        self.n_agents  = len(self.robots)
        self.r  = float(self.get_parameter('r').value)
        self.k_phi  = float(self.get_parameter('k_phi').value)
        self.phi_dot  = float(self.get_parameter('phi_dot').value)
        self.tactic  = self.get_parameter('tactic').value
        self.reboot_client = self.create_client(Empty,  '/'+'all'+'/reboot')
        self.order = None
        self.has_initial_pose = False#np.zeros((3,self.n_agents))
        self.initial_phases = {}
        self.final_pose = None
        self.has_taken_off = False
        self.has_hovered = False
        self.has_landed = False
        self.land_flag = False
        self.hover_height = 0.3
        self.i_landing = 0
        self.i_takeoff = 0
        self.has_final = False

        #for robot in self.robots:
        self.create_subscription(
            Bool,
            'landing',
            self._landing_callback,
            10)
        qos_profile = QoSProfile(reliability =QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            deadline=Duration(seconds=0, nanoseconds=0))
        self.create_subscription(
            NamedPoseArray, "/poses",
            self._poses_changed, qos_profile
        )
        while self.order == None:
            rclpy.spin_once(self, timeout_sec=0.1)
        while self.has_initial_pose == False:
            rclpy.spin_once(self, timeout_sec=0.1)
        
        self.get_logger().info(f"Initial pose: {self.initial_pose}")
        self.get_logger().info("First pose received. Moving on...")
        self.get_logger().info(f"Order: {self.order}")

        self.my_publishers = []
        for robot in self.order:
            self.my_publishers.append(self.create_publisher(FullState,'/'+ robot + '/cmd_full_state', 10))
        for publish in self.my_publishers:
            self.get_logger().info(f"Publishers: {publish.topic_name}")
        #initiating some variables
        self.target_v_new = np.zeros((3,self.n_agents))
        self.Ca_r = np.zeros((3,3,self.n_agents))
        self.Ca_b = np.zeros((3,3,self.n_agents))
        self.agents_r = self.initial_pose
        self.agents_v = np.zeros((3, self.n_agents))
        self.kx = 2
        self.kv = 2.5*np.sqrt(2)
        self.phi_cur = np.zeros(self.n_agents)
        self.timer_period = 0.01
        quat = np.array([0,0,0,1])
        self.quat_default = np.tile(quat[:,np.newaxis],(1,self.n_agents))
        self.Wr_r_default = np.zeros((3,self.n_agents))
        self.embedding = Embedding(self.r, self.phi_dot,self.k_phi, self.tactic,self.n_agents,self.timer_period)
        

        self.takeoff_traj(5)
        self.landing_traj(3)

        input("Press Enter to takeoff")

        self.timer = self.create_timer(self.timer_period, self.timer_callback)

    def timer_callback(self):
        # phi_new, target_r_new, target_v_new, _, _, _ = self.embedding.targets(self.agents_r[:,:],target_r_new, self.phi_cur)
        # self.phi_cur = phi_new

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
                    self.get_logger().info('Exiting circle node')

            elif not self.has_taken_off and not self.has_landed:
                self.takeoff()

            elif (not self.has_hovered) and (self.has_taken_off) and not self.has_landed:
                # self.get_logger().info(f"initial pose: {self.initial_pose}")
                self.hover()   
                if ((time.time()-self.t_init) > 5):
                    self.has_hovered = True
                    self.get_logger().info('Hovering finished')
                else:
                    _, target_r_new, self.target_v_new, target_a_new, _, _, _ = self.embedding.targets(self.agents_r,self.target_v_new, self.phi_cur)
                    Wr_r_new, _, _,quat_new, self.Ca_r = generate_reference(target_a_new,self.Ca_r,self.Ca_b,self.target_v_new,self.timer_period)

            elif not self.has_landed and self.has_hovered:# and self.pose.position.z > 0.10:#self.ra_r[:,0]:
                #self.get_logger().info("encirclement")
                self.phi_cur, target_r_new, target_v_new, target_a_new, _, _, _ = self.embedding.targets(self.agents_r,self.target_v_new, self.phi_cur)
                self.get_logger().info(f"target_r_new: {target_r_new}")
                self.get_logger().info(f"target_v_new: {target_v_new}")
                self.get_logger().info(f"agents_r: {self.agents_r}")
                Wr_r_new, _, _,quat_new, self.Ca_r = generate_reference(target_a_new,self.Ca_r,self.Ca_b,self.target_v_new,self.timer_period)
                # accels = self.kx*(target_r_new - self.agents_r) + self.kv*(self.target_v_new - self.agents_v)
                # agents_v = self.agents_v + accels*self.timer_period
                # self.agents_r = self.agents_r + agents_v*self.timer_period + 0.5*accels*self.timer_period**2
                # self.agents_v = agents_v
                self.next_point(target_r_new,target_v_new,target_a_new,Wr_r_new,quat_new)
                self.target_v_new = target_v_new
                #self.landing()
                

                

        except KeyboardInterrupt:
            self.landing()
            self.get_logger().info('Exiting open loop command node')

            #self.publishers[i].publish(msg)
    
    
    def _poses_changed(self, msg):
        """
        Topic update callback to the motion capture lib's
           poses topic to send through the external position
           to the crazyflie 
        """

        if bool(self.initial_phases) == False:
            for pose in msg.poses:
                if pose.name in self.robots:
                    self.initial_phases[str(pose.name)] = np.mod(np.arctan2(pose.pose.position.y, pose.pose.position.x),2*np.pi)
            
            self.order = sorted(self.initial_phases, key=lambda x: self.initial_phases[x],reverse=True)
        elif self.has_initial_pose == False:
            self.has_initial_pose = True
            
            self.initial_pose = np.zeros((3,self.n_agents))
            
            for pose in msg.poses:
                if pose.name in self.robots:
                    self.initial_pose[0,self.order.index(pose.name)] = pose.pose.position.x
                    self.initial_pose[1,self.order.index(pose.name)] = pose.pose.position.y
                    self.initial_pose[2,self.order.index(pose.name)] = pose.pose.position.z
            
            
        elif self.land_flag != True:
            for pose in msg.poses:
                if pose.name in self.robots:
                    self.agents_r[0, self.order.index(pose.name)] = pose.pose.position.x
                    self.agents_r[1, self.order.index(pose.name)] = pose.pose.position.y
                    self.agents_r[2, self.order.index(pose.name)] = pose.pose.position.z
                    quat = [pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w]
                    self.Ca_b[:,:,self.order.index(pose.name)] = R.from_quat(quat).as_matrix()
        elif self.has_final == False and self.land_flag == True:
            self.has_final = True
            self.final_pose = np.zeros((3,self.n_agents))
            self.get_logger().info("Landing...")
            for pose in msg.poses:
                if pose.name in self.robots:
                    self.final_pose[0,self.order.index(pose.name)] = pose.pose.position.x
                    self.final_pose[1,self.order.index(pose.name)] = pose.pose.position.y
                    self.final_pose[2,self.order.index(pose.name)] = pose.pose.position.z
            
            self.r_landing[0,:,:] += np.tile(self.final_pose[0,:,np.newaxis],(1,len(self.t_landing)))
            self.r_landing[1,:,:] += np.tile(self.final_pose[1,:,np.newaxis],(1,len(self.t_landing)))
            

    def takeoff(self):

        self.next_point(self.r_takeoff[:,:,self.i_takeoff],self.r_dot_takeoff[:,:,self.i_takeoff],np.zeros((3,self.n_agents)),self.Wr_r_default,self.quat_default)
        #self.get_logger().info(f"Publishing to {msg.pose.position.x}, {msg.pose.position.y}, {msg.pose.position.z}")
        if self.i_takeoff < len(self.t_takeoff)-1:
            self.i_takeoff += 1
        else:
            self.has_taken_off = True
        self.t_init = time.time()

    def takeoff_traj(self,t_max):
        #takeoff trajectory
        self.t_takeoff = np.arange(0,t_max,self.timer_period)
        #self.t_takeoff = np.tile(t_takeoff[:,np.newaxis],(1,self.n_agents))
        self.r_takeoff = np.zeros((3,self.n_agents,len(self.t_takeoff))) 
        # for i in range(len(self.t_takeoff)):
        #     self.r_takeoff[0,:,i] += self.initial_pose[0,:]
        #     self.r_takeoff[1,:,i] += self.initial_pose[1,:]
        # for j in range(self.n_agents):
        #     self.r_takeoff[2,j,:] = self.hover_height*(self.t_takeoff/t_max) #+


        self.r_takeoff[0,:,:] += np.tile(self.initial_pose[0,:,np.newaxis],(1,len(self.t_takeoff)))
        self.r_takeoff[1,:,:] += np.tile(self.initial_pose[1,:,np.newaxis],(1,len(self.t_takeoff)))
        hover_traj = self.hover_height*(self.t_takeoff/t_max)
        self.r_takeoff[2,:,:] = np.tile(hover_traj,(self.n_agents,1)) #+ self.initial_pose.position.z
        v,_ = self.trajectory(self.r_takeoff)
        self.r_dot_takeoff = v
    def landing_traj(self,t_max):
        #landing trajectory
        self.t_landing = np.arange(t_max,1,-self.timer_period)
        self.i_landing = 0
        self.r_landing = np.zeros((3,self.n_agents,len(self.t_landing)))
        landing_traj = self.hover_height*(self.t_landing/t_max)
        self.r_landing[2,:,:] = np.tile(landing_traj,(self.n_agents,1)) #+ self.initial_pose.position.z
        v,_ = self.trajectory(self.r_landing)
        self.r_dot_landing = v
    
    def _landing_callback(self, msg):
        self.land_flag = msg.data

    def hover(self):
        i = 0
        for publisher in self.my_publishers:
            msg = FullState()
            msg.pose.position.x = self.r_takeoff[0,i,-1]
            msg.pose.position.y = self.r_takeoff[1,i,-1]
            msg.pose.position.z = self.hover_height
            publisher.publish(msg)
            i+=1

    def landing(self):
        self.next_point(self.r_landing[:,:,self.i_landing],self.r_dot_landing[:,:,self.i_landing],np.zeros((3,self.n_agents)),self.Wr_r_default,self.quat_default)

    def reboot(self):
        req = Empty.Request()
        self.reboot_client.call_async(req)
        time.sleep(2.0)

    def next_point(self,r,v,v_dot,Wr_r_new,quat_new):
        i = 0
        for publisher in self.my_publishers:
            msg = FullState()
            msg.pose.position.x = float(r[0,i])
            msg.pose.position.y = float(r[1,i])
            msg.pose.position.z = float(r[2,i])
            msg.acc.x = float(v_dot[0,i])
            msg.acc.y = float(v_dot[1,i])
            msg.acc.z = float(v_dot[2,i])
            msg.pose.orientation.x = float(quat_new[0,i])
            msg.pose.orientation.y = float(quat_new[1,i])
            msg.pose.orientation.z = float(quat_new[2,i])
            msg.pose.orientation.w = float(quat_new[3,i])
            msg.twist.linear.x = float(v[0,i])
            msg.twist.linear.y = float(v[1,i])
            msg.twist.linear.z = float(v[2,i])
            msg.twist.angular.x = np.rad2deg(float(Wr_r_new[0,i]))
            msg.twist.angular.y = np.rad2deg(float(Wr_r_new[1,i]))
            msg.twist.angular.z = np.rad2deg(float(Wr_r_new[2,i]))
            i+=1
            
            publisher.publish(msg)
    def trajectory(self, r):
        v = np.zeros_like(r)
        v_dot = np.zeros_like(v)
        for agent in range(self.n_agents):
            v[:,agent,0:-1] = np.diff(r[:,agent,:],axis=1)/self.timer_period
            v_dot[:,agent,0:-2] = np.diff(v[:,agent,:-1],axis=1)/self.timer_period
            return v,v_dot
def main():
    rclpy.init()
    encirclement = Encirclement()
    rclpy.spin(encirclement)
    encirclement.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
