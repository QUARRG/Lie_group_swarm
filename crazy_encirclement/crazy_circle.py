import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from rclpy.duration import Duration

from crazyflie_interfaces.srv import Takeoff, Land, NotifySetpointsStop
from std_srvs.srv import Empty
from geometry_msgs.msg import Pose, PoseStamped
from std_msgs.msg import Bool
#from nav_msgs.msg import Odometry
from crazy_encirclement.utils2 import generate_reference

from crazyflie_interfaces.msg import FullState
from motion_capture_tracking_interfaces.msg import NamedPoseArray
#the reference angular velocity is in degrees
import time
import numpy as np
from crazy_encirclement.set_parameter_client import SetParameterClient

class Circle(Node):
    def __init__(self):
        super().__init__('circle_trajectory_node')
        self.declare_parameter('robot_prefix', 'C120')  

        self.robot = self.get_parameter('robot_prefix').value
        #clients
        self.notify_client = self.create_client(NotifySetpointsStop, '/'+self.robot+'/notify_setpoints_stop')  
        self.reboot_client = self.create_client(Empty,  '/'+self.robot+'/reboot')
        qos_profile = QoSProfile(reliability =QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            deadline=Duration(seconds=0, nanoseconds=0))
            #deadline = Duration(seconds=0, nanoseconds=1e9/100.0))

        self.create_subscription(
            NamedPoseArray, "/poses",
            self._pose_callback, qos_profile
        )
        self.subscription = self.create_subscription(
            Bool,
            'landing',
            self._landing_callback,
            10)
        self.publisher_pose = self.create_publisher(Pose, self.robot + '/cmd_position', 10)
        self.full_state_publisher = self.create_publisher(FullState,'/'+ self.robot + '/cmd_full_state', 10)

        self.has_taken_off = False
        self.has_landed = False
        self.has_hovered = False
        self.msg = FullState()
        
        self.pose = Pose()
        self.land_flag = False
        self.initial_pose = None
        self.final_pose = None
        self.timer_period = 0.01
        self.Tp = 60
        self.dt = self.timer_period
        self.t = np.arange(0,self.Tp,self.dt)
        self.N = len(self.t)
        self.t_init = 0
        
        self.Ca_r = np.zeros((3,3,self.N))
        self.i = 0
        self.i_takeoff = 0
        
        while self.initial_pose is None:
            rclpy.spin_once(self, timeout_sec=0.1)
        self.get_logger().info("First pose received. Moving on...")
        x= self.initial_pose.position.x
        y= self.initial_pose.position.y
        self.r = np.sqrt(x**2 + y**2)
        phase = np.arctan2(y,x)
        #center = np.array([np.sign(x)*(np.abs(x)-self.r),np.sign(y)*(np.abs(y)-self.r)])
        self.hover_height = 0.3
       
        
        #generating the cicle trajectory
        self.ra_r = np.vstack((self.r*np.cos(2*np.pi*self.t/self.Tp + phase),self.r*np.sin(2*np.pi*self.t/self.Tp+phase),self.hover_height*np.ones(self.N)))
        self.va_r = np.zeros_like(self.ra_r)
        self.va_r_dot = np.zeros_like(self.va_r)
        v,v_dot = self.trajectory(self.ra_r)
        self.va_r = v
        self.va_r_dot = v_dot

        #takeoff trajectory
        self.t_max = 5
        self.t_takeoff = np.arange(0,self.t_max,self.timer_period)
        self.r_takeoff = np.zeros((3,len(self.t_takeoff))) 
        self.r_takeoff[0,:] += self.initial_pose.position.x
        self.r_takeoff[1,:] += self.initial_pose.position.y
        self.r_takeoff[2,:] = self.hover_height*(self.t_takeoff/self.t_max) #+ self.initial_pose.position.z
        v,_ = self.trajectory(self.r_takeoff)
        self.r_dot_takeoff = v

        #landing trajectory
        self.t_landing = np.arange(self.t_max,1,-self.timer_period)
        self.i_landing = 0
        self.r_landing = np.zeros((3,len(self.t_landing)))
        self.r_landing[2,:] = self.hover_height*(self.t_landing/self.t_max) #+ self.initial_pose.position.z
        v,_ = self.trajectory(self.r_landing)
        self.r_dot_landing = v

        self.get_logger().info('Circle node has been started.')
        
        for i in range(7):            
            _, _, _,_, Ca_r_new = generate_reference(self.va_r_dot[:,0],self.Ca_r[:,:,0],self.va_r[:,0],self.dt)
            self.Ca_r[:,:,0] = Ca_r_new
        self.get_logger().info(f"Initial pose: {self.initial_pose.position.x}, {self.initial_pose.position.y}, {self.initial_pose.position.z}")
        #input("Press Enter to takeoff")
        time.sleep(3.0)
        
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

    def _landing_callback(self, msg):
        self.land_flag = msg.data

    def _pose_callback(self, msg):
        for pose in msg.poses:
            if pose.name == self.robot:
                if self.initial_pose is None:
                    self.get_logger().info('Initial pose received')
                    self.initial_pose = pose.pose
                    self.pose = pose.pose

    def takeoff(self):

        self.next_point(self.r_takeoff[:,self.i_takeoff],self.r_dot_takeoff[:,self.i_takeoff])
        #self.get_logger().info(f"Publishing to {msg.pose.position.x}, {msg.pose.position.y}, {msg.pose.position.z}")
        if self.i_takeoff < len(self.t_takeoff)-1:
            self.i_takeoff += 1
        else:
            self.has_taken_off = True
        self.t_init = time.time()

    def hover(self):
        msg = FullState()
        msg.pose.position.x = self.initial_pose.position.x
        msg.pose.position.y = self.initial_pose.position.y
        msg.pose.position.z = self.hover_height
        self.full_state_publisher.publish(msg)

    def reboot(self):
        req = Empty.Request()
        self.reboot_client.call_async(req)
        time.sleep(2.0)

    def timer_callback(self):

        try:
            if self.land_flag or (self.i > len(self.t)-2):
                if self.final_pose is None:
                    self.final_pose = self.pose
                    self.r_landing[0,:] += self.final_pose.position.x
                    self.r_landing[1,:] += self.final_pose.position.y

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

            elif not self.has_taken_off:
                self.takeoff()

            elif (not self.has_hovered) and (self.has_taken_off):
                self.hover()   
                if ((time.time()-self.t_init) > 5):
                    self.has_hovered = True
                    self.get_logger().info('Hovering finished')

            elif not self.has_landed:# and self.pose.position.z > 0.10:#self.ra_r[:,0]:
                Wr_r_new, _, _, quat_new, Ca_r_new = generate_reference(self.va_r_dot[:,self.i],self.Ca_r[:,:,self.i],self.va_r[:,self.i],self.dt)
                self.Ca_r[:,:,self.i+1] = Ca_r_new
                self.next_point(self.ra_r[:,self.i],self.va_r[:,self.i],self.va_r_dot[:,self.i],Wr_r_new,quat_new,self.Ca_r[:,:,self.i])
                
                if self.i <= len(self.t)-2:
                #     self.Ca_r[:,:,self.i+1] = Ca_r_new
                    self.i += 1


        except KeyboardInterrupt:
            self.landing()
            self.get_logger().info('Exiting open loop command node')
    
                #quat = pose.pose.orientation

    
    def landing(self):
        self.next_point(self.r_landing[:,self.i_landing],self.r_dot_landing[:,self.i_landing])

    def next_point(self,r,v,v_dot=np.zeros(3),Wr_r_new=np.zeros(3),quat_new=np.array([0,0,0,1]),Ca_r_new=np.eye(3)):
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
        #self.get_logger().info(f"Publishing to {msg.pose.position.x}, {msg.pose.position.y}, {msg.pose.position.z}")
        self.full_state_publisher.publish(msg)

    def trajectory(self, r):
        v = np.zeros_like(r)
        v_dot = np.zeros_like(v)
        v[:,0:-1] = np.diff(r,axis=1)/self.timer_period
        v_dot[:,0:-2] = np.diff(v[:,:-1],axis=1)/self.timer_period
        return v,v_dot
    
def main():
    rclpy.init()
    encirclement = Circle()
    rclpy.spin(encirclement)
    encirclement.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
