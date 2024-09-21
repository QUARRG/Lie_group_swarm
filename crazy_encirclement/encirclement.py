import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from motion_capture_tracking_interfaces.msg import NamedPoseArray
from rclpy.duration import Duration
import numpy as np
from crazy_encirclement.embedding import Embedding
from geometry_msgs.msg import Pose
from nav_msgs.msg import Odometry

class Encirclement(Node):
    def __init__(self):
        super().__init__('encirclement')
        self.declare_parameter('r', '0.5')
        self.declare_parameter('k_phi', '5')
        self.declare_parameter('robot_data', ['C103', 'C104', 'C105'])
        self.declare_parameter('phi_dot', '0.5')
        self.declare_parameter('tactic', 'dumbbell')    

        self.robots = self.get_parameter('robot_data').value
        self.n_agents  = len(self.robots)
        self.r  = float(self.get_parameter('r').value)
        self.k_phi  = float(self.get_parameter('k_phi').value)
        self.phi_dot  = float(self.get_parameter('phi_dot').value)
        self.tactic  = self.get_parameter('tactic').value


        self.agents_r = np.zeros((3, self.n_agents))

        self.phi_cur = np.zeros(self.n_agents)

        self.embedding = Embedding(self.r, self.phi_dot,self.k_phi, self.tactic,self.n_agents)
        self.my_publishers = []

        self.get_logger().info('Encirclement node has been started.')
        timer_period = 0.5

            #deadline = Duration(seconds=0, nanoseconds=1e9/100.0))
        for robot in self.robots:
            self.my_publishers.append(self.create_publisher(Pose,'/'+ robot + '/cmd_position', 10))
        
        #self.subscriber = []
        #for robot in self.robots:
        #    self.subscriber.append(self.create_subscription(
        #        Odometry, robot + '/odom', self._odom_callback, qos_profile
        #    ))
        qos_profile = QoSProfile(reliability =QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            deadline=Duration(seconds=0, nanoseconds=0))

        self.create_subscription(
            NamedPoseArray, "/poses",
            self._poses_changed, qos_profile
        )
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        phi_new, target_r_new, target_v_new, _, _, _ = self.embedding.targets(self.agents_r[:,:], self.phi_cur)
        self.phi_cur = phi_new
        for i in range(self.n_agents):
            msg = Pose()
            msg.position.x = target_r_new[0,i]
            msg.position.y = target_r_new[1,i]
            msg.position.z = target_r_new[2,i]
            self.get_logger().info(f"Publishing to {msg.position.x}, {msg.position.y}, {msg.position.z}")

            #self.publishers[i].publish(msg)
    
    
    def _poses_changed(self, msg):
        """
        Topic update callback to the motion capture lib's
           poses topic to send through the external position
           to the crazyflie 
        """
        # self.get_logger().error(f"Received poses")
        poses = msg.poses
        for pose in poses:
            if pose.name in self.robots:
                x = pose.pose.position.x
                y = pose.pose.position.y
                z = pose.pose.position.z
                self.agents_r[0, self.robots.index(pose.name)] = x
                self.agents_r[1, self.robots.index(pose.name)] = y
                self.agents_r[2, self.robots.index(pose.name)] = z
                #quat = pose.pose.orientation


def main():
    rclpy.init()
    encirclement = Encirclement()
    rclpy.spin(encirclement)
    encirclement.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
