import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from motion_capture_tracking_interfaces.msg import NamedPoseArray
from crazyflie_interfaces.msg import StringArray
from std_msgs.msg import Bool
from rclpy.duration import Duration
from crazy_encirclement.embedding import Embedding
from std_srvs.srv import Empty

from crazy_encirclement.utils import generate_reference

import time
import numpy as np
from scipy.spatial.transform import Rotation as R

class AgentsOrder(Node):
    """
    Subscribes to the poses topic and publishes the agents' order
    """
    def __init__(self):
        super().__init__('encirclement')
        self.info = self.get_logger().info
        self.info('Agents order node has been started.')
        self.declare_parameter('robot_data', ['C04', 'C20', 'C05']) 

        self.robots = self.get_parameter('robot_data').value
        self.n_agents  = len(self.robots)

        self.order = StringArray()
        self.has_order = False#np.zeros((3,self.n_agents))
        self.initial_phases = {}        

        qos_profile = QoSProfile(reliability =QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            deadline=Duration(seconds=0, nanoseconds=0))
        self.create_subscription(
            NamedPoseArray, "/poses",
            self._poses_changed, qos_profile
        )
        while self.has_order == False:
            rclpy.spin_once(self, timeout_sec=0.1)
        # self.order.data = ['C04', 'C20', 'C05']

        self.order_publisher = self.create_publisher(StringArray, '/agents_order', 10)
        self.timer_period = 0.01
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

    def timer_callback(self):
        try:
            self.order_publisher.publish(self.order)          

        except KeyboardInterrupt:
            self.info('Exiting node...')
            self.destroy_node()
            rclpy.shutdown()

    def _poses_changed(self, msg):
        """
        Topic update callback to the motion capture lib's
           poses topic to send through the external position
           to the crazyflie 
        """

        if not self.has_order:
            
            self.initial_pose = np.zeros((3,self.n_agents))
            for pose in msg.poses:
                if pose.name in self.robots:
                    self.initial_phases[str(pose.name)] = np.mod(np.arctan2(pose.pose.position.y, pose.pose.position.x),2*np.pi)
            
            self.order.data = sorted(self.initial_phases, key=lambda x: self.initial_phases[x],reverse=True)
            self.get_logger().info(f'Order of agents: {self.order.data}')
            self.has_order = True         
            self.info(f'Order of agents: {self.order}')    


def main():
    rclpy.init()
    order = AgentsOrder()
    rclpy.spin(order)
    order.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
