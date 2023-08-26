from robot_library_py import *
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy, JointState
from intera_core_msgs.msg import JointCommand

from threading import Thread
import time
import numpy as np


class ROSInterface:
    def __init__(self, node, robot):
        self.joy_msg = None
        self.joint_states_msg = None
        self.robot = robot
        self.node = node
        self.jointMap = {name: ind for ind, name in enumerate(self.robot.jointNames)}

    def joy_callback(self, msg):
        self.joy_msg = msg

    def joint_states_callback(self, msg):
        self.joint_states_msg = msg
        q = self.robot.getJoints()
        for ind, name in enumerate(msg.name):
            if name in self.jointMap:
                q[self.jointMap[name]] = msg.position[ind]
        self.robot.setJoints(q)

    def spin_thread(self):
        self.node.create_subscription(JointState, 'robot/joint_states', self.joint_states_callback, 10)
        self.node.create_subscription(Joy, '/joy', self.joy_callback, 10)
        rclpy.spin(self.node)


trigger = lambda msg: msg.buttons[0]
mode = lambda msg: msg.buttons[6] == 0
forward = lambda msg: (-msg.axes[1] / 5) * mode(msg)
right = lambda msg: (-(msg.axes[0] - 0.12488976866006851 * 0) / 5) * mode(msg)
up = lambda msg: (msg.axes[2] / 5) * mode(msg)
roll = lambda msg: (msg.axes[0] / 1) * (1 - mode(msg))
pitch = lambda msg: (-msg.axes[1] / 1) * (1 - mode(msg))
yaw = lambda msg: (msg.axes[2] / 1) * (1 - mode(msg))


def run():
    rclpy.init()
    robot = URDFModel('sawyer.urdf')
    node = Node('joy_node')
    q = robot.getJoints()
    ros_interface = ROSInterface(node, robot)

    t1 = Thread(target=ros_interface.spin_thread)
    t1.start()

    I = np.eye(q.shape[0])
    alpha = .0001
    cmd_pub = node.create_publisher(JointCommand, '/robot/limb/right/joint_command', 10)
    cmd_msg = JointCommand()
    cmd_msg.names = robot.jointNames
    cmd_msg.mode = cmd_msg.VELOCITY_MODE
    gripper_msg = JointCommand()
    gripper_msg.mode = cmd_msg.POSITION_MODE
    gripper_msg.names = ['right_gripper_l_finger_joint', 'right_gripper_r_finger_joint']
    gripper_msg.position = [0.0, 0.0]
    last_trigger = 0
    while rclpy.ok():
        if ros_interface.joy_msg and ros_interface.joint_states_msg:
            print(ros_interface.joy_msg)
            m = ros_interface.joy_msg
            xd = np.array([forward(m), right(m), up(m), roll(m), pitch(m), yaw(m)])
            J = robot.getJacobian('camera_link')
            qd = np.linalg.inv(J.T @ J + alpha * I) @ J.T @ xd
            cmd_msg.velocity = list(qd)

            cmd_pub.publish(cmd_msg)
            if trigger(ros_interface.joy_msg) == 1 and last_trigger == 0:
                if gripper_msg.position[0] == 0.0:
                    gripper_msg.position = [.03, -.03]
                else:
                    gripper_msg.position = [0.0, 0.0]
                cmd_pub.publish(gripper_msg)
                last_trigger = 1
            elif trigger(ros_interface.joy_msg) == 0:
                last_trigger = 0
            time.sleep(.1)


run()
