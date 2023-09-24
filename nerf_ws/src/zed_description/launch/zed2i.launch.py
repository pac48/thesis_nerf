# Copyright 2022 Stereolabs
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import sys

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node as LaunchNode

from dt_apriltags import Detector
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
import cv2
import numpy as np
import rclpy

from tf2_ros.transform_listener import TransformListener
from tf2_ros.buffer import Buffer
from tf2_ros import TransformException
from tf2_kdl import transform_to_kdl


def get_mono_image(node, image_topic):
    img_list = []

    def creat_callback():
        def callback(msg):
            arr = np.asarray(msg.data, dtype=np.uint8)
            arr = np.reshape(arr, (msg.height, msg.width))
            img_list.append(arr[:, :])

        return callback

    node.create_subscription(Image, image_topic, creat_callback(), 10)
    while len(img_list) == 0:
        rclpy.spin_once(node)

    return img_list[0]


def rotation_matrix_quaternion(R):
    t = np.matrix.trace(R)
    q = np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    if (t > 0):
        t = np.sqrt(t + 1)
        q[3] = 0.5 * t
        t = 0.5 / t
        q[0] = (R[2, 1] - R[1, 2]) * t
        q[1] = (R[0, 2] - R[2, 0]) * t
        q[2] = (R[1, 0] - R[0, 1]) * t

    else:
        i = 0
        if (R[1, 1] > R[0, 0]):
            i = 1
        if (R[2, 2] > R[i, i]):
            i = 2
        j = (i + 1) % 3
        k = (j + 1) % 3

        t = np.sqrt(R[i, i] - R[j, j] - R[k, k] + 1)
        q[i] = 0.5 * t
        t = 0.5 / t
        q[3] = (R[k, j] - R[j, k]) * t
        q[j] = (R[j, i] + R[i, j]) * t
        q[k] = (R[k, i] + R[i, k]) * t

    return q


def get_transform_args():
    if not rclpy.ok():
        rclpy.init()
    node = Node('calibration_node')
    tf_buffer = Buffer()
    tf_listener = TransformListener(tf_buffer, node)

    detections = []
    while len(detections) == 0:
        image = get_mono_image(node, '/io/internal_camera/right_hand_camera/image_raw')
        fx = 623.955994
        fy = 622.718994
        cx = 349.213013
        cy = 214.837997
        camera_matrix = np.eye(3, 3)
        camera_matrix[0, 0] = fx
        camera_matrix[1, 1] = fy
        camera_matrix[0, 2] = cx
        camera_matrix[1, 2] = cy

        dist_coeffs = np.array([-0.439, 0.265, 0.001, 0.0, -0.12])

        image = cv2.undistort(image, camera_matrix, dist_coeffs)
        at_detector = Detector("tagStandard41h12")

        camera_params = [fx, fy, cx, cy]
        tag_size = 0.015
        detections = at_detector.detect(image, estimate_tag_pose=True, camera_params=camera_params, tag_size=tag_size)

    detection = detections[0]
    T1 = np.eye(4, 4)
    T1[:3, :3] = detection.pose_R
    T1[:3, 3] = np.reshape(detection.pose_t, (-1,))

    lookup_succeed = False
    while not lookup_succeed:
        rclpy.spin_once(node)
        try:
            msg = tf_buffer.lookup_transform('world', 'right_hand_camera', rclpy.time.Time())
            lookup_succeed = True
        except TransformException as ex:
            pass

    T2 = np.eye(4, 4)
    kdlFrame = transform_to_kdl(msg)
    for r in range(3):
        for c in range(3):
            T2[r, c] = kdlFrame[r, c]

    T2[0, 3] = kdlFrame[0, 3]
    T2[1, 3] = kdlFrame[1, 3]
    T2[2, 3] = kdlFrame[2, 3]

    T = T2 @ T1

    q = rotation_matrix_quaternion(T[:3, :3])

    args = ['--x', T[0, 3],
            '--y', T[1, 3],
            '--z', T[2, 3],
            '--qx', q[0],
            '--qy', q[1],
            '--qz', q[2],
            '--qw', q[3],
            "--frame-id", "world",
            "--child-frame-id", "zed2i_april_tag"]
    args = [str(v) for v in args]
    return args


def generate_launch_description():
    # Camera model (force value)
    camera_model = 'zed2i'

    # ZED Wrapper node
    zed_wrapper_launch = IncludeLaunchDescription(
        launch_description_source=PythonLaunchDescriptionSource([
            get_package_share_directory('zed_description'),
            '/launch/include/zed_camera.launch.py'
        ]),
        launch_arguments={
            'camera_model': camera_model
        }.items()
    )

    # Define LaunchDescription variable
    ld = LaunchDescription()

    # Add nodes to LaunchDescription
    ld.add_action(zed_wrapper_launch)

    # geometry_msgs.msg.Quaternion(x=-0.039709407710967384, y=-0.2186098863391901, z=0.9681829347305384, w=0.11512899474316864)
    node = LaunchNode(package="tf2_ros",
                      executable="static_transform_publisher",
                      arguments=get_transform_args())

    ld.add_action(node)

    return ld
