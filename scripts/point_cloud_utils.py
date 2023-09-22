import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import PointField
import struct
import numpy as np


def send_point_cloud(rgba_points, has_alpha=True, topic='point_cloud', wait_time=1):
    if not rclpy.ok():
        rclpy.init()
    node = Node('nerf_node')
    pub = node.create_publisher(PointCloud2, topic, 10)

    msg = PointCloud2()
    x_field = PointField()
    y_field = PointField()
    z_field = PointField()

    x_field.name = "x"
    x_field.count = 1
    x_field.datatype = PointField.FLOAT32
    x_field.offset = 0

    y_field.name = "y"
    y_field.count = 1
    y_field.datatype = PointField.FLOAT32
    y_field.offset = 4

    z_field.name = "z"
    z_field.count = 1
    z_field.datatype = PointField.FLOAT32
    z_field.offset = 8

    color_field = PointField()
    if has_alpha:
        color_field.name = "rgba"
    else:
        color_field.name = "rgb"
    color_field.count = 1
    color_field.datatype = PointField.UINT32
    color_field.offset = 12

    msg.height = 1
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.header.frame_id = "unity"

    msg.fields = [x_field, y_field, z_field, color_field]
    msg.point_step = 16

    format_string = ''
    data_list = []
    for i in range(0, rgba_points.shape[0]):
        # if has_alpha:
        #     alpha = int(255 * rgba_points[i, 6])
        #     alpha = 255 * (alpha > 100)
        # else:
        #     alpha = 255
        alpha = int(255 * rgba_points[i, 6])
        entry = [float(rgba_points[i, 0]), float(rgba_points[i, 1]), float(rgba_points[i, 2]),
                 int(255 * rgba_points[i, 5]),
                 int(255 * rgba_points[i, 4]), int(255 * rgba_points[i, 3]), int(alpha)]

        if alpha > 100:
            format_string += 'f' * 3 + 'B' * 4
            data_list.extend(entry)

    msg.width = len(data_list) // 7
    msg.row_step = msg.width * msg.point_step
    tmp = struct.pack(format_string, *data_list)
    msg.data = tmp
    for i in range(3):
        time.sleep(wait_time / 3)
        pub.publish(msg)


def rotation_matrix_quaternion(R):
    # q0 = qw
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


def quaternion_rotation_matrix(quat):
    q0 = quat.w
    q1 = quat.x
    q2 = quat.y
    q3 = quat.z
    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])

    return rot_matrix


def save_obj(points):
    with open("/tmp/tmp.obj", 'w') as f:
        f.write('o tmp\n')
        f.write('\n'.join([f'v {points[i, 0]} {points[i, 1]} {points[i, 2]}' for i in range(points.shape[0])]))
