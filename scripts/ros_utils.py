import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
import numpy as np
from tf2_ros.transform_listener import TransformListener
from tf2_ros.buffer import Buffer
from tf2_ros import TransformException
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from scripts.point_cloud_utils import quaternion_rotation_matrix, rotation_matrix_quaternion


def get_image_tf_pair(image_topics, tf_names):
    assert len(image_topics) == len(tf_names)
    if not rclpy.ok():
        rclpy.init()
    node = Node('nerf_node')
    images_map = dict()
    tf_map = dict()

    def creat_callback(topic_name):
        def callback(msg):
            arr = np.asarray(msg.data, dtype=np.uint8)
            arr = np.reshape(arr, (msg.height, msg.width, 4))
            images_map[topic_name] = arr[:, :, :3]

        return callback

    tf_buffer = Buffer()
    tf_listener = TransformListener(tf_buffer, node)
    for topic in image_topics:
        node.create_subscription(Image, topic, creat_callback(topic), 10)
    while len(images_map) < len(image_topics) or len(tf_map) < len(image_topics):
        rclpy.spin_once(node)
        for frame in tf_names:
            try:
                msg = tf_buffer.lookup_transform('unity', frame,
                                                 rclpy.time.Time())
                tf_map[frame] = ros_tf_to_matrix(msg)
            except TransformException as ex:
                pass

    images_map = dict(sorted(images_map.items()))
    tf_map = dict(sorted(tf_map.items()))

    imgs = [images_map[key] for key in images_map]
    transforms = [tf_map[key] for key in tf_map]

    return imgs, transforms


def send_tf(tf_name, transform):
    if not rclpy.ok():
        rclpy.init()
    node = Node('nerf_node')

    tf_broadcaster = TransformBroadcaster(node)
    t = TransformStamped()

    # Read message content and assign it to
    # corresponding tf variables
    t.header.stamp = node.get_clock().now().to_msg()
    t.header.frame_id = 'unity'
    t.child_frame_id = tf_name

    # Turtle only exists in 2D, thus we get x and y translation
    # coordinates from the message and set the z coordinate to 0
    t.transform.translation.x = transform[0, 3]
    t.transform.translation.y = transform[1, 3]
    t.transform.translation.z = transform[2, 3]

    # For the same reason, turtle can only rotate around one axis
    # and this why we set rotation in x and y to 0 and obtain
    # rotation in z axis from the message
    q = rotation_matrix_quaternion(transform[:3, :3].transpose())
    t.transform.rotation.x = q[0]
    t.transform.rotation.y = q[1]
    t.transform.rotation.z = q[2]
    t.transform.rotation.w = q[3]

    # Send the transformation
    tf_broadcaster.sendTransform(t)


def ros_tf_to_matrix(msg):
    rot = quaternion_rotation_matrix(msg.transform.rotation)
    pos = msg.transform.translation
    pos = np.array([pos.x, pos.y, pos.z]).reshape(-1, 1)
    T = np.eye(4)
    T[:3, :4] = np.hstack((rot, pos))
    return T
