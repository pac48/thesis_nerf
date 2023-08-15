import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
# import cv2
from PIL import Image as PILImage
import numpy as np
from threading import Thread
from matplotlib import pyplot as plt
from tf2_ros.transform_listener import TransformListener
from tf2_ros.buffer import Buffer
from tf2_ros import TransformException


def get_images():
    node = Node('nerf_node')
    images_map = dict()
    tf_map = dict()

    def creat_callback(topic_name):
        def callback(msg):
            arr = np.asarray(msg.data, dtype=np.uint8)
            arr = np.reshape(arr, (msg.height, msg.width, 4))
            images_map[topic_name] = arr

        return callback

    tf_buffer = Buffer()
    tf_listener = TransformListener(tf_buffer, node)
    for i in range(1, 13):
        topic = '/camera/color/image_raw_' + str(i)
        node.create_subscription(Image, topic, creat_callback(topic), 10)
    while len(images_map) < 12 and len(tf_map) < 12:
        rclpy.spin_once(node)
        for i in range(1, 13):
            try:
                name = f'camera_color_optical_frame_{i}'
                tf_map[name] = tf_buffer.lookup_transform(name, 'unity',
                                                          rclpy.time.Time())
            except TransformException as ex:
                pass

    imgs = [images_map[key] for key in images_map]
    for img in imgs:
        fig = plt.figure()  # Create a new figure
        plt.imshow(img, interpolation='nearest')
    plt.show()

    return imgs


if __name__ == "__main__":
    rclpy.init()

    imgs = get_images()
