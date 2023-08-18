import os
import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
# import cv2
from PIL import Image as PILImage
import numpy as np
import json
from threading import Thread
from matplotlib import pyplot as plt
from tf2_ros.transform_listener import TransformListener
from tf2_ros.buffer import Buffer
from tf2_ros import TransformException


def get_images():
    node = Node('nerf_node')
    images_map = dict()
    tf_map = dict()
    info_msg = []

    def creat_callback(topic_name):
        def callback(msg):
            arr = np.asarray(msg.data, dtype=np.uint8)
            arr = np.reshape(arr, (msg.height, msg.width, 4))
            images_map[topic_name] = arr[:, :, :3]

        return callback

    def info_callback(msg):
        if len(info_msg) == 0:
            info_msg.append(msg)

    tf_buffer = Buffer()
    tf_listener = TransformListener(tf_buffer, node)
    # num_cams = 12
    num_cams = 12
    for i in range(1, num_cams + 1):
        if i == 1:
            node.create_subscription(CameraInfo, '/camera/color/camera_info_1', info_callback, 10)
        topic = '/camera/color/image_raw_' + str(i)
        node.create_subscription(Image, topic, creat_callback(topic), 10)
    while len(images_map) < num_cams or len(tf_map) < num_cams or len(info_msg) == 0:
        rclpy.spin_once(node)
        for i in range(1, num_cams + 1):
            try:
                name = f'camera_color_optical_frame_{i}'
                tf_map[name] = tf_buffer.lookup_transform('unity', name,
                                                          rclpy.time.Time())
                # tf_map[name] = tf_buffer.lookup_transform(name, 'unity',
                #                                           rclpy.time.Time())
            except TransformException as ex:
                pass

    images_map = dict(sorted(images_map.items()))
    tf_map = dict(sorted(tf_map.items()))

    imgs = [images_map[key] for key in images_map]
    transforms = [tf_map[key] for key in tf_map]
    # for img in imgs:
    #     plt.figure()
    #     plt.imshow(img, interpolation='nearest')
    # plt.show()

    return imgs, transforms, info_msg[0]


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


def main():
    rclpy.init()
    imgs, transforms, info = get_images()
    frames = []
    data_folder = 'nerf'
    pos_mean = np.zeros((3, 1))
    # for transform in transforms:
    #     pos = transform.transform.translation
    #     pos_mean += np.array([pos.x, pos.y, pos.z]).reshape(-1, 1)
    # pos_mean = pos_mean / len(transforms)

    for ind, img in enumerate(imgs):
        transform = transforms[ind]
        rot = quaternion_rotation_matrix(transform.transform.rotation)
        pos = transform.transform.translation
        pos = np.array([pos.x, pos.y, pos.z]).reshape(-1, 1) - pos_mean
        # pos = 2*pos
        T = np.eye(4)
        T[:3, :4] = np.hstack((rot, pos))

        c2w = T
        c2w[0:3, 2] *= -1  # flip the y and z axis
        c2w[0:3, 1] *= -1
        c2w = c2w[[1, 0, 2, 3], :]
        c2w[2, :] *= -1  # flip whole world upside down
        T = c2w

        frame = {
            "file_path": f"images/{ind}.png",
            "sharpness": 50,
            "transform_matrix": T.tolist()
        }
        image_pil = PILImage.fromarray(img)
        image_pil.save(f'{data_folder}/images/{ind}.png')

        frames.append(frame)

    w = info.width
    cx = w / 2
    h = info.height
    cy = h / 2
    fl_x = info.k[0] * 1.0
    fl_y = info.k[4] * 1.0
    camera_angle_x = math.atan(w / (fl_x * 2)) * 2
    camera_angle_y = math.atan(h / (fl_y * 2)) * 2
    config = {"camera_angle_x": camera_angle_x,
              "camera_angle_y": camera_angle_y,
              "fl_x": fl_x,
              "fl_y": fl_y,
              "k1": 0,  # camera distortion?
              "k2": 0,
              "p1": 0,
              "p2": 0,
              "cx": cx,
              "cy": cy,
              "w": w,
              "h": h,
              "aabb_scale": 1,
              "frames": frames}
    config = json.dumps(config)
    with open(os.path.join(data_folder, 'transforms.json'), 'w') as f:
        f.write(str(config))


if __name__ == "__main__":
    main()
