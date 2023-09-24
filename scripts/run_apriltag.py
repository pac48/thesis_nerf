from dt_apriltags import Detector
from rclpy.node import Node
from scripts.point_cloud_utils import send_point_cloud
from scripts.ros_utils import send_tf
from sensor_msgs.msg import Image, CameraInfo
import cv2
import numpy as np
import pyzed.sl as sl
import rclpy
import time


def get_mono_image(image_topic):
    if not rclpy.ok():
        rclpy.init()
    node = Node('nerf_node')
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


init = sl.InitParameters(depth_mode=sl.DEPTH_MODE.ULTRA,
                         coordinate_units=sl.UNIT.METER,
                         coordinate_system=sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP)

init.camera_resolution = sl.RESOLUTION.HD2K
zed = sl.Camera()
zed.close()
status = zed.open(init)
if status != sl.ERROR_CODE.SUCCESS:
    print(repr(status))
    exit()

zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, 60)
zed.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, 0)
zed.set_camera_settings(sl.VIDEO_SETTINGS.BRIGHTNESS, 5)
zed.set_camera_settings(sl.VIDEO_SETTINGS.CONTRAST, 2)

res = sl.Resolution()
res.width = 720
res.height = 404
point_cloud = sl.Mat(res.width, res.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)

while True:
    image = get_mono_image('/io/internal_camera/right_hand_camera/image_raw')

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

    image = undistorted_image = cv2.undistort(image, camera_matrix, dist_coeffs)
    at_detector = Detector("tagStandard41h12")

    camera_params = [fx, fy, cx, cy]
    tag_size = 0.015
    detections = at_detector.detect(image, estimate_tag_pose=True, camera_params=camera_params, tag_size=tag_size)
    if len(detections) == 0:
        continue
    detection = detections[0]

    T = np.eye(4, 4)
    T[:3, :3] = detection.pose_R.transpose()
    T[:3, 3] = np.reshape(detection.pose_t, (-1,))

    send_tf('zed2i_april_tag', T, base_frame='right_hand_camera')

    if zed.grab() == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, res)
        data = point_cloud.get_data()
        nan_mask = np.isnan(data).any(axis=2).flatten()
        x = -data[:, :, 0].flatten()[~nan_mask]
        y = -data[:, :, 1].flatten()[~nan_mask]
        z = data[:, :, 2].flatten()[~nan_mask]
        points = np.vstack([x, y, z]).transpose()
        colors = data[:, :, 3].flatten()[~nan_mask]  # .astype(np.uint8)
        colors.dtype = np.uint8
        colors = np.reshape(colors, (-1, 4))
        colors = colors.astype(np.float32)
        colors = colors / 255
        send_point_cloud(np.hstack((points, colors)), has_alpha=False, topic='point_cloud',
                         base_frame='zed2i_right_camera_optical_frame')

        pass
