import time
import zmq
import base64
import cv2
import numpy as np
import threading

from scripts.run_apriltag import load_model, get_disparity, get_points
from scripts.point_cloud_utils import send_point_cloud
import math

context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.bind("tcp://*:5555")

left = None
right = None


def worker():
    global left
    global right

    while True:
        socket.send(b"Hello")
        message = socket.recv()
        encoded_image = base64.b64decode(message)
        np_img = np.asarray(bytearray(encoded_image), dtype=np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        left = img[:, :img.shape[1] // 2, :]
        right = img[:, img.shape[1] // 2:, :]
        cv2.imshow("stereo", img)
        cv2.waitKey(1)


t = threading.Thread(target=worker, args=())
t.start()


def epipolar_rectify(imL, imR):
    orb = cv2.ORB_create()
    # Detect and compute keypoints and descriptors for both images
    keypoints_left, descriptors_left = orb.detectAndCompute(imL, None)
    keypoints_right, descriptors_right = orb.detectAndCompute(imR, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors
    matches = bf.match(descriptors_left, descriptors_right)

    # Sort matches by distance (smaller distances are better)
    matches = sorted(matches, key=lambda x: x.distance)

    N = 50
    good_matches = matches[:N]

    points_left = [keypoints_left[match.queryIdx].pt for match in good_matches]
    points_right = [keypoints_right[match.trainIdx].pt for match in good_matches]

    points_left = np.int32(points_left)
    points_right = np.int32(points_right)

    F, mask = cv2.findFundamentalMat(points_left, points_right, cv2.FM_RANSAC)

    retval, h1, h2 = cv2.stereoRectifyUncalibrated(points_left, points_right, F, (imR.shape[1], imR.shape[0]))

    # Rectify the left image
    left_rectified = cv2.warpPerspective(imL, h1, (imL.shape[1], imL.shape[0]))

    # Rectify the right image
    right_rectified = cv2.warpPerspective(imR, h2, (imR.shape[1], imR.shape[0]))

    print(F)

    fx = 506.4702
    fy = 506.7368
    cx = 322.3706
    cy = 247.0331
    baseline = 0.0978
    # Define the camera matrices for both left and right cameras
    K_left = np.array([[fx, 0, cx],
                       [0, fy, cy],
                       [0, 0, 1]])

    K_right = np.array([[fx, 0, cx],
                        [0, fy, cy],
                        [0, 0, 1]])

    # Define the distortion coefficients for both cameras
    dist_coeffs_left = np.array([0.0949, -0.0949, 0, 0, 0])
    dist_coeffs_right = np.array([0.0949, -0.0949, 0, 0, 0])

    # Define the relative transformation between the cameras (rotation and translation matrices)
    R = np.array([[1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0]])

    T = np.array([baseline, 0, 0])  # Baseline is the distance between the camera centers

    # Compute the rectification transformations
    image_width = imL.shape[1]
    image_height = imL.shape[0]
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K_left, dist_coeffs_left, K_right, dist_coeffs_right,
                                                (image_width, image_height), R, T)

    # Compute the rectification maps for each camera
    map1_left, map2_left = cv2.initUndistortRectifyMap(K_left, dist_coeffs_left, R1, P1, (image_width, image_height),
                                                       cv2.CV_32F)
    map1_right, map2_right = cv2.initUndistortRectifyMap(K_right, dist_coeffs_right, R2, P2,
                                                         (image_width, image_height), cv2.CV_32F)

    # Apply the rectification maps to the images
    rectified_left = cv2.remap(imL, map1_left, map2_left, interpolation=cv2.INTER_LINEAR)
    rectified_right = cv2.remap(imR, map1_right, map2_right, interpolation=cv2.INTER_LINEAR)

    return rectified_left, rectified_right


def send_model_point_cloud(model_stereo, left, right):
    if right is None or left is None:
        return

    # left, right = epipolar_rectify(left, right)

    baseline = 0.0978
    h_fov = 90
    fov = h_fov * (math.pi / 180.0)  # 2.28308824

    fx = 506.4702
    fy = 506.7368
    cx = 322.3706
    cy = 247.0331

    width = left.shape[1]
    height = left.shape[0]
    focal = 2 * np.tan(fov / 2.0)

    right = right[:, :, :3].astype(np.float32)
    left = left[:, :, :3].astype(np.float32)
    disp = get_disparity(model_stereo, left, right)

    focal = 1.0 * 1.05
    depth = baseline * focal / disp
    depth = depth / 2  # correct for scaling
    stereo_points = 1000 * get_points(depth, fx, fy, cx, cy, width, height)

    tmp = left / 255
    tmp[tmp > 1.0] = 1.0

    stereo_colors = np.reshape(tmp, (-1, 3))
    stereo_colors = stereo_colors[:, [2, 1, 0]]
    stereo_colors = np.hstack([stereo_colors, np.ones((stereo_colors.shape[0], 1))])
    points = np.hstack((stereo_points, stereo_colors))
    send_point_cloud(points[::1, :], has_alpha=False, topic='point_cloud',
                     base_frame='left_camera', wait_time=.1)


stereo_model = load_model()
while True:
    send_model_point_cloud(stereo_model, left, right)
