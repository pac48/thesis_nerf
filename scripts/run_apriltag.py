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

import sys

sys.path.append('../unimatch_repo')
import torch
import torch.nn.functional as F
from unimatch.unimatch import UniMatch
from dataloader.stereo import transforms
from utils.visualization import viz_depth_tensor
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model():
    model_stereo = UniMatch(feature_channels=128,
                            num_scales=2,
                            upsample_factor=4,
                            num_head=1,
                            ffn_dim_expansion=4,
                            num_transformer_layers=6,
                            reg_refine=True,
                            task='stereo').to(device)

    trained_depth_path = '../unimatch_repo/pretrained/gmstereo-scale2-regrefine3-resumeflowthings-middleburyfthighres-a82bec03.pth'
    loc = 'cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(trained_depth_path, map_location=loc)
    model_stereo.load_state_dict(checkpoint['model'], strict=False)
    model_stereo.eval()
    return model_stereo


def get_disparity(model_stereo, left, right):
    padding_factor = 32
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    sample = {'left': left,
              'right': right
              }
    val_transform_list = [transforms.ToTensor(), transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]
    val_transform = transforms.Compose(val_transform_list)
    sample = val_transform(sample)
    left = sample['left'].to(device).unsqueeze(0)  # [1, 3, H, W]
    right = sample['right'].to(device).unsqueeze(0)  # [1, 3, H, W]

    nearest_size = [int(np.ceil(left.size(-2) / padding_factor)) * padding_factor,
                    int(np.ceil(left.size(-1) / padding_factor)) * padding_factor]

    # fixed_inference_size = [val // 2 for val in nearest_size]
    #     print(fixed_inference_size)
    # fixed_inference_size = [544, 960]
    fixed_inference_size = [480, 640]
    #     fixed_inference_size = None

    # resize to nearest size or specified size
    inference_size = nearest_size if fixed_inference_size is None else fixed_inference_size

    ori_size = left.shape[-2:]

    if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
        left = F.interpolate(left, size=inference_size,
                             mode='bilinear',
                             align_corners=True)
        right = F.interpolate(right, size=inference_size,
                              mode='bilinear',
                              align_corners=True)

    with torch.no_grad():
        pred_disp = model_stereo(left, right,
                                 attn_type='self_swin2d_cross_swin1d',
                                 attn_splits_list=[2, 8],
                                 prop_radius_list=[-1, 1],
                                 corr_radius_list=[-1, 4],
                                 num_reg_refine=3,
                                 task='stereo',
                                 )['flow_preds'][-1]  # [1, H, W]

    if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
        pred_disp = F.interpolate(pred_disp.unsqueeze(1), size=ori_size, mode='nearest').squeeze(1)  # [1, H, W]

    disp = pred_disp[0].cpu().numpy()
    return disp


def get_points(depth, fx, fy, cx, cy, width, height):
    Z = depth
    # x_scale = (width - cx + 40) / width  # TODO why are these needed??
    # y_scale = (height - cy + 32) / height

    # x_scale = (width - cx) / width
    # y_scale = (height - cy - 10) / height

    x_scale = (width - cx + 10) / width
    y_scale = (height - cy - 20) / height

    [Xgrid, Ygrid] = np.meshgrid(np.linspace(-(1.0 - x_scale) * width, x_scale * width, width),
                                 np.linspace(-(1.0 - y_scale) * height, y_scale * height, height))
    X = (Xgrid) * Z / fx  # / (fx / width)
    Y = (Ygrid) * Z / fy  # / (fy / height)
    points = np.stack([X, Y, Z], axis=2)  # + 0.025/1000
    points = np.reshape(points, (-1, 3))

    return points


def send_zed_point_cloud(point_cloud, res):
    if zed.grab() == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, res)
    else:
        return

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


def send_model_point_cloud(model_stereo, zed_param):
    image_l = sl.Mat()
    image_r = sl.Mat()
    if zed.grab() == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(image_l, sl.VIEW.LEFT)  # Retrieve left image
        zed.retrieve_image(image_r, sl.VIEW.RIGHT)  # Retrieve left image
    else:
        return

    left = image_l.get_data()
    right = image_r.get_data()

    baseline = .120
    h_fov = zed_param.calibration_parameters.left_cam.h_fov
    # v_fov = zed_param.calibration_parameters.left_cam.v_fov
    fov = h_fov * (math.pi / 180.0)  # 2.28308824

    fx = zed_param.calibration_parameters.left_cam.fx
    fy = zed_param.calibration_parameters.left_cam.fy
    cx = zed_param.calibration_parameters.left_cam.cx
    cy = zed_param.calibration_parameters.left_cam.cy

    width = left.shape[1]
    height = left.shape[0]
    focal = 2 * np.tan(fov / 2.0)

    right = right[:, :, :3].astype(np.float32)
    left = left[:, :, :3].astype(np.float32)
    disp = get_disparity(model_stereo, left, right)

    focal = 1.0 * 1.05
    depth = baseline * focal / disp
    depth = depth / 2.3  # correct for scaling
    stereo_points = 1000 * get_points(depth, fx, fy, cx, cy, width, height)

    tmp = left / 255
    tmp[tmp > 1.0] = 1.0

    stereo_colors = np.reshape(tmp, (-1, 3))
    stereo_colors = stereo_colors[:, [2, 1, 0]]
    stereo_colors = np.hstack([stereo_colors, np.ones((stereo_colors.shape[0], 1))])
    points = np.hstack((stereo_points, stereo_colors))
    send_point_cloud(points[::2, :], has_alpha=False, topic='point_cloud',
                     base_frame='zed2i_right_camera_optical_frame', wait_time=.1)


if __name__ == "__main__":
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

    zed_param = zed.get_camera_information()

    # res = sl.Resolution()
    # res.width = 720
    # res.height = 404
    # point_cloud = sl.Mat(res.width, res.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)
    # while True:
    #     send_zed_point_cloud(point_cloud, res)

    stereo_model = load_model()
    while True:
        send_model_point_cloud(stereo_model, zed_param)
