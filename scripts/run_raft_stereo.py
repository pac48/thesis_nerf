import sys
#
# sys.path.append('core')

import argparse
import glob
import numpy as np
import rclpy
import torch
from tqdm import tqdm
from pathlib import Path
from raft_stereo import RAFTStereo
from utils.utils import InputPadder
from PIL import Image
from matplotlib import pyplot as plt

import plotly.graph_objects as go
from pathlib import Path
from imageio.v3 import imread

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image as ImageMsg
# import cv2
import numpy as np
from tf2_ros.transform_listener import TransformListener
from tf2_ros.buffer import Buffer
from tf2_ros import TransformException
from scripts.point_cloud_utils import send_point_cloud, save_obj
from scripts.point_cloud_utils import quaternion_rotation_matrix


def get_stereo_images():
    node = Node('stereo_node')
    images_map = dict()
    tf_map = dict()
    info_map = dict()

    def image_callback(topic_name):
        def callback(msg):
            arr = np.asarray(msg.data, dtype=np.uint8)
            arr = np.reshape(arr, (msg.height, msg.width, 4))
            images_map[topic_name] = arr[:, :, :3]

        return callback

    def info_callback(topic_name):
        def callback(msg):
            info_map[topic_name] = msg

        return callback

    tf_buffer = Buffer()
    tf_listener = TransformListener(tf_buffer, node)
    num_cams = 3
    for i in range(1, num_cams + 1):
        node.create_subscription(CameraInfo, f'/unity_camera/rgb/camera_info_l_{i}',
                                 info_callback(f'/unity_camera/rgb/camera_info_l_{i}'), 10)
        node.create_subscription(CameraInfo, f'/unity_camera/rgb/camera_info_r_{i}',
                                 info_callback(f'/unity_camera/rgb/camera_info_r_{i}'), 10)
        node.create_subscription(ImageMsg, f'/unity_camera/color/image_raw_l_{i}',
                                 image_callback(f'/unity_camera/color/image_raw_l_{i}'), 10)
        node.create_subscription(ImageMsg, f'/unity_camera/color/image_raw_r_{i}',
                                 image_callback(f'/unity_camera/color/image_raw_r_{i}'), 10)
    while len(images_map) < 2 * num_cams or len(tf_map) < 2 * num_cams or len(info_map) < 2 * num_cams:
        rclpy.spin_once(node)
        for i in range(1, num_cams + 1):
            try:
                name_l = f'camera_color_optical_frame_l_{i}'
                tf_map[name_l] = tf_buffer.lookup_transform('unity', name_l,
                                                            rclpy.time.Time())
                name_r = f'camera_color_optical_frame_r_{i}'
                tf_map[name_r] = tf_buffer.lookup_transform('unity', name_r,
                                                            rclpy.time.Time())
            except TransformException as ex:
                pass

    images_map = dict(sorted(images_map.items()))
    tf_map = dict(sorted(tf_map.items()))

    imgs = [images_map[key] for key in images_map]
    transforms = [tf_map[key] for key in tf_map]

    return images_map, tf_map, info_map


DEVICE = 'cuda'


def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def image_to_tensor(img):
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def demo(args):
    rclpy.init()
    imgs, transforms, info = get_stereo_images()

    model = torch.nn.DataParallel(RAFTStereo(args), device_ids=[0])
    model.load_state_dict(torch.load(args.restore_ckpt))

    model = model.module
    model.to(DEVICE)
    model.eval()

    output_directory = Path(args.output_directory)
    output_directory.mkdir(exist_ok=True)
    # fig = go.Figure(layout=dict(
    #     scene=dict(
    #         xaxis=dict(visible=True),
    #         yaxis=dict(visible=True),
    #         zaxis=dict(visible=True),
    #     )
    # ))
    all_points = []
    all_colors = []
    with torch.no_grad():
        for cam in range(1, 4):
            image1 = image_to_tensor(imgs[f'/unity_camera/color/image_raw_l_{cam}'])
            image2 = image_to_tensor(imgs[f'/unity_camera/color/image_raw_r_{cam}'])

            padder = InputPadder(image1.shape, divis_by=32)
            image1, image2 = padder.pad(image1, image2)

            _, flow_up = model(image1, image2, iters=args.valid_iters, test_mode=True)
            flow_up = padder.unpad(flow_up).squeeze()

            # file_stem = imfile1.split('/')[-2]
            # if args.save_numpy:
            #     np.save(output_directory / f"{file_stem}.npy", flow_up.cpu().numpy().squeeze())
            # plt.imsave(output_directory / f"{file_stem}.png", -flow_up.cpu().numpy().squeeze(), cmap='jet')

            # inverse-project
            msg1 = info[f'/unity_camera/rgb/camera_info_l_{cam}']
            msg2 = info[f'/unity_camera/rgb/camera_info_r_{cam}']
            # assert (msg2 == msg)
            w1 = msg1.width
            w2 = msg2.width
            cx1 = w1 / 2
            cx2 = w2 / 2
            h1 = msg1.height
            h2 = msg2.height
            assert h2 == h1
            cy1 = h1 / 2
            baseline = 100
            fx = msg1.k[0]
            fy = msg1.k[4]

            flow_up = flow_up.cpu().numpy().squeeze()
            depth = (fx * baseline) / (-flow_up + (cx2 - cx1))
            H, W = depth.shape
            xx, yy = np.meshgrid(np.arange(W), np.arange(H))
            points_grid = np.stack(((xx - cx1) / fx, (yy - cy1) / fy, np.ones_like(xx)), axis=0) * depth

            mask = np.ones((H, W), dtype=bool)

            # Remove flying points
            mask[1:][np.abs(depth[1:] - depth[:-1]) > 1] = False
            mask[:, 1:][np.abs(depth[:, 1:] - depth[:, :-1]) > 1] = False

            points = points_grid.transpose(1, 2, 0)[mask]
            image = imgs[f'/unity_camera/color/image_raw_l_{cam}']
            colors = image[mask].astype(np.float64) / 255

            NUM_POINTS_TO_DRAW = 50000

            subset = np.random.choice(points.shape[0], size=(NUM_POINTS_TO_DRAW,), replace=False)
            points_subset = points[subset]
            colors_subset = colors[subset]

            transform_l = transforms[f'camera_color_optical_frame_l_{cam}']
            transform_r = transforms[f'camera_color_optical_frame_r_{cam}']
            rot = quaternion_rotation_matrix(transform_l.transform.rotation).transpose()
            get_pos = lambda T: np.array(
                [T.transform.translation.x, T.transform.translation.y, T.transform.translation.z])
            # pos_l = get_pos(transform_l)
            # pos_r = get_pos(transform_r)
            # pos = (pos_l + pos_r) / 2
            pos = get_pos(transform_l)
            points_subset = points_subset / 1000.0
            points_subset = (points_subset @ rot) + pos

            all_points.append(points_subset)
            all_colors.append(colors_subset)

    #         print("""
    #         Controls:
    #         ---------
    #         Zoom:      Scroll Wheel
    #         Translate: Right-Click + Drag
    #         Rotate:    Left-Click + Drag
    #         """)
    #
    #         x, y, z = points_subset.T
    #
    #         trace = go.Scatter3d(
    #             x=x, y=-z, z=-y,  # flipped to make visualization nicer
    #             mode='markers',
    #             marker=dict(size=1, color=colors_subset)
    #         )
    #         fig.add_trace(trace)
    #
    # fig.show()

    points = np.vstack(all_points)
    colors = np.vstack(all_colors)
    rgb_points = np.hstack((points, colors))
    send_point_cloud(rgb_points, has_alpha=False)
    save_obj(points)
    from pykdtree.kdtree import KDTree
    KDTree


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", required=True)
    parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')
    parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames",
                        default="datasets/Middlebury/MiddEval3/testH/*/im0.png")
    parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames",
                        default="datasets/Middlebury/MiddEval3/testH/*/im1.png")
    parser.add_argument('--output_directory', help="directory to save output", default="demo_output")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=32,
                        help='number of flow-field updates during forward pass')

    # Architecture choices
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128] * 3,
                        help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg",
                        help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true',
                        help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--context_norm', type=str, default="batch", choices=['group', 'batch', 'instance', 'none'],
                        help="normalization of context encoder")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")

    args = parser.parse_args()

    demo(args)
