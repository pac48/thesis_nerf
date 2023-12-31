#!/usr/bin/env python3
import time
from pyngp.common import *
from tqdm import tqdm
import pyngp.pyngp as ngp  # noqa
from scripts.point_cloud_utils import send_point_cloud


import rclpy


print(dir(ngp.Testbed))





def ours_real_converted(path, frameidx=0):
    return {
        "data_dir": os.path.join(NERF_DATA_FOLDER, path),
        "dataset_train": "transforms.json",
        "dataset_test": "transforms.json",
        "dataset": "",
        "test_every": 5,
        "frameidx": frameidx
    }


if __name__ == "__main__":
    rclpy.init()

    testbed = ngp.Testbed()
    testbed.root_dir = 'instant-ngp'
    testbed.reload_network_from_file('base_14.json')
    # testbed.reload_network_from_file('base.json')

    scene = '/home/paul/CLionProjects/thesis_nerf/nerf'
    # scene = '/home/paul/CLionProjects/thesis_nerf/box'
    scene_info = ours_real_converted(scene)
    scene = os.path.join(scene_info["data_dir"], scene_info["dataset"])
    testbed.load_training_data(scene)
    testbed.init_window(1920, 1080)
    testbed.shall_train = True
    testbed.nerf.render_with_lens_distortion = True
    testbed.nerf.training.near_distance = .21

    n_steps = 100*3500 / 1
    old_training_step = 0

    tqdm_last_update = 0
    if n_steps > 0:
        with tqdm(desc="Training", total=n_steps, unit="steps") as t:
            while testbed.frame():
                if testbed.want_repl():
                    repl(testbed)
                # What will happen when training is done?
                if testbed.training_step >= n_steps:
                    testbed.shall_train = False
                    break

                # Update progress bar
                if testbed.training_step < old_training_step or old_training_step == 0:
                    old_training_step = 0
                    t.reset()
                now = time.monotonic()
                if now - tqdm_last_update > 0.1:
                    t.update(testbed.training_step - old_training_step)
                    t.set_postfix(loss=testbed.loss)
                    old_training_step = testbed.training_step
                    tqdm_last_update = now

        box_diff = testbed.aabb.max - testbed.aabb.min
        min_val = -box_diff / 2
        max_val = box_diff / 2
        # x_segments = np.linspace(min_val[0], max_val[0], 64)
        # y_segments = np.linspace(min_val[1], max_val[1], 64)
        # z_segments = np.linspace(min_val[2], max_val[2], 64)
        x_segments = np.linspace(.25, .75, 3*int(256*(.75-.25)))
        y_segments = np.linspace(.45, .6, 3*int(256*(.6-.45)))
        z_segments = np.linspace(.4, .6, 3*int(256*(.6-.4)))
        x, y, z = np.meshgrid(x_segments, y_segments, z_segments, indexing='ij')
        points = np.stack((x.flatten(), y.flatten(), z.flatten()), axis=1)
        points = points[0:256 * (points.shape[0] // 256), :]
        # out_all = []
        # for i in range(0, points_all.shape[0], 256):
        #     points = points_all[i:i + 256, :]
        dirs = np.zeros(points.shape)
        dirs[:, 0] = 1
        dirs[:, 1] = .5
        dirs[:, 2] = .5
        # dirs[:, 1] = 1
        # dirs[:, 2] = 1
        dt = np.zeros((points.shape[0], 1))
        coords = np.hstack((points, dt, dirs))
        # coords = coords.T
        # coords = np.reshape(coords, (-1, coords.shape[0])).T

        out = testbed.sample_nerf(list(coords.flatten()))
        out = np.reshape(out, (coords.shape[0], -1))
        # out_all.append(out)
        # out = np.reshape(out, (-1, points.shape[0])).T
        # points = np.reshape(points, (-1, points.shape[0])).T

        # out = np.vstack(out_all)
        # points = points_all
        points = np.vstack((points[:, 0], points[:, 2], 1 - points[:, 1])).T
        points = points * (max_val - min_val) + min_val
        points = 2 * points
        send_point_cloud(np.hstack((points, out)), has_alpha=False)

    # if False:
    #     os.makedirs(os.path.dirname(args.save_snapshot), exist_ok=True)
    #     testbed.save_snapshot(args.save_snapshot, False)
    #
    # if False:
    #     res = args.marching_cubes_res or 256
    #     thresh = args.marching_cubes_density_thresh or 2.5
    #     print(
    #         f"Generating mesh via marching cubes and saving to {args.save_mesh}. Resolution=[{res},{res},{res}], Density Threshold={thresh}")
    #     testbed.compute_and_save_marching_cubes_mesh(args.save_mesh, [res, res, res], thresh=thresh)
