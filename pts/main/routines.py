import numpy as np

from pts.models import MaskRG, Trainer
from pts.sim.train_scene import TrainScene
from pts.utils.image_helper import get_heightmap


def train_dqn(cfg_dqn, cfg_rg, cfg_env):
    mask_rg_net = MaskRG(cfg_rg)  # network is in testing mode
    success_threshold = cfg_dqn.success_threshold
    ws_limits = cfg_env.workspace_limit
    heightmap_res = cfg_env.heightmap_resolution

    train_scene = TrainScene(cfg_env)
    trainer = Trainer(cfg_dqn)
    train_scene.create_testing_scenario()

    def best_push_prediction(push_prediction, exploration=False):
        if not exploration and push_prediction is not None:
            return np.unravel_index(np.argmax(push_prediction), push_prediction.shape)
        else:
            return (
                np.random.random_integers(0, 15),
                np.random.random_integers(0, 224 - 1),
                np.random.random_integers(0, 224 - 1),
            )

    points, colors = train_scene.get_point_cloud()
    rgb, depth = train_scene.get_camera_data()
    seg = train_scene.get_data_mask_rg()
    seg[seg == 1] = 0

    # ### Preprocessing ###
    color_heightmap, depth_heightmap = get_heightmap(
        points,
        colors,
        train_scene.get_cam_pose(),
        ws_limits,
        heightmap_res,
    )
    valid_depth_heightmap = depth_heightmap.copy()
    valid_depth_heightmap[np.isnan(valid_depth_heightmap)] = 0

    prev_seg_reward = 0
    push_pred = trainer.forward(color_heightmap, valid_depth_heightmap)

    # first push is uninformed (will not be counted as iteration)
    # train_scene.push_at(0.5, 0.0)

    for i in range(cfg_dqn.train.maximum_iterations):
        prev_valid_depth_heightmap = valid_depth_heightmap.copy()
        prev_pix_ind = best_push_prediction(push_pred, exploration=True)

        # ### Execute Push ###
        mask_rg_net.set_reward_generator(depth, seg)
        pred_ids, seg_rew, err = mask_rg_net.get_current_rewards()
        color_heightmap, depth_heightmap = get_heightmap(
            rgb,
            depth,
            train_scene.cam_intrinsics,
            train_scene.cam_pose,
            ws_limits,
            heightmap_res,
        )

        valid_depth_heightmap = depth_heightmap.copy()
        valid_depth_heightmap[np.isnan(valid_depth_heightmap)] = 0
        push_pred = trainer.forward(
            color_heightmap, valid_depth_heightmap, is_volatile=True
        )

        pix_ind = best_push_prediction(push_pred, exploration=True)

        x_push, y_push = (
            pix_ind[2] * heightmap_res + ws_limits[0][0],
            pix_ind[1] * heightmap_res + ws_limits[1][0],
        )
        train_scene.push_at(x_push, y_push)

        rgb, depth = train_scene.get_camera_data()
        seg = train_scene.get_data_mask_rg()

        # ### Preprocessing ###

        seg[seg == 1] = 0  # two stands for table
        mask_rg_net.set_reward_generator(depth, seg)

        # ### Compute change for evaluation ###
        diff_depth = abs(valid_depth_heightmap - prev_valid_depth_heightmap)
        diff_depth[diff_depth > 0] = 1
        change_detected = np.sum(diff_depth) > cfg_dqn.change_threshold
        success_action = seg_rew > success_threshold

        if success_action:
            print("Success after:", i, " iterations.")
            train_scene.create_testing_scenario()  # resets simulation

        # ### Update network ###

        label_val, prev_reward_val, seg_val = trainer.get_reward_value(
            color_heightmap,
            valid_depth_heightmap,
            prev_seg_reward,
            seg_rew,
            change_detected,
        )
        trainer.backprop(prev_pix_ind, label_val)


def test_dqn(cfg_dqn, cfg_env):
    ws_limits = cfg_env.workspace_limit
    heightmap_res = cfg_env.heightmap_resolution
    train_scene = TrainScene(cfg_env)
    train_scene.create_testing_scenario()
    trainer = Trainer(cfg_dqn)

    # ### Compute prediction ###
    points, colors = train_scene.get_point_cloud()
    color_heightmap, depth_heightmap = get_heightmap(
        points,
        colors,
        train_scene.get_cam_pose(),
        ws_limits,
        heightmap_res,
    )
    valid_depth_heightmap = depth_heightmap.copy()
    valid_depth_heightmap[np.isnan(valid_depth_heightmap)] = 0
    push_pred = trainer.forward(color_heightmap, valid_depth_heightmap)

    # ### Execute push ###
    max_pred = np.unravel_index(np.argmax(push_pred), push_pred.shape)
    x_push, y_push = (
        max_pred[2] * heightmap_res + ws_limits[0][0],
        max_pred[1] * heightmap_res + ws_limits[1][0],
    )
    train_scene.push_at(x_push, y_push)

    # ### Evaluate result ###

    # mask_rg_net = MaskRG(cfg_rg)  # network is in testing mode
    # success_threshold = cfg_dqn.success_threshold
