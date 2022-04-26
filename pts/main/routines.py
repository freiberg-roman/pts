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

    def predict_push_position(push_prediction, exploration=False, size=None):
        if not exploration and push_prediction is not None:
            pix_ind = np.unravel_index(
                np.argmax(push_prediction), push_prediction.shape
            )
        else:
            pix_ind = (
                np.random.random_integers(0, size[0]),
                np.random.random_integers(0, size[1]),
                np.random.random_integers(0, size[2]),
            )
        x_push, y_push = (
            pix_ind[2] * heightmap_res + ws_limits[0][0],
            pix_ind[1] * heightmap_res + ws_limits[1][0],
        )

        return x_push, y_push

    # ### Get initial scene data ###
    points, colors = train_scene.get_point_cloud()
    color_heightmap, depth_heightmap = get_heightmap(
        points,
        colors,
        train_scene.get_cam_pose(),
        ws_limits,
        heightmap_res,
    )
    seg_reward = 0

    for i in range(cfg_dqn.train.maximum_iterations):
        # ### Execute Push ###
        push_pred = trainer.forward(color_heightmap, depth_heightmap)
        x_push, y_push = predict_push_position(
            push_pred, exploration=False, size=push_pred.shape
        )
        train_scene.push_at(x_push, y_push)
        train_scene.beam_back()

        # ### Store old data and get new scene data ###
        prev_depth_heightmap = depth_heightmap
        prev_seg_reward = seg_reward

        points, colors = train_scene.get_point_cloud()
        color_heightmap, depth_heightmap = get_heightmap(
            points,
            colors,
            train_scene.get_cam_pose(),
            ws_limits,
            heightmap_res,
        )
        rgb, depth = train_scene.get_camera_data()
        seg, num_obj = train_scene.get_object_segmentation()

        # ### Compute reward ###
        mask_rg_net.set_reward_generator(depth, seg)
        _, seg_reward, _ = mask_rg_net.get_current_rewards()

        # ### Compute change for evaluation ###
        diff_depth = abs(depth_heightmap - prev_depth_heightmap)
        diff_depth[diff_depth > 0] = 1
        change_detected = np.sum(diff_depth) > cfg_dqn.change_threshold
        success_action = seg_reward > success_threshold

        if success_action:
            print("Success after:", i, " iterations.")
            train_scene.create_testing_scenario()  # resets simulation

        # ### Update network ###

        label_val, _, _ = trainer.get_reward_value(
            color_heightmap,
            depth_heightmap,
            prev_seg_reward,
            seg_reward,
            change_detected,
        )
        trainer.backprop((x_push, y_push), label_val)


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
    push_pred = trainer.forward(color_heightmap, depth_heightmap)

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
