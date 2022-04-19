import numpy as np
import torch
from scipy import ndimage

from pts.models import MaskRG, ReinforcementNet, Trainer
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

    # ### Preprocessing ###
    rgb, depth = train_scene.get_camera_data()
    seg = train_scene.get_data_mask_rg()
    seg[seg == 1] = 0

    prev_seg_reward = 0
    color_heightmap, depth_heightmap = get_heightmap(
        rgb,
        depth,
        train_scene.cam_intrinsics,
        train_scene.get_cam_pose(),
        ws_limits,
        heightmap_res,
    )
    valid_depth_heightmap = depth_heightmap.copy()
    valid_depth_heightmap[np.isnan(valid_depth_heightmap)] = 0
    push_pred = trainer.forward(
        color_heightmap, valid_depth_heightmap, is_volatile=True
    )

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


def test_dqn(cfg_dqn, cfg_rg, cfg_env):
    # ### Set up ###
    r_net = ReinforcementNet(
        use_cuda=cfg_dqn.train.use_cuda and torch.cuda.is_available()
    )
    # TODO
    # r_net.load_state_dict(
    #     torch.load(
    #         cfg_dqn.path_load_weights,
    #         map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    #     )
    # )
    ws_limits = cfg_env.workspace_limit
    heightmap_res = cfg_env.heightmap_resolution
    train_scene = TrainScene(cfg_env)
    train_scene.create_testing_scenario()

    def forward(color_heightmap, depth_heightmap, specific_rotation=-1):

        # Apply 2x scale to input heightmaps
        color_heightmap_2x = ndimage.zoom(color_heightmap, zoom=[2, 2, 1], order=0)
        depth_heightmap_2x = ndimage.zoom(depth_heightmap, zoom=[2, 2], order=0)
        assert color_heightmap_2x.shape[0:2] == depth_heightmap_2x.shape[0:2]

        # Add extra padding (to handle rotations inside network)
        diag_length = float(color_heightmap_2x.shape[0]) * np.sqrt(2)
        diag_length = np.ceil(diag_length / 32) * 32
        padding_width = int((diag_length - color_heightmap_2x.shape[0]) / 2)
        color_heightmap_2x_r = np.pad(
            color_heightmap_2x[:, :, 0], padding_width, "constant", constant_values=0
        )
        color_heightmap_2x_r.shape = (
            color_heightmap_2x_r.shape[0],
            color_heightmap_2x_r.shape[1],
            1,
        )
        color_heightmap_2x_g = np.pad(
            color_heightmap_2x[:, :, 1], padding_width, "constant", constant_values=0
        )
        color_heightmap_2x_g.shape = (
            color_heightmap_2x_g.shape[0],
            color_heightmap_2x_g.shape[1],
            1,
        )
        color_heightmap_2x_b = np.pad(
            color_heightmap_2x[:, :, 2], padding_width, "constant", constant_values=0
        )
        color_heightmap_2x_b.shape = (
            color_heightmap_2x_b.shape[0],
            color_heightmap_2x_b.shape[1],
            1,
        )
        color_heightmap_2x = np.concatenate(
            (color_heightmap_2x_r, color_heightmap_2x_g, color_heightmap_2x_b), axis=2
        )
        depth_heightmap_2x = np.pad(
            depth_heightmap_2x, padding_width, "constant", constant_values=0
        )

        # Pre-process color image (scale and normalize)
        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]
        input_color_image = color_heightmap_2x.astype(float) / 255
        for c in range(3):
            input_color_image[:, :, c] = (
                input_color_image[:, :, c] - image_mean[c]
            ) / image_std[c]

        # Pre-process depth image (normalize)
        image_mean = [0.01, 0.01, 0.01]
        image_std = [0.03, 0.03, 0.03]
        depth_heightmap_2x.shape = (
            depth_heightmap_2x.shape[0],
            depth_heightmap_2x.shape[1],
            1,
        )
        input_depth_image = np.concatenate(
            (depth_heightmap_2x, depth_heightmap_2x, depth_heightmap_2x), axis=2
        )
        for c in range(3):
            input_depth_image[:, :, c] = (
                input_depth_image[:, :, c] - image_mean[c]
            ) / image_std[c]

        # Construct minibatch of size 1 (b,c,h,w)
        input_color_image.shape = (
            input_color_image.shape[0],
            input_color_image.shape[1],
            input_color_image.shape[2],
            1,
        )
        input_depth_image.shape = (
            input_depth_image.shape[0],
            input_depth_image.shape[1],
            input_depth_image.shape[2],
            1,
        )
        input_color_data = torch.from_numpy(
            input_color_image.astype(np.float32)
        ).permute(3, 2, 0, 1)
        input_depth_data = torch.from_numpy(
            input_depth_image.astype(np.float32)
        ).permute(3, 2, 0, 1)

        # Pass input data through model
        # output_prob, state_feat
        output_prob = r_net.forward(
            input_color_data, input_depth_data, specific_rotation
        )

        # Return Q values (and remove extra padding)
        for rotate_idx in range(len(output_prob)):
            # if first rotation
            if rotate_idx == 0:
                push_predictions = (
                    output_prob[rotate_idx][0]
                    .cpu()
                    .data.numpy()[
                        :,
                        0,
                        int(padding_width / 2) : int(
                            color_heightmap_2x.shape[0] / 2 - padding_width / 2
                        ),
                        int(padding_width / 2) : int(
                            color_heightmap_2x.shape[0] / 2 - padding_width / 2
                        ),
                    ]
                )
            else:
                push_predictions = np.concatenate(
                    (
                        push_predictions,
                        output_prob[rotate_idx][0]
                        .cpu()
                        .data.numpy()[
                            :,
                            0,
                            int(padding_width / 2) : int(
                                color_heightmap_2x.shape[0] / 2 - padding_width / 2
                            ),
                            int(padding_width / 2) : int(
                                color_heightmap_2x.shape[0] / 2 - padding_width / 2
                            ),
                        ],
                    ),
                    axis=0,
                )

        return push_predictions

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
    push_pred = forward(color_heightmap, valid_depth_heightmap, is_volatile=True)

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
