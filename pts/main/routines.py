import cv2
import numpy as np

from pts.models import MaskRG, Trainer
from pts.sim.train_scene import TrainScene
from pts.utils.image_helper import CrossEntropyLoss2d, get_heightmap


def train_dqn(cfg_dqn, cfg_rg, cfg_env):
    # ### Initializing ###
    mask_rg_net = MaskRG(cfg_rg, train=False)  # network is in testing mode
    success_threshold = cfg_dqn.success_threshold
    ws_limits = cfg_env.workspace_limit
    heightmap_res = cfg_env.heightmap_resolution
    min_obj = cfg_env.min_number_objects
    max_obj = cfg_env.max_number_objects
    session_limits = cfg_env.session_limit
    gamma = cfg_dqn.discount
    seed = cfg_dqn.seed
    exploration = True

    train_scene = TrainScene(cfg_env)
    trainer = Trainer(cfg_dqn.train.lr, cfg_dqn.train.gamma, cfg_dqn.train.rg_model)
    train_scene.create_testing_scenario()

    # ### Main loop including initial run ###

    # ### Init run ###
    color_img, depth_img = train_scene.get_camera_data()
    seg, num_obj = train_scene.get_data_mask_rg()
    dim = (int(color_img.shape[0] * 1.024), int(color_img.shape[1] * 1.024))

    color_img = cv2.resize(color_img, dim, interpolation=cv2.INTER_AREA)
    depth_img = cv2.resize(depth_img, dim, interpolation=cv2.INTER_AREA)
    seg = cv2.resize(np.uint8(seg), dim, interpolation=cv2.INTER_NEAREST)

    # ### Preprocessing ###

    seg[seg == 1] = 0
    depth_img[depth_img > 1.50] = 0
    depth_img[depth_img == 0] = 0.8
    depth_img = depth_img * 5000

    mask_rg_net.set_reward_generator(depth_img, seg)
    pred_ids, seg_rew, err = mask_rg_net.get_current_rewards()
    color_heightmap, depth_heightmap = get_heightmap(
        color_img,
        depth_img,
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

    best_pix_ind = (
        np.random.random_integers(0, 15),
        np.random.random_integers(0, 150),
        np.random.random_integers(75),
    )
    best_rot_angle = np.deg2rad(best_pix_ind[0] * (360.0 / trainer.model.num_rotations))

    prim_pos = [
        best_pix_ind[2] * heightmap_res + ws_limits[0][0],
        best_pix_ind[1] * heightmap_res + ws_limits[1][0],
        valid_depth_heightmap[best_pix_ind[1]][best_pix_ind[2]] + ws_limits[2][0],
    ]
    safe_kernel_width = int(
        np.round((0.02 / 2) / heightmap_res)
    )  # 0.02 is the finger width
    local_region = valid_depth_heightmap[
        max(best_pix_ind[1] - safe_kernel_width, 0) : min(
            best_pix_ind[1] + safe_kernel_width + 1, valid_depth_heightmap.shape[0]
        ),
        max(best_pix_ind[2] - safe_kernel_width, 0) : min(
            best_pix_ind[2] + safe_kernel_width + 1, valid_depth_heightmap.shape[1]
        ),
    ]
    if local_region.size == 0:
        safe_z_position = ws_limits[2][0]
    elif np.max(local_region) > 0.02:
        # elif (np.max(local_region) - workspace_limits[2][0]) > finger_width:
        # in z  --> - finger_width/4
        safe_z_position = ws_limits[2][0] + np.max(local_region) - 0.02 / 4
    else:
        safe_z_position = ws_limits[2][0] + np.max(local_region)

    prim_pos[2] = safe_z_position

    prim_pos[2] = safe_z_position
    train_scene.push(prim_pos, best_rot_angle)

    # ### Loop ###
    for it in range(cfg_dqn.train.maximum_iterations):

        # ### Execute Push ###

        prev_pix_ind = best_pix_ind
        if it > 0 and not exploration:
            best_pix_ind = np.unravel_index(
                np.argmax(push_pred), push_pred.shape
            )
        else:
            best_pix_ind = (
                np.random.random_integers(0, 15),
                np.random.random_integers(0, 223),
                np.random.random_integers(223),
            )
        best_rotation_angle = np.deg2rad(
            best_pix_ind[0] * (360.0 / trainer.model.num_rotations)
        )
        best_pix_x = best_pix_ind[2]
        best_pix_y = best_pix_ind[1]
        primitive_position = [
            best_pix_x * heightmap_res + ws_limits[0][0],
            best_pix_y * heightmap_res + ws_limits[1][0],
            valid_depth_heightmap[best_pix_y][best_pix_x] + ws_limits[2][0],
        ]
        finger_width = 0.02
        safe_kernel_width = int(np.round((finger_width / 2) / heightmap_res))
        local_region = valid_depth_heightmap[
            max(best_pix_y - safe_kernel_width, 0) : min(
                best_pix_y + safe_kernel_width + 1, valid_depth_heightmap.shape[0]
            ),
            max(best_pix_x - safe_kernel_width, 0) : min(
                best_pix_x + safe_kernel_width + 1, valid_depth_heightmap.shape[1]
            ),
        ]
        if local_region.size == 0:
            safe_z_position = ws_limits[2][0]
        elif np.max(local_region) > finger_width:
            safe_z_position = ws_limits[2][0] + np.max(local_region) - finger_width / 4
        else:
            safe_z_position = ws_limits[2][0] + np.max(local_region)

        primitive_position[2] = safe_z_position

        push_success = train_scene.push(primitive_position, best_rotation_angle)

        # ### Get Image ###
        color_img, depth_img = train_scene.get_camera_data()
        [seg, num_obj] = train_scene.get_data_mask_rg()
        width = int(color_img.shape[1] * 1.6)
        height = int(color_img.shape[0] * 1.6)

        mask_rg_net.set_reward_generator(depth_img, seg)
        prev_seg_reward = seg_reward
        pred_ids, seg_reward, err_rate = mask_rg_net.get_current_rewards()

        # ### Save images for change calculation ###

        prev_color_img = color_img.copy()
        prev_depth_img = depth_img.copy()

        push_pred = trainer.forward(
            color_heightmap, valid_depth_heightmap, is_volatile=True
        )

        diff_depth = abs(valid_depth_heightmap - prev_depth_img)
        diff_depth[diff_depth > 0] = 1
        change_detected = np.sum(diff_depth) > cfg_dqn.change_threshold
        success_action = seg_reward > success_threshold

        # ### Update network ###

        label_val, prev_reward_val, seg_val = trainer.get_reward_value(
            color_heightmap,
            valid_depth_heightmap,
            prev_seg_reward,
            seg_reward,
            change_detected,
        )
        loss_val = trainer.backprop(prev_pix_ind, label_val)
