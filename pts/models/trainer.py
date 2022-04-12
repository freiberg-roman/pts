import numpy as np
import torch
from scipy import ndimage

from pts.models import ReinforcementNet


class Trainer:
    def __init__(self, cfg_dqn):
        self.gamma = cfg_dqn.train.gamma
        self.criterion = torch.nn.SmoothL1Loss(reduce=False)

        if cfg_dqn.train.use_cuda:
            if torch.cuda.is_available():
                self.device = "cuda"
                print("CUDA device found!")
            else:
                print("WARNING: CUDA is not available!!! CPU will be used instead!")
                self.device = "cpu"
        else:
            print("CPU will be used to train/evaluate the network!")
            self.device = "cpu"

        self.model = ReinforcementNet(
            use_cuda=cfg_dqn.train.use_cuda and torch.cuda.is_available()
        )
        if cfg_dqn.train.pretrained_weights != "":
            self.model.load_state_dict(
                torch.load(
                    cfg_dqn.train.pretrained_weights,
                    map_location=torch.device(self.device),
                )
            )
        self.model.train()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg_dqn.train.lr)
        self.epoch = 0

    def forward(
        self, color_heightmap, depth_heightmap, is_volatile=False, specific_rotation=-1
    ):

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
        output_prob = self.model.forward(
            input_color_data, input_depth_data, is_volatile, specific_rotation
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

    def get_reward_value(
        self,
        next_color_heightmap,
        next_depth_heightmap,
        prev_seg_score,
        curr_seg_score,
        change_detected,
    ):

        diff = curr_seg_score - prev_seg_score
        if change_detected:
            current_reward = (diff * 10) ** 2
            if diff < 0:
                current_reward *= -1
        else:
            current_reward = -0.5

        seg_vals = [
            curr_seg_score,
            prev_seg_score,
            diff,
            current_reward,
            change_detected,
        ]
        next_push_predictions = self.forward(
            next_color_heightmap, next_depth_heightmap, is_volatile=True
        )

        future_reward = np.max(next_push_predictions)
        expected_reward = current_reward + self.gamma * future_reward
        return expected_reward, current_reward, seg_vals

    def backprop(self, best_pix_ind, label_value):
        label = np.zeros((1, 320, 320))
        action_area = np.zeros((224, 224))
        action_area[best_pix_ind[1]][best_pix_ind[2]] = 1
        tmp_label = np.zeros((224, 224))
        tmp_label[action_area > 0] = label_value
        label[0, 48 : (320 - 48), 48 : (320 - 48)] = tmp_label

        # Compute label mask
        label_weights = np.zeros(label.shape)
        tmp_label_weights = np.zeros((224, 224))
        tmp_label_weights[action_area > 0] = 1
        label_weights[0, 48 : (320 - 48), 48 : (320 - 48)] = tmp_label_weights

        self.optimizer.zero_grad()
        loss_value = 0

        loss = self.criterion(0)

        # TODO figure out
        # loss = self.criterion(
        # self.model.output_prob[0][0].view(1, 320, 320),
        # Variable(torch.from_numpy(label).float())) *
        # Variable( torch.from_numpy(label_weights).float(), requires_grad=True)

        loss = loss.sum()
        loss.backward()
        loss_value = loss.cpu().data.numpy()
        l_value = loss_value
        self.optimizer.step()

        return l_value
