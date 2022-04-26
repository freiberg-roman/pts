import numpy as np
import torch
from scipy import ndimage
from torch.autograd import Variable

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

        # TODO enable pretrained weights
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

    def forward(self, color_heightmap, depth_heightmap):
        # Apply 2x scale to input heightmaps
        color_heightmap_2x = ndimage.zoom(color_heightmap, zoom=[2, 2, 1], order=0)
        depth_heightmap_2x = ndimage.zoom(depth_heightmap, zoom=[2, 2], order=0)
        assert color_heightmap_2x.shape[0:2] == depth_heightmap_2x.shape[0:2]

        # Add extra padding (to handle rotations inside network)
        diag_length = float(color_heightmap_2x.shape[0]) * np.sqrt(2)
        diag_length = np.ceil(diag_length / 32) * 32
        padding_width = int((diag_length - color_heightmap_2x.shape[0]) / 2)
        color_heightmap_2x = np.pad(
            color_heightmap_2x,
            pad_width=(
                (padding_width, padding_width),
                (padding_width, padding_width),
                (0, 0),
            ),
        )
        depth_heightmap_2x = np.pad(
            depth_heightmap_2x, padding_width, "constant", constant_values=0
        )

        # Pre-process color image (scale and normalize)
        image_mean = np.array([[[0.485, 0.456, 0.406]]])
        image_std = np.array([[[0.229, 0.224, 0.225]]])
        color_heightmap_2x = (
            (color_heightmap_2x.astype(float) / 255) - image_mean
        ) / image_std

        # Pre-process depth image (normalize)
        depth_mean = 0.01
        depth_std = 0.03
        depth_heightmap_2x = (depth_heightmap_2x - depth_mean) / depth_std
        depth_heightmap_2x = np.repeat(depth_heightmap_2x[:, :, np.newaxis], 3, axis=2)

        # Construct minibatch of size 1 (b,c,h,w)
        input_depth_image = (
            torch.from_numpy(depth_heightmap_2x[np.newaxis])
            .permute(0, 3, 1, 2)
            .to(torch.float32)
        )
        input_color_image = (
            torch.from_numpy(color_heightmap_2x[np.newaxis])
            .permute(0, 3, 1, 2)
            .to(torch.float32)
        )

        # Feed through network
        output_prob = self.model.forward(input_color_image, input_depth_image)

        # Return Q values (and remove extra padding)
        push_pred = []
        for rotate_idx in range(len(output_prob)):
            push_pred.append(
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
        return np.concatenate(push_pred)

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

        next_push_predictions = self.forward(next_color_heightmap, next_depth_heightmap)

        future_reward = np.max(next_push_predictions)
        rew = current_reward + self.gamma * future_reward
        return rew

    def backprop(self, pix_ind, reward, height_map_dim):
        out_prob_dim = self.model.output_prob[0][0].size()
        out_prob_dim = (out_prob_dim[2], out_prob_dim[3])
        pad_width = (out_prob_dim[0] - height_map_dim[0]) // 2
        pad_height = (out_prob_dim[1] - height_map_dim[1]) // 2
        label = np.zeros((1, *out_prob_dim))
        label[0, pad_width + pix_ind[1], pad_height + pix_ind[2]] = reward

        # Compute label mask
        label_weights = np.zeros(label.shape)
        label_weights[0, pad_width + pix_ind[1], pad_height + pix_ind[2]] = 1

        self.optimizer.zero_grad()
        loss = self.criterion(
            self.model.output_prob[0][0].view(1, *out_prob_dim),
            Variable(torch.from_numpy(label).float())
            * Variable(torch.from_numpy(label_weights).float(), requires_grad=True),
        )
        loss = loss.sum()
        loss.backward()
        self.optimizer.step()

        return loss.cpu().data.numpy()
