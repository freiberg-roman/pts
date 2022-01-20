import os
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage
from torch.autograd import Variable

from pts.models import ReinforcementNet
from pts.utils.image_helper import CrossEntropyLoss2d, get_heightmap


class Trainer(object):
    def __init__(self, lr, gamma, model_file):

        self.model = ReinforcementNet(use_cuda=False)
        self.gamma = gamma

        # Initialize Huber loss
        self.criterion = torch.nn.SmoothL1Loss(reduce=False)  # Huber loss

        # Load pre-trained model
        if not model_file == "":
            self.model.load_state_dict(torch.load(model_file))

        # Set model to training mode
        self.model.train()

        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.epoch = 0

    # Compute forward pass through model to compute affordances/Q
    def forward(self, color_heightmap, depth_heightmap, is_volatile=False, specific_rotation=-1):

        # Apply 2x scale to input heightmaps
        color_heightmap_2x = ndimage.zoom(color_heightmap, zoom=[2, 2, 1], order=0)
        depth_heightmap_2x = ndimage.zoom(depth_heightmap, zoom=[2, 2], order=0)
        assert (color_heightmap_2x.shape[0:2] == depth_heightmap_2x.shape[0:2])

        # Add extra padding (to handle rotations inside network)
        diag_length = float(color_heightmap_2x.shape[0]) * np.sqrt(2)
        diag_length = np.ceil(diag_length / 32) * 32
        padding_width = int((diag_length - color_heightmap_2x.shape[0]) / 2)
        color_heightmap_2x_r = np.pad(color_heightmap_2x[:, :, 0], padding_width, 'constant', constant_values=0)
        color_heightmap_2x_r.shape = (color_heightmap_2x_r.shape[0], color_heightmap_2x_r.shape[1], 1)
        color_heightmap_2x_g = np.pad(color_heightmap_2x[:, :, 1], padding_width, 'constant', constant_values=0)
        color_heightmap_2x_g.shape = (color_heightmap_2x_g.shape[0], color_heightmap_2x_g.shape[1], 1)
        color_heightmap_2x_b = np.pad(color_heightmap_2x[:, :, 2], padding_width, 'constant', constant_values=0)
        color_heightmap_2x_b.shape = (color_heightmap_2x_b.shape[0], color_heightmap_2x_b.shape[1], 1)
        color_heightmap_2x = np.concatenate((color_heightmap_2x_r, color_heightmap_2x_g, color_heightmap_2x_b), axis=2)
        depth_heightmap_2x = np.pad(depth_heightmap_2x, padding_width, 'constant', constant_values=0)

        # Pre-process color image (scale and normalize)
        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]
        input_color_image = color_heightmap_2x.astype(float) / 255
        for c in range(3):
            input_color_image[:, :, c] = (input_color_image[:, :, c] - image_mean[c]) / image_std[c]

        # Pre-process depth image (normalize)
        image_mean = [0.01, 0.01, 0.01]
        image_std = [0.03, 0.03, 0.03]
        depth_heightmap_2x.shape = (depth_heightmap_2x.shape[0], depth_heightmap_2x.shape[1], 1)
        input_depth_image = np.concatenate((depth_heightmap_2x, depth_heightmap_2x, depth_heightmap_2x), axis=2)
        for c in range(3):
            input_depth_image[:, :, c] = (input_depth_image[:, :, c] - image_mean[c]) / image_std[c]

        # Construct minibatch of size 1 (b,c,h,w)
        input_color_image.shape = (
        input_color_image.shape[0], input_color_image.shape[1], input_color_image.shape[2], 1)
        input_depth_image.shape = (
        input_depth_image.shape[0], input_depth_image.shape[1], input_depth_image.shape[2], 1)
        input_color_data = torch.from_numpy(input_color_image.astype(np.float32)).permute(3, 2, 0, 1)
        input_depth_data = torch.from_numpy(input_depth_image.astype(np.float32)).permute(3, 2, 0, 1)

        # Pass input data through model
        # output_prob, state_feat
        output_prob = self.model.forward(input_color_data, input_depth_data, is_volatile, specific_rotation)

        # Return Q values (and remove extra padding)
        for rotate_idx in range(len(output_prob)):
            # if first rotation
            if rotate_idx == 0:
                push_predictions = output_prob[rotate_idx][0].cpu().data.numpy()[:,
                                   0, int(padding_width / 2):int(color_heightmap_2x.shape[0] / 2 - padding_width / 2),
                                   int(padding_width / 2):int(color_heightmap_2x.shape[0] / 2 - padding_width / 2)]
            else:
                push_predictions = np.concatenate((push_predictions,
                                                   output_prob[rotate_idx][0].cpu().data.numpy()[:,
                                                   0, int(padding_width / 2):int(
                                                       color_heightmap_2x.shape[0] / 2 - padding_width / 2),
                                                   int(padding_width / 2):int(
                                                       color_heightmap_2x.shape[0] / 2 - padding_width / 2)]),
                                                  axis=0)

        return push_predictions

    def get_reward_value(self, next_color_heightmap, next_depth_heightmap, prev_seg_score, curr_seg_score,
                         change_detected):

        diff = curr_seg_score - prev_seg_score
        if change_detected:
            current_reward = (diff * 10) ** 2
            if diff < 0:
                current_reward *= -1
        else:
            current_reward = -0.5

        seg_vals = [curr_seg_score, prev_seg_score, diff, current_reward, change_detected]
        next_push_predictions = self.forward(next_color_heightmap, next_depth_heightmap, is_volatile=True)

        future_reward = np.max(next_push_predictions)
        expected_reward = current_reward + self.gamma * future_reward
        return expected_reward, current_reward, seg_vals

    # Compute labels and backpropagate
    def backprop(self, best_pix_ind, label_value):
        label = np.zeros((1, 320, 320))
        action_area = np.zeros((224, 224))
        action_area[best_pix_ind[1]][best_pix_ind[2]] = 1
        tmp_label = np.zeros((224, 224))
        tmp_label[action_area > 0] = label_value
        label[0, 48:(320 - 48), 48:(320 - 48)] = tmp_label

        # Compute label mask
        label_weights = np.zeros(label.shape)
        tmp_label_weights = np.zeros((224, 224))
        tmp_label_weights[action_area > 0] = 1
        label_weights[0, 48:(320 - 48), 48:(320 - 48)] = tmp_label_weights

        # Compute loss and backward pass
        self.optimizer.zero_grad()
        loss_value = 0

        loss = self.criterion(self.model.output_prob[0][0].view(1, 320, 320),
                              Variable(torch.from_numpy(label).float())) * Variable(
            torch.from_numpy(label_weights).float(),
            requires_grad=False)
        loss = loss.sum()
        loss.backward()
        loss_value = loss.cpu().data.numpy()
        l_value = loss_value
        self.optimizer.step()

        return l_value

    def get_prediction_vis(self, predictions, color_heightmap, best_pix_ind):

        canvas = None
        num_rotations = predictions.shape[0]
        for canvas_row in range(int(num_rotations / 4)):
            tmp_row_canvas = None
            for canvas_col in range(4):
                rotate_idx = canvas_row * 4 + canvas_col
                prediction_vis = predictions[rotate_idx, :, :].copy()
                prediction_vis = np.clip(prediction_vis, 0, 1)
                prediction_vis.shape = (predictions.shape[1], predictions.shape[2])
                prediction_vis = cv2.applyColorMap((prediction_vis * 255).astype(np.uint8), cv2.COLORMAP_JET)
                if rotate_idx == best_pix_ind[0]:
                    prediction_vis = cv2.circle(prediction_vis, (int(best_pix_ind[2]), int(best_pix_ind[1])), 7,
                                                (0, 0, 255), 2)
                prediction_vis = ndimage.rotate(prediction_vis, rotate_idx * (360.0 / num_rotations), reshape=False,
                                                order=0)
                background_image = ndimage.rotate(color_heightmap, rotate_idx * (360.0 / num_rotations), reshape=False,
                                                  order=0)
                prediction_vis = (
                            0.5 * cv2.cvtColor(background_image, cv2.COLOR_RGB2BGR) + 0.5 * prediction_vis).astype(
                    np.uint8)
                if tmp_row_canvas is None:
                    tmp_row_canvas = prediction_vis
                else:
                    tmp_row_canvas = np.concatenate((tmp_row_canvas, prediction_vis), axis=1)
            if canvas is None:
                canvas = tmp_row_canvas
            else:
                canvas = np.concatenate((canvas, tmp_row_canvas), axis=0)

        return canvas

    def push_heuristic(self, depth_heightmap):

        num_rotations = 16

        for rotate_idx in range(num_rotations):
            rotated_heightmap = ndimage.rotate(depth_heightmap, rotate_idx * (360.0 / num_rotations), reshape=False,
                                               order=0)
            valid_areas = np.zeros(rotated_heightmap.shape)
            valid_areas[
                ndimage.interpolation.shift(rotated_heightmap, [0, -25], order=0) - rotated_heightmap > 0.02] = 1
            # valid_areas = np.multiply(valid_areas, rotated_heightmap)
            blur_kernel = np.ones((25, 25), np.float32) / 9
            valid_areas = cv2.filter2D(valid_areas, -1, blur_kernel)
            tmp_push_predictions = ndimage.rotate(valid_areas, -rotate_idx * (360.0 / num_rotations), reshape=False,
                                                  order=0)
            tmp_push_predictions.shape = (1, rotated_heightmap.shape[0], rotated_heightmap.shape[1])

            if rotate_idx == 0:
                push_predictions = tmp_push_predictions
            else:
                push_predictions = np.concatenate((push_predictions, tmp_push_predictions), axis=0)

        best_pix_ind = np.unravel_index(np.argmax(push_predictions), push_predictions.shape)
        return best_pix_ind

    def grasp_heuristic(self, depth_heightmap):

        num_rotations = 16

        for rotate_idx in range(num_rotations):
            rotated_heightmap = ndimage.rotate(depth_heightmap, rotate_idx * (360.0 / num_rotations), reshape=False,
                                               order=0)
            valid_areas = np.zeros(rotated_heightmap.shape)
            valid_areas[
                np.logical_and(rotated_heightmap - ndimage.interpolation.shift(rotated_heightmap, [0, -25], order=0)
                               > 0.02, rotated_heightmap - ndimage.interpolation.shift(rotated_heightmap, [0, 25],
                                                                                       order=0) > 0.02)] = 1
            blur_kernel = np.ones((25, 25), np.float32) / 9
            valid_areas = cv2.filter2D(valid_areas, -1, blur_kernel)
            tmp_grasp_predictions = ndimage.rotate(valid_areas, -rotate_idx * (360.0 / num_rotations), reshape=False,
                                                   order=0)
            tmp_grasp_predictions.shape = (1, rotated_heightmap.shape[0], rotated_heightmap.shape[1])

            if rotate_idx == 0:
                grasp_predictions = tmp_grasp_predictions
            else:
                grasp_predictions = np.concatenate((grasp_predictions, tmp_grasp_predictions), axis=0)

        best_pix_ind = np.unravel_index(np.argmax(grasp_predictions), grasp_predictions.shape)
        return best_pix_ind
