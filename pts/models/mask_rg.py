import datetime
import math
import os
import sys
import time

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from omegaconf import OmegaConf
from torch.utils import data as td
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from pts.utils.train_eval import reduce_dict

HEIGHT = 1024
WIDTH = 1024


class MaskRGNetwork(nn.Module):
    def __init__(self, cfg_rg):
        super(MaskRGNetwork, self).__init__()

        self.num_epochs = cfg_rg.train.epochs
        self.lr = cfg_rg.train.lr
        self.batch_size = cfg_rg.train.batch_size
        self.backbone = cfg_rg.train.backbone
        self.bb_pretrained = cfg_rg.train.backbone_pretrained
        self.saving_path = cfg_rg.path_store_weights
        self.weights_path = cfg_rg.path_load_weights
        is_cuda = cfg_rg.train.cuda_available

        if is_cuda:
            if torch.cuda.is_available():
                self.device = "cuda"
                print("CUDA device found!")
            else:
                print("WARNING: CUDA is not available!!! CPU will be used instead!")
                self.device = "cpu"
        else:
            print("CPU will be used to train/evaluate the network!")
            self.device = "cpu"

        # self.backbone.out_channels = [512, 512, 512]
        if self.backbone == "resnet50":
            self.mask_r_cnn = torchvision.models.detection.maskrcnn_resnet50_fpn(
                pretrained=True, progress=True, pretrained_backbone=self.bb_pretrained
            )

        # get number of input features for the classifier
        self.input_features = (
            self.mask_r_cnn.roi_heads.box_predictor.cls_score.in_features
        )

        # replace the pre-trained head with a new one --> category-agnostic so number of classes=2
        self.mask_r_cnn.roi_heads.box_predictor = FastRCNNPredictor(
            self.input_features, 2
        )

        # now get the number of input features for the mask classifier
        self.input_features_mask = (
            self.mask_r_cnn.roi_heads.mask_predictor.conv5_mask.in_channels
        )

        hidden_layer = 256
        # and replace the mask predictor with a new one
        self.mask_r_cnn.roi_heads.mask_predictor = MaskRCNNPredictor(
            self.input_features_mask, hidden_layer, 2
        )

        self.mask_r_cnn.to(self.device)

        # construct an optimizer
        self.params = [p for p in self.mask_r_cnn.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(self.params, lr=self.lr)
        self.conf = OmegaConf.to_container(cfg_rg)

    def train_model(self):
        import wandb

        wandb.init(project="train_rg_model", entity="freiberg-roman", config=self.conf)

        for _ in range(self.num_epochs):
            self.mask_r_cnn.train()

            for imgs, targets in self.data_loader:
                imgs = [img.to(self.device) for img in imgs]
                targets = [
                    {k: v.to(self.device) for k, v in t.items()} for t in targets
                ]

                loss_dict = self.mask_r_cnn(imgs, targets)
                loss_dict_reduced = reduce_dict(
                    loss_dict
                )  # reduce losses over all GPUs
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())

                loss_value = losses_reduced.item()

                if not math.isfinite(loss_value):
                    print("Loss is {}, stopping training".format(loss_value))
                    print(loss_dict_reduced)
                    sys.exit(1)

                self.optimizer.zero_grad()
                losses_reduced.backward()
                self.optimizer.step()
                wandb.log({"loss": loss_value})

            if self.conf.train.store_model_each_epoch:
                self.save_model()

    def eval_single_img(self, img):
        self.mask_r_cnn.eval()
        with torch.no_grad():
            preds = self.mask_r_cnn(img)

        return preds

    def load_weights(self):
        self.mask_r_cnn.load_state_dict(
            torch.load(self.weights_path, map_location=self.device)
        )

    def set_data(self, data):
        self.data_loader = td.DataLoader(
            data,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=lambda x: tuple(zip(*x)),
        )

    def save_model(self):
        t = time.time()
        timestamp = datetime.datetime.fromtimestamp(t)
        file_name = timestamp.strftime("%Y-%m-%d.%H:%M:%S") + ".pth"
        torch.save(
            self.mask_r_cnn.state_dict(), os.path.join(self.saving_path, file_name)
        )

    @staticmethod
    def print_boxes(image, input_tensor, score_threshold=0.75):
        boxes = input_tensor[0]["boxes"]
        scores = input_tensor[0]["scores"]

        fig, ax = plt.subplots(1)
        ax.imshow(image)
        num_box = 0
        for box in boxes:
            box_index = np.where(boxes == box)
            box_index = int(box_index[0][0])
            if scores[box_index] > score_threshold:
                rect = patches.Rectangle(
                    (box[0], box[1]),
                    box[2] - box[0],
                    box[3] - box[1],
                    linewidth=3,
                    edgecolor="r",
                    facecolor="none",
                )
                ax.add_patch(rect)
                num_box += 1
        print("num boxes that have a score higher than .75 --> ", num_box)
        plt.show()


class RewardGenerator:
    def __init__(self, confidence_threshold=0.75, mask_threshold=0.75, device="cuda"):

        self.mask_threshold = mask_threshold
        self.confidence_threshold = confidence_threshold

        self.obj_ids = None
        self.num_objs = None
        self.height = 1000
        self.width = 1000
        self.gts = None
        self.scores = None
        self.masks = None
        self.boxes = None
        self.num_pred = None
        self.valid_masks = []
        self.valid_boxes = []
        self.num_valid_pred = 0

    def set_element(self, pred_tensor, gt):
        obj_ids = np.unique(gt)
        self.obj_ids = obj_ids[1:]

        # combine the masks
        self.num_objs = len(self.obj_ids)
        self.height = gt.shape[0]
        self.width = gt.shape[1]
        self.gts = np.zeros((self.num_objs, self.height, self.width), dtype=np.uint8)
        for i in range(self.num_objs):
            self.gts[i][np.where(gt == i + 1)] = 1

        self.scores = pred_tensor[0]["scores"]
        self.masks = pred_tensor[0]["masks"]
        self.boxes = pred_tensor[0]["boxes"]
        self.num_pred = self.masks.shape[0]
        self.valid_masks = []
        self.valid_boxes = []
        self.num_valid_pred = 0

        for pred_no in range(0, self.num_pred):
            if self.scores[pred_no] > self.confidence_threshold:
                self.valid_masks.append(self.masks[pred_no])
                self.valid_boxes.append(self.boxes[pred_no])
                self.num_valid_pred += 1

    @staticmethod
    def _jaccard(gt, pred):

        intersection = gt * pred
        union = (gt + pred) / 2
        area_i = np.count_nonzero(intersection)
        area_u = np.count_nonzero(union)
        if area_u > 0:
            iou = area_i / area_u
        else:
            return [0, 0, 0]
        return [iou, area_i, area_u]

    def get_reward(self):
        res = self._detect_mask_id()

        if self.num_objs <= 0:
            raise ValueError("no objects detected for reward generation")

        sum_iou = np.sum(res, axis=0)[1]
        reward_iou = sum_iou / self.num_objs

        true_detections = np.count_nonzero(res, axis=0)[1]
        num_non_detected = self.num_objs - true_detections
        reward = reward_iou + num_non_detected * (-0.02)
        return (
            res,
            reward,
            [self.num_objs, num_non_detected, num_non_detected / self.num_objs],
        )

    def _detect_mask_id(self):

        results = []
        pred_order = np.zeros((self.num_objs, 2), dtype=np.float32)

        for idx_gt, gt in enumerate(self.gts):
            for inx_pred, mask in enumerate(self.valid_masks):
                pred_arr = np.asarray(mask.cpu().detach()).reshape(
                    (self.height, self.width)
                )
                pred_arr = np.where(pred_arr > self.mask_threshold, 1, 0)
                res = self._jaccard(gt, pred_arr)
                if res[0] > 0:
                    results.append([inx_pred, res[0]])

            if results != []:
                res_arr = np.asarray(results)
                max_index = np.argmax(res_arr, axis=0)
                max_index = max_index[1]
                if res_arr[max_index][1] > self.mask_threshold:
                    pred_order[idx_gt] = res_arr[max_index]
                else:
                    pred_order[idx_gt] = [255, 0]
            else:
                pred_order[idx_gt] = [254, 0]
            results = []
        return pred_order


class MaskRG:
    def __init__(self, cfg_rg):
        self.model = MaskRGNetwork(cfg_rg)
        self.model.load_weights()
        self.rg = RewardGenerator(cfg_rg.confidence_threshold, cfg_rg.mask_threshold)
        self.prediction = None
        self.gt = None

    def set_reward_generator(self, depth_image, gt_segmentation):
        depth_image = np.repeat(depth_image.reshape(1000, 1000, 1), 3, axis=2)
        with torch.no_grad():
            depth_tensor = T.ToTensor()(depth_image)
            self.prediction = self.model.eval_single_img([depth_tensor])

        self.gt = gt_segmentation

        self.rg.set_element(self.prediction, self.gt)

    def get_current_rewards(self):
        return self.rg.get_reward()
