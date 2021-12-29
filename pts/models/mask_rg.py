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
from torch.utils import data as td
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from pts.utils.train_eval import evaluate, reduce_dict

HEIGHT = 1024
WIDTH = 1024


class MaskRGNetwork(nn.Module):
    def __init__(self, cfg_rg, train=True):
        super(MaskRGNetwork, self).__init__()

        self.num_epochs = cfg_rg.train.epochs
        self.lr = cfg_rg.train.lr
        self.batch_size = cfg_rg.train.batch_size
        self.backbone = cfg_rg.train.backbone
        self.bb_pretrained = cfg_rg.train.backbone_pretrained
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

    def train_model(self):

        for epoch in range(self.num_epochs):

            # ### Use adaptive lr for first epoch ###
            def scheduler(warm_iters, warm_factor):
                def f(x):
                    if x >= warm_iters:
                        return 1
                    alpha = float(x) / warm_iters
                    return warm_factor * (1 - alpha) + alpha

                return torch.optim.lr_scheduler.LambdaLR(self.optimizer, f)

            warm_scheduler = scheduler(min(1000, len(self.data_loader), 1.0 / 1000))

            # ### Train for one epoch ###

            self.mask_r_cnn.train()

            for imgs, targets in self.data_loader:
                imgs = list(imgs.to(self.device) for img in imgs)
                targets = [
                    {k: v.to(self.device) for k, v in t.items()} for t in targets
                ]

                loss_dict = self.mask_r_cnn(imgs, targets)

                losses = sum(loss for loss in loss_dict.values())

                # reduce losses over all GPUs for logging purposes
                loss_dict_reduced = reduce_dict(loss_dict)
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())

                loss_value = losses_reduced.item()

                if not math.isfinite(loss_value):
                    print("Loss is {}, stopping training".format(loss_value))
                    print(loss_dict_reduced)
                    sys.exit(1)

                self.optimizer.zero_grad()
                losses.backwards()
                self.optimizer.step()

                if epoch == 0:
                    warm_scheduler.step()

    def evaluate_model(self):
        # evaluate on the test dataset
        res = evaluate(
            self.mask_r_cnn, self.data_loader, device=self.device, num_threads=12
        )
        return res

    def eval_single_img(self, img):
        # self.device = torch.device("cpu")

        self.mask_r_cnn.eval()
        with torch.no_grad():
            preds = self.mask_r_cnn(img)

        return preds

    def load_weights(self):
        # this should be called after the initialisation of the model
        self.mask_r_cnn.load_state_dict(torch.load(self.weigths_path))

    def set_data(self, data, is_test=False):

        data_subset = (
            td.Subset(data, self.train_indices)
            if not is_test
            else td.Subset(data, self.test_indices)
        )

        # init a data loader either for training or testing
        self.data_loader = td.DataLoader(
            data_subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=6,
            collate_fn=lambda x: tuple(zip(*x)),
        )

    def _save_model(self, string=None):
        t = time.time()
        timestamp = datetime.datetime.fromtimestamp(t)
        file_name = (
            self.saving_prefix
            + string
            + timestamp.strftime("%Y-%m-%d.%H:%M:%S")
            + ".pth"
        )
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

    @staticmethod
    def print_masks(image, input_tensor, score_threshold=0.75):
        masks = input_tensor[0]["masks"]
        scores = input_tensor[0]["scores"]
        num_pred = masks.shape[0]
        num_masks = 0
        all = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        for mask in range(0, num_pred):
            if scores[mask] > score_threshold:
                # TODO if cuda, add a control here

                mask_arr = np.asarray(masks[mask].cpu().detach()).reshape(
                    (HEIGHT, WIDTH)
                )
                mask_arr = np.where(mask_arr > score_threshold, 1, 0)
                all[np.where(mask_arr > 0)] = num_masks

                num_masks += 1
        plt.imshow(all)
        plt.show()
        print("num masks that have a score higher than .75 --> ", num_masks)
