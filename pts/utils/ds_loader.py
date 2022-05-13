import os

import cv2 as cv
import numpy as np
import torch
import torchvision.transforms as T


class PTSDataset:
    def __init__(self, cfg):
        self.root_dir = cfg.dataset_path
        self.mask_dir = cfg.mask_dir
        self.depth_dir = cfg.depth_dir

        self.base_rgb = cfg.base_rgb
        self.base_depth = cfg.base_depth
        self.base_seg = cfg.base_seg

        # assume for each depth image exists a matching mask
        self.length = len(list(os.listdir(self.root_dir + self.depth_dir)))

    def __getitem__(self, idx):
        img_depth = np.load(
            self.root_dir + self.depth_dir + str(idx) + self.base_depth + ".npy"
        )
        mask = np.load(
            self.root_dir + self.mask_dir + str(idx) + self.base_seg + ".npy"
        )

        #  make all object indices unique and remove bg
        obj_ids = np.unique(mask)[1:]
        num_obj = len(obj_ids)
        masks = np.zeros((num_obj, 1000, 1000), dtype=np.uint8)
        boxes = []
        valid_masks_idxs = []
        # create separate binary masks and bboxes for each object
        for i in range(num_obj):
            masks[i][mask == obj_ids[i]] = 1
            masks[i] = cv.erode(masks[i], kernel=np.ones((6, 6)))
            masks[i] = cv.dilate(masks[i], kernel=np.ones((6, 6)))
            if np.sum(masks[i]) != 0:
                valid_masks_idxs.append(i)

        valid_masks = np.zeros((len(valid_masks_idxs), 1000, 1000), dtype=np.uint8)
        for i, valid_idx in enumerate(valid_masks_idxs):
            valid_masks[i] = masks[valid_idx]
            pos = np.where(valid_masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        masks = valid_masks

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_obj,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        # If there is no object on the scene,
        # this will be used to ignore that instance during coco eval
        iscrowd = torch.zeros((num_obj,), dtype=torch.int64)

        # this control is to solve 'loss is NaN error'
        # during training which can be a result of an invalid box
        keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes = boxes[keep]
        masks = masks[keep]

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd,
        }

        img_depth = T.ToTensor()(img_depth)

        return img_depth, target

    def __len__(self):
        return self.length
