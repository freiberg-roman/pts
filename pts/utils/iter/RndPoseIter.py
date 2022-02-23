import alr_sim.utils.geometric_transformation as gt
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class ObjectPose:
    pos: List[float]
    orientation: List[float]


class RndPoseIter:
    def __init__(self, limits, drop_heigth):
        self.drop_height = drop_heigth
        self.limits = limits

    def __iter__(self):
        return self

    def __next__(self):
        drop_x = (
            self.limits[0][1] - self.limits[0][0]
        ) * np.random.random_sample() + self.limits[0][0]
        drop_y = (
            self.limits[1][1] - self.limits[1][0]
        ) * np.random.random_sample() + self.limits[1][0]
        pos = [drop_x, drop_y, self.drop_height]
        quat = gt.euler2quat([
            2 * np.pi * np.random.random_sample(),
            2 * np.pi * np.random.random_sample(),
            2 * np.pi * np.random.random_sample(),
        ])
        return ObjectPose(pos, quat)
