import numpy as np
from alr_sim.sims.mujoco.mj_utils.mujoco_scene_object import MujocoObject


class RndMJObjectIter:
    def __init__(self, min, max, total, path, idx=0):
        self.num = -1
        self.min = min
        self.max = max
        self.total = total
        self.path = path
        self.idx = idx
        self.num_object = np.random.random_integers(min, max)
        self.obj_list = np.random.random_integers(0, total - 1, self.num_object)

    def __iter__(self):
        return self

    def __next__(self):
        self.num += 1
        name = "shape_run_%03d_it_%06d" % (self.idx, self.num)
        if self.num >= self.num_object:
            raise StopIteration

        return MujocoObject(
            name,
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            obj_path=self.path
            + "%03d/%d.xml" % (self.obj_list[self.num], self.obj_list[self.num]),
        )
