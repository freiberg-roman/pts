import os

import numpy as np
from alr_sim.sims.pybullet.pb_utils.pybullet_scene_object import PyBulletObject


def create_clutter(
    scene, robot, min, max, total, limits, drop_height, obj_path, iter=0
):
    num_obj = np.random.random_integers(min, max)

    # ### Dropping objets to table ###

    object_order = []
    objects = np.random.random_integers(0, total - 1, num_obj)

    for idx, obj_id in enumerate(objects):
        obj_name = "shape_run_%03d_it_%06d" % (iter, idx)

        drop_x = (limits[0][1] - limits[0][0]) * np.random.random_sample() + limits[0][
            0
        ]
        drop_y = (limits[1][1] - limits[1][0]) * np.random.random_sample() + limits[1][
            0
        ]
        obj_pos = [drop_x, drop_y, drop_height]
        obj_orientation = [
            2 * np.pi * np.random.random_sample(),
            2 * np.pi * np.random.random_sample(),
            2 * np.pi * np.random.random_sample(),
        ]

        pb_obj = PyBulletObject(
            urdf_name="%03d" % obj_id,
            object_name=obj_name,
            position=obj_pos,
            orientation=obj_orientation,
            data_dir=os.path.dirname(os.path.abspath(__file__))
            + "/../../"
            + obj_path
            + "%03d" % obj_id,
        )

        print("dropping  -->", pb_obj)
        object_order.append(["%03d.urdf" % idx, obj_pos, obj_orientation])
        scene.add_object(pb_obj)

        # wait ...
        for _ in range(256):
            robot.nextStep()

    # wait a little bit, once all the objects were dropped
    for _ in range(4096):
        robot.nextStep()
