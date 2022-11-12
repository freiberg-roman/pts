import os
import time
from pathlib import Path

import alr_sim.utils.geometric_transformation as gt
import cv2 as cv
import numpy as np
from alr_sim.sims.mj_beta import MjRobot, MjScene
from alr_sim.sims.SimFactory import SimRepository
from alr_sim.sims.universal_sim.PrimitiveObjects import Box

from pts.utils.iter.RndObjectIter import RndMJObjectIter, RndPoseIter


def generate(sessions: int, save_to: str):

    # ### Scene creation ###

    glob_path = os.path.dirname(os.path.abspath(__file__))
    # for type in ["depth", "seg", "rgb"]:
    #     Path(save_to + type).mkdir(parents=True, exist_ok=True)

    # ### Main Generation Loop ###
    for i in range(sessions):
        start_time = time.time()

        # create environment
        sim_factory = SimRepository.get_factory("mj_beta")
        obj = [
            sim_factory.create_camera(
                "cage_cam",
                512,  # width
                384,  # height
                [0.0, 0.0, 0.7 + 0.5],  # init pos.
                gt.euler2quat([-np.pi * 7 / 8, 0, np.pi / 2]),
            ),
            Box(
                name="drop_zone",
                init_pos=[0.6, 0.0, -0.01],
                rgba=[1.0, 1.0, 1.0, 1.0],
                init_quat=[0.0, 1.0, 0.0, 0.0],
                size=[0.6, 0.6, 0.005],
                static=True,
            ),
            Box(
                name="wall",
                init_pos=[1.2, 0.0, 0.35],
                rgba=[1.0, 1.0, 1.0, 0.1],
                init_quat=[0.0, 1.0, 0.0, 0.0],
                size=[0.005, 1.2, 0.4],
                static=True,
            ),
        ]

        scene: MjScene = sim_factory.create_scene(object_list=obj, dt=0.001)
        sim_factory.create_robot(scene)
        scene.start()

        gen_obj_pose = RndPoseIter(
            limits=[[0.55, 0.998], [-0.224, 0.224], [0.02, 0.62]], drop_heigth=0.2
        )
        gen_obj_iter = RndMJObjectIter(min=10, max=20, pose_generator=gen_obj_pose)

        # Drop objects onto the scene table
        for new_obj in gen_obj_iter:
            scene.add_object_rt(new_obj)

            for _ in range(200):
                scene.next_step()

        # ### Saving image ###

        print("saving images ...")
        rgb, depth = scene.get_cage_cam().get_image(depth=True)
        seg_img = scene.get_cage_cam().get_segmentation(
            height=1000, width=1000, depth=False
        )
        cv.imwrite(
            save_to + "rgb/" + "%d.png" % i,
            cv.cvtColor(rgb, cv.COLOR_RGB2BGR),
        )
        np.save(save_to + "depth/" + "%d.npy" % i, depth)
        np.save(save_to + "seg/" + "%d.npy" % i, seg_img)
        print(
            "Iteration: "
            + str(i + 1)
            + " finished in "
            + str(time.time() - start_time)
            + " s."
        )
        scene.reset()
