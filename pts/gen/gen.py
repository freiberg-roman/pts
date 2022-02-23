import os
import time

import cv2 as cv
import matplotlib.pyplot as plt

import alr_sim.utils.geometric_transformation as gt
import imageio
import numpy as np
import pybullet as p
from alr_sim.sims.mujoco.FreezableMujocoEnvironment import FreezableMujocoEnvironment
from alr_sim.sims.SimFactory import SimRepository
from alr_sim.sims.universal_sim.PrimitiveObjects import Box
from alr_sim.utils.unique_dict import UniqueDict
from omegaconf import DictConfig

from pts.utils.iter.RndObjectIter import RndMJObjectIter
from pts.utils.iter.RndPoseIter import RndPoseIter
from pts.utils.sim_helper import create_clutter


def generate(cfg_gen: DictConfig):

    # ### Scene creation ###

    sim_factory = SimRepository.get_factory(cfg_gen.simulator)
    scene = sim_factory.create_scene()
    robot = sim_factory.create_robot(scene)
    cam = sim_factory.create_camera(
        "cage_cam",
        cfg_gen.data.cam_width,
        cfg_gen.data.cam_height,
        [0.0, 0.0, cfg_gen.drop_height + 0.7],  # init pos.
        gt.euler2quat([-np.pi * 7 / 8, 0, np.pi / 2]),
    )

    drop_zone = Box(
        name="drop_zone",
        init_pos=[0.6, 0.0, -0.01],
        rgba=[1.0, 1.0, 1.0, 1.0],
        init_quat=[0.0, 1.0, 0.0, 0.0],
        size=[0.6, 0.6, 0.005],
        static=True,
    )

    wall = Box(
        name="wall",
        init_pos=[1.2, 0.0, 0.35],
        rgba=[1.0, 1.0, 1.0, 0.1],
        init_quat=[0.0, 1.0, 0.0, 0.0],
        size=[0.005, 1.2, 0.4],
        static=True,
    )
    freezable = FreezableMujocoEnvironment(scene, robot, [])
    glob_path = os.path.dirname(os.path.abspath(__file__))


    # ### Main Generation Loop ###
    for i in range(cfg_gen.session_limit):
        scene = sim_factory.create_scene()
        robot = sim_factory.create_robot(scene)
        scene.add_object(cam)
        scene.add_object(drop_zone)
        scene.add_object(wall)

        # ### Start Simulation ###
        scene.start()

        # create freezable context
        freezable = FreezableMujocoEnvironment(scene, robot, [])

        # create object generator
        gen_obj_iter = RndMJObjectIter(
            cfg_gen.min_number_objects,
            cfg_gen.max_number_objects,
            cfg_gen.total_number_objects,
            glob_path + "/../../" + cfg_gen.object_path,
            idx=i,
        )

        gen_obj_pose = RndPoseIter(cfg_gen.workspace_limit, cfg_gen.drop_height)
        start_time = time.time()
        for new_obj, pose in zip(gen_obj_iter, gen_obj_pose):

            with freezable as f:
                f.add_obj_rt(new_obj)
                f.set_obj_pose(new_obj, pose.pos, pose.orientation)

            # wait a bit
            for _ in range(2000):
                freezable.robot.nextStep()

            # ### This is a hack which will be changed in the future version ###

            # ### Saving image ###

        rgb, depth = freezable.scene.get_cage_cam().get_image(depth=True)

        print("saving images ...")
        plt.imshow(rgb), plt.show()
        cv.imwrite(
             cfg_gen.path + "%d.png" % i,
                cv.cvtColor(rgb, cv.COLOR_RGB2BGR),
        )
        # remove dropped objects from simulation
        print(
            "Iteration: "
            + str(i)
            + " finished in "
            + str(time.time() - start_time)
            + " ms."
        )