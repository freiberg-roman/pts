"""
Simple setting for performing predefined push actions
"""
import os

import alr_sim.utils.geometric_transformation as gt
import numpy as np
import pybullet as p
from alr_sim.core.Logger import RobotPlotFlags
from alr_sim.sims.mujoco.MujocoCamera import MujocoCamera
from alr_sim.sims.mujoco.MujocoFactory import MujocoFactory
from alr_sim.sims.mujoco.MujocoRobot import MujocoRobot
from alr_sim.sims.mujoco.MujocoScene import MujocoScene
from alr_sim.sims.SimFactory import SimRepository
from alr_sim.sims.mujoco.mj_utils.mujoco_scene_object import MujocoObject
from alr_sim.sims.universal_sim.PrimitiveObjects import Box
from alr_sim.utils.geometric_transformation import quat2mat
from alr_sim.sims.mujoco.MujocoLoadable import MujocoXmlLoadable

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
    for _ in range(256):
        robot.nextStep()

def load_obj(order):
    obj_list = []
    for item in order:
        obj = MujocoXmlLoadable()
        obj.xml_file_path = os.path.abspath(__file__) + "/../resources/obj/random_urdfs/" + "%03d" % item + "/" + "%03d.xml" % item
        obj_list.append(obj)

    return obj_list

def create_scene():
    num_obj = np.random.random_integers(0, 4)
    objects_order = np.random.random_integers(0, 3 - 1, num_obj)

    mujoco_factory = MujocoFactory()
    scene = mujoco_factory.create_scene()
    robot = mujoco_factory.create_robot(scene)
    cam = mujoco_factory.create_camera(
        "cage_cam",
        512,
        384,
        [0.7, 0.0, 0.5 + 0.7],
        gt.euler2quat([-np.pi * 7 / 8, 0, np.pi / 2]),
    )
    scene.add_object(cam)

    scene.add_object(
        Box(
            name="drop_zone",
            init_pos=[0.6, 0.0, -0.01],
            rgba=[1.0, 1.0, 1.0, 1.0],
            init_quat=[0.0, 1.0, 0.0, 0.0],
            size=[0.6, 0.6, 0.005],
            static=True,
        )
    )
    scene.add_object(
        Box(
            name="wall",
            init_pos=[1.2, 0.0, 0.35],
            rgba=[1.0, 1.0, 1.0, 0.1],
            init_quat=[0.0, 1.0, 0.0, 0.0],
            size=[0.005, 1.2, 0.4],
            static=True,
        )
    )

    scene.add_object(
        MujocoObject("test",
                     [0.3, 0.0, 0.3],
                     [0.0, 1.0, 0.0, 0.0],
                     obj_path=os.path.dirname(os.path.abspath(__file__)) + "/../../resources/obj/random_urdfs/000/000.xml"
                     )
    )

    scene.start()
    return scene, robot


def push_at(robot, x, y, height=0.5, intermediate_steps=15):
    # goto position above
    robot.gotoCartPositionAndQuat(
        desiredPos=[x, y, height], desiredQuat=[0, 1, 0, 0], duration=4.0
    )

    # straight push down
    interp = np.linspace(robot.current_c_pos, [x, y, 0.0], intermediate_steps)
    for i in range(interp.shape[0]):
        robot.gotoCartPositionAndQuat(
            desiredPos=interp[i, :],
            desiredQuat=[0, 1, 0, 0],
            duration=4.0 / intermediate_steps,
        )
    robot.gotoCartPositionAndQuat(
        desiredPos=[x, y, 0.0],
        desiredQuat=[0, 1, 0, 0],
        duration=2.0,
    )

    # go back to position above
    robot.gotoCartPositionAndQuat(
        desiredPos=[x, y, height], desiredQuat=[0, 1, 0, 0], duration=4.0
    )


def run():
    # ### Creating scene ###
    # workspace limits are x: [0.3, 0.9], y: [-0.6, 0.6]
    scene, robot = create_scene()
    robot.set_gripper_width = 0.0

    # ### Create clutter ###

    # ### Perform push ###
    push_at(robot, 0.5, 0.0)

    # ### Save positions of objects ###

    # ### Reset scene ###
    scene.reset()


if __name__ == "__main__":
    run()
