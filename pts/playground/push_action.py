"""
Simple setting for performing predefined push actions
"""
import os
import xml.etree.ElementTree as Et

import alr_sim.utils.geometric_transformation as gt
import numpy as np
from alr_sim.sims.mujoco.FreezableMujocoEnvironment import FreezableMujocoEnvironment
from alr_sim.sims.mujoco.mj_utils.mujoco_scene_object import MujocoObject
from alr_sim.sims.mujoco.MujocoFactory import MujocoFactory
from alr_sim.sims.mujoco.MujocoLoadable import MujocoXmlLoadable
from alr_sim.sims.SimFactory import SimRepository
from alr_sim.sims.universal_sim.PrimitiveObjects import Box

base_path = "/home/roman/projects/SimulationFramework/models/mujoco/surroundings/"


def create_clutter(scene, robot, objects, limits, drop_height):
    # ### Dropping objets to table ###

    for i in len(objects):

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

        objects = freeze(scene, objects)  # should be a python context
        set_pos(objects[i], obj_pos, obj_orientation)
        scene, robot = unfreeze(scene, robot, objects)

        # wait ...
        for _ in range(256):
            robot.nextStep()

    # wait a little bit, once all the objects were dropped
    for _ in range(256):
        robot.nextStep()

    return scene, robot


def load_obj(order):
    obj_list = []
    for item in order:
        obj = MujocoXmlLoadable()
        obj.xml_file_path = (
            os.path.abspath(__file__)
            + "/../resources/obj/random_urdfs/"
            + "%03d" % item
            + "/"
            + "%03d.xml" % item
        )
        obj_list.append(obj)

    return obj_list


def freeze(scene, freezable):
    """
    Saves current scene to xml -> enables position setting
    Returns: nothing

    """
    file = open(base_path + "test.xml", mode="w")
    scene.sim.save(file, "xml")
    file.close()

    etree = Et.parse(base_path + "test.xml")
    for obj in freezable:

        root = etree.getroot()
        body_element = root.find(".//*[@name='%s']" % obj.name)

        pos_str = " ".join(map(str, scene._get_obj_pos(obj.name, obj)))
        obj.pos = scene._get_obj_pos(obj.name, obj)
        obj.quat = scene._get_obj_quat(obj.name, obj)
        quat_str = " ".join(map(str, scene._get_obj_quat(obj.name, obj)))

        body_element.set("pos", pos_str)
        body_element.set("quat", quat_str)

    etree.write(base_path + "test.xml")
    return freezable


def unfreeze(scene, robot, freezable):
    """
    Unfreeze basically reloads the whole scene -> does all the reconstruction
    Args:
        scene:

    Returns:

    """
    del scene, robot
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

    for obj in freezable:
        scene.add_object(obj)

    scene.start()
    return scene, robot


def set_pos(obj, pos, quat):
    etree = Et.parse(base_path + "test.xml")
    root = etree.getroot()
    pos_str = " ".join(map(str, pos))
    quat_str = " ".join(map(str, quat))

    body_element = root.find(".//*[@name='%s']" % obj.name)
    body_element.set("pos", pos_str)
    body_element.set("quat", quat_str)
    obj.quat = quat
    obj.pos = pos

    # ### WRITE BACK ###
    etree.write(base_path + "test.xml")


def create_scene():
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

    obj1 = MujocoObject(
        "rnd_0",
        [0.55, 0.0, 0.3],
        [0.0, 1.0, 0.0, 0.0],
        obj_path=os.path.dirname(os.path.abspath(__file__))
        + "/../../resources/obj/mujoco_xml/000/0.xml",
    )
    obj2 = MujocoObject(
        "rnd_1",
        [0.55, 0.0, 0.4],
        [0.0, 1.0, 0.0, 0.0],
        obj_path=os.path.dirname(os.path.abspath(__file__))
        + "/../../resources/obj/mujoco_xml/001/1.xml",
    )
    obj3 = MujocoObject(
        "rnd_2",
        [0.55, 0.0, 0.5],
        [0.0, 1.0, 0.0, 0.0],
        obj_path=os.path.dirname(os.path.abspath(__file__))
        + "/../../resources/obj/mujoco_xml/002/2.xml",
    )
    obj4 = MujocoObject(
        "rnd_3",
        [0.55, 0.0, 0.6],
        [0.0, 1.0, 0.0, 0.0],
        obj_path=os.path.dirname(os.path.abspath(__file__))
        + "/../../resources/obj/mujoco_xml/003/3.xml",
    )
    obj5 = MujocoObject(
        "rnd_4",
        [0.55, 0.0, 0.7],
        [0.0, 1.0, 0.0, 0.0],
        obj_path=os.path.dirname(os.path.abspath(__file__))
        + "/../../resources/obj/mujoco_xml/004/4.xml",
    )
    obj6 = MujocoObject(
        "rnd_5",
        [0.55, 0.0, 0.8],
        [0.0, 1.0, 0.0, 0.0],
        obj_path=os.path.dirname(os.path.abspath(__file__))
        + "/../../resources/obj/mujoco_xml/005/5.xml",
    )
    obj7 = MujocoObject(
        "rnd_6",
        [0.55, 0.0, 0.8],
        [0.0, 1.0, 0.0, 0.0],
        obj_path=os.path.dirname(os.path.abspath(__file__))
        + "/../../resources/obj/mujoco_xml/006/6.xml",
    )
    obj8 = MujocoObject(
        "rnd_7",
        [0.55, 0.0, 0.8],
        [0.0, 1.0, 0.0, 0.0],
        obj_path=os.path.dirname(os.path.abspath(__file__))
        + "/../../resources/obj/mujoco_xml/007/7.xml",
    )
    scene.add_object(obj1)
    scene.add_object(obj2)
    scene.add_object(obj3)
    scene.add_object(obj4)
    scene.add_object(obj5)
    scene.add_object(obj6)
    scene.add_object(obj7)
    scene.add_object(obj8)

    freezable = [obj1, obj2, obj3, obj4, obj5, obj6, obj7]

    scene.start()
    return scene, robot, freezable


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
    mj_factory = MujocoFactory()
    scene = mj_factory.create_scene()
    robot = mj_factory.create_robot(scene)
    obj_list = [
        SimRepository.get_factory("mujoco").create_camera(
            "cage_cam",
            384,
            512,
            [0.0, 0.0, 0.5 + 0.7],  # init pos.
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
    for o in obj_list:
        scene.add_object(o)
    scene.start()

    for i in range(2000):
        robot.nextStep()
    import glfw

    glfw.destroy_window(scene.viewer.viewer.window)
    scene.viewer.viewer = None
    scene.viewer = None
    scene = mj_factory.create_scene()
    robot = mj_factory.create_robot(scene)
    obj_list = [
        SimRepository.get_factory("mujoco").create_camera(
            "cage_cam",
            384,
            512,
            [0.0, 0.0, 0.5 + 0.7],  # init pos.
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
    for o in obj_list:
        scene.add_object(o)
    scene.start()

    for i in range(2000):
        robot.nextStep()


def run2():
    scene, robot, freezable = create_scene()
    robot.set_gripper_width = 0.0
    mj_freezable = FreezableMujocoEnvironment([])

    push_at(mj_freezable.robot, 0.5, 0.0)

    with mj_freezable as f:
        f.set_obj_pose(freezable[-1], [0.55, 0.0, 0.4], [0.0, 1.0, 0.0, 0.0])
        box = Box(
            name="simple_box",
            init_pos=[0.55, 0.0, 0.3],
            rgba=[1.0, 1.0, 1.0, 1.0],
            init_quat=[0.0, 1.0, 0.0, 0.0],
            size=[0.05, 0.05, 0.05],
        )
        f.add_obj_rt(box)

    push_at(mj_freezable.robot, 0.5, 0.0)


if __name__ == "__main__":
    run()
