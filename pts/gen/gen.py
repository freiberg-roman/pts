import os
import time

import alr_sim.utils.geometric_transformation as gt
import imageio
import numpy as np
import pybullet as p
from alr_sim.sims.pybullet.pb_utils.pybullet_scene_object import PyBulletObject
from alr_sim.sims.SimFactory import SimRepository
from alr_sim.sims.universal_sim.PrimitiveObjects import Box
from alr_sim.utils.unique_dict import UniqueDict
from omegaconf import DictConfig


def generate(cfg_gen: DictConfig):

    settings = cfg_gen.settings
    # ### Scene creation ###

    sim_factory = SimRepository.get_factory(settings.simulator)
    robot = sim_factory.create_robot()
    scene = sim_factory.create_scene(robot, object_list=[])
    cam = sim_factory.create_camera(
        "cage_cam",
        cfg_gen.data.cam_width,
        cfg_gen.data.cam_height,
        [0.7, 0.0, settings.drop_height + 0.7],  # init pos.
        gt.euler2quat([-np.pi * 7 / 8, 0, np.pi / 2]),
    )

    for i in range(settings.database_size):

        # ### This is a hack which will be changed in the future version ###

        scene.object_list = []
        scene.name2id_map = {}  # dict mapping names to obj_ids
        scene._objects = UniqueDict(
            err_msg="Duplicate object name:"
        )  # dict mapping names to object instances
        scene.add_object(cam)

        # ### Adding wall and platform ###
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

        scene.start()

        # ### Begin generation ###

        start_time = time.time()
        num_obj = np.random.random_integers(settings.min_num_obj, settings.max_num_obj)

        # ### Dropping objets to table ###

        object_order = []
        objects = np.random.random_integers(0, settings.total_num_obj - 1, num_obj)

        for idx, obj_id in enumerate(objects):
            obj_name = "shape_run_%03d_it_%06d" % (i, idx)

            drop_x = (
                settings.limits[0][1] - settings.limits[0][0]
            ) * np.random.random_sample() + settings.limits[0][0]
            drop_y = (
                settings.limits[1][1] - settings.limits[1][0]
            ) * np.random.random_sample() + settings.limits[1][0]
            obj_pos = [drop_x, drop_y, settings.drop_height]
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
                + cfg_gen.meshes.object_path
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

        # ### Saving image ###

        rgb, depth = scene.get_cage_cam().get_image(depth=True)
        seg = scene.get_cage_cam().get_segmentation(depth=True)

        print("saving images ...")
        if cfg_gen.data.save_numpy:
            os.makedirs(settings.path + "depth_img/numpy/", exist_ok=True)
            np.save(settings.path + "depth_img/numpy/" + "image_%06d.npy" % i, depth)
            os.makedirs(settings.path + "segmentation/numpy/", exist_ok=True)
            np.save(
                settings.path + "segmentation/numpy/" + "seg_mask_%06d.npy" % i, seg
            )
        if cfg_gen.data.save_png:
            os.makedirs(settings.path + "depth_img/png/", exist_ok=True)
            imageio.imsave(
                settings.path + "depth_img/png/" + "image_%06d.png" % i,
                depth.astype(np.uint8),
            )

            os.makedirs(settings.path + "segmentation/png/", exist_ok=True)
            imageio.imsave(
                settings.path + "segmentation/png/" + "seg_mask_id_%06d.png" % i,
                seg[0].astype(np.uint8),
            )
            imageio.imsave(
                settings.path + "segmentation/png/" + "seg_mask_value_%06d.png" % i,
                seg[1].astype(np.uint8),
            )
        if cfg_gen.data.save_color_img:
            os.makedirs(settings.path + "color_img/", exist_ok=True)
            imageio.imsave(settings.path + "color_img/" + "image_%06d.png" % i, rgb)

        # remove dropped objects from simulation
        print(
            "Iteration: "
            + str(i)
            + " finished in "
            + str(time.time() - start_time)
            + " ms."
        )

        # ### Restoring state ###
        p.resetSimulation(scene.physics_client_id)
