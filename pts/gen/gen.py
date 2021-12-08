import os
import time

import alr_sim.utils.geometric_transformation as gt
import imageio
import numpy as np
from alr_sim.sims.SimFactory import SimRepository
from omegaconf import DictConfig


def generate(cfg_gen: DictConfig):

    # ### Scene creation ###

    settings = cfg_gen.settings

    sim_factory = SimRepository.get_factory(settings.simulator)
    robot = sim_factory.create_robot()
    scene = sim_factory.create_scene(robot, object_list=[])  # TODO add objects
    cam = sim_factory.create_camera(
        "cage_cam",
        cfg_gen.data.cam_width,
        cfg_gen.data.cam_height,
        [0.7, 0.0, settings.drop_height + 0.7],  # init pos.
        gt.euler2quat([-np.pi * 7 / 8, 0, np.pi / 2]),
    )  # init rot.
    scene.add_object(cam)

    # ### Adding bin ###

    # TODO

    scene.start()

    # ### Generation loop ###

    for i in range(settings.database_size):
        start_time = time.time()
        num_obj = np.random.random_integers(settings.min_num_obj, settings.may_num_obj)

        # ### Dropping objets to table ###

        object_order = []
        objects = np.random.random_integers(0, settings.total_num_obj - 1, num_obj)

        for obj_id in objects:
            mesh_path = os.path.join(
                cfg_gen.meshes.object_path, "%03d/%03d.urdf" % (obj_id, obj_id)
            )
            shape_name = "shape_%03d_it_%06d" % (obj_id, i)

            drop_x = (
                settings.limits[0][1] - settings.limits[0][0]
            ) * np.random.random_sample() + settings.limits[0][0]
            drop_y = (
                settings.limits[1][1] - settings.limits[1][0]
            ) * np.random.random_sample() + settings.limits[1][0]
            obj_pos = [drop_x, drop_y, settings.drop_height]
            obj_angle = [
                2 * np.pi * np.random.random_sample(),
                2 * np.pi * np.random.random_sample(),
                2 * np.pi * np.random.random_sample(),
            ]

            print("dropping  -->", mesh_path)
            object_order.append(["%03d.urdf" % i, obj_pos, obj_angle])
            shape = scene.load_object_to_scene(
                path_to_urdf=mesh_path,
                position=obj_pos,
                orientation=obj_angle,
                id_name=shape_name,
            )
            print(shape)

            # wait ...
            for i in range(64):
                robot.nextStep()

        # wait a little bit, once all the objects were dropped
        for i in range(256):  # 4096
            robot.nextStep()

        # ### Saving image ###

        rgb, depth, seg = scene.get_cage_cam().get_image("cage_cam")

    print("saving scene info...")
    np.save(object_order + "scene_info_%06d.npy" % i, object_order)

    print("saving images ...")
    if cfg_gen.data.save_numpy:
        np.save(settings.path + "depth_img/numpy/" + "image_%06d.npy" % i, depth)
        np.save(settings.path + "segmentation/numpy/" + "seg_mask_%06d.npy" % i, seg)
    if cfg_gen.data.save_png:
        imageio.imsave(
            settings.path + "depth_img/png/" + "image_%06d.png" % i,
            depth.astype(np.uint8),
        )
        imageio.imsave(settings.path + "segmentation/png/" + "seg_mask_%06d.png" % i, seg.astype(np.uint8))
    if cfg_gen.data.save_color_img:
        imageio.imsave(settings.path + "color_img/" + "image_%06d.png" % i, rgb)

    # remove dropped objects from simulation
    print("Reset the scene")
    scene.reset()
    print("Iteration: " + i + " finished in " + time.time() - start_time + " ms.")
