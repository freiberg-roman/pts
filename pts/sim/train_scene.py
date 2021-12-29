import alr_sim.utils.geometric_transformation as gt
import numpy as np
import pybullet as p
from alr_sim.sims.SimFactory import SimRepository
from alr_sim.sims.universal_sim.PrimitiveObjects import Box
from alr_sim.utils.unique_dict import UniqueDict
from utils.sim_helper import create_clutter


class TrainScene:
    def __init__(self, settings):
        self.ws_limits = settings.workspace_limits
        self.min_obj = settings.min_num_obj
        self.max_obj = settings.max_num_obj
        self.total_num = settings.total_num_obj
        self.drop_limits = settings.drop_limits
        self.drop_height = settings.drop_height

        # ### Scene creating ###

        sim_factory = SimRepository.get_factory(settings.simulator)
        robot = sim_factory.create_robot()
        self.scene = sim_factory.create_scene(robot, object_list=[])
        self.cam = sim_factory.create_camera(
            "cage_cam",
            settings.cam_width,
            settings.cam_height,
            [0.7, 0.0, settings.drop_height + 0.7],  # init pos.
            gt.euler2quat([-np.pi * 7 / 8, 0, np.pi / 2]),
        )
        self.scene.object_list = []
        self.scene.name2id_map = {}  # dict mapping names to obj_ids
        self.scene._objects = UniqueDict(
            err_msg="Duplicate object name:"
        )  # dict mapping names to object instances
        self.scene.add_object(self.cam)

        # ### Adding wall and platform ###
        self.scene.add_object(
            Box(
                name="drop_zone",
                init_pos=[0.6, 0.0, -0.01],
                rgba=[1.0, 1.0, 1.0, 1.0],
                init_quat=[0.0, 1.0, 0.0, 0.0],
                size=[0.6, 0.6, 0.005],
                static=True,
            )
        )
        self.scene.add_object(
            Box(
                name="wall",
                init_pos=[1.2, 0.0, 0.35],
                rgba=[1.0, 1.0, 1.0, 0.1],
                init_quat=[0.0, 1.0, 0.0, 0.0],
                size=[0.005, 1.2, 0.4],
                static=True,
            )
        )
        self.scene.start()

    def create_testing_scenario(self, it=0):
        # note: For real application this needs to be reimplemented
        # ### Restoring state ###
        p.resetSimulation(self.scene.physics_client_id)

        self.scene.object_list = []
        self.scene.name2id_map = {}  # dict mapping names to obj_ids
        self.scene._objects = UniqueDict(
            err_msg="Duplicate object name:"
        )  # dict mapping names to object instances
        self.scene.add_object(self.cam)

        # ### Adding wall and platform ###
        self.scene.add_object(
            Box(
                name="drop_zone",
                init_pos=[0.6, 0.0, -0.01],
                rgba=[1.0, 1.0, 1.0, 1.0],
                init_quat=[0.0, 1.0, 0.0, 0.0],
                size=[0.6, 0.6, 0.005],
                static=True,
            )
        )
        self.scene.add_object(
            Box(
                name="wall",
                init_pos=[1.2, 0.0, 0.35],
                rgba=[1.0, 1.0, 1.0, 0.1],
                init_quat=[0.0, 1.0, 0.0, 0.0],
                size=[0.005, 1.2, 0.4],
                static=True,
            )
        )

        self.scene.start()

        # ### Generation ###

        create_clutter(
            self.scene,
            self.robot,
            self.min_obj,
            self.max_obj,
            self.total_num,
            self.drop_limits,
            self.drop_height,
            self.path,
            it,
        )
