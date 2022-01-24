import alr_sim.utils.geometric_transformation as gt
import numpy as np
import pybullet as p
from alr_sim.sims.SimFactory import SimRepository
from alr_sim.sims.universal_sim.PrimitiveObjects import Box
from alr_sim.utils.geometric_transformation import quat2mat
from alr_sim.utils.unique_dict import UniqueDict

from pts.utils.sim_helper import create_clutter


class TrainScene:
    def __init__(self, settings):
        self.min_obj = settings.min_number_objects
        self.max_obj = settings.max_number_objects
        self.total_num = settings.total_number_objects
        self.ws_limits = settings.workspace_limit
        self.drop_height = settings.drop_height
        self.path = settings.object_path

        # ### Scene creating ###

        sim_factory = SimRepository.get_factory(settings.simulator)
        self.robot = sim_factory.create_robot()
        self.scene = sim_factory.create_scene(self.robot, object_list=[])
        self.cam = sim_factory.create_camera(
            "cage_cam",
            settings.data.cam_width,
            settings.data.cam_height,
            [0.7, 0.0, self.drop_height + 0.7],  # init pos.
            gt.euler2quat([-np.pi * 7 / 8, 0, np.pi / 2]),
        )
        self.scene.object_list = []
        self.scene.name2id_map = {}  # dict mapping names to obj_ids
        self.scene._objects = UniqueDict(
            err_msg="Duplicate object name:"
        )  # dict mapping names to object instances
        self.scene.add_object(self.cam)

        pos, rot_quat = self.cam.get_cart_pos_quat()
        self.cam_rot_mat = quat2mat(rot_quat)

        cam_trans = np.eye(4, 4)
        cam_trans[0:3, 3] = np.asarray(pos)
        rotation = np.eye(4, 4)
        rotation[0:3, 0:3] = self.cam_rot_mat

        self.cam_pose = np.dot(cam_trans, rotation)
        self.cam_intrinsics = ((self.cam.fx, self.cam.cx), (self.cam.fy, self.cam.cy))

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
        self._duration = 2

    def create_testing_scenario(self, it=0):
        # note: For real application this needs to be reimplemented

        # ### Restoring state ###
        p.resetSimulation(self.scene.physics_client_id)  # hack version

        self.scene.object_list = []
        self.scene.name2id_map = {}  # dict mapping names to obj_ids
        self.scene._objects = {}
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
        self._origin_pos = self.robot.current_j_pos

        # ### Generation ###

        create_clutter(
            self.scene,
            self.robot,
            self.min_obj,
            self.max_obj,
            self.total_num,
            self.ws_limits,
            self.drop_height,
            self.path,
            it,
        )

    def get_camera_data(self):
        color, depth = self.scene.get_cage_cam().get_image(depth=True)
        return color, depth

    def get_data_mask_rg(self):
        gt = self.scene.get_cage_cam().get_segmentation()[0]
        return gt, -1

    def move_to(
        self, tool_position, tool_orientation, beam=False, duration=None, nsteps=1
    ):
        tool_orientation = 0 if tool_position is None else tool_orientation
        duration = self._duration if duration is None else duration
        desiredEul = [np.pi + tool_orientation, -np.pi / 2, np.pi]
        desiredQuat = np.round(np.array(p.getQuaternionFromEuler(desiredEul)), 5)

        if beam:
            self.robot.beam_to_cart_pos_and_quat(
                desiredPos=tool_position, desiredQuat=desiredQuat
            )
            return

        curp = self.robot.current_c_pos
        interp = np.linspace(curp, tool_position, nsteps)
        for i in range(interp.shape[0]):
            self.robot.gotoCartPositionAndQuat(
                desiredPos=interp[i, :],
                desiredQuat=desiredQuat,
                use_fictive=True,
                duration=duration / nsteps,
            )
        self.robot.gotoCartPositionAndQuat(
            desiredPos=tool_position,
            desiredQuat=desiredQuat,
            use_fictive=True,
            duration=duration / 2,
        )

    def push(self, pos, heightmap_rotation_angle):
        return
        tool_rotation_angle = (heightmap_rotation_angle % np.pi) - np.pi / 2
        pos[2] += 0.1

        push_orientation = [1.0, 0.0]
        push_direction = np.asarray(
            [
                push_orientation[0] * np.cos(heightmap_rotation_angle)
                - push_orientation[1] * np.sin(heightmap_rotation_angle),
                push_orientation[0] * np.sin(heightmap_rotation_angle)
                + push_orientation[1] * np.cos(heightmap_rotation_angle),
            ]
        )

        pushing_point_margin = 0.15
        location_above_pushing_point = (pos[0], pos[1], pos[2] + pushing_point_margin)

        # goto position before pushing
        self.move_to(location_above_pushing_point, tool_rotation_angle, beam=True)

        # close gripper
        self.robot.set_gripper_width = 0.0

        # prepare for push
        self.move_to(pos, tool_rotation_angle, duration=self._duration)
        push_length = 0.1
        target_x = min(
            max(pos[0] + push_direction[0] * push_length, self.ws_limits[0][0]),
            self.ws_limits[0][1],
        )
        target_y = min(
            max(pos[1] + push_direction[1] * push_length, self.ws_limits[1][0]),
            self.ws_limits[1][1],
        )

        # perform push
        self.move_to(
            [target_x, target_y, pos[2]],
            tool_rotation_angle,
            duration=self._duration,
        )

        # Done --> go back to staring position
        self.robot.beam_to_joint_pos(self._origin_pos)
