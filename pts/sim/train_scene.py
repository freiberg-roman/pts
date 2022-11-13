import os.path

import alr_sim.utils.geometric_transformation as gt
import numpy as np
import pybullet as p
from alr_sim.sims.mj_beta import MjScene
from alr_sim.sims.SimFactory import SimRepository
from alr_sim.sims.universal_sim.PrimitiveObjects import Box
from alr_sim.utils.geometric_transformation import quat2mat

from pts.utils.iter.RndObjectIter import RndMJObjectIter, RndPoseIter


class TrainScene:
    def __init__(self, cfg):
        self.min_obj = cfg.min_number_objects
        self.max_obj = cfg.max_number_objects
        self.total_num = cfg.total_number_objects
        self.ws_limits = cfg.workspace_limit
        self.drop_height = cfg.drop_height
        self.path = cfg.object_path
        self.cam_width = cfg.data.cam_width
        self.cam_height = cfg.data.cam_height

        self.scene = None
        self.robot = None
        self.cam_intrinsics = None
        self.glob_path = os.path.dirname(os.path.abspath(__file__))
        self.obj_path = cfg.object_path
        self._duration = 2

    def create_testing_scenario(self, it=0):
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
                init_pos=[0.6, 0.0, 0.08],
                rgba=[1.0, 1.0, 1.0, 1.0],
                init_quat=[0.0, 1.0, 0.0, 0.0],
                size=[0.6, 1.0, 0.005],
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

        self.scene: MjScene = sim_factory.create_scene(object_list=obj, dt=0.001)
        self.robot = sim_factory.create_robot(self.scene)
        self.scene.start()

        gen_obj_pose = RndPoseIter(limits=self.ws_limits, drop_heigth=0.2)
        gen_obj_iter = RndMJObjectIter(
            min=self.min_obj, max=self.min_obj, pose_generator=gen_obj_pose
        )

        # Drop objects onto the scene table
        for new_obj in gen_obj_iter:
            self.scene.add_object_rt(new_obj)

            for _ in range(200):
                self.scene.next_step()

        self.cam_intrinsics = (
            (
                self.scene.get_cage_cam().fx,
                self.scene.get_cage_cam().cx,
            ),
            (
                self.scene.get_cage_cam().fy,
                self.scene.get_cage_cam().cy,
            ),
        )

    def get_camera_data(self):
        rgb, depth = self.scene.get_cage_cam().get_image(depth=True)
        return rgb, depth

    def get_point_cloud(self):
        points, colors = self.scene.get_cage_cam().calc_point_cloud()
        return points, colors

    def get_object_segmentation(self):
        seg = self.scene.get_cage_cam().get_segmentation(
            height=1000, width=1000, depth=False
        )

        # ### Make segmentations labels unique ###
        obj_ids = np.unique(seg)
        num_obj = len(obj_ids) - 1  # remove background
        for i in range(num_obj):
            seg[seg == obj_ids[i]] = i

        return seg, num_obj

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

    def beam_back(self):
        ...

    def push_at(self, x, y, height=0.5, intermediate_steps=5):
        # goto position above
        self.robot.gotoCartPositionAndQuat(
            desiredPos=[x, y, height], desiredQuat=[0, 1, 0, 0], duration=4.0
        )

        # straight push down
        interp = np.linspace(self.robot.current_c_pos, [x, y, 0.0], intermediate_steps)
        for i in range(interp.shape[0]):
            self.robot.gotoCartPositionAndQuat(
                desiredPos=interp[i, :],
                desiredQuat=[0, 1, 0, 0],
                duration=6.0 / intermediate_steps,
            )
        self.robot.gotoCartPositionAndQuat(
            desiredPos=[x, y, 0.0],
            desiredQuat=[0, 1, 0, 0],
            duration=2.0,
        )

        # go back to position above
        self.robot.gotoCartPositionAndQuat(
            desiredPos=[x, y, height], desiredQuat=[0, 1, 0, 0], duration=4.0
        )

    def get_cam_pose(self):
        pos, rot_quat = self.scene.get_cage_cam().get_cart_pos_quat()
        self.cam_rot_mat = quat2mat(rot_quat)

        cam_trans = np.eye(4, 4)
        cam_trans[0:3, 3] = np.asarray(pos)
        rotation = np.eye(4, 4)
        rotation[0:3, 0:3] = self.cam_rot_mat

        self.cam_pose = np.dot(cam_trans, rotation)
        return self.cam_pose
