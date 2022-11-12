import os
import xml.etree.ElementTree as Et
from dataclasses import dataclass
from typing import List, Tuple

import alr_sim.utils.geometric_transformation as gt
import numpy as np
from alr_sim.sims.mj_beta.mj_utils.mj_helper import IncludeType
from alr_sim.sims.mj_beta.MjLoadable import MjFreezable, MjLoadable


class TemplateObject(MjLoadable, MjFreezable):
    def __init__(
        self,
        unique_name: str,
        id: int,
        init_pos: Tuple[float, float, float],
        init_quat: Tuple[float, float, float, float],
        rgba: Tuple[float, float, float, float],
    ):
        assert 0 <= id < 8

        self.name = unique_name
        self.id = id
        self.init_pos = init_pos
        self.init_quat = init_quat
        self.rgba = rgba
        self.asset_dir = os.path.join(
            os.path.dirname(__file__), "../../../resources/assets/"
        )
        self.xml_include_template: str = """
        <mujocoinclude>
            <compiler angle="radian"/>
            <size njmax="500" nconmax="100" />
            <asset>
                <mesh name="{name}_mesh" file="{id}.stl" />
            </asset>
            <worldbody>
                <body name="{name}" pos="{pos}" quat="{quat}">
                    <freejoint name="{name}:joint" />
                    <geom quat="0.707107 0 -0.707107 0" rgba="{rgba}" mesh="{name}_mesh" type="mesh" solimp="0.998 0.998 0.001" solref="0.001 1" density="50" friction="0.95 0.3 0.1" condim="4"/>
                </body>
            </worldbody>
        </mujocoinclude>
        """

    def mj_load(self) -> Tuple[Et.Element, list, IncludeType]:
        xml_include = self.xml_include_template.format(
            name=self.name,
            id=self.id,
            pos=" ".join(map(str, self.init_pos)),
            quat=" ".join(map(str, self.init_quat)),
            rgba=" ".join(map(str, self.rgba)),
        )
        assets = {}
        with open(os.path.join(self.asset_dir, str(self.id) + ".stl"), "rb") as file:
            assets[str(self.id) + ".stl"] = file.read()
        return Et.fromstring(xml_include), assets, IncludeType.MJ_INCLUDE

    def freeze(self, model, data) -> None:
        pos: np.ndarray = data.jnt(self.name + ":joint").qpos
        self.init_pos = pos[0:3].tolist()
        self.init_quat = pos[3:].tolist()


@dataclass
class ObjectPose:
    pos: List[float]
    orientation: List[float]


class RndPoseIter:
    def __init__(self, limits, drop_heigth):
        self.drop_height = drop_heigth
        self.limits = limits

    def __iter__(self):
        return self

    def __next__(self):
        drop_x = (
            self.limits[0][1] - self.limits[0][0]
        ) * np.random.random_sample() + self.limits[0][0]
        drop_y = (
            self.limits[1][1] - self.limits[1][0]
        ) * np.random.random_sample() + self.limits[1][0]
        pos = [drop_x, drop_y, self.drop_height]
        quat = gt.euler2quat(
            [
                2 * np.pi * np.random.random_sample(),
                2 * np.pi * np.random.random_sample(),
                2 * np.pi * np.random.random_sample(),
            ]
        )
        return ObjectPose(pos, quat)


class RndMJObjectIter:
    def __init__(self, min: int, max: int, pose_generator: RndPoseIter):
        self.num = -1
        self.num_object = np.random.randint(min, max + 1)
        self.generator = pose_generator

    def __iter__(self):
        return self

    def __next__(self):
        self.num += 1
        name = "shape_{number}".format(number=self.num)
        id = np.random.randint(0, 7 + 1)
        rgba = np.random.random_sample(4)
        rgba[3] = 1.0
        pose = next(self.generator)
        if self.num >= self.num_object:
            raise StopIteration

        return TemplateObject(
            unique_name=name,
            id=id,
            init_pos=pose.pos,
            init_quat=pose.orientation,
            rgba=rgba.tolist(),
        )
