"""
environment.py — PyBullet scene with 8 obstacles.

Two "blocker" obstacles sit directly on the straight-line EEF path between
Q_START and Q_GOAL, forcing the RRT to explore around them. Six more
surround the workspace to create the cluttered canopy.
"""
import numpy as np
import pybullet as p
import pybullet_data
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Obstacle:
    position: np.ndarray
    radius:   float
    shape:    str
    height:   float = 0.0
    body_id:  int   = -1


OBSTACLE_DEFS: List[Tuple] = [
    # Two blockers on the direct EEF path (forces RRT to detour)
    ( 0.38,  0.12,  0.52, 0.09, "sphere",   0.0),
    ( 0.44, -0.10,  0.55, 0.09, "sphere",   0.0),
    # Surrounding canopy obstacles
    ( 0.55,  0.25,  0.35, 0.06, "sphere",   0.0),
    ( 0.55, -0.25,  0.35, 0.06, "sphere",   0.0),
    ( 0.65,  0.15,  0.50, 0.06, "sphere",   0.0),
    ( 0.65, -0.15,  0.50, 0.06, "sphere",   0.0),
    ( 0.45,  0.30,  0.65, 0.05, "cylinder", 0.12),
    ( 0.45, -0.30,  0.65, 0.05, "cylinder", 0.12),
]

Q_START = np.array([ 0.0,  -0.785,  0.0, -2.356,  0.0,  1.571,  0.785])
Q_GOAL  = np.array([ 0.5,  -0.400,  0.2, -1.800,  0.3,  1.600,  0.800])


class Environment:
    def __init__(self, gui=False):
        self.gui = gui
        self.obstacles: List[Obstacle] = []
        self.obstacle_ids = []
        self._connect()
        self._load_scene()
        self._spawn_obstacles()

    def _connect(self):
        self.client = p.connect(p.GUI if self.gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(),
                                  physicsClientId=self.client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)

    def _load_scene(self):
        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client)
        self.robot_id = p.loadURDF(
            "franka_panda/panda.urdf", basePosition=[0, 0, 0],
            useFixedBase=True, physicsClientId=self.client)

    def _spawn_obstacles(self):
        for x, y, z, r, shape, h in OBSTACLE_DEFS:
            if shape == "sphere":
                col = p.createCollisionShape(p.GEOM_SPHERE, radius=r,
                                             physicsClientId=self.client)
                vis = p.createVisualShape(p.GEOM_SPHERE, radius=r,
                                          rgbaColor=[0.85, 0.25, 0.15, 0.7],
                                          physicsClientId=self.client)
            else:
                col = p.createCollisionShape(p.GEOM_CYLINDER, radius=r,
                                             height=h, physicsClientId=self.client)
                vis = p.createVisualShape(p.GEOM_CYLINDER, radius=r, length=h,
                                          rgbaColor=[0.15, 0.35, 0.85, 0.7],
                                          physicsClientId=self.client)
            bid = p.createMultiBody(baseMass=0,
                                    baseCollisionShapeIndex=col,
                                    baseVisualShapeIndex=vis,
                                    basePosition=[x, y, z],
                                    physicsClientId=self.client)
            self.obstacles.append(Obstacle(np.array([x, y, z]), r, shape, h, bid))
            self.obstacle_ids.append(bid)
        print(f"[Env] Spawned {len(self.obstacles)} obstacles")

    def set_joint_state(self, q):
        for i, a in enumerate(q):
            p.resetJointState(self.robot_id, i, float(a),
                              physicsClientId=self.client)

    def check_collision(self, q) -> bool:
        """
        True if any robot link penetrates an obstacle (contactDistance < 0).
        Uses getClosestPoints — no simulation step required.
        """
        self.set_joint_state(q)
        for obs_id in self.obstacle_ids:
            pts = p.getClosestPoints(
                bodyA=self.robot_id,
                bodyB=obs_id,
                distance=0.001,
                physicsClientId=self.client,
            )
            if pts:
                for pt in pts:
                    if pt[8] < 0.0:
                        return True
        return False

    def get_link_positions(self, q):
        self.set_joint_state(q)
        return [np.array(p.getLinkState(self.robot_id, l,
                                        computeForwardKinematics=True,
                                        physicsClientId=self.client)[4])
                for l in range(7)]

    def eef_position(self, q):
        self.set_joint_state(q)
        return np.array(p.getLinkState(self.robot_id, 11,
                                       computeForwardKinematics=True,
                                       physicsClientId=self.client)[4])

    def disconnect(self):
        p.disconnect(physicsClientId=self.client)
