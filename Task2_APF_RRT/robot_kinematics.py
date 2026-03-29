"""
robot_kinematics.py — joint limits, collision checking via PyBullet built-in.
"""
import numpy as np
from environment import Environment

JOINT_LOWER = np.array([-2.90,-1.76,-2.90,-3.07,-2.90,-0.02,-2.90])
JOINT_UPPER = np.array([ 2.90, 1.76, 2.90,-0.07, 2.90, 3.75, 2.90])

class RobotKinematics:
    def __init__(self, env: Environment):
        self.env = env

    @staticmethod
    def within_limits(q):
        return bool(np.all(q >= JOINT_LOWER) and np.all(q <= JOINT_UPPER))

    @staticmethod
    def clamp(q):
        return np.clip(q, JOINT_LOWER, JOINT_UPPER)

    @staticmethod
    def random_config():
        return np.random.uniform(JOINT_LOWER, JOINT_UPPER)

    def is_collision_free(self, q) -> bool:
        if not self.within_limits(q):
            return False
        return not self.env.check_collision(q)

    def is_segment_free(self, qa, qb, n=5) -> bool:
        for t in np.linspace(0, 1, n):
            if not self.is_collision_free(qa + t*(qb-qa)):
                return False
        return True
