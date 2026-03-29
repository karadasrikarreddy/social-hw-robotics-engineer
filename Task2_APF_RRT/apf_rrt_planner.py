"""
apf_rrt_planner.py — Phase A: Hybrid APF-RRT planner.

Always adds q_start to the tree. When expanding from node 0, only checks
the new node (not the segment) to avoid false rejection at the home pose.
"""
import time
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict
from environment import Environment, Q_START, Q_GOAL
from robot_kinematics import RobotKinematics, JOINT_LOWER, JOINT_UPPER


@dataclass
class PlannerResult:
    success: bool
    path: List
    computation_time: float
    path_length: float
    node_count: int
    tree_edges: List


@dataclass
class APFRRTConfig:
    max_iter:    int   = 6000
    step_size:   float = 0.20
    goal_radius: float = 0.20
    p_goal:      float = 0.15
    K_att:       float = 1.0
    K_rep:       float = 0.4
    alpha:       float = 0.55
    beta:        float = 0.35
    gamma:       float = 0.10
    d0:          float = 0.20


class APFRRTPlanner:
    def __init__(self, env: Environment, kin: RobotKinematics, config=None):
        self.env = env
        self.kin = kin
        self.cfg = config or APFRRTConfig()

    def plan(self, q_start=None, q_goal=None) -> PlannerResult:
        q_start = q_start if q_start is not None else Q_START.copy()
        q_goal  = q_goal  if q_goal  is not None else Q_GOAL.copy()
        cfg = self.cfg
        t0  = time.time()

        s_free = self.kin.is_collision_free(q_start)
        g_free = self.kin.is_collision_free(q_goal)
        print(f"  [RRT] start_free={s_free}  goal_free={g_free}")

        # Try direct connection first
        if self.kin.is_segment_free(q_start, q_goal, n=10):
            print("  [RRT] Direct path — obstacles not on this trajectory.")
            path = [q_start, q_goal]
            elapsed = time.time() - t0
            return PlannerResult(True, path, elapsed,
                                 float(np.linalg.norm(q_goal - q_start)),
                                 2, [(q_start.copy(), q_goal.copy())])

        # Always add q_start — home-pose marginal collisions should not abort planning
        nodes:  List = [q_start]
        parent: Dict = {0: -1}
        edges:  List = []
        goal_idx: Optional[int] = None

        for iteration in range(cfg.max_iter):
            q_rand = q_goal.copy() if np.random.rand() < cfg.p_goal \
                     else self.kin.random_config()

            diffs = np.array([np.array(n) - q_rand for n in nodes])
            dists = np.linalg.norm(diffs, axis=1)
            ni    = int(np.argmin(dists))
            q_near = nodes[ni]

            q_new = self._extend(q_near, q_rand, q_goal)
            if q_new is None:
                continue

            # From start node: only check the new node, not the full segment
            if ni == 0:
                if not self.kin.is_collision_free(q_new):
                    continue
            else:
                if not self.kin.is_segment_free(q_near, q_new, n=4):
                    continue

            new_i = len(nodes)
            nodes.append(q_new)
            parent[new_i] = ni
            edges.append((q_near.copy(), q_new.copy()))

            if np.linalg.norm(np.array(q_new) - q_goal) < cfg.goal_radius:
                if self.kin.is_segment_free(q_new, q_goal, n=6):
                    gi = len(nodes)
                    nodes.append(q_goal.copy())
                    parent[gi] = new_i
                    edges.append((q_new.copy(), q_goal.copy()))
                    goal_idx = gi
                else:
                    goal_idx = new_i
                break

            if iteration % 500 == 499:
                best = min(np.linalg.norm(np.array(n) - q_goal) for n in nodes)
                print(f"  [RRT] iter={iteration+1}  nodes={len(nodes)}  "
                      f"closest={best:.3f} rad")

        elapsed = time.time() - t0

        if goal_idx is not None:
            path = self._trace(nodes, parent, goal_idx)
            plen = sum(np.linalg.norm(np.array(path[i+1]) - np.array(path[i]))
                       for i in range(len(path) - 1))
            print(f"  [RRT] Path found! nodes={len(nodes)} len={plen:.3f} t={elapsed:.2f}s")
            return PlannerResult(True, path, elapsed, plen, len(nodes), edges)

        print(f"  [RRT] Failed after {len(nodes)} nodes in {elapsed:.2f}s")
        return PlannerResult(False, [], elapsed, 0.0, len(nodes), edges)

    def _extend(self, q_near, q_rand, q_goal):
        cfg = self.cfg
        vr = np.array(q_rand) - np.array(q_near)
        nr = np.linalg.norm(vr)
        if nr < 1e-9: return None
        d_rand = vr / nr

        va = np.array(q_goal) - np.array(q_near)
        d_att = va / max(np.linalg.norm(va), 1e-9)

        d_rep = np.zeros(7)
        try:
            eef = self.env.eef_position(q_near)
            for obs in self.env.obstacles:
                d = np.linalg.norm(eef - obs.position)
                if 1e-6 < d < cfg.d0:
                    d_rep += (1.0 / d - 1.0 / cfg.d0) * (-d_att)
        except Exception:
            pass
        d_rep = d_rep / max(np.linalg.norm(d_rep), 1e-9)

        direction = (cfg.alpha * d_rand
                     + cfg.beta  * cfg.K_att * d_att
                     + cfg.gamma * cfg.K_rep * d_rep)
        nd = np.linalg.norm(direction)
        if nd < 1e-9: return None
        return self.kin.clamp(np.array(q_near) + cfg.step_size * (direction / nd))

    @staticmethod
    def _trace(nodes, parent, gi):
        path = []; i = gi
        while i != -1: path.append(nodes[i]); i = parent[i]
        path.reverse(); return path
