"""
simulation_env.py
-----------------
Manages the PyBullet physics simulation environment for the Pick-and-Place pipeline.
Handles scene initialisation, table/robot loading, and random object spawning.
"""

import pybullet as p
import pybullet_data
import numpy as np
import random
import time


# ─── Panda joint indices ───────────────────────────────────────────────────────
PANDA_ARM_JOINTS   = list(range(7))   # joints 0–6 control the arm
PANDA_FINGER_JOINT_1 = 9             # left finger
PANDA_FINGER_JOINT_2 = 10            # right finger
PANDA_EEF_LINK     = 11              # end-effector link index
PANDA_DOFS         = 7               # degrees of freedom we plan for

# Cube dimensions and colour palette
CUBE_HALF_EXTENT   = 0.025           # 5 cm cubes
CUBE_COLOURS = {
    "red":    [1.0, 0.1, 0.1, 1.0],
    "green":  [0.1, 0.8, 0.1, 1.0],
    "blue":   [0.1, 0.3, 1.0, 1.0],
    "yellow": [1.0, 0.9, 0.0, 1.0],
    "orange": [1.0, 0.5, 0.0, 1.0],
}


class SimulationEnv:
    """
    Wraps all PyBullet setup and scene management.

    Parameters
    ----------
    gui : bool
        True  → opens the interactive visualiser window.
        False → headless (no window), useful for automated testing.
    timestep : float
        Physics integration step in seconds.
    """

    def __init__(self, gui: bool = True, timestep: float = 1.0 / 240.0):
        self.gui      = gui
        self.timestep = timestep
        self.objects  = {}   # { object_id: {"colour": str, "position": np.ndarray} }

        self._connect()
        self._configure_physics()
        self._load_scene()

    # ──────────────────────────────────────────────────────────────────────────
    # Connection & physics
    # ──────────────────────────────────────────────────────────────────────────

    def _connect(self):
        """Open the PyBullet connection in GUI or DIRECT (headless) mode."""
        mode = p.GUI if self.gui else p.DIRECT
        self.client = p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        print(f"[SimEnv] Connected to PyBullet (mode={'GUI' if self.gui else 'HEADLESS'})")

    def _configure_physics(self):
        """Set gravity and the simulation time-step."""
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.setTimeStep(self.timestep, physicsClientId=self.client)
        p.setRealTimeSimulation(0, physicsClientId=self.client)  # manual stepping

    # ──────────────────────────────────────────────────────────────────────────
    # Scene loading
    # ──────────────────────────────────────────────────────────────────────────

    def _load_scene(self):
        """Load the ground plane, table, and robot arm."""
        # Ground plane
        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client)

        # Table (a flat box as a static body)
        table_collision = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.05], physicsClientId=self.client
        )
        table_visual = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.05],
            rgbaColor=[0.76, 0.60, 0.42, 1.0], physicsClientId=self.client
        )
        self.table_id = p.createMultiBody(
            baseMass=0,                          # mass=0 → static object
            baseCollisionShapeIndex=table_collision,
            baseVisualShapeIndex=table_visual,
            basePosition=[0.5, 0.0, 0.05],      # table surface at z = 0.10
            physicsClientId=self.client,
        )
        self.table_surface_z = 0.10             # top of table in world frame

        # Franka Panda robot arm (mounted at the origin, floor-based)
        self.robot_id = p.loadURDF(
            "franka_panda/panda.urdf",
            basePosition=[0.0, 0.0, 0.0],
            useFixedBase=True,                  # fixed to the floor
            physicsClientId=self.client,
        )
        self._set_home_pose()
        print("[SimEnv] Scene loaded: plane, table, Franka Panda")

    def _set_home_pose(self):
        """
        Reset the arm to a neutral 'home' joint configuration.
        These angles position the arm upright above the table.
        """
        home_angles = [0.0, -np.pi/4, 0.0, -3*np.pi/4, 0.0, np.pi/2, np.pi/4]
        for i, angle in zip(PANDA_ARM_JOINTS, home_angles):
            p.resetJointState(self.robot_id, i, angle, physicsClientId=self.client)

        # Open gripper fully
        for finger_joint in [PANDA_FINGER_JOINT_1, PANDA_FINGER_JOINT_2]:
            p.resetJointState(self.robot_id, finger_joint, 0.04, physicsClientId=self.client)

    # ──────────────────────────────────────────────────────────────────────────
    # Object spawning
    # ──────────────────────────────────────────────────────────────────────────

    def spawn_random_cubes(self, n: int = 3, seed: int = None) -> dict:
        """
        Place n coloured cubes at random non-overlapping positions on the table.

        Parameters
        ----------
        n    : number of cubes to spawn
        seed : random seed for reproducibility

        Returns
        -------
        dict mapping object_id → {"colour": str, "position": np.ndarray}
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Workspace bounds on the table (x: 0.3–0.7, y: -0.3–0.3)
        x_min, x_max = 0.30, 0.70
        y_min, y_max = -0.30, 0.30
        spawn_z      = self.table_surface_z + CUBE_HALF_EXTENT + 0.001

        occupied = []   # list of (x, y) centres already placed
        colours   = list(CUBE_COLOURS.keys())
        chosen    = random.sample(colours, min(n, len(colours)))

        for colour_name in chosen:
            # Rejection-sample until a position is far enough from others
            for _ in range(100):
                x = random.uniform(x_min, x_max)
                y = random.uniform(y_min, y_max)
                too_close = any(
                    np.linalg.norm([x - ox, y - oy]) < 0.10
                    for ox, oy in occupied
                )
                if not too_close:
                    break
            else:
                print(f"[SimEnv] WARNING: could not find clear spot for cube {colour_name}")
                continue

            object_id = self._create_cube([x, y, spawn_z], CUBE_COLOURS[colour_name])
            self.objects[object_id] = {
                "colour":   colour_name,
                "position": np.array([x, y, spawn_z]),
            }
            occupied.append((x, y))
            print(f"[SimEnv] Spawned {colour_name} cube at ({x:.3f}, {y:.3f}, {spawn_z:.3f})")

        # Let the scene settle
        self.step(n_steps=120)
        return self.objects

    def _create_cube(self, position: list, rgba: list) -> int:
        """Spawns a single 5cm cube with enhanced physics."""
        # Create shapes
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.025, 0.025, 0.025])
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.025, 0.025, 0.025], rgbaColor=rgba)
        
        # Create the body first
        cube_id = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=position,
            physicsClientId=self.client,
        )

        # Apply dynamics to the newly created cube_id
        p.changeDynamics(
            cube_id, 
            -1, 
            lateralFriction=1.0, 
            spinningFriction=1.0,
            rollingFriction=0.001,
            restitution=0.1
        )
        
        return cube_id

    # ──────────────────────────────────────────────────────────────────────────
    # Simulation stepping
    # ──────────────────────────────────────────────────────────────────────────

    def step(self, n_steps: int = 1, real_time: bool = False):
        """Advance the simulation by n_steps physics steps."""
        for _ in range(n_steps):
            p.stepSimulation(physicsClientId=self.client)
            if real_time:
                time.sleep(self.timestep)

    def get_object_pose(self, object_id: int):
        """Return (position, orientation) of an object in world frame."""
        pos, orn = p.getBasePositionAndOrientation(
            object_id, physicsClientId=self.client
        )
        return np.array(pos), np.array(orn)

    def reset(self):
        """Remove all spawned cubes and return the arm to home."""
        for obj_id in list(self.objects.keys()):
            p.removeBody(obj_id, physicsClientId=self.client)
        self.objects.clear()
        self._set_home_pose()
        print("[SimEnv] Scene reset")

    def disconnect(self):
        """Cleanly shut down the physics server."""
        p.disconnect(physicsClientId=self.client)
        print("[SimEnv] Disconnected from PyBullet")
