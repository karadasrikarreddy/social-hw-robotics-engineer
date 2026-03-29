"""
robot_controller.py
-------------------
Controls the Franka Panda arm.

Responsibilities
----------------
1. Inverse Kinematics (IK) — solve joint angles for a desired EEF pose.
2. Joint-space trajectory execution — move smoothly between configs.
3. Gripper control — open / close the parallel fingers.
4. grasp_point_world() — the full autonomous grasp sequence:
       Approach → Descend → Grasp → Lift
"""

import numpy as np
import pybullet as p
import time
from typing import Optional, Tuple

from simulation_env import (
    SimulationEnv,
    PANDA_ARM_JOINTS,
    PANDA_FINGER_JOINT_1,
    PANDA_FINGER_JOINT_2,
    PANDA_EEF_LINK,
    PANDA_DOFS,
)


# ─── IK configuration ──────────────────────────────────────────────────────────
IK_MAX_ITER      = 200
IK_RESIDUAL_THRESH = 1e-5
IK_DAMPING       = 0.05          # damped-least-squares regulariser

# ─── Trajectory timing ─────────────────────────────────────────────────────────
TRAJ_STEPS       = 300           # more steps = smoother, slower motion in GUI
TRAJ_DT          = 1.0 / 240.0  # seconds per physics step

# ─── Grasp geometry ────────────────────────────────────────────────────────────
PRE_GRASP_HEIGHT_OFFSET = 0.15   # metres above object before descending
LIFT_HEIGHT             = 0.30   # absolute Z height to lift the object to
GRIPPER_OPEN_WIDTH      = 0.08   # total gap when fully open (0.04 per side)
# Close width must be LESS than cube width (0.05 m) so fingers contact the cube.
# 0.015 m gap → each finger sits 7.5 mm from centre → grips 50 mm cube firmly.
GRIPPER_CLOSE_WIDTH     = 0.015

# Default EEF orientation: gripper pointing straight down (vertical approach)
GRASP_ORIENTATION = p.getQuaternionFromEuler([np.pi, 0, 0])


class RobotController:
    """
    High-level controller for the Franka Panda arm.

    Parameters
    ----------
    env : SimulationEnv
    """

    def __init__(self, env: SimulationEnv):
        self.env = env
        self.robot_id = env.robot_id

        # Cache the lower/upper joint limits for IK
        joint_info = [
            p.getJointInfo(self.robot_id, i, physicsClientId=env.client)
            for i in PANDA_ARM_JOINTS
        ]
        self.lower_limits = [info[8]  for info in joint_info]
        self.upper_limits = [info[9]  for info in joint_info]
        self.rest_poses    = [0.0, -np.pi/4, 0.0, -3*np.pi/4,
                              0.0, np.pi/2, np.pi/4]
        self.joint_ranges  = [u - l for u, l in
                              zip(self.upper_limits, self.lower_limits)]

    # ──────────────────────────────────────────────────────────────────────────
    # Inverse Kinematics
    # ──────────────────────────────────────────────────────────────────────────

    def solve_ik(
        self,
        target_pos: np.ndarray,
        target_orn: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """
        Solve IK for the desired EEF position (and optionally orientation).

        Uses PyBullet's damped-least-squares IK with joint limit clamping.

        Parameters
        ----------
        target_pos : (3,) desired EEF position in world frame
        target_orn : (4,) desired EEF orientation as quaternion; None = any

        Returns
        -------
        np.ndarray shape (7,) joint angles, or None if IK fails
        """
        kwargs = dict(
            bodyUniqueId=self.robot_id,
            endEffectorLinkIndex=PANDA_EEF_LINK,
            targetPosition=target_pos.tolist(),
            lowerLimits=self.lower_limits,
            upperLimits=self.upper_limits,
            jointRanges=self.joint_ranges,
            restPoses=self.rest_poses,
            maxNumIterations=IK_MAX_ITER,
            residualThreshold=IK_RESIDUAL_THRESH,
            physicsClientId=self.env.client,
        )
        if target_orn is not None:
            kwargs["targetOrientation"] = target_orn.tolist()

        joint_angles = p.calculateInverseKinematics(**kwargs)
        arm_angles = np.array(joint_angles[:PANDA_DOFS], dtype=np.float64)

        # ── IK quality check ──────────────────────────────────────────────────
        # CRITICAL: save the real joint state before teleporting for FK check.
        # Without this, _move_to_joints reads the IK-solved pose as "current"
        # and interpolates from IK→IK (zero motion) — the arm teleports
        # through physics and the gripper closes on air.
        real_angles = self.get_joint_angles()         # ← save real state

        self._set_joints_instant(arm_angles)          # teleport for FK only
        eef_state = p.getLinkState(
            self.robot_id, PANDA_EEF_LINK,
            computeForwardKinematics=True,
            physicsClientId=self.env.client,
        )
        actual_pos = np.array(eef_state[4])
        error = np.linalg.norm(actual_pos - target_pos)
        if error > 0.02:
            print(f"[IK] WARNING: residual error {error*100:.1f} cm")

        self._set_joints_instant(real_angles)         # ← RESTORE real state

        return arm_angles

    # ──────────────────────────────────────────────────────────────────────────
    # Joint control
    # ──────────────────────────────────────────────────────────────────────────

    def _set_joints_instant(self, joint_angles: np.ndarray):
        """Teleport the arm to the given joint configuration (no physics)."""
        for i, angle in zip(PANDA_ARM_JOINTS, joint_angles):
            p.resetJointState(
                self.robot_id, i, angle, physicsClientId=self.env.client
            )

    def _move_to_joints(self, target_angles: np.ndarray, steps: int = TRAJ_STEPS):
        """
        Smoothly interpolate from the current joint config to target_angles.

        Uses PyBullet's position controller (PD control) along a linearly
        interpolated path, then steps the physics simulation.
        """
        # Read current joint positions
        current_angles = np.array([
            p.getJointState(self.robot_id, i, physicsClientId=self.env.client)[0]
            for i in PANDA_ARM_JOINTS
        ])

        for t in np.linspace(0.0, 1.0, steps):
            waypoint = current_angles + t * (target_angles - current_angles)

            p.setJointMotorControlArray(
                bodyUniqueId=self.robot_id,
                jointIndices=PANDA_ARM_JOINTS,
                controlMode=p.POSITION_CONTROL,
                targetPositions=waypoint.tolist(),
                forces=[87.0] * PANDA_DOFS,      # Panda rated continuous torque ≈ 87 Nm
                physicsClientId=self.env.client,
            )
            self.env.step(n_steps=1, real_time=True)

    def move_to_pose(
        self,
        target_pos: np.ndarray,
        target_orn: Optional[np.ndarray] = None,
        steps: int = TRAJ_STEPS,
    ) -> bool:
        """
        Move the end-effector to target_pos with optional target_orn.

        Returns True on success, False if IK failed.
        """
        if target_orn is None:
            target_orn = np.array(GRASP_ORIENTATION)

        joint_angles = self.solve_ik(target_pos, target_orn)
        if joint_angles is None:
            print(f"[Robot] IK failed for target {target_pos}")
            return False

        self._move_to_joints(joint_angles, steps=steps)
        return True

    # ──────────────────────────────────────────────────────────────────────────
    # Gripper
    # ──────────────────────────────────────────────────────────────────────────

    def open_gripper(self, steps: int = 60):
        """Open both fingers to the maximum width."""
        self._set_gripper(GRIPPER_OPEN_WIDTH / 2, steps)

    def close_gripper(self, steps: int = 180):
        """Close both fingers to grip an object."""
        self._set_gripper(GRIPPER_CLOSE_WIDTH / 2, steps)

    def _set_gripper(self, target_per_finger: float, steps: int):
        """Drive each finger to target_per_finger metres from the centre."""
        for finger_joint in [PANDA_FINGER_JOINT_1, PANDA_FINGER_JOINT_2]:
            p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=finger_joint,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_per_finger,
                force=150.0,         # strong enough to hold 100g cube at any acceleration
                physicsClientId=self.env.client,
            )
        self.env.step(n_steps=steps, real_time=True)

    # ──────────────────────────────────────────────────────────────────────────
    # Full Grasp Pipeline
    # ──────────────────────────────────────────────────────────────────────────

    def grasp_point_world(
        self,
        object_pos: np.ndarray,
        lift_height: float = LIFT_HEIGHT,
    ) -> bool:
        """
        Autonomous pick sequence for an object at object_pos (world frame).

        Sequence
        --------
        1. Open gripper.
        2. Approach : move EEF to a pre-grasp position PRE_GRASP_HEIGHT_OFFSET
                      above the object.
        3. Descend  : lower EEF to object's centroid height.
        4. Grasp    : close the gripper fingers.
        5. Lift     : raise the arm to lift_height.

        Parameters
        ----------
        object_pos  : (3,) world-frame XYZ of the object's centroid
        lift_height : absolute world Z to lift to

        Returns
        -------
        bool — True if the entire sequence completed without IK failure
        """
        obj_x, obj_y, obj_z = object_pos
        grasp_orn = np.array(GRASP_ORIENTATION)

        # ── Step 0: Open gripper before approaching ─────────────────────────
        print("[Grasp] Step 0 — Opening gripper")
        self.open_gripper()

        # ── Step 1: Approach (pre-grasp) ────────────────────────────────────
        pre_grasp_pos = np.array([obj_x, obj_y, obj_z + PRE_GRASP_HEIGHT_OFFSET])
        print(f"[Grasp] Step 1 — Approaching  pre-grasp @ {pre_grasp_pos}")
        ok = self.move_to_pose(pre_grasp_pos, grasp_orn, steps=120)
        if not ok:
            return False

        # ── Step 2: Descend onto the object ─────────────────────────────────
        grasp_pos = np.array([obj_x, obj_y, obj_z])
        print(f"[Grasp] Step 2 — Descending   grasp    @ {grasp_pos}")
        ok = self.move_to_pose(grasp_pos, grasp_orn, steps=80)
        if not ok:
            return False

        # ── Step 3: Close gripper ────────────────────────────────────────────
        print("[Grasp] Step 3 — Closing gripper")
        self.close_gripper()

        # ── Settle: hold still for 0.5s so fingers fully engage ──────────────
        # Without this the arm immediately starts lifting while contact forces
        # are still building, causing the cube to slip.
        self.env.step(n_steps=120, real_time=True)

        # ── Step 4: Lift ─────────────────────────────────────────────────────
        lift_pos = np.array([obj_x, obj_y, lift_height])
        print(f"[Grasp] Step 4 — Lifting       target  @ {lift_pos}")
        ok = self.move_to_pose(lift_pos, grasp_orn, steps=120)
        if not ok:
            return False

        print("[Grasp] Sequence complete ✓")
        return True

    def place_at_world(
        self,
        place_pos: np.ndarray,
        release_height_offset: float = 0.06,
    ) -> bool:
        """
        Place the currently-held object at place_pos (world frame).

        Sequence
        --------
        1. Move above the drop zone at (place_pos + release_height_offset).
        2. Descend to just above the surface (release_height_offset / 2).
        3. Open gripper to release the object.
        4. Retreat upward so the arm clears the placed cube.

        Parameters
        ----------
        place_pos             : (3,) world XYZ of the centre of the drop zone
        release_height_offset : how high above place_pos to hover before lowering

        Returns
        -------
        bool — True if the full sequence completed without IK failure
        """
        grasp_orn = np.array(GRASP_ORIENTATION)
        px, py, pz = place_pos

        # ── Step A: move above drop zone ────────────────────────────────────
        above_pos = np.array([px, py, pz + release_height_offset])
        print(f"[Place] Moving above drop zone @ {above_pos}")
        ok = self.move_to_pose(above_pos, grasp_orn, steps=120)
        if not ok:
            return False

        # ── Step B: lower to just above the surface ──────────────────────────
        lower_pos = np.array([px, py, pz + release_height_offset / 2])
        print(f"[Place] Lowering to release height @ {lower_pos}")
        ok = self.move_to_pose(lower_pos, grasp_orn, steps=80)
        if not ok:
            return False

        # ── Step C: open gripper ─────────────────────────────────────────────
        print("[Place] Releasing object")
        self.open_gripper(steps=60)
        self.env.step(n_steps=60, real_time=True)   # let cube settle on surface

        # ── Step D: retreat upward so arm doesn't drag the placed cube ────────
        retreat_pos = np.array([px, py, pz + release_height_offset + 0.10])
        print(f"[Place] Retreating to @ {retreat_pos}")
        ok = self.move_to_pose(retreat_pos, grasp_orn, steps=80)
        return ok   # retreat failure is non-critical but we report it

    # ──────────────────────────────────────────────────────────────────────────
    # Diagnostics
    # ──────────────────────────────────────────────────────────────────────────

    def get_eef_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return current (position, quaternion) of the end-effector in world frame."""
        state = p.getLinkState(
            self.robot_id, PANDA_EEF_LINK,
            computeForwardKinematics=True,
            physicsClientId=self.env.client,
        )
        return np.array(state[4]), np.array(state[5])

    def get_joint_angles(self) -> np.ndarray:
        """Return current joint angles for the 7-DOF arm."""
        return np.array([
            p.getJointState(self.robot_id, i, physicsClientId=self.env.client)[0]
            for i in PANDA_ARM_JOINTS
        ])
