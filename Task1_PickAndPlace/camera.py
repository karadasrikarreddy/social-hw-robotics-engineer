"""
camera.py
---------
Handles both overhead and wrist-mounted cameras.

Key responsibilities
--------------------
1. Compute camera intrinsics (K matrix) from PyBullet projection parameters.
2. Compute camera extrinsics (T_cam→world) from view matrix.
3. Back-project 2D pixel (u, v) + depth d → 3D world point P_world.
   Formula:  P_world = T_cam→world · K⁻¹ · [u, v, 1]ᵀ · d

Math reference
--------------
  K = | fx   0  cx |       (camera intrinsic matrix)
      |  0  fy  cy |
      |  0   0   1 |

  P_cam  = K⁻¹ · [u, v, 1]ᵀ · d
         = [(u-cx)/fx * d,  (v-cy)/fy * d,  d]

  P_world = T_cam→world · [P_cam; 1]   (homogeneous multiplication)
"""

import numpy as np
import pybullet as p
from dataclasses import dataclass
from simulation_env import SimulationEnv, PANDA_EEF_LINK


# ─── Camera specs ──────────────────────────────────────────────────────────────
IMG_W  = 640
IMG_H  = 480
FOV    = 60.0         # vertical field of view in degrees
NEAR   = 0.01
FAR    = 5.0


@dataclass
class CameraCapture:
    """Holds all data from a single camera render call."""
    rgb:   np.ndarray   # (H, W, 3) uint8
    depth: np.ndarray   # (H, W)    float32  — true metric depth in metres
    seg:   np.ndarray   # (H, W)    int32    — segmentation mask (object IDs)
    K:     np.ndarray   # (3, 3)    intrinsic matrix
    T_cam_world: np.ndarray  # (4, 4) camera-to-world transform


class Camera:
    """
    Represents a single virtual camera in the PyBullet scene.

    Parameters
    ----------
    env  : SimulationEnv
    kind : "overhead" | "wrist"
           overhead → fixed position above the table
           wrist    → attached to the robot end-effector
    """

    def __init__(self, env: SimulationEnv, kind: str = "overhead"):
        assert kind in ("overhead", "wrist"), "kind must be 'overhead' or 'wrist'"
        self.env  = env
        self.kind = kind

    # ──────────────────────────────────────────────────────────────────────────
    # View & projection matrices
    # ──────────────────────────────────────────────────────────────────────────

    def _get_view_matrix(self):
        """
        Compute the 4×4 view matrix (world→camera transform).

        Overhead  : fixed camera looking straight down at the table centre.
        Wrist     : camera rigidly attached to the EEF, looking forward.
        """
        if self.kind == "overhead":
            # Camera at 1.2 m above the table centre, looking straight down
            cam_eye    = [0.5,  0.0, 1.2]
            cam_target = [0.5,  0.0, 0.1]   # table surface
            cam_up     = [0.0, -1.0, 0.0]   # –Y is "up" in image

        else:  # wrist
            # Get the EEF pose in world frame
            eef_state = p.getLinkState(
                self.env.robot_id,
                PANDA_EEF_LINK,
                computeForwardKinematics=True,
                physicsClientId=self.env.client,
            )
            eef_pos = np.array(eef_state[4])   # world position
            eef_orn = np.array(eef_state[5])   # world orientation (quaternion)

            # Camera looks along the EEF's –Z axis (pointing down from wrist)
            rot_mat = np.array(
                p.getMatrixFromQuaternion(eef_orn, physicsClientId=self.env.client)
            ).reshape(3, 3)

            cam_eye    = eef_pos + rot_mat @ np.array([0, 0, -0.05])
            cam_target = eef_pos + rot_mat @ np.array([0, 0,  0.10])
            cam_up     = rot_mat @ np.array([0, -1, 0])

        view = p.computeViewMatrix(
            cameraEyePosition=cam_eye,
            cameraTargetPosition=cam_target,
            cameraUpVector=cam_up,
            physicsClientId=self.env.client,
        )
        return np.array(view).reshape(4, 4).T   # PyBullet gives column-major

    def _get_projection_matrix(self):
        """Return the OpenGL-style 4×4 projection matrix."""
        aspect = IMG_W / IMG_H
        proj = p.computeProjectionMatrixFOV(
            fov=FOV, aspect=aspect, nearVal=NEAR, farVal=FAR,
            physicsClientId=self.env.client,
        )
        return np.array(proj).reshape(4, 4).T   # column-major → row-major

    # ──────────────────────────────────────────────────────────────────────────
    # Intrinsic matrix K
    # ──────────────────────────────────────────────────────────────────────────

    def compute_K(self) -> np.ndarray:
        """
        Derive the 3×3 camera intrinsic matrix from FOV and image dimensions.

              fy = (H/2) / tan(fov_y/2)
              fx = fy   (square pixels assumed)
              cx = W/2,  cy = H/2

        Returns
        -------
        K : (3, 3) float64
        """
        fov_y_rad = np.deg2rad(FOV)
        fy = (IMG_H / 2.0) / np.tan(fov_y_rad / 2.0)
        fx = fy                    # assume square pixels (aspect ratio handled by FOV)
        cx = IMG_W / 2.0
        cy = IMG_H / 2.0
        K = np.array([
            [fx,  0, cx],
            [ 0, fy, cy],
            [ 0,  0,  1],
        ], dtype=np.float64)
        return K

    # ──────────────────────────────────────────────────────────────────────────
    # Capture
    # ──────────────────────────────────────────────────────────────────────────

    def capture(self) -> CameraCapture:
        """
        Render the scene from this camera's viewpoint.

        Returns
        -------
        CameraCapture with rgb, depth (metric), seg, K, T_cam_world
        """
        view_mat = self._get_view_matrix()        # world→cam  (4×4)
        proj_mat = self._get_projection_matrix()  # OpenGL projection (4×4)

        _, _, rgba_raw, depth_raw, seg_raw = p.getCameraImage(
            width=IMG_W,
            height=IMG_H,
            viewMatrix=view_mat.T.flatten().tolist(),
            projectionMatrix=proj_mat.T.flatten().tolist(),
            renderer=p.ER_TINY_RENDERER,
            physicsClientId=self.env.client,
        )

        # ── RGB ────────────────────────────────────────────────────────────────
        rgb = np.array(rgba_raw, dtype=np.uint8).reshape(IMG_H, IMG_W, 4)[:, :, :3]

        # ── Depth (OpenGL normalised [0,1] → metric metres) ────────────────────
        #   OpenGL z_ndc is non-linear; linearise with the standard formula:
        #       z_eye = (2 * near * far) / (far + near - z_ndc*(far - near))
        # PyBullet's getCameraImage returns window-space depth in [0, 1].
        # Correct linearisation for window-space z:
        #     d_metric = near * far / (far - z_buf * (far - near))
        #
        # WRONG formula (for OpenGL NDC z in [-1,1], NOT window space):
        #     d = 2*n*f / (f+n - z*(f-n))   ← this over-estimates depth ~1.65x
        z_buf  = np.array(depth_raw, dtype=np.float32).reshape(IMG_H, IMG_W)
        depth  = (NEAR * FAR) / (FAR - z_buf * (FAR - NEAR))

        # ── Segmentation ───────────────────────────────────────────────────────
        seg = np.array(seg_raw, dtype=np.int32).reshape(IMG_H, IMG_W)

        # ── T_cam→world (invert the view matrix which is world→cam) ────────────
        T_world_cam = np.linalg.inv(view_mat)   # cam→world

        K = self.compute_K()

        return CameraCapture(rgb=rgb, depth=depth, seg=seg, K=K, T_cam_world=T_world_cam)

    # ──────────────────────────────────────────────────────────────────────────
    # 3-D Back-Projection
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def pixel_to_world(
        u: float, v: float, d: float,
        K: np.ndarray, T_cam_world: np.ndarray
    ) -> np.ndarray:
        """
        Convert a single 2-D pixel + metric depth into a 3-D world point.

        Formula
        -------
          P_cam   = K⁻¹ · [u, v, 1]ᵀ · d
          P_world = T_cam_world · [P_cam; 1]

        Parameters
        ----------
        u, v         : pixel coordinates (column, row)
        d            : metric depth at that pixel (metres)
        K            : (3, 3) camera intrinsic matrix
        T_cam_world  : (4, 4) camera-to-world transform

        Returns
        -------
        np.ndarray  shape (3,) — world-frame XYZ
        """
        # Back-project into camera frame using the correct per-component signs.
        #
        # OpenGL vs image coordinate convention:
        #   x_cam ↔ u : SAME direction  → x_cam = (u−cx)/fx * d   (no extra sign)
        #   y_cam ↔ v : OPPOSITE sense  → y_cam = −(v−cy)/fy * d  (image v goes
        #               DOWN but camera y goes UP, so we need the minus)
        #   z_cam      : camera looks along −Z in OpenGL → z_cam = −d
        #
        # Applying (−d) uniformly to all of K⁻¹·[u,v,1] is WRONG: it also
        # flips the x sign, placing every off-centre object at the mirror
        # position across the optical axis.
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        x_cam  = (u - cx) / fx * d       # +d: u and x_cam agree in direction
        y_cam  = -(v - cy) / fy * d      # −d: v and y_cam are opposite
        z_cam  = -d                       # −d: camera looks along −Z
        p_cam  = np.array([x_cam, y_cam, z_cam], dtype=np.float64)

        # Lift to homogeneous coordinates and transform to world frame
        p_cam_h   = np.array([p_cam[0], p_cam[1], p_cam[2], 1.0])
        p_world_h = T_cam_world @ p_cam_h
        return p_world_h[:3]

    def get_3d_point(
        self, u: int, v: int, capture: CameraCapture
    ) -> np.ndarray:
        """Convenience wrapper: look up depth from a capture and back-project."""
        d = float(capture.depth[v, u])
        return self.pixel_to_world(u, v, d, capture.K, capture.T_cam_world)
