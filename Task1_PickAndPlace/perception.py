"""
perception.py
-------------
Object detection and classification using colour segmentation on the RGB image.

Pipeline
--------
1. Convert RGB → HSV colour space.
2. Apply per-colour HSV masks.
3. Find connected blobs with OpenCV contours.
4. For each blob: compute 2-D centroid (u, v) and back-project to 3-D using depth.
5. Return a list of DetectedObject structs.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from camera import Camera, CameraCapture


# ─── HSV colour ranges ─────────────────────────────────────────────────────────
# Each entry: (lower_hsv, upper_hsv)
# H is 0–179 in OpenCV, S and V are 0–255.
HSV_RANGES = {
    "red":    [(  0,  80, 80), ( 10, 255, 255)],   # red wraps; add second range below
    "red2":   [(165,  80, 80), (179, 255, 255)],
    "green":  [( 40, 100, 80), ( 80, 255, 255)],
    "blue":   [(100,  80, 80), (130, 255, 255)],
    "yellow": [( 18, 100, 100), ( 35, 255, 255)],
    "orange": [( 10,  80, 80), ( 20, 255, 255)],
}

# Colours that use two HSV ranges (like red which wraps around hue 0/180)
DUAL_RANGE = {"red": "red2"}

# Minimum blob area in pixels (filters noise)
MIN_BLOB_AREA = 200


@dataclass
class DetectedObject:
    """Represents one detected object in the scene."""
    label:       str              # colour class name
    centroid_px: Tuple[int, int]  # (u, v) — pixel centre
    centroid_3d: np.ndarray       # (3,)   — world XYZ
    bbox_px:     Tuple[int, int, int, int]  # (x, y, w, h)
    confidence:  float            # fraction of mask pixels within blob


class Perception:
    """
    Detects and classifies coloured cubes from an overhead (or wrist) camera.

    Parameters
    ----------
    camera : Camera  — the camera used for capturing images
    """

    def __init__(self, camera: Camera):
        self.camera = camera

    # ──────────────────────────────────────────────────────────────────────────
    # Main detection entry point
    # ──────────────────────────────────────────────────────────────────────────

    def detect(self, capture: Optional[CameraCapture] = None) -> List[DetectedObject]:
        """
        Run the full detection pipeline on a camera image.

        Parameters
        ----------
        capture : if None, a fresh capture is taken automatically.

        Returns
        -------
        List of DetectedObject, one per detected cube.
        """
        if capture is None:
            capture = self.camera.capture()

        hsv = cv2.cvtColor(capture.rgb, cv2.COLOR_RGB2HSV)
        objects: List[DetectedObject] = []

        # Iterate over each colour class
        primary_colours = [c for c in HSV_RANGES if "2" not in c]
        for colour in primary_colours:
            lo, hi = np.array(HSV_RANGES[colour]), np.array(HSV_RANGES[colour][1])
            lo = np.array(HSV_RANGES[colour][0])
            hi = np.array(HSV_RANGES[colour][1])
            mask = cv2.inRange(hsv, lo, hi)

            # Merge second range for colours that wrap the hue axis
            if colour in DUAL_RANGE:
                alt = DUAL_RANGE[colour]
                lo2 = np.array(HSV_RANGES[alt][0])
                hi2 = np.array(HSV_RANGES[alt][1])
                mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo2, hi2))

            # Clean the mask with morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=2)
            mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

            blobs = self._extract_blobs(mask, capture, colour)
            objects.extend(blobs)

        print(f"[Perception] Detected {len(objects)} object(s): "
              f"{[o.label for o in objects]}")
        return objects

    # ──────────────────────────────────────────────────────────────────────────
    # Blob extraction
    # ──────────────────────────────────────────────────────────────────────────

    def _extract_blobs(
        self, mask: np.ndarray, capture: CameraCapture, colour: str
    ) -> List[DetectedObject]:
        """
        Find connected components in the binary mask and back-project their
        centroids to 3-D world coordinates.
        """
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        blobs = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_BLOB_AREA:
                continue

            # ── 2-D centroid ───────────────────────────────────────────────────
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            u = int(M["m10"] / M["m00"])
            v = int(M["m01"] / M["m00"])

            # ── Bounding box ───────────────────────────────────────────────────
            x, y, w, h = cv2.boundingRect(cnt)

            # ── Depth at centroid (median over a small patch for robustness) ───
            patch_r = 5
            v0, v1 = max(0, v - patch_r), min(capture.depth.shape[0], v + patch_r)
            u0, u1 = max(0, u - patch_r), min(capture.depth.shape[1], u + patch_r)
            depth_patch = capture.depth[v0:v1, u0:u1]
            valid = depth_patch[depth_patch > 0]
            if len(valid) == 0:
                continue
            d = float(np.median(valid))

            # ── 3-D back-projection ────────────────────────────────────────────
            world_xyz = Camera.pixel_to_world(u, v, d, capture.K, capture.T_cam_world)

            # Confidence: normalised blob size as a proxy
            img_area = capture.rgb.shape[0] * capture.rgb.shape[1]
            confidence = min(1.0, area / (img_area * 0.01))  # 1% of image = full conf.

            blobs.append(DetectedObject(
                label=colour,
                centroid_px=(u, v),
                centroid_3d=world_xyz,
                bbox_px=(x, y, w, h),
                confidence=round(confidence, 3),
            ))

        return blobs

    # ──────────────────────────────────────────────────────────────────────────
    # Debug / visualisation
    # ──────────────────────────────────────────────────────────────────────────

    def draw_detections(
        self, capture: CameraCapture, objects: List[DetectedObject]
    ) -> np.ndarray:
        """
        Return an annotated BGR image with bounding boxes, labels, and centroids.
        Useful for debugging and video demos.
        """
        vis = cv2.cvtColor(capture.rgb, cv2.COLOR_RGB2BGR)
        for obj in objects:
            x, y, w, h = obj.bbox_px
            u, v        = obj.centroid_px
            xyz         = obj.centroid_3d

            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(vis, (u, v), 5, (0, 0, 255), -1)
            label_txt = (
                f"{obj.label} "
                f"({xyz[0]:.2f},{xyz[1]:.2f},{xyz[2]:.2f})"
            )
            cv2.putText(vis, label_txt, (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

        return vis
