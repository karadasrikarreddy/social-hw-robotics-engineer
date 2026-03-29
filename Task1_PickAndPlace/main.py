"""
main.py
-------
Orchestrates the full Pick-and-Place pipeline.

Usage
-----
    # GUI (interactive window):
    python main.py

    # Headless (automated testing):
    python main.py --headless

    # Specific number of cubes and random seed:
    python main.py --cubes 3 --seed 42
"""

import argparse
import csv
import time
from pathlib import Path
import numpy as np

from simulation_env import SimulationEnv
from camera          import Camera
from perception      import Perception
from robot_controller import RobotController


# ─── Telemetry CSV output ──────────────────────────────────────────────────────
LOG_FILE = "telemetry.csv"


def parse_args():
    parser = argparse.ArgumentParser(description="Pick-and-Place pipeline")
    parser.add_argument("--headless", action="store_true",
                        help="Run without GUI (for automated testing)")
    parser.add_argument("--cubes", type=int, default=3,
                        help="Number of cubes to spawn (default: 3)")
    parser.add_argument("--seed",  type=int, default=None,
                        help="Random seed for cube placement reproducibility")
    return parser.parse_args()


def run_pipeline(gui: bool = True, n_cubes: int = 3, seed: int = None):
    """
    Full end-to-end Pick-and-Place pipeline.

    1. Initialise simulation.
    2. Spawn random coloured cubes.
    3. Detect objects with the overhead camera.
    4. For each object: run grasp_point_world() and log telemetry.
    5. Return the robot home and clean up.
    """

    # ── 1. Initialise ──────────────────────────────────────────────────────────
    print("=" * 60)
    print("  Social HW Robotics Assessment — Pick-and-Place Pipeline")
    print("=" * 60)

    env      = SimulationEnv(gui=gui)
    overhead = Camera(env, kind="overhead")
    wrist    = Camera(env, kind="wrist")
    perceive = Perception(overhead)
    robot    = RobotController(env)

    # Drop zone: to the left of the robot base, clear of the table workspace.
    # Table occupies x=0.3–0.7, y=−0.3–0.3.  We place cubes at x=0.0, y=−0.45
    # spaced 0.07 m apart so they don't stack on each other.
    DROP_ZONE_BASE = np.array([0.10, -0.45, 0.10])   # first cube lands here
    DROP_ZONE_STEP = np.array([0.0,   0.07, 0.0 ])   # shift for each subsequent cube

    # ── 2. Spawn cubes ─────────────────────────────────────────────────────────
    print(f"\n[Main] Spawning {n_cubes} cube(s)...")
    env.spawn_random_cubes(n=n_cubes, seed=seed)
    env.step(n_steps=240)   # let objects settle

    # ── 3. Detect objects ──────────────────────────────────────────────────────
    print("\n[Main] Capturing overhead image and running perception...")
    capture  = overhead.capture()
    detected = perceive.detect(capture)

    if not detected:
        print("[Main] No objects detected — exiting.")
        env.disconnect()
        return

    print(f"\n[Main] Detected {len(detected)} object(s):")
    for i, obj in enumerate(detected):
        print(f"       [{i}] {obj.label:8s} | px=({obj.centroid_px[0]},{obj.centroid_px[1]}) "
              f"| world=({obj.centroid_3d[0]:.3f}, {obj.centroid_3d[1]:.3f}, {obj.centroid_3d[2]:.3f})")

    # ── 4. Grasp loop ──────────────────────────────────────────────────────────
    log_rows = []

    for idx, obj in enumerate(detected):
        print(f"\n[Main] ── Grasping object {idx+1}/{len(detected)}: {obj.label} ──")

        # Re-detect before each grasp so any disturbance from prior drops is handled
        if idx > 0:
            capture  = overhead.capture()
            detected_now = perceive.detect(capture)
            # Find the matching label in the fresh detection
            match = next((o for o in detected_now if o.label == obj.label), None)
            if match:
                obj = match
                print(f"[Main] ↺ Refreshed position for {obj.label}: "
                      f"({obj.centroid_3d[0]:.3f}, {obj.centroid_3d[1]:.3f}, {obj.centroid_3d[2]:.3f})")
            else:
                print(f"[Main] ⚠ Could not re-detect {obj.label}, using original position")

        # Use wrist camera for a closer look just before grasping (optional)
        wrist_capture = wrist.capture()
        wrist_detect  = Perception(wrist).detect(wrist_capture)
        print(f"       Wrist camera confirms {len(wrist_detect)} object(s) in close range")

        t0 = time.time()

        # ── Clamp/validate the detected Z to the known table height ──────────
        CUBE_HALF_EXTENT = 0.025
        fixed_pos = obj.centroid_3d.copy()
        fixed_pos[2] -= CUBE_HALF_EXTENT          # top surface → centroid

        CENTROID_Z  = 0.126
        Z_TOLERANCE = 0.04
        if abs(fixed_pos[2] - CENTROID_Z) > Z_TOLERANCE:
            print(f"[Main] ⚠ z={fixed_pos[2]:.3f} out of range → clamped to {CENTROID_Z}")
            fixed_pos[2] = CENTROID_Z

        success = robot.grasp_point_world(
            object_pos=fixed_pos,
            lift_height=0.35,
        )
        elapsed = time.time() - t0

        # ── Place the cube in the drop zone (don't release in mid-air) ────────
        if success:
            drop_pos = DROP_ZONE_BASE + idx * DROP_ZONE_STEP
            print(f"[Main] Placing {obj.label} at drop zone {drop_pos}")
            robot.place_at_world(drop_pos)
        else:
            # Grasp failed — open gripper and retreat cleanly
            robot.open_gripper()
            env.step(n_steps=60)

        eef_pos, _ = robot.get_eef_pose()
        log_rows.append({
            "object_index":  idx,
            "label":         obj.label,
            "target_x":      obj.centroid_3d[0],
            "target_y":      obj.centroid_3d[1],
            "target_z":      obj.centroid_3d[2],
            "eef_x":         eef_pos[0],
            "eef_y":         eef_pos[1],
            "eef_z":         eef_pos[2],
            "success":       success,
            "elapsed_s":     round(elapsed, 3),
        })

        if success:
            print(f"[Main] ✓ Grasp succeeded in {elapsed:.2f}s")
        else:
            print(f"[Main] ✗ Grasp FAILED for {obj.label}")

        # Brief settle before the next grasp
        env.step(n_steps=60)

    # ── 5. Telemetry logging ────────────────────────────────────────────────────
    _write_csv(log_rows)

    # ── 6. Summary ─────────────────────────────────────────────────────────────
    n_success = sum(r["success"] for r in log_rows)
    print("\n" + "=" * 60)
    print(f"  Results: {n_success}/{len(log_rows)} grasps successful")
    for r in log_rows:
        icon = "✓" if r["success"] else "✗"
        print(f"  {icon} {r['label']:8s}  {r['elapsed_s']:.2f}s")
    print(f"  Telemetry logged to {LOG_FILE}")
    print("=" * 60)

    env.disconnect()
    return log_rows


def _write_csv(rows: list):
    """Write the telemetry rows to a CSV file."""
    if not rows:
        return
    path = Path(LOG_FILE)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"[Telemetry] Wrote {len(rows)} rows to {path.resolve()}")


if __name__ == "__main__":
    args = parse_args()
    #env = SimulationEnv(headless=args.headless)
    run_pipeline(
        gui    =not args.headless,
        n_cubes=args.cubes,
        seed   =args.seed,
    )
