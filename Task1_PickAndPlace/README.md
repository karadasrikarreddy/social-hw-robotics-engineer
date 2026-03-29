# Task 1 — Autonomous Pick-and-Place
**Social HW Lab — Robotics Engineer Assessment**
> Author: Srikar Reddy | March 2026

---

## Demo

![Pick-and-Place Demo](demo.gif)

*Franka Panda autonomously detecting, grasping, and placing 3 coloured cubes using RGB-D perception and IK-based motion control.*

---

## Overview

Full autonomous pick-and-place pipeline using a simulated **Franka Panda 7-DOF arm** in PyBullet. An overhead RGB-D camera detects coloured cubes, back-projects their pixel centroids to 3-D world coordinates, and the arm executes a complete **grasp → lift → place** sequence for every cube.

### Key Features
- Overhead + wrist-mounted synthetic RGB-D cameras
- HSV colour segmentation with depth back-projection (`P_world = T · K⁻¹ · P_pixel · d`)
- PyBullet IK-based motion control with smooth joint-space trajectories
- Full `grasp_point_world()` pipeline: Approach → Descend → Grasp → Lift → Place
- GUI and headless modes; configurable cube count and random seed
- Telemetry logging to `telemetry.csv`

---

## File Structure
```
Task1_PickAndPlace/
├── main.py               # Pipeline orchestrator
├── simulation_env.py     # PyBullet scene — robot, table, cube spawning
├── camera.py             # RGB-D camera model + 3-D back-projection
├── perception.py         # HSV colour segmentation → world XYZ
├── robot_controller.py   # IK, trajectories, grasp_point_world(), place_at_world()
├── telemetry.csv         # Auto-generated per-run log
└── demo.gif              # Demo of 3/3 successful grasps
```

---

## Installation
```bash
git clone https://github.com/karadasrikarreddy/social-hw-robotics-engineer.git
cd social-hw-robotics-engineer/Task1_PickAndPlace
pip install numpy pybullet opencv-python matplotlib
```

---

## Usage
```bash
# GUI mode — opens PyBullet window (default, 3 cubes)
python main.py

# Set number of cubes
python main.py --cubes 3

# Headless — no window, faster for automated testing
python main.py --headless --cubes 3

# Fixed seed for reproducible placement
python main.py --headless --cubes 3 --seed 42
```

---

## How It Works

### 1. Camera Model

Intrinsics derived analytically from FOV and image size (640×480, FOV=60°):
```
fy = (H/2) / tan(fov_y/2)
fx = fy          # square pixels
cx = W/2, cy = H/2
```

PyBullet window-space depth linearised to metric:
```
d_metric = (near × far) / (far − z_buf × (far − near))
```

### 2. 3-D Back-Projection
```
x_cam =  (u − cx) / fx × d
y_cam = −(v − cy) / fy × d     # image v and camera y are opposite
z_cam = −d                       # camera looks along −Z

P_world = T_cam_world @ [x_cam, y_cam, z_cam, 1]
```

### 3. Grasp Sequence — `grasp_point_world()`

| Step | Action |
|---|---|
| 0 — Open | Open gripper fully (0.08 m gap) |
| 1 — Approach | Move EEF to 15 cm above object centroid |
| 2 — Descend | Lower to object centroid height |
| 3 — Grasp | Close fingers to 0.015 m; hold 0.5 s for contact forces to build |
| 4 — Lift | Raise to 0.35 m absolute height |
| 5 — Place | Move to drop zone; lower to surface; release |

---

## Results
```
Results: 3/3 grasps successful
  ✓ red       3.23s
  ✓ yellow    3.23s
  ✓ orange    3.19s
Telemetry logged to telemetry.csv
```

---

## Telemetry

`telemetry.csv` records per-grasp:

| Column | Description |
|---|---|
| `label` | Detected colour class |
| `target_x/y/z` | Detected cube centroid (world frame) |
| `eef_x/y/z` | End-effector position after sequence |
| `success` | True if full sequence completed without IK failure |
| `elapsed_s` | Wall-clock time for the grasp |
