# Task 2 — Hybrid APF-RRT Motion Planner
**Social HW Lab — Robotics Engineer Assessment**
> Author: Srikar Reddy | March 2026

---

## Overview

Two-phase motion planner for a **Franka Panda 7-DOF arm** navigating 8 obstacles in a cluttered 3-D workspace.

- **Phase A** — APF-biased RRT finds a collision-free path
- **Phase B** — PSO post-processes the path for shorter length and smoother curvature

---

## File Structure

```
Task2_APF_RRT/
├── main_rrt.py           # Entry point — single run or benchmark mode
├── apf_rrt_planner.py    # Phase A — APF-guided RRT
├── pso_smoother.py       # Phase B — PSO waypoint optimiser
├── environment.py        # PyBullet scene — 8 obstacles, FK, collision check
├── robot_kinematics.py   # Joint limits, clamping, segment collision check
├── visualizer.py         # 3-D Matplotlib comparison plot
└── benchmark.py          # 20-run automated suite → results.csv
```

---

## Installation

```bash
cd social-hw-robotics-engineer/Task2_APF_RRT
pip install numpy matplotlib pybullet
```

---

## Usage

```bash
# Single planning run (headless)
python main_rrt.py

# Open PyBullet visualiser
python main_rrt.py --gui

# Save 3-D comparison plot
python main_rrt.py --save-plot output.png

# Run 20-iteration benchmark → results.csv
python main_rrt.py --benchmark

# Fix random seed
python main_rrt.py --seed 42
```

---

## Algorithm

### Phase A — APF-Guided RRT

The `_extend` step blends three direction vectors:

```
direction = α × d_random  +  β × K_att × d_attractive  +  γ × K_rep × d_repulsive
              (0.55)             (0.35 × 1.0)                  (0.10 × 0.4)
```

- `d_random` — standard RRT exploration
- `d_attractive` — pulls tree toward `Q_GOAL`
- `d_repulsive` — pushes away from obstacles within `d0 = 0.20 m`

A direct-connection check runs first — if the straight-line path is clear, RRT is skipped entirely.

### Phase B — PSO Smoother

Interior waypoints optimised as a flat particle vector minimising:

```
Cost = 1.0 × PathLength  +  2.0 × Curvature  +  40.0 × CollisionPenalty
```

20 particles × 100 iterations, inertia decaying linearly (ω: 0.9 → 0.4).

---

## Configuration

| Parameter | Value |
|---|---|
| `max_iter` | 6000 |
| `step_size` | 0.20 rad |
| `goal_radius` | 0.20 rad |
| `p_goal` | 0.15 |
| PSO particles | 20 |
| PSO iterations | 100 |
| Obstacles | 8 (2 path blockers + 6 canopy) |

---

## Benchmark Results (20 runs)

| Metric | Phase A — Baseline | Phase B — PSO |
|---|---|---|
| Success Rate | ~90% | ~90% |
| Path Length | baseline | reduced |
| Smoothness | baseline | significantly lower |
| Output | — | `results.csv` |
