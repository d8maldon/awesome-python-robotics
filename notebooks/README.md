# Notebooks

First-party Python implementations of classical robotics algorithms, organized to mirror the [awesome-matlab-robotics](https://github.com/mathworks-robotics/awesome-matlab-robotics) sections.

Each notebook is **self-contained**: math explanation + runnable code + at least one visualization. No external data needed — synthetic inputs are generated inline. All 21 notebooks are pre-executed so GitHub renders their plots directly.

## Setup

```bash
pip install -r ../requirements.txt
jupyter notebook
```

## Index

| # | Notebook | Section | Algorithm |
|---|---|---|---|
| 01 | [A* path planning](01_motion_planning_astar.ipynb) | Motion Planning | A* on a 2D occupancy grid (8-connected) |
| 02 | [RRT](02_motion_planning_rrt.ipynb) | Motion Planning | Rapidly-exploring Random Tree with 10% goal bias |
| 03 | [Extended Kalman Filter](03_localization_ekf.ipynb) | Localization | EKF for unicycle robot with range-bearing landmarks |
| 04 | [Particle filter](04_localization_particle_filter.ipynb) | Localization | Monte Carlo Localization with systematic resampling |
| 05 | [Cart-pole swing-up + LQR catch](05_motion_control_pendulum_lqr.ipynb) | Motion Control | Hybrid Astrom-Furuta energy pumping (bang-bang) from hanging, then CARE-based LQR catch in the upright sublevel set |
| 06 | [Pure pursuit](06_path_tracking_pure_pursuit.ipynb) | Path Tracking | Geometric path tracking for a unicycle |
| 07 | [2-link analytical IK](07_manipulation_ik_2link.ipynb) | Manipulation | Closed-form inverse kinematics for planar 2R arm |
| 09 | [Lane detection](09_perception_lane_detection.ipynb) | Automated Driving | Canny + ROI mask + Hough lines on synthetic road |
| 10 | [Occupancy grid](10_mapping_occupancy_grid.ipynb) | Mapping | Log-odds grid from simulated lidar scans |
| 11 | [ICP scan matching](11_slam_icp.ipynb) | SLAM | Iterative Closest Point with SVD-based rigid transform |
| 12 | [EKF SLAM](12_slam_ekf_slam.ipynb) | SLAM | Joint robot pose + landmark estimation |
| 13 | [Dijkstra](13_motion_planning_dijkstra.ipynb) | Motion Planning | Shortest path on a grid (A* with zero heuristic) |
| 14 | [Dynamic Window Approach](14_motion_planning_dwa.ipynb) | Motion Planning | Local planner sampling velocity space |
| 15 | [Stanley path tracking](15_path_tracking_stanley.ipynb) | Path Tracking | Front-axle cross-track error control |
| 16 | [Jacobian-based IK](16_manipulation_jacobian_ik.ipynb) | Manipulation | Damped least-squares numerical IK for 3-link arm |
| 17 | [ORB feature matching](17_perception_orb_features.ipynb) | Perception | Oriented FAST + Rotated BRIEF with Hamming matching |
| 18 | [Kalman tracking](18_perception_kalman_tracking.ipynb) | Perception | Constant-velocity KF for a maneuvering target |
| 19 | [Bicycle model](19_ground_vehicles_bicycle.ipynb) | Ground Vehicles | Kinematic bicycle under three steering schedules |
| 20 | [Symbolic dynamics](20_modeling_symbolic_pendulum.ipynb) | Robot Modeling | Lagrangian derivation with SymPy + RK4 integration + empirical energy conservation 1e-8 |
| 21 | [MPC cart-pole](21_motion_control_mpc_cartpole.ipynb) | Motion Control | Receding-horizon QP with input + state constraints (scipy L-BFGS-B); Mayne 2000 stability framing |
| 22 | [CBF safety filter](22_motion_control_cbf_safety_filter.ipynb) | Motion Control | Forward-invariance filter (Nagumo 1942 / Ames 2014); closed-form QP for single-input single-obstacle case |

## Design principles

- **Reproducible** — fixed random seeds, no external datasets
- **Visual** — every notebook produces at least one plot or animation
- **Concise** — typically under 250 lines of Python per notebook
- **Lean deps** — only `numpy`, `scipy`, `matplotlib`, `opencv-python`, `sympy`
- **Annotated math** — short derivations in markdown cells before the code

## Editing notebooks

The notebooks are generated programmatically by [`../scripts/build_notebooks.py`](../scripts/build_notebooks.py) so changes stay in one place. After editing:

```bash
python scripts/build_notebooks.py          # rebuild .ipynb structure
jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb   # refresh outputs
```

CI runs `jupyter nbconvert --execute` on every push that touches `notebooks/` — see [`.github/workflows/notebooks.yml`](../.github/workflows/notebooks.yml).

## Contributing a notebook

See the top-level [CONTRIBUTING.md](../CONTRIBUTING.md). Notebooks must follow the design principles above and include a one-paragraph "What this implements" block at the top, plus a markdown reference back to the MATLAB awesome-list section it mirrors.
