# Notebooks

First-party Python implementations of classical robotics algorithms, organized to mirror the [awesome-matlab-robotics](https://github.com/mathworks-robotics/awesome-matlab-robotics) sections.

Each notebook is **self-contained**: math explanation + runnable code + at least one visualization. No external data needed; synthetic data is generated inline.

## Setup

```bash
pip install -r ../requirements.txt
jupyter notebook
```

## Index

| # | Notebook | Section | Algorithm |
|---|---|---|---|
| 01 | [A* path planning](01_motion_planning_astar.ipynb) | Motion Planning | A* search on a 2D occupancy grid |
| 02 | [RRT in 2D](02_motion_planning_rrt.ipynb) | Motion Planning | Rapidly-exploring Random Tree |
| 03 | [Extended Kalman Filter](03_localization_ekf.ipynb) | Localization | EKF for unicycle robot with range-bearing landmarks |
| 04 | [Particle filter](04_localization_particle_filter.ipynb) | Localization | Monte Carlo Localization for 2D pose |
| 05 | [LQR inverted pendulum](05_motion_control_pendulum_lqr.ipynb) | Motion Control | Continuous-time LQR balancing |
| 06 | [Pure pursuit](06_path_tracking_pure_pursuit.ipynb) | Path Tracking | Geometric tracking for unicycle |
| 07 | [2-link inverse kinematics](07_manipulation_ik_2link.ipynb) | Manipulation | Analytical IK for planar 2R arm |
| 08 | [Quadrotor PID](08_uav_quadrotor_pid.ipynb) | UAV | Altitude + attitude PID stabilization |
| 09 | [Lane detection](09_perception_lane_detection.ipynb) | Automated Driving | Canny + Hough lane detection |
| 10 | [Occupancy grid](10_mapping_occupancy_grid.ipynb) | Mapping | 2D occupancy grid from simulated lidar scans |

## Design principles

- **Reproducible** — fixed random seeds, no external datasets
- **Visual** — every notebook produces at least one plot or animation
- **Concise** — under 250 lines of Python per notebook
- **Pure Python** — only `numpy`, `scipy`, `matplotlib`, `opencv-python` (no ROS, no PyTorch)
- **Annotated math** — short derivations in markdown cells before the code

## Contributing a notebook

See the top-level [CONTRIBUTING.md](../CONTRIBUTING.md). Notebooks must follow the design principles above and include a one-paragraph "What this implements" block at the top.
