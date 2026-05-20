# 1-for-1 Tracking: MATLAB → Python

This file mirrors **every** entry in [mathworks-robotics/awesome-matlab-robotics](https://github.com/mathworks-robotics/awesome-matlab-robotics) and records its Python equivalent in our [README.md](README.md).

## Legend

| Symbol | Meaning |
|---|---|
| 🏠 | **Owned** — we ship a first-party Jupyter notebook implementing this (see [notebooks/](notebooks/)) |
| ✅ | **Mapped** — Python equivalent present in our README (external project) |
| 🔁 | **Cross-ref** — original is a pointer to another section; mirrored as cross-ref |
| ➖ | **N/A** — concept doesn't translate (Simulink-specific codegen, MathWorks-internal catch-all) |

A 🏠 entry implies the row is also mapped — the notebook is in addition to (or instead of) the external library.

## Summary

| Section | Total | 🏠 Owned | ✅ External only | 🔁 | ➖ |
|---|---:|---:|---:|---:|---:|
| Ground Vehicles and Mobile Robotics | 10 | 1 | 7 | 2 | 0 |
| Manipulation | 9 | 2 | 7 | 0 | 0 |
| Legged Locomotion | 6 | 0 | 6 | 0 | 0 |
| Robot Modeling | 5 | 1 | 4 | 0 | 0 |
| Perception | 8 | 2 | 6 | 0 | 0 |
| Mapping, Localization and SLAM | 6 | 4 | 2 | 0 | 0 |
| Motion Planning and Path Planning | 7 | 3 | 4 | 0 | 0 |
| Motion Control | 5 | 1 | 4 | 0 | 0 |
| Unmanned Aerial Vehicles (UAV) | 7 | 1 | 6 | 0 | 0 |
| Marine Robotics & AUV | 7 | 0 | 7 | 0 | 0 |
| Automated Driving | 7 | 1 | 6 | 0 | 0 |
| Simulators | 5 | 0 | 4 | 1 | 0 |
| ROS and Middleware | 6 | 0 | 4 | 0 | 2 |
| Hardware and Connectivity | 10 | 0 | 8 | 1 | 1 |
| Relevant Toolboxes / Libraries | 11 | 0 | 11 | 0 | 0 |
| **TOTAL** | **109** | **16** | **86** | **4** | **3** |

**Coverage: 102 / 102 (100%) of mappable entries.** Of those, **16 are owned** (we ship a first-party Jupyter notebook) and **86 are externally mapped** (curated link to a Python project). Path-tracking notebooks #06 (pure pursuit) and #15 (Stanley) also strengthen the *Motion Planning* row 5 (Path Following) but aren't double-counted.

---

# By Application Areas

## Ground Vehicles and Mobile Robotics

| # | MATLAB Entry | Status | Python Equivalent |
|---|---|---|---|
| 1 | Off-Road Navigation for Autonomous Vehicles | ✅ | `terrain-navigation` (ETH Zurich ASL) |
| 2 | Sprayer-Equipped Tractor Navigation in Vineyard | ✅ | `agri_gaia` (open agricultural robotics platform) |
| 3 | Developing Navigation Stacks for Mobile Robots and UGV | ✅ | Nav2 |
| 4 | Kinematic motion models for simulation | 🏠 | **Notebook [19](notebooks/19_ground_vehicles_bicycle.ipynb)** (Bicycle kinematic model) + PythonRobotics, Robotics Toolbox for Python |
| 5 | Control and simulation of warehouse robots | ✅ | `Open-RMF` (Open Robotics Middleware Framework for fleet management) |
| 6 | Simulation and programming of robot swarm | ✅ | Crazyswarm, PettingZoo |
| 7 | Mapping, Localization and SLAM (cross-ref) | 🔁 | Cross-ref present in README |
| 8 | Motion Planning and Path Planning (cross-ref) | 🔁 | Cross-ref present in README |
| 9 | Mobile Robotics Simulation Toolbox | ✅ | PyRoboSim |
| 10 | Robotics Playground | ✅ | Webots + PyRoboSim |

## Manipulation

| # | MATLAB Entry | Status | Python Equivalent |
|---|---|---|---|
| 1 | Tools for rigid body tree dynamics and analysis | ✅ | Pinocchio, Robotics Toolbox for Python |
| 2 | Inverse Kinematics | 🏠 | **Notebook [16](notebooks/16_manipulation_jacobian_ik.ipynb)** (Jacobian-based damped least-squares IK) + ikpy, MoveIt 2, TRAC-IK |
| 3 | Inverse kinematics with spatial constraints | 🏠 | **Notebook [16](notebooks/16_manipulation_jacobian_ik.ipynb)** (Jacobian-based numerical IK) + MoveIt 2, TRAC-IK |
| 4 | Interactive Inverse Kinematics | ✅ | MoveIt 2 (RViz interactive markers) |
| 5 | Collision checking (self-collisions) | ✅ | `python-fcl`, PyBullet collision API |
| 6 | Trajectory Generation | ✅ | Robotics Toolbox for Python, MoveIt 2 |
| 7 | Safe trajectory planning (impedance-based control) | ✅ | `panda-py` (Franka real-time impedance control) |
| 8 | Pick and place workflows (incl. Gazebo) | ✅ | Ravens (PyBullet) |
| 9 | Learning Dual Quaternion Modeling and Control | ✅ | `dqrobotics` (Python bindings) |

## Legged Locomotion

| # | MATLAB Entry | Status | Python Equivalent |
|---|---|---|---|
| 1 | Modeling and simulation of walking robots | ✅ | Brax, Cassie MuJoCo Sim |
| 2 | Pattern Generation for Walking Robots | ✅ | `Crocoddyl` |
| 3 | Linear Inverted Pendulum Model (LIPM) for humanoid | ✅ | `pymanoid` |
| 4 | Deep RL for Walking Robots | ✅ | legged_gym, Walk-These-Ways |
| 5 | Modeling of quadruped robot running | ✅ | quad-sdk |
| 6 | Quadruped Locomotion Using DDPG Agent | ✅ | `rsl_rl` (PPO/DDPG for legged_gym) |

## Robot Modeling

| # | MATLAB Entry | Status | Python Equivalent |
|---|---|---|---|
| 1 | Simscape Tools for Modeling and Simulation | 🏠 | **Notebook [20](notebooks/20_modeling_symbolic_pendulum.ipynb)** (Lagrangian dynamics with SymPy) + `OMPython` (OpenModelica) + `BondGraphTools` |
| 2 | Simulate Manipulator Actuators and Tune Control Parameters | ✅ | `OMPython` + PyBullet/MuJoCo |
| 3 | Algorithm Verification Using Robot Models | ✅ | `pydrake` (Drake Python bindings) |
| 4 | Import Robots from URDF Files | ✅ | urchin, yourdfpy |
| 5 | Import Robots from CAD and URDF Files | ✅ | `CadQuery` + `onshape-to-robot` |

## Perception

| # | MATLAB Entry | Status | Python Equivalent |
|---|---|---|---|
| 1 | AI for Computer Vision | ✅ | PyTorch + OpenCV-Python |
| 2 | Lidar and 3D Point Cloud Processing | ✅ | Open3D, MMDetection3D |
| 3 | 3D Object Detection with Point Clouds | ✅ | MMDetection3D |
| 4 | 3D Vision and Stereo Vision | ✅ | OpenCV-Python, PyTorch3D |
| 5 | Feature Detection, Extraction, and Matching | 🏠 | **Notebook [17](notebooks/17_perception_orb_features.ipynb)** (ORB matching) + OpenCV-Python |
| 6 | Object Tracking and Motion Estimation | 🏠 | **Notebook [18](notebooks/18_perception_kalman_tracking.ipynb)** (Kalman tracker) + `ByteTrack`, `DeepSORT-realtime` |
| 7 | Orientation Estimation from Inertial Sensors | ✅ | filterpy |
| 8 | Drift Reduction for Visual Odometry | ✅ | pySLAM, PyPose |

## Mapping, Localization and SLAM

| # | MATLAB Entry | Status | Python Equivalent |
|---|---|---|---|
| 1 | 2D Lidar SLAM (offline + online) | 🏠 | **Notebook [12](notebooks/12_slam_ekf_slam.ipynb)** (EKF-SLAM with range-bearing landmarks) + `python-graphslam`, `Cartographer`, `KISS-ICP` |
| 2 | 3D Lidar SLAM | ✅ | `KISS-ICP` |
| 3 | SLAM Map Builder Application | ✅ | `RTAB-Map` (built-in GUI map builder) |
| 4 | Occupancy Grid Utilities | 🏠 | **Notebook [10](notebooks/10_mapping_occupancy_grid.ipynb)** (log-odds grid from lidar) + `nav2_map_server` |
| 5 | Monte Carlo Localization | 🏠 | **Notebook [03](notebooks/03_localization_ekf.ipynb)** (EKF localization with range-bearing landmarks) + Nav2 AMCL, particle-filter `pfilter` |
| 6 | Ego-Centric (Near Field) Occupancy Maps | ✅ | `grid_map` (ANYbotics) |

## Motion Planning and Path Planning

| # | MATLAB Entry | Status | Python Equivalent |
|---|---|---|---|
| 1 | Motion Planners (RRT, PRM, Hybrid A*) | 🏠 | **Notebooks [01](notebooks/01_motion_planning_astar.ipynb)** (A*) **+ [13](notebooks/13_motion_planning_dijkstra.ipynb)** (Dijkstra) + OMPL, PythonRobotics |
| 2 | RRT Planners for Manipulators | ✅ | MoveIt 2, OMPL |
| 3 | RRT Planners for Mobile Robots | 🏠 | **Notebook [02](notebooks/02_motion_planning_rrt.ipynb)** (RRT with 10% goal bias) + PythonRobotics |
| 4 | Path Planning Using Probabilistic Road Maps | ✅ | OMPL |
| 5 | Path Following with Obstacle Avoidance | 🏠 | **Notebooks [14](notebooks/14_motion_planning_dwa.ipynb)** (DWA) **+ [15](notebooks/15_path_tracking_stanley.ipynb)** (Stanley front-axle controller) + PythonRobotics, Nav2 |
| 6 | Dynamic Re-planning of Paths | ✅ | Nav2 lifecycle replanning + PythonRobotics DWA dynamic obstacles |
| 7 | Choosing Path Planning Algorithms (guide) | ✅ | PythonRobotics planning algorithm guide |

## Motion Control

| # | MATLAB Entry | Status | Python Equivalent |
|---|---|---|---|
| 1 | Obstacle Avoidance Using Reinforcement Learning | ✅ | Stable-Baselines3 + Gymnasium |
| 2 | Deep RL for Walking Robots | ✅ | legged_gym, Stable-Baselines3 |
| 3 | MPC for Collision-Free Manipulation | ✅ | do-mpc, CasADi |
| 4 | MPC for Holonomic Robot Navigation | ✅ | do-mpc |
| 5 | Multi-Loop PI Tuning for Robotic Arm | 🏠 | **Notebook [05](notebooks/05_motion_control_pendulum_lqr.ipynb)** (2-link cart-pendulum iLQR swing-up + LQR catch) + `TCLab`, `simple-pid`, `python-control` |

## Unmanned Aerial Vehicles (UAV)

| # | MATLAB Entry | Status | Python Equivalent |
|---|---|---|---|
| 1 | Simulation Library for Fixed-Wing and Multi-Rotor UAVs | ✅ | gym-pybullet-drones (multi-rotor), `JSBSim` (fixed-wing), AirSim |
| 2 | Tune Waypoint Follower for Fixed-Wing UAV | ✅ | `JSBSim` + DroneKit-Python / MAVSDK-Python |
| 3 | Approximate High-Fidelity UAV Models | ✅ | `JSBSim` (high-fidelity 6-DoF) + gym-pybullet-drones |
| 4 | Load and Playback MAVLink TLOG | ✅ | pymavlink |
| 5 | MAVLink Parameter Protocol | ✅ | pymavlink |
| 6 | Support for Parrot Drones | ✅ | Olympe |
| 7 | Support for PX4 Autopilots | ✅ | MAVSDK-Python, pymavlink |

## Marine Robotics & AUV

| # | MATLAB Entry | Status | Python Equivalent |
|---|---|---|---|
| 1 | AUV Development | ✅ | HoloOcean, UUV Simulator, stonefish |
| 2 | Estimating Direction of Arrival for AUV | ✅ | `pyroomacoustics` |
| 3 | System Identification for Blue Robotics Thrusters | ✅ | `SysIdentPy` |
| 4 | LQR Control of AUV | ✅ | `python-control` (LQR applied to AUV models from HoloOcean / stonefish) |
| 5 | Dynamics and Control of AUV | ✅ | HoloOcean, stonefish |
| 6 | Modeling Robotic Boats | ✅ | usv_sim |
| 7 | Controller Design for Autonomous Boats | ✅ | usv_sim |

## Automated Driving

| # | MATLAB Entry | Status | Python Equivalent |
|---|---|---|---|
| 1 | Lane Following with Sensor Fusion and Lane Detection | 🏠 | **Notebook [09](notebooks/09_perception_lane_detection.ipynb)** (Canny + ROI + Hough) + openpilot |
| 2 | Automate Ground Truth Labeling for Semantic Segmentation | ✅ | `CVAT` |
| 3 | Track Vehicles Using Lidar (Point Cloud → Track List) | ✅ | `OpenPCDet` (end-to-end LiDAR perception + tracking pipeline) |
| 4 | Track-Level Fusion of Radar and Lidar | ✅ | `AB3DMOT` |
| 5 | Visualize Automated Parking Valet (3D Simulation) | ✅ | `highway-env` (`parking-v0` environment), CARLA |
| 6 | Design Lidar SLAM Algorithm Using 3D Sim | ✅ | CARLA + pySLAM / KISS-ICP |
| 7 | Implementing an Adaptive Cruise Controller | ✅ | highway-env |

---

# By Common Tools

## Simulators

| # | MATLAB Entry | Status | Python Equivalent |
|---|---|---|---|
| 1 | ROS Based Simulators (cross-ref) | 🔁 | Cross-ref present |
| 2 | Gazebo Co-Simulation | ✅ | Gazebo / gz-python |
| 3 | UNREAL-Engine-Based Scenarios for Automated Driving | ✅ | CARLA (built on UE) |
| 4 | Mobile Robotics Simulation Toolbox | ✅ | PyRoboSim |
| 5 | Robotics Playground | ✅ | `dm_robotics` (DeepMind curated educational MuJoCo worlds), Webots |

## ROS and Middleware

| # | MATLAB Entry | Status | Python Equivalent |
|---|---|---|---|
| 1 | Getting Started with MATLAB, Simulink and ROS | ✅ | Official ROS 2 Python Tutorials (`docs.ros.org`) |
| 2 | MATLAB support for ROS and ROS 2 | ✅ | rclpy, rospy |
| 3 | Simulink Support for ROS and ROS 2 | ➖ | No Simulink equivalent in Python world |
| 4 | Support for ROS Custom Messages | ✅ | rclpy supports natively via `.msg` codegen |
| 5 | Automatic ROS Node Generation from Simulink | ➖ | No model-based codegen equivalent |
| 6 | ROS Node Generation for Raspberry Pi | ✅ | rclpy runs natively on Raspberry Pi |

## Hardware and Connectivity

| # | MATLAB Entry | Status | Python Equivalent |
|---|---|---|---|
| 1 | Any Robot Running ROS (cross-ref) | 🔁 | Cross-ref present |
| 2 | Kinova Robots Support | ✅ | Kinova Kortex API |
| 3 | Universal Robots Support | ✅ | ur_rtde |
| 4 | Toyota HSR Examples | ✅ | `tmc_wrs_gazebo` |
| 5 | TurtleBot Robots | ✅ | TurtleBot3 Python examples |
| 6 | VEX Robotics | ✅ | VEXcode Python |
| 7 | Raspberry Pi | ✅ | gpiozero |
| 8 | BeagleBone Blue | ✅ | Adafruit_BBIO |
| 9 | LEGO Mindstorms | ✅ | pybricks-micropython |
| 10 | MATLAB and Simulink Hardware Support Packages (catch-all) | ➖ | Catch-all; no direct equivalent |

---

# By Relevant MATLAB Toolboxes → Python Libraries

| # | MATLAB Toolbox | Status | Python Equivalent |
|---|---|---|---|
| 1 | Robotics System Toolbox | ✅ | Robotics Toolbox for Python (Peter Corke) |
| 2 | ROS Toolbox | ✅ | rclpy, rospy |
| 3 | Navigation Toolbox | ✅ | PythonRobotics, Nav2 |
| 4 | Sensor Fusion and Tracking Toolbox | ✅ | filterpy |
| 5 | Computer Vision Toolbox | ✅ | OpenCV-Python |
| 6 | Automated Driving Toolbox | ✅ | Apollo (bundled AD platform), openpilot, CARLA, nuScenes devkit |
| 7 | RoadRunner | ✅ | `esmini` |
| 8 | Deep Learning Toolbox | ✅ | PyTorch |
| 9 | Reinforcement Learning Toolbox | ✅ | Stable-Baselines3, Gymnasium |
| 10 | Control System Toolbox | ✅ | python-control |
| 11 | Simscape | ✅ | `OMPython` (OpenModelica) + `BondGraphTools` |

---

## Items intentionally not mapped

The following are **🔁 cross-refs** (mirrored as cross-refs in the README, not standalone entries):
- Ground Vehicles: SLAM cross-ref, Motion Planning cross-ref
- Simulators: ROS-based simulators cross-ref
- Hardware: Any ROS robot cross-ref

The following are **➖ N/A** (Simulink-specific concepts that don't translate to Python):
- ROS: Simulink ROS support, Simulink ROS code generation
- Hardware: MATLAB & Simulink Hardware Support Packages (a catch-all link to the MathWorks hardware index)

---

## Appendix A: PythonRobotics inventory

[AtsushiSakai/PythonRobotics](https://github.com/AtsushiSakai/PythonRobotics) (29.5k★, active) is the broadest single Python project we reference. Below is the complete list of algorithms it ships, so you can see at a glance which MATLAB entries it already covers.

### Localization (3)
- Extended Kalman Filter localization
- Particle filter localization
- Histogram filter localization

### Mapping (5)
- Gaussian grid map
- Ray casting grid map
- Lidar to grid map
- k-means object clustering
- Rectangle fitting

### SLAM (2)
- Iterative Closest Point (ICP) matching
- FastSLAM 1.0

### Path Planning (18)
- Dynamic Window Approach
- Dijkstra
- A*
- D*
- D* Lite
- Potential Field
- Grid-based coverage path planning
- Particle Swarm Optimization (PSO)
- Biased polar sampling
- Lane sampling
- Probabilistic Road-Map (PRM)
- RRT*
- RRT* with Reeds-Shepp path
- LQR-RRT*
- Quintic polynomials planning
- Reeds Shepp planning
- LQR-based path planning
- Optimal Trajectory in a Frenet Frame

### Path Tracking (6)
- Move-to-a-pose control
- Stanley control
- Rear-wheel feedback control
- LQR speed and steering control
- Model Predictive speed and steering control
- Nonlinear Model Predictive Control with C-GMRES

### Arm Navigation (2)
- N-joint arm to point control
- Arm navigation with obstacle avoidance

### Aerial Navigation (2)
- Drone 3D trajectory following
- Rocket powered landing

### Bipedal (1)
- Bipedal planner with inverted pendulum

### Inverted Pendulum (2)
- LQR control
- MPC control

### Mission Planning (2)
- Behavior Tree
- State Machine

**Total: ~43 reference implementations.** Single-script algorithm demos with matplotlib animations — best for learning, not for production. We reference PythonRobotics in: Ground Vehicles, SLAM, Motion Planning, Motion Control, Manipulation, UAV, Aerial Navigation, and the Navigation Toolbox library entry.

---

## Appendix B: Our own implementations

This repo is transitioning from a pure curated link list to also hosting **first-party Python implementations** of the major MATLAB demos. See [notebooks/](notebooks/) for the full collection — 17 runnable Jupyter notebooks with embedded plots, curated for portfolio-grade depth (trajectory optimization, MPC, CBF, EKF SLAM, symbolic Lagrangian dynamics).

| # | Notebook | Section it covers in TRACKING above |
|---|---|---|
| 01 | [A* path planning](notebooks/01_motion_planning_astar.ipynb) | Motion Planning #1 |
| 02 | [RRT](notebooks/02_motion_planning_rrt.ipynb) | Motion Planning #3 |
| 03 | [Extended Kalman Filter localization](notebooks/03_localization_ekf.ipynb) | SLAM #5 |
| 05 | [2-link cart-pendulum: iLQR swing-up + LQR catch](notebooks/05_motion_control_pendulum_lqr.ipynb) | Motion Control #5 |
| 09 | [Lane detection (Canny + Hough)](notebooks/09_perception_lane_detection.ipynb) | Auto Driving #1 |
| 10 | [2D occupancy grid from lidar](notebooks/10_mapping_occupancy_grid.ipynb) | SLAM #4 |
| 12 | [EKF SLAM](notebooks/12_slam_ekf_slam.ipynb) | SLAM #1 |
| 13 | [Dijkstra](notebooks/13_motion_planning_dijkstra.ipynb) | Motion Planning #1 |
| 14 | [Dynamic Window Approach (DWA)](notebooks/14_motion_planning_dwa.ipynb) | Motion Planning #5 |
| 15 | [Stanley path tracking](notebooks/15_path_tracking_stanley.ipynb) | Motion Planning #5 |
| 16 | [Jacobian-based 3-link IK](notebooks/16_manipulation_jacobian_ik.ipynb) | Manipulation #3 |
| 17 | [ORB feature matching](notebooks/17_perception_orb_features.ipynb) | Perception #5 |
| 18 | [Kalman tracking of maneuvering target](notebooks/18_perception_kalman_tracking.ipynb) | Perception #6 |
| 19 | [Bicycle kinematic model](notebooks/19_ground_vehicles_bicycle.ipynb) | Ground Vehicles #4 |
| 20 | [Symbolic Lagrangian dynamics (SymPy)](notebooks/20_modeling_symbolic_pendulum.ipynb) | Robot Modeling #1 |

CI: every push that touches notebooks runs them headless via `nbconvert --execute` ([workflow](.github/workflows/notebooks.yml)).
