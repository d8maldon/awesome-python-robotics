# 1-for-1 Tracking: MATLAB → Python

This file mirrors **every** entry in [mathworks-robotics/awesome-matlab-robotics](https://github.com/mathworks-robotics/awesome-matlab-robotics) and records whether a Python equivalent is present in our [README.md](README.md), partially covered, or still missing.

## Legend

| Symbol | Meaning |
|---|---|
| ✅ | **Mapped** — direct Python equivalent already in our README |
| ⚠️ | **Partial** — covered tangentially or by a more general entry; refinement welcome |
| ❌ | **Missing** — needs a Python equivalent added |
| 🔁 | **Cross-ref** — original is a pointer to another section; mirrored as cross-ref |
| ➖ | **N/A** — concept doesn't translate (e.g., Simulink-specific codegen, MathWorks-internal tools) |

## Summary

| Section | Total | ✅ | ⚠️ | ❌ | 🔁 | ➖ |
|---|---:|---:|---:|---:|---:|---:|
| Ground Vehicles and Mobile Robotics | 10 | 5 | 3 | 0 | 2 | 0 |
| Manipulation | 9 | 8 | 1 | 0 | 0 | 0 |
| Legged Locomotion | 6 | 6 | 0 | 0 | 0 | 0 |
| Robot Modeling | 5 | 1 | 4 | 0 | 0 | 0 |
| Perception | 8 | 7 | 1 | 0 | 0 | 0 |
| Mapping, Localization and SLAM | 6 | 6 | 0 | 0 | 0 | 0 |
| Motion Planning and Path Planning | 7 | 6 | 1 | 0 | 0 | 0 |
| Motion Control | 5 | 4 | 1 | 0 | 0 | 0 |
| Unmanned Aerial Vehicles (UAV) | 7 | 5 | 2 | 0 | 0 | 0 |
| Marine Robotics & AUV | 7 | 6 | 1 | 0 | 0 | 0 |
| Automated Driving | 7 | 5 | 2 | 0 | 0 | 0 |
| Simulators | 5 | 3 | 1 | 0 | 1 | 0 |
| ROS and Middleware | 6 | 3 | 1 | 0 | 0 | 2 |
| Hardware and Connectivity | 10 | 8 | 0 | 0 | 1 | 1 |
| Relevant Toolboxes / Libraries | 11 | 9 | 2 | 0 | 0 | 0 |
| **TOTAL** | **109** | **82** | **20** | **0** | **4** | **3** |

**Coverage: 109 / 109 (100%) — 82 fully mapped, 20 partially mapped, 0 explicit gaps.** The 4 cross-refs and 3 N/A entries do not require Python equivalents.

---

# By Application Areas

## Ground Vehicles and Mobile Robotics

| # | MATLAB Entry | Status | Python Equivalent / Notes |
|---|---|---|---|
| 1 | Off-Road Navigation for Autonomous Vehicles | ⚠️ | Nav2 covers general nav; no off-road-specific Python project listed. Candidate to add: `BARN-challenge` benchmarks |
| 2 | Sprayer-Equipped Tractor Navigation in Vineyard (Unreal) | ⚠️ | Indirectly via CARLA / Isaac Sim; no agricultural-specific Python project. Candidate: `agri_gaia` |
| 3 | Developing Navigation Stacks for Mobile Robots and UGV | ✅ | Nav2 |
| 4 | Kinematic motion models for simulation | ⚠️ | PythonRobotics, Robotics Toolbox for Python cover this |
| 5 | Control and simulation of warehouse robots | ⚠️ | PettingZoo + PyRoboSim partially; no warehouse-specific Python project |
| 6 | Simulation and programming of robot swarm | ✅ | Crazyswarm, PettingZoo |
| 7 | Mapping, Localization and SLAM (cross-ref) | 🔁 | Cross-ref present in README |
| 8 | Motion Planning and Path Planning (cross-ref) | 🔁 | Cross-ref present in README |
| 9 | Mobile Robotics Simulation Toolbox | ✅ | PyRoboSim |
| 10 | Robotics Playground | ✅ | Webots + PyRoboSim |

## Manipulation

| # | MATLAB Entry | Status | Python Equivalent / Notes |
|---|---|---|---|
| 1 | Tools for rigid body tree dynamics and analysis | ✅ | Pinocchio, Robotics Toolbox for Python |
| 2 | Inverse Kinematics | ✅ | ikpy, MoveIt 2, TRAC-IK |
| 3 | Inverse kinematics with spatial constraints | ✅ | MoveIt 2, TRAC-IK |
| 4 | Interactive Inverse Kinematics | ⚠️ | MoveIt 2 RViz markers; no dedicated interactive-IK Python lib |
| 5 | Collision checking (self-collisions) | ✅ | `python-fcl`, PyBullet collision API |
| 6 | Trajectory Generation | ✅ | Robotics Toolbox for Python, MoveIt 2 |
| 7 | Safe trajectory planning (impedance-based control) | ✅ | `panda-py` (Franka real-time impedance control) |
| 8 | Pick and place workflows (incl. Gazebo) | ✅ | Ravens (PyBullet) |
| 9 | Learning Dual Quaternion Modeling and Control | ✅ | `dqrobotics` (Python bindings) |

## Legged Locomotion

| # | MATLAB Entry | Status | Python Equivalent / Notes |
|---|---|---|---|
| 1 | Modeling and simulation of walking robots | ✅ | Brax, Cassie MuJoCo Sim |
| 2 | Pattern Generation for Walking Robots | ✅ | `Crocoddyl` (DDP trajectory optimization for legged robots) |
| 3 | Linear Inverted Pendulum Model (LIPM) for humanoid | ✅ | `pymanoid` (LIPM-based humanoid walking) |
| 4 | Deep RL for Walking Robots | ✅ | legged_gym, Walk-These-Ways |
| 5 | Modeling of quadruped robot running | ✅ | quad-sdk |
| 6 | Quadruped Locomotion Using DDPG Agent | ✅ | `rsl_rl` (PPO/DDPG implementations used by legged_gym) |

## Robot Modeling

| # | MATLAB Entry | Status | Python Equivalent / Notes |
|---|---|---|---|
| 1 | Simscape Tools for Modeling and Simulation | ⚠️ | PyBullet, MuJoCo, Drake fill this role (no 1:1 multi-physics symbolic equivalent) |
| 2 | Simulate Manipulator Actuators and Tune Control Parameters | ⚠️ | PyBullet, MuJoCo |
| 3 | Algorithm Verification Using Robot Models | ⚠️ | Drake, Robotics Toolbox for Python |
| 4 | Import Robots from URDF Files | ✅ | urchin, yourdfpy |
| 5 | Import Robots from CAD and URDF Files | ⚠️ | URDF covered; CAD import gap — could add `cadquery` or `onshape-to-robot` |

## Perception

| # | MATLAB Entry | Status | Python Equivalent / Notes |
|---|---|---|---|
| 1 | AI for Computer Vision | ✅ | PyTorch + OpenCV-Python |
| 2 | Lidar and 3D Point Cloud Processing | ✅ | Open3D, MMDetection3D |
| 3 | 3D Object Detection with Point Clouds | ✅ | MMDetection3D |
| 4 | 3D Vision and Stereo Vision | ✅ | OpenCV-Python, PyTorch3D |
| 5 | Feature Detection, Extraction, and Matching | ✅ | OpenCV-Python |
| 6 | Object Tracking and Motion Estimation | ⚠️ | OpenCV-Python; consider adding `bytetrack` or `SORT` for modern tracking |
| 7 | Orientation Estimation from Inertial Sensors | ✅ | filterpy |
| 8 | Drift Reduction for Visual Odometry | ✅ | pySLAM, PyPose |

## Mapping, Localization and SLAM

| # | MATLAB Entry | Status | Python Equivalent / Notes |
|---|---|---|---|
| 1 | 2D Lidar SLAM (offline + online) | ✅ | `python-graphslam`, `Cartographer` |
| 2 | 3D Lidar SLAM | ✅ | `KISS-ICP` |
| 3 | SLAM Map Builder Application | ✅ | `RTAB-Map` (has built-in GUI map builder) |
| 4 | Occupancy Grid Utilities | ✅ | `nav2_map_server` |
| 5 | Monte Carlo Localization | ✅ | Nav2 AMCL |
| 6 | Ego-Centric (Near Field) Occupancy Maps | ✅ | `grid_map` (ANYbotics) |

## Motion Planning and Path Planning

| # | MATLAB Entry | Status | Python Equivalent / Notes |
|---|---|---|---|
| 1 | Motion Planners (RRT, PRM, Hybrid A*) | ✅ | OMPL, PythonRobotics |
| 2 | RRT Planners for Manipulators | ✅ | MoveIt 2, OMPL |
| 3 | RRT Planners for Mobile Robots | ✅ | PythonRobotics |
| 4 | Path Planning Using Probabilistic Road Maps | ✅ | OMPL |
| 5 | Path Following with Obstacle Avoidance | ✅ | PythonRobotics, Nav2 |
| 6 | Dynamic Re-planning of Paths | ⚠️ | Nav2 covers this conceptually |
| 7 | Choosing Path Planning Algorithms (guide) | ✅ | PythonRobotics planning algorithm guide |

## Motion Control

| # | MATLAB Entry | Status | Python Equivalent / Notes |
|---|---|---|---|
| 1 | Obstacle Avoidance Using Reinforcement Learning | ✅ | Stable-Baselines3 + Gymnasium |
| 2 | Deep RL for Walking Robots | ✅ | legged_gym, Stable-Baselines3 |
| 3 | MPC for Collision-Free Manipulation | ✅ | do-mpc, CasADi |
| 4 | MPC for Holonomic Robot Navigation | ✅ | do-mpc |
| 5 | Multi-Loop PI Tuning for Robotic Arm | ⚠️ | python-control (no GUI tuner equivalent) |

## Unmanned Aerial Vehicles (UAV)

| # | MATLAB Entry | Status | Python Equivalent / Notes |
|---|---|---|---|
| 1 | Simulation Library for Fixed-Wing and Multi-Rotor UAVs | ✅ | gym-pybullet-drones, AirSim |
| 2 | Tune Waypoint Follower for Fixed-Wing UAV | ⚠️ | DroneKit-Python, MAVSDK-Python (mostly multi-rotor) |
| 3 | Approximate High-Fidelity UAV Models | ⚠️ | gym-pybullet-drones |
| 4 | Load and Playback MAVLink TLOG | ✅ | pymavlink |
| 5 | MAVLink Parameter Protocol | ✅ | pymavlink |
| 6 | Support for Parrot Drones | ✅ | Olympe |
| 7 | Support for PX4 Autopilots | ✅ | MAVSDK-Python, pymavlink |

## Marine Robotics & AUV

| # | MATLAB Entry | Status | Python Equivalent / Notes |
|---|---|---|---|
| 1 | AUV Development | ✅ | HoloOcean, UUV Simulator, stonefish |
| 2 | Estimating Direction of Arrival for AUV | ✅ | `pyroomacoustics` (DOA estimation for array signal processing) |
| 3 | System Identification for Blue Robotics Thrusters | ✅ | `SysIdentPy` |
| 4 | LQR Control of AUV | ⚠️ | python-control provides general LQR |
| 5 | Dynamics and Control of AUV | ✅ | HoloOcean, stonefish |
| 6 | Modeling Robotic Boats | ✅ | usv_sim |
| 7 | Controller Design for Autonomous Boats | ✅ | usv_sim |

## Automated Driving

| # | MATLAB Entry | Status | Python Equivalent / Notes |
|---|---|---|---|
| 1 | Lane Following with Sensor Fusion and Lane Detection | ✅ | openpilot |
| 2 | Automate Ground Truth Labeling for Semantic Segmentation | ✅ | `CVAT` |
| 3 | Track Vehicles Using Lidar | ⚠️ | MMDetection3D + tracking, no dedicated lidar-tracking Python project |
| 4 | Track-Level Fusion of Radar and Lidar | ✅ | `AB3DMOT` |
| 5 | Visualize Automated Parking Valet (3D Simulation) | ⚠️ | CARLA, MetaDrive (general) |
| 6 | Design Lidar SLAM Algorithm Using 3D Sim | ✅ | CARLA + pySLAM / KISS-ICP |
| 7 | Implementing an Adaptive Cruise Controller | ✅ | highway-env |

---

# By Common Tools

## Simulators

| # | MATLAB Entry | Status | Python Equivalent / Notes |
|---|---|---|---|
| 1 | ROS Based Simulators (cross-ref) | 🔁 | Cross-ref present |
| 2 | Gazebo Co-Simulation | ✅ | Gazebo / gz-python |
| 3 | UNREAL-Engine-Based Scenarios for Automated Driving | ✅ | CARLA (built on UE) |
| 4 | Mobile Robotics Simulation Toolbox | ✅ | PyRoboSim |
| 5 | Robotics Playground | ⚠️ | Webots, PyRoboSim |

## ROS and Middleware

| # | MATLAB Entry | Status | Python Equivalent / Notes |
|---|---|---|---|
| 1 | Getting Started with MATLAB, Simulink and ROS | ⚠️ | rclpy + rospy listed but no dedicated tutorial linked |
| 2 | MATLAB support for ROS and ROS 2 | ✅ | rclpy, rospy |
| 3 | Simulink Support for ROS and ROS 2 | ➖ | No Simulink equivalent in Python world |
| 4 | Support for ROS Custom Messages | ✅ | rclpy supports natively via `.msg` codegen |
| 5 | Automatic ROS Node Generation from Simulink | ➖ | No model-based codegen equivalent |
| 6 | ROS Node Generation for Raspberry Pi | ✅ | rclpy runs natively on Raspberry Pi |

## Hardware and Connectivity

| # | MATLAB Entry | Status | Python Equivalent / Notes |
|---|---|---|---|
| 1 | Any Robot Running ROS (cross-ref) | 🔁 | Cross-ref present |
| 2 | Kinova Robots Support | ✅ | Kinova Kortex API |
| 3 | Universal Robots Support | ✅ | ur_rtde |
| 4 | Toyota HSR Examples | ✅ | `tmc_wrs_gazebo` (HSR simulator + Python ROS interfaces) |
| 5 | TurtleBot Robots | ✅ | TurtleBot3 Python examples (under Ground Vehicles) |
| 6 | VEX Robotics | ✅ | VEXcode Python |
| 7 | Raspberry Pi | ✅ | gpiozero |
| 8 | BeagleBone Blue | ✅ | Adafruit_BBIO |
| 9 | LEGO Mindstorms | ✅ | pybricks-micropython |
| 10 | MATLAB and Simulink Hardware Support Packages (catch-all) | ➖ | Catch-all; no direct equivalent |

---

# By Relevant MATLAB Toolboxes → Python Libraries

| # | MATLAB Toolbox | Status | Python Equivalent / Notes |
|---|---|---|---|
| 1 | Robotics System Toolbox | ✅ | Robotics Toolbox for Python (Peter Corke) |
| 2 | ROS Toolbox | ✅ | rclpy, rospy |
| 3 | Navigation Toolbox | ✅ | PythonRobotics, Nav2 |
| 4 | Sensor Fusion and Tracking Toolbox | ✅ | filterpy |
| 5 | Computer Vision Toolbox | ✅ | OpenCV-Python |
| 6 | Automated Driving Toolbox | ⚠️ | CARLA + openpilot together; no single library |
| 7 | RoadRunner | ✅ | `esmini` (OpenSCENARIO player, closest open equivalent) |
| 8 | Deep Learning Toolbox | ✅ | PyTorch |
| 9 | Reinforcement Learning Toolbox | ✅ | Stable-Baselines3, Gymnasium |
| 10 | Control System Toolbox | ✅ | python-control |
| 11 | Simscape | ⚠️ | MuJoCo + PyBullet + Drake fill this role; no symbolic multi-physics equivalent |

---

## Remaining ⚠️ partial coverage

These 20 entries have a Python project that *covers* the topic but the mapping isn't as crisp as the MATLAB original. Improvements welcome via PR:

- **Ground Vehicles** (3) — off-road-specific, vineyard/agricultural, warehouse-specific projects
- **Robot Modeling** (4) — no direct Simscape multi-physics symbolic equivalent; CAD-to-URDF could add `onshape-to-robot`
- **Perception** (1) — modern object tracking (`bytetrack`, `SORT`) could be added explicitly
- **Motion Planning** (1) — dynamic re-planning is implicit in Nav2
- **Motion Control** (1) — no GUI PI-tuner equivalent for `python-control`
- **UAV** (2) — fixed-wing-specific tooling is sparser than multi-rotor
- **Marine** (1) — LQR is covered by `python-control`, not a marine-specific library
- **Automated Driving** (2) — lidar object tracking + parking valet simulation
- **Simulators** (1) — Robotics Playground equivalent
- **ROS** (1) — no consolidated Python ROS getting-started tutorial linked
- **Libraries** (2) — Automated Driving Toolbox + Simscape have no single Python equivalent

## Items intentionally not mapped

The following are **🔁 cross-refs** (already mirrored as cross-refs in the README):
- Ground Vehicles: SLAM cross-ref, Motion Planning cross-ref
- Simulators: ROS-based simulators cross-ref
- Hardware: Any ROS robot cross-ref

The following are **➖ N/A** (concepts that don't translate to Python):
- ROS: Simulink ROS support, Simulink ROS code generation
- Hardware: MATLAB hardware support packages (catch-all)
