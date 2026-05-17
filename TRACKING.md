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
| Manipulation | 9 | 5 | 1 | 3 | 0 | 0 |
| Legged Locomotion | 6 | 3 | 0 | 3 | 0 | 0 |
| Robot Modeling | 5 | 1 | 4 | 0 | 0 | 0 |
| Perception | 8 | 7 | 1 | 0 | 0 | 0 |
| Mapping, Localization and SLAM | 6 | 1 | 0 | 5 | 0 | 0 |
| Motion Planning and Path Planning | 7 | 5 | 1 | 1 | 0 | 0 |
| Motion Control | 5 | 4 | 1 | 0 | 0 | 0 |
| Unmanned Aerial Vehicles (UAV) | 7 | 5 | 2 | 0 | 0 | 0 |
| Marine Robotics & AUV | 7 | 4 | 1 | 2 | 0 | 0 |
| Automated Driving | 7 | 3 | 2 | 2 | 0 | 0 |
| Simulators | 5 | 3 | 1 | 0 | 1 | 0 |
| ROS and Middleware | 6 | 3 | 1 | 0 | 0 | 2 |
| Hardware and Connectivity | 10 | 6 | 0 | 2 | 1 | 1 |
| Relevant Toolboxes / Libraries | 11 | 8 | 2 | 1 | 0 | 0 |
| **TOTAL** | **109** | **63** | **20** | **19** | **4** | **3** |

**Coverage: 87 / 109 (80%) fully or partially mapped. 19 explicit gaps to close.**

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
| 5 | Collision checking (self-collisions) | ❌ | Need to add: `python-fcl`, PyBullet collision API |
| 6 | Trajectory Generation | ✅ | Robotics Toolbox for Python, MoveIt 2 |
| 7 | Safe trajectory planning (impedance-based control) | ❌ | Need to add: `franka_ros2`, `panda-py`, or Pinocchio impedance examples |
| 8 | Pick and place workflows (incl. Gazebo) | ✅ | Ravens (PyBullet) |
| 9 | Learning Dual Quaternion Modeling and Control | ❌ | Need to add: [`dqrobotics` Python bindings](https://github.com/dqrobotics/python) |

## Legged Locomotion

| # | MATLAB Entry | Status | Python Equivalent / Notes |
|---|---|---|---|
| 1 | Modeling and simulation of walking robots | ✅ | Brax, Cassie MuJoCo Sim |
| 2 | Pattern Generation for Walking Robots | ❌ | Need to add: `Crocoddyl`, `TOWR`, or `bipedal-locomotion-framework` |
| 3 | Linear Inverted Pendulum Model (LIPM) for humanoid | ❌ | Need to add: `lipm_walking_controller` (Stéphane Caron) or Pinocchio LIPM examples |
| 4 | Deep RL for Walking Robots | ✅ | legged_gym, Walk-These-Ways |
| 5 | Modeling of quadruped robot running | ✅ | quad-sdk |
| 6 | Quadruped Locomotion Using DDPG Agent | ❌ | Implicitly via Stable-Baselines3 + legged_gym, but no specific DDPG quadruped Python project listed |

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
| 6 | Object Tracking and Motion Estimation | ⚠️ | OpenCV-Python; consider adding [`bytetrack`](https://github.com/ifzhang/ByteTrack) or `SORT` for modern tracking |
| 7 | Orientation Estimation from Inertial Sensors | ✅ | filterpy |
| 8 | Drift Reduction for Visual Odometry | ✅ | pySLAM, PyPose |

## Mapping, Localization and SLAM

| # | MATLAB Entry | Status | Python Equivalent / Notes |
|---|---|---|---|
| 1 | 2D Lidar SLAM (offline + online) | ❌ | Need to add: [`python-graphslam`](https://github.com/JeffLIrion/python-graphslam), Cartographer Python bindings |
| 2 | 3D Lidar SLAM | ❌ | Need to add: [`KISS-ICP`](https://github.com/PRBonn/kiss-icp) (Python), `nano-slam` |
| 3 | SLAM Map Builder Application | ❌ | No direct Python GUI app; possibly `RTAB-Map` Python interface |
| 4 | Occupancy Grid Utilities | ❌ | Need to add: Nav2 `nav2_map_server` Python utilities or `occupancy_grid_utils` |
| 5 | Monte Carlo Localization | ✅ | Nav2 AMCL |
| 6 | Ego-Centric (Near Field) Occupancy Maps | ❌ | Need to add: [`grid_map`](https://github.com/ANYbotics/grid_map) Python bindings |

## Motion Planning and Path Planning

| # | MATLAB Entry | Status | Python Equivalent / Notes |
|---|---|---|---|
| 1 | Motion Planners (RRT, PRM, Hybrid A*) | ✅ | OMPL, PythonRobotics |
| 2 | RRT Planners for Manipulators | ✅ | MoveIt 2, OMPL |
| 3 | RRT Planners for Mobile Robots | ✅ | PythonRobotics |
| 4 | Path Planning Using Probabilistic Road Maps | ✅ | OMPL |
| 5 | Path Following with Obstacle Avoidance | ✅ | PythonRobotics, Nav2 |
| 6 | Dynamic Re-planning of Paths | ⚠️ | Nav2 covers this conceptually |
| 7 | Choosing Path Planning Algorithms (guide) | ❌ | This is documentation, not a library; could add link to PythonRobotics docs |

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
| 2 | Estimating Direction of Arrival for AUV | ❌ | Need to add: `pyroomacoustics` (DOA estimation) |
| 3 | System Identification for Blue Robotics Thrusters | ❌ | Need to add: [`sysidentpy`](https://github.com/wilsonrljr/sysidentpy) |
| 4 | LQR Control of AUV | ⚠️ | python-control provides general LQR |
| 5 | Dynamics and Control of AUV | ✅ | HoloOcean, stonefish |
| 6 | Modeling Robotic Boats | ✅ | usv_sim |
| 7 | Controller Design for Autonomous Boats | ✅ | usv_sim |

## Automated Driving

| # | MATLAB Entry | Status | Python Equivalent / Notes |
|---|---|---|---|
| 1 | Lane Following with Sensor Fusion and Lane Detection | ✅ | openpilot |
| 2 | Automate Ground Truth Labeling for Semantic Segmentation | ❌ | Need to add: [CVAT](https://github.com/cvat-ai/cvat) or [Label Studio](https://github.com/HumanSignal/label-studio) |
| 3 | Track Vehicles Using Lidar | ⚠️ | MMDetection3D + tracking, no dedicated lidar-tracking Python project |
| 4 | Track-Level Fusion of Radar and Lidar | ❌ | Need to add: [AB3DMOT](https://github.com/xinshuoweng/AB3DMOT) |
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
| 4 | Toyota HSR Examples | ❌ | Need to add: [`hsrb_interface_py`](https://github.com/hsr-project) |
| 5 | TurtleBot Robots | ✅ | TurtleBot3 Python examples (under Ground Vehicles) |
| 6 | VEX Robotics | ❌ | Need to add: [VEXcode Python](https://www.vexrobotics.com/iq/downloads) |
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
| 7 | RoadRunner | ❌ | Need to add: [CARLA Map Editor](https://carla.readthedocs.io/en/latest/tuto_M_generate_map/) or [Esmini](https://github.com/esmini/esmini) Python |
| 8 | Deep Learning Toolbox | ✅ | PyTorch |
| 9 | Reinforcement Learning Toolbox | ✅ | Stable-Baselines3, Gymnasium |
| 10 | Control System Toolbox | ✅ | python-control |
| 11 | Simscape | ⚠️ | MuJoCo + PyBullet + Drake fill this role; no symbolic multi-physics equivalent |

---

## Gaps to close (19 explicit ❌)

Quick-wins to bring coverage to 100%:

1. **Manipulation** — add `python-fcl`, `panda-py`, `dqrobotics/python`
2. **Legged** — add `Crocoddyl`, `lipm_walking_controller`, explicit DDPG-quadruped example
3. **SLAM** — add `python-graphslam`, `KISS-ICP`, `grid_map` Python, Cartographer Python
4. **Motion Planning** — add PythonRobotics decision-guide link
5. **Marine** — add `pyroomacoustics` (DOA), `sysidentpy`
6. **Automated Driving** — add `CVAT`/`Label Studio`, `AB3DMOT`
7. **Hardware** — add `hsrb_interface_py`, VEXcode Python
8. **Libraries** — add `esmini` for RoadRunner equivalent

When each is added to [README.md](README.md), update this file's table from ❌ → ✅.
