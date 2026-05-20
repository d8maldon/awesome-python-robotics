# Interactive Demos

Three pygame-based interactive demos that let you **drive, plan, and pose** the algorithms from our notebooks. Each runs in a standalone window — no Jupyter required.

## Setup

```bash
pip install pygame
```

(pygame is intentionally left out of the main `requirements.txt` because it's only needed for these demos — the 17 notebooks don't depend on it.)

## Run any demo

```bash
python demos/drive_bicycle.py     # Drive the kinematic bicycle model
python demos/click_to_plan.py     # Paint obstacles, run A* live
python demos/move_arm.py          # Mouse-controlled 2-link IK
```

## What each one does

### `drive_bicycle.py` — Drive the Bicycle
Mirrors [Notebook 19](../notebooks/19_ground_vehicles_bicycle.ipynb). The kinematic bicycle model with arrow-key throttle + steering. The camera follows the car as it leaves a trail across an infinite gridded ground plane.

| Key | Action |
|---|---|
| ↑ / ↓ | accelerate / brake |
| ← / → | steer |
| R | reset |
| Q / Esc | quit |

### `click_to_plan.py` — Click-to-plan A\*
Mirrors [Notebook 01](../notebooks/01_motion_planning_astar.ipynb). Paint or erase obstacles on a 60×40 grid and A\* replans automatically. Green cells = the search frontier; red cells = the chosen path.

| Action | Result |
|---|---|
| Left-click + drag | paint obstacles |
| Right-click + drag | erase obstacles |
| Hold `S` + click | place START |
| Hold `G` + click | place GOAL |
| `C` | clear all obstacles |
| `R` | reset start/goal |
| Space | re-run A* |
| Q / Esc | quit |

### `move_arm.py` — Mouse-Controlled 2-Link IK
Mirrors [Notebook 16](../notebooks/16_manipulation_jacobian_ik.ipynb) (Jacobian-based IK). The end-effector tracks your cursor in real time using a closed-form 2-link inverse-kinematics solution; the cursor turns orange when you leave the reachable workspace.

| Key | Action |
|---|---|
| Mouse | target end-effector position |
| E | toggle elbow-up / elbow-down |
| Q / Esc | quit |

## Adding your own demo

Pattern to follow:

```python
import pygame, math
pygame.init()
screen = pygame.display.set_mode((W, H))
clock = pygame.time.Clock()
while running:
    dt = clock.tick(60) / 1000.0
    for e in pygame.event.get():
        if e.type == pygame.QUIT: running = False
    # step algorithm with dt
    # draw to screen
    pygame.display.flip()
pygame.quit()
```

Keep each demo to one file. Reuse algorithm code from `notebooks/` by copy-paste — these are show-off demos, not a library.
