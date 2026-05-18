"""Build all starter notebooks for awesome-python-robotics.

Run: python scripts/build_notebooks.py

Produces 10 self-contained Jupyter notebooks in notebooks/.
"""
from pathlib import Path
import nbformat as nbf

NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent / "notebooks"
NOTEBOOKS_DIR.mkdir(exist_ok=True)


def md(text):
    return nbf.v4.new_markdown_cell(text)


def code(text):
    return nbf.v4.new_code_cell(text)


def write(name, cells):
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.9"},
    }
    path = NOTEBOOKS_DIR / name
    with open(path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"  wrote {path.name}")


# ---------------------------------------------------------------------------
# 01. A* path planning
# ---------------------------------------------------------------------------
def nb_01_astar():
    cells = [
        md("""# 01 — A\\* Path Planning on an Occupancy Grid

**Section:** Motion Planning · **Mirrors MATLAB:** *Motion Planners (RRT, PRM, Hybrid A\\*)*

This notebook implements **A\\*** search on a 2-D occupancy grid with 8-connected motion. A\\* expands grid cells in order of `f = g + h`, where `g` is the cost-to-come and `h` is the heuristic estimate of cost-to-go.

We use the **Euclidean distance** heuristic, which is admissible (never overestimates) for 8-connected grids with unit-or-√2 edge costs.
"""),
        md("""## Math

For each cell `n` we keep two numbers:

- `g(n)` — actual cost from the start to `n`
- `h(n)` — heuristic estimate from `n` to the goal: `‖n − goal‖₂`

A\\* always expands the node with smallest `f(n) = g(n) + h(n)`. When the goal is popped, the path is optimal (because `h` is admissible).
"""),
        code("""import numpy as np
import matplotlib.pyplot as plt
import heapq

np.random.seed(42)
"""),
        code("""# Build a 30 x 50 occupancy grid with random rectangular obstacles
H, W = 30, 50
grid = np.zeros((H, W), dtype=np.uint8)
for _ in range(40):
    y, x = np.random.randint(2, H - 3), np.random.randint(2, W - 3)
    grid[y - 1:y + 2, x - 1:x + 2] = 1

start = (2, 2)
goal = (H - 3, W - 3)
grid[start] = 0
grid[goal] = 0
print(f"Grid: {H}x{W}, {int(grid.sum())} obstacle cells")
"""),
        code("""def astar(grid, start, goal):
    H, W = grid.shape

    def h(a, b):
        return np.hypot(a[0] - b[0], a[1] - b[1])

    open_heap = [(h(start, goal), 0.0, start, None)]
    came_from = {}
    g_score = {start: 0.0}
    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)]

    while open_heap:
        f, g, current, parent = heapq.heappop(open_heap)
        if current in came_from:
            continue
        came_from[current] = parent
        if current == goal:
            path = []
            while current is not None:
                path.append(current)
                current = came_from[current]
            return path[::-1], g_score
        for dy, dx in nbrs:
            ny, nx = current[0] + dy, current[1] + dx
            if not (0 <= ny < H and 0 <= nx < W) or grid[ny, nx]:
                continue
            tentative = g + np.hypot(dy, dx)
            neighbor = (ny, nx)
            if tentative < g_score.get(neighbor, np.inf):
                g_score[neighbor] = tentative
                heapq.heappush(
                    open_heap,
                    (tentative + h(neighbor, goal), tentative, neighbor, current),
                )
    return None, g_score


path, g_score = astar(grid, start, goal)
print(f"Path length: {len(path)} cells  ·  expanded {len(g_score)} nodes")
"""),
        code("""# Visualize
fig, ax = plt.subplots(figsize=(11, 6))
ax.imshow(grid, cmap='Greys', origin='lower')

# Heat-map of expanded nodes
expanded = np.full_like(grid, np.nan, dtype=float)
for (y, x), c in g_score.items():
    expanded[y, x] = c
ax.imshow(expanded, cmap='viridis', alpha=0.4, origin='lower')

ys, xs = zip(*path)
ax.plot(xs, ys, 'r-', lw=2.5, label='A* path')
ax.plot(start[1], start[0], 'go', markersize=14, label='Start')
ax.plot(goal[1], goal[0], 'b*', markersize=18, label='Goal')
ax.legend(loc='upper right')
ax.set_title('A* Path Planning — expanded cells shaded by g-score')
plt.tight_layout()
plt.show()
"""),
        md("""## Extensions

- **D\\* / D\\* Lite** — incremental replanning when the grid changes (use for dynamic environments).
- **Hybrid A\\*** — extends A\\* to continuous heading + kinematic constraints (used for car-like robots).
- **Jump Point Search** — a symmetry-breaking optimization for uniform grids; often 10×+ faster than A\\*.
"""),
    ]
    write("01_motion_planning_astar.ipynb", cells)


# ---------------------------------------------------------------------------
# 02. RRT
# ---------------------------------------------------------------------------
def nb_02_rrt():
    cells = [
        md("""# 02 — Rapidly-Exploring Random Tree (RRT)

**Section:** Motion Planning · **Mirrors MATLAB:** *RRT Planners for Mobile Robots*

RRT incrementally builds a tree by sampling random points in the configuration space, finding the nearest tree node, and growing a small step toward the sample. This biases the tree to rapidly explore unexplored regions.

We add a **10% goal bias** so the tree occasionally tries to extend toward the goal directly.
"""),
        code("""import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

world = 100.0
obstacles = [(30, 30, 15), (60, 70, 12), (75, 30, 10), (20, 70, 8), (50, 45, 8)]
start = np.array([5.0, 5.0])
goal = np.array([95.0, 95.0])
"""),
        code("""def in_collision(pt, obs):
    for cx, cy, r in obs:
        if (pt[0] - cx) ** 2 + (pt[1] - cy) ** 2 < r * r:
            return True
    return False


def edge_collision(p1, p2, obs, steps=25):
    for t in np.linspace(0, 1, steps):
        if in_collision(p1 * (1 - t) + p2 * t, obs):
            return True
    return False


def rrt(start, goal, obs, world=100.0, max_iter=3000, step=4.0, goal_bias=0.1, goal_tol=4.0):
    tree = [start]
    parent = {0: -1}
    for _ in range(max_iter):
        rnd = goal if np.random.random() < goal_bias else np.random.uniform(0, world, 2)
        dists = [np.linalg.norm(p - rnd) for p in tree]
        i_near = int(np.argmin(dists))
        near = tree[i_near]
        direction = rnd - near
        d = np.linalg.norm(direction)
        new = near + (direction / d) * step if d > step else rnd
        if in_collision(new, obs) or edge_collision(near, new, obs):
            continue
        tree.append(new)
        parent[len(tree) - 1] = i_near
        if np.linalg.norm(new - goal) < goal_tol:
            tree.append(goal)
            parent[len(tree) - 1] = len(tree) - 2
            return tree, parent, True
    return tree, parent, False


tree, parent, success = rrt(start, goal, obstacles)
print(f"Success: {success}  ·  tree size: {len(tree)} nodes")
"""),
        code("""fig, ax = plt.subplots(figsize=(8, 8))
for cx, cy, r in obstacles:
    ax.add_patch(plt.Circle((cx, cy), r, color='lightgrey'))
for idx, p_idx in parent.items():
    if p_idx >= 0:
        ax.plot([tree[idx][0], tree[p_idx][0]], [tree[idx][1], tree[p_idx][1]],
                'g-', lw=0.5, alpha=0.6)
if success:
    path, idx = [], len(tree) - 1
    while idx != -1:
        path.append(tree[idx])
        idx = parent[idx]
    path = np.array(path[::-1])
    ax.plot(path[:, 0], path[:, 1], 'r-', lw=2.5, label='Path')
ax.plot(*start, 'go', markersize=14, label='Start')
ax.plot(*goal, 'b*', markersize=18, label='Goal')
ax.set_xlim(0, world); ax.set_ylim(0, world); ax.set_aspect('equal')
ax.legend(loc='upper left')
ax.set_title(f'RRT — {len(tree)} nodes, goal-bias 10%')
plt.tight_layout()
plt.show()
"""),
        md("""## Notes

- **RRT\\*** rewires the tree on each insertion to converge toward an optimal path.
- **BiRRT** grows two trees (one from start, one from goal) and tries to connect them — usually faster in cluttered spaces.
- The goal-bias parameter trades exploration vs exploitation.
"""),
    ]
    write("02_motion_planning_rrt.ipynb", cells)


# ---------------------------------------------------------------------------
# 03. EKF localization
# ---------------------------------------------------------------------------
def nb_03_ekf():
    cells = [
        md("""# 03 — Extended Kalman Filter Localization

**Section:** Localization · **Mirrors MATLAB:** *Monte Carlo Localization* (similar problem, EKF approach)

We track the pose `[x, y, θ]` of a unicycle robot moving in a circle, using **range + bearing** measurements to known landmarks. The EKF linearizes the nonlinear motion and observation models around the current estimate at each step.
"""),
        md("""## Motion and observation models

**Motion:**
$x_{t+1} = x_t + \\Delta t \\cdot v \\cos\\theta_t$
$y_{t+1} = y_t + \\Delta t \\cdot v \\sin\\theta_t$
$\\theta_{t+1} = \\theta_t + \\Delta t \\cdot \\omega$

**Observation** of landmark $\\ell$:
$z = [r,\\ \\phi]^T = [\\|\\ell - p\\|,\\ \\arctan2(\\ell_y - y, \\ell_x - x) - \\theta]^T$

The EKF linearizes both with the Jacobians $G_x$ and $H$ at each step.
"""),
        code("""import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

T = 200
dt = 0.1
v, omega = 1.0, 0.1

# Ground-truth trajectory (circle)
true_x = np.zeros((T, 3))
for t in range(1, T):
    th = true_x[t - 1, 2]
    true_x[t] = true_x[t - 1] + dt * np.array([v * np.cos(th), v * np.sin(th), omega])

landmarks = np.array([[5, 5], [-5, 5], [-5, -5], [5, -5], [0, 10], [10, 0]])
"""),
        code("""Q = np.diag([0.05, 0.05, 0.01]) ** 2      # process noise
R = np.diag([0.2, 0.05]) ** 2             # measurement noise (range, bearing)


def motion(x, u, dt):
    return x + dt * np.array([u[0] * np.cos(x[2]), u[0] * np.sin(x[2]), u[1]])


def G_x(x, u, dt):
    return np.array([[1, 0, -u[0] * np.sin(x[2]) * dt],
                     [0, 1,  u[0] * np.cos(x[2]) * dt],
                     [0, 0, 1]])


def observe(x, lm):
    dx, dy = lm[0] - x[0], lm[1] - x[1]
    return np.array([np.hypot(dx, dy), np.arctan2(dy, dx) - x[2]])


def H_jacobian(x, lm):
    dx, dy = lm[0] - x[0], lm[1] - x[1]
    q = dx * dx + dy * dy
    r = np.sqrt(q)
    return np.array([[-dx / r, -dy / r, 0],
                     [ dy / q, -dx / q, -1]])


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi
"""),
        code("""mu = np.array([0.0, 0.0, 0.0])
Sigma = np.eye(3) * 0.1
mu_hist = [mu.copy()]

for t in range(1, T):
    u_noisy = [v + np.random.randn() * 0.05, omega + np.random.randn() * 0.01]
    mu = motion(mu, u_noisy, dt)
    G = G_x(mu, u_noisy, dt)
    Sigma = G @ Sigma @ G.T + Q

    for lm in landmarks:
        z = observe(true_x[t], lm) + np.random.multivariate_normal([0, 0], R)
        z_hat = observe(mu, lm)
        innov = z - z_hat
        innov[1] = wrap(innov[1])
        H = H_jacobian(mu, lm)
        S = H @ Sigma @ H.T + R
        K = Sigma @ H.T @ np.linalg.inv(S)
        mu = mu + K @ innov
        Sigma = (np.eye(3) - K @ H) @ Sigma

    mu_hist.append(mu.copy())

mu_hist = np.array(mu_hist)
err = np.linalg.norm(mu_hist[:, :2] - true_x[:, :2], axis=1)
print(f"Mean position error: {err.mean():.3f} m  ·  max: {err.max():.3f} m")
"""),
        code("""fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(true_x[:, 0], true_x[:, 1], 'b-', lw=2, label='True')
ax.plot(mu_hist[:, 0], mu_hist[:, 1], 'r--', lw=1.5, label='EKF estimate')
ax.scatter(landmarks[:, 0], landmarks[:, 1], c='k', marker='*', s=180, label='Landmarks')
ax.set_aspect('equal'); ax.grid(); ax.legend()
ax.set_title('EKF Localization with Range-Bearing Landmarks')
plt.tight_layout()
plt.show()
"""),
    ]
    write("03_localization_ekf.ipynb", cells)


# ---------------------------------------------------------------------------
# 04. Particle filter
# ---------------------------------------------------------------------------
def nb_04_particle_filter():
    cells = [
        md("""# 04 — Particle Filter (Monte Carlo) Localization

**Section:** Localization · **Mirrors MATLAB:** *Monte Carlo Localization*

The particle filter represents the belief over robot pose with a set of weighted samples (particles). At each step we (1) propagate each particle through the motion model, (2) weight by observation likelihood, (3) resample proportional to weights.
"""),
        code("""import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

T = 120
dt = 0.1

# Figure-8 ground-truth path
true_x = np.zeros((T, 3))
for t in range(T):
    a = t * dt
    true_x[t] = [5 * np.sin(a), 3 * np.sin(2 * a), a]

landmarks = np.array([[6, 0], [-6, 0], [0, 4], [0, -4], [4, 4], [-4, -4]])
"""),
        code("""N = 600
# Initial particles spread around the origin
particles = np.zeros((N, 3))
particles[:, :2] = np.random.uniform(-2, 2, (N, 2))
particles[:, 2] = np.random.uniform(-np.pi, np.pi, N)

est_hist, particle_hist = [], []
range_noise = 0.25

for t in range(T):
    u = (true_x[t] - true_x[t - 1]) if t > 0 else np.zeros(3)
    particles += u + np.random.randn(N, 3) * np.array([0.08, 0.08, 0.04])

    # Range observations to each landmark
    z = np.linalg.norm(landmarks - true_x[t, :2], axis=1) + np.random.randn(len(landmarks)) * range_noise
    expected = np.linalg.norm(landmarks[:, None, :] - particles[None, :, :2], axis=2)  # (M, N)
    err = (z[:, None] - expected) / range_noise
    log_w = -0.5 * np.sum(err ** 2, axis=0)
    w = np.exp(log_w - log_w.max())
    w /= w.sum()

    # Systematic resample
    idx = np.searchsorted(np.cumsum(w), (np.arange(N) + np.random.random()) / N)
    particles = particles[idx]

    est_hist.append(particles.mean(axis=0))
    if t in (0, T // 3, 2 * T // 3, T - 1):
        particle_hist.append((t, particles.copy()))

est_hist = np.array(est_hist)
err = np.linalg.norm(est_hist[:, :2] - true_x[:, :2], axis=1)
print(f"Mean position error: {err.mean():.3f} m")
"""),
        code("""fig, axs = plt.subplots(2, 2, figsize=(12, 10))
for ax, (t, p) in zip(axs.flat, particle_hist):
    ax.scatter(p[:, 0], p[:, 1], c='g', s=4, alpha=0.4, label='Particles')
    ax.plot(true_x[:t + 1, 0], true_x[:t + 1, 1], 'b-', lw=2, label='True')
    ax.plot(est_hist[:t + 1, 0], est_hist[:t + 1, 1], 'r--', lw=1, label='Estimate')
    ax.scatter(landmarks[:, 0], landmarks[:, 1], c='k', marker='*', s=140)
    ax.set_title(f't = {t * dt:.1f}s')
    ax.set_aspect('equal'); ax.grid(); ax.legend(loc='upper right', fontsize=8)
plt.tight_layout()
plt.show()
"""),
    ]
    write("04_localization_particle_filter.ipynb", cells)


# ---------------------------------------------------------------------------
# 05. LQR inverted pendulum
# ---------------------------------------------------------------------------
def nb_05_lqr_pendulum():
    cells = [
        md("""# 05 — LQR Control of an Inverted Pendulum on a Cart

**Section:** Motion Control · **Mirrors MATLAB:** *Multi-Loop PI Tuning for Robotic Arm* (LQR replaces PID here)

We balance a pendulum on a cart by linearizing around the upright equilibrium and solving the **continuous-time algebraic Riccati equation** for the optimal state-feedback gain `K = R⁻¹ Bᵀ P`.

Simulation uses the full **nonlinear** cart-pole dynamics — LQR is computed once from the linearization and remains stable over a wide region.
"""),
        code("""import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are

M, m, l, g = 1.0, 0.2, 0.5, 9.81

# Pendulum-UP convention: theta = 0 is the unstable upright equilibrium.
# Linearization at theta = 0:
A = np.array([[0, 1,             0, 0],
              [0, 0, -m * g / M,  0],
              [0, 0,             0, 1],
              [0, 0,  (M + m) * g / (M * l), 0]])
B = np.array([[0], [1 / M], [0], [-1 / (M * l)]])

Q = np.diag([1.0, 1.0, 10.0, 10.0])
R = np.array([[0.1]])

P = solve_continuous_are(A, B, Q, R)
K = np.linalg.inv(R) @ B.T @ P
print("LQR gain K =", K.flatten())
"""),
        code("""def nonlinear_dyn(x, u):
    # Pendulum-UP nonlinear cart-pole dynamics (theta = 0 is upright)
    pos, vel, th, om = x
    s, c = np.sin(th), np.cos(th)
    den = M + m * s ** 2
    acc = (u + m * l * s * om ** 2 - m * g * s * c) / den
    a_th = ((M + m) * g * s - u * c - m * l * s * c * om ** 2) / (l * den)
    return np.array([vel, acc, om, a_th])


x = np.array([0.0, 0.0, 0.35, 0.0])  # start ~20° off upright
T = 5.0; dt = 0.005
N = int(T / dt)
xs = np.zeros((N, 4)); us = np.zeros(N)
for i in range(N):
    u = float(-(K @ x).item())
    us[i] = u
    xs[i] = x
    x = x + dt * nonlinear_dyn(x, u)
"""),
        code("""t = np.arange(N) * dt
fig, axs = plt.subplots(2, 2, figsize=(11, 6))
labels = ['cart position (m)', 'cart velocity (m/s)', 'pendulum angle (rad)', 'angular velocity (rad/s)']
for k, ax in enumerate(axs.flat):
    if k < 4:
        ax.plot(t, xs[:, k])
        ax.set_title(labels[k]); ax.set_xlabel('time (s)'); ax.grid()

fig2, ax2 = plt.subplots(figsize=(11, 3))
ax2.plot(t, us, 'r')
ax2.set_title('Control input (N)'); ax2.set_xlabel('time (s)'); ax2.grid()
plt.tight_layout()
plt.show()
"""),
    ]
    write("05_motion_control_pendulum_lqr.ipynb", cells)


# ---------------------------------------------------------------------------
# 06. Pure pursuit
# ---------------------------------------------------------------------------
def nb_06_pure_pursuit():
    cells = [
        md("""# 06 — Pure Pursuit Path Tracking

**Section:** Path Tracking · **Mirrors MATLAB:** *Path Following with Obstacle Avoidance*

Pure pursuit is a geometric controller that finds a point on the reference path at a fixed **look-ahead distance** ahead of the robot, then computes the curvature that would carry the robot to that point in a single arc.

For a unicycle with linear velocity `v`, the commanded angular velocity is `ω = 2v sin(α) / L_d` where `α` is the angle from the robot heading to the look-ahead point.
"""),
        code("""import numpy as np
import matplotlib.pyplot as plt

# Sinusoidal reference path
xs_ref = np.linspace(0, 30, 400)
ys_ref = 3 * np.sin(xs_ref * 0.3)
path = np.column_stack([xs_ref, ys_ref])

x = np.array([0.0, -2.0, 0.0])      # x, y, theta
v = 1.5
Ld = 1.8
dt = 0.05
T = 25.0
N = int(T / dt)

hist = np.zeros((N, 3))
lookaheads = np.zeros((N, 2))
"""),
        code("""for i in range(N):
    d = np.linalg.norm(path - x[:2], axis=1)
    j = int(np.argmin(d))
    while j < len(path) - 1 and np.linalg.norm(path[j] - x[:2]) < Ld:
        j += 1
    target = path[j]
    alpha = np.arctan2(target[1] - x[1], target[0] - x[0]) - x[2]
    omega = 2 * v * np.sin(alpha) / Ld

    x = x + dt * np.array([v * np.cos(x[2]), v * np.sin(x[2]), omega])
    hist[i] = x
    lookaheads[i] = target
"""),
        code("""fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(xs_ref, ys_ref, 'b--', lw=1.5, label='Reference')
ax.plot(hist[:, 0], hist[:, 1], 'r-', lw=2, label='Robot')
ax.plot(0, -2, 'go', markersize=12, label='Start')
ax.legend(loc='upper right'); ax.grid(); ax.set_aspect('equal')
ax.set_title('Pure Pursuit Path Tracking')
plt.tight_layout()
plt.show()
"""),
    ]
    write("06_path_tracking_pure_pursuit.ipynb", cells)


# ---------------------------------------------------------------------------
# 07. 2-link IK
# ---------------------------------------------------------------------------
def nb_07_ik_2link():
    cells = [
        md("""# 07 — Analytical Inverse Kinematics for a 2-Link Planar Arm

**Section:** Manipulation · **Mirrors MATLAB:** *Inverse Kinematics*

For a planar 2-link arm with link lengths $l_1$, $l_2$ and end-effector position $(x, y)$:

$$\\cos\\theta_2 = \\frac{x^2 + y^2 - l_1^2 - l_2^2}{2 l_1 l_2}$$

$$\\theta_1 = \\arctan2(y, x) - \\arctan2(l_2 \\sin\\theta_2,\\ l_1 + l_2 \\cos\\theta_2)$$

The choice of $+\\arccos$ vs $-\\arccos$ for $\\theta_2$ selects the **elbow-up** or **elbow-down** branch.
"""),
        code("""import numpy as np
import matplotlib.pyplot as plt

l1, l2 = 1.0, 1.0


def fk(theta):
    th1, th2 = theta
    p1 = np.array([l1 * np.cos(th1), l1 * np.sin(th1)])
    p2 = p1 + np.array([l2 * np.cos(th1 + th2), l2 * np.sin(th1 + th2)])
    return p1, p2


def ik(target, elbow_up=True):
    x, y = target
    r2 = x * x + y * y
    c2 = np.clip((r2 - l1 ** 2 - l2 ** 2) / (2 * l1 * l2), -1, 1)
    th2 = np.arccos(c2) if elbow_up else -np.arccos(c2)
    th1 = np.arctan2(y, x) - np.arctan2(l2 * np.sin(th2), l1 + l2 * np.cos(th2))
    return np.array([th1, th2])
"""),
        code("""# Trace a circular trajectory
N = 60
phi = np.linspace(0, 2 * np.pi, N)
targets = np.column_stack([1.0 + 0.4 * np.cos(phi), 0.7 + 0.4 * np.sin(phi)])

fig, ax = plt.subplots(figsize=(8, 8))
for k, t in enumerate(targets):
    theta = ik(t)
    p1, p2 = fk(theta)
    a = 0.15 + 0.7 * (k / N)
    ax.plot([0, p1[0], p2[0]], [0, p1[1], p2[1]], 'b-', alpha=a, lw=1.2)
    ax.plot(p1[0], p1[1], 'ko', markersize=3, alpha=a)

ax.plot(targets[:, 0], targets[:, 1], 'r--', lw=1.5, label='Target circle')
ax.plot(0, 0, 'gs', markersize=12, label='Base')
ax.set_xlim(-2.5, 2.5); ax.set_ylim(-1.0, 2.5)
ax.set_aspect('equal'); ax.grid(); ax.legend()
ax.set_title('2-Link IK: arm poses traced through a circle (elbow-up)')
plt.tight_layout()
plt.show()
"""),
    ]
    write("07_manipulation_ik_2link.ipynb", cells)


# ---------------------------------------------------------------------------
# 08. Quadrotor PID
# ---------------------------------------------------------------------------
def nb_08_quadrotor_pid():
    cells = [
        md("""# 08 — Quadrotor Altitude + Roll PID Stabilization

**Section:** UAV · **Mirrors MATLAB:** *Approximate High-Fidelity UAV Models*

We use a planar (2-D) quadrotor model with state `[z, ż, φ, φ̇]` (altitude, vertical velocity, roll, roll rate) and two control inputs: total **thrust** and **roll torque**.

A cascaded PD controller stabilizes both axes:
- Outer altitude loop computes thrust to track `z_ref`.
- Inner attitude loop computes torque to drive `φ → 0`.
"""),
        code("""import numpy as np
import matplotlib.pyplot as plt

g = 9.81
m = 0.5
I = 0.01

Kp_z, Kd_z = 8.0, 4.5
Kp_phi, Kd_phi = 18.0, 5.5

z_ref = 2.0
phi_ref = 0.0
"""),
        code("""x = np.array([0.0, 0.0, 0.30, 0.0])  # start at ground, 0.3 rad roll
dt = 0.005
T = 5.0
N = int(T / dt)
hist = np.zeros((N, 4))
u_hist = np.zeros((N, 2))

for i in range(N):
    z, z_dot, phi, phi_dot = x
    thrust = m * (g + Kp_z * (z_ref - z) - Kd_z * z_dot)
    torque = Kp_phi * (phi_ref - phi) - Kd_phi * phi_dot
    z_ddot = thrust * np.cos(phi) / m - g
    phi_ddot = torque / I
    x = x + dt * np.array([z_dot, z_ddot, phi_dot, phi_ddot])
    hist[i] = x
    u_hist[i] = [thrust, torque]
"""),
        code("""t = np.arange(N) * dt
fig, axs = plt.subplots(2, 2, figsize=(11, 6))
axs[0, 0].plot(t, hist[:, 0]); axs[0, 0].axhline(z_ref, color='r', ls='--')
axs[0, 0].set_title('Altitude (m)')
axs[0, 1].plot(t, hist[:, 2]); axs[0, 1].axhline(phi_ref, color='r', ls='--')
axs[0, 1].set_title('Roll (rad)')
axs[1, 0].plot(t, u_hist[:, 0]); axs[1, 0].set_title('Thrust (N)')
axs[1, 1].plot(t, u_hist[:, 1]); axs[1, 1].set_title('Roll torque (N·m)')
for ax in axs.flat:
    ax.set_xlabel('time (s)'); ax.grid()
plt.tight_layout()
plt.show()
"""),
    ]
    write("08_uav_quadrotor_pid.ipynb", cells)


# ---------------------------------------------------------------------------
# 09. Lane detection
# ---------------------------------------------------------------------------
def nb_09_lane_detection():
    cells = [
        md("""# 09 — Lane Detection with Canny + Hough Transform

**Section:** Automated Driving · **Mirrors MATLAB:** *Lane Following Control with Sensor Fusion and Lane Detection*

Classical lane detection pipeline:

1. Convert to grayscale.
2. Apply **Canny** edge detection.
3. Mask a **region-of-interest** trapezoid (road ahead).
4. Run the **probabilistic Hough transform** to extract line segments.

We synthesize a road scene so the notebook needs no external images.
"""),
        code("""import numpy as np
import cv2
import matplotlib.pyplot as plt

np.random.seed(1)

H, W = 360, 640
img = np.full((H, W, 3), (60, 60, 60), dtype=np.uint8)
# Road surface
pts_road = np.array([[100, H], [W - 100, H], [W // 2 + 70, H - 220], [W // 2 - 70, H - 220]])
cv2.fillPoly(img, [pts_road], (95, 95, 95))
# Center dashed line
for y in range(H - 200, H, 35):
    cv2.line(img, (W // 2, y), (W // 2, y + 18), (240, 220, 60), 3)
# Solid white edges
cv2.line(img, (130, H - 5), (W // 2 - 60, H - 220), (240, 240, 240), 5)
cv2.line(img, (W - 130, H - 5), (W // 2 + 60, H - 220), (240, 240, 240), 5)
img = cv2.GaussianBlur(img, (3, 3), 0)
noise = np.random.randint(-12, 12, img.shape, dtype=np.int16)
img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
"""),
        code("""gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 60, 160)

mask = np.zeros_like(edges)
roi = np.array([[(60, H), (W - 60, H), (W // 2 + 60, H - 220), (W // 2 - 60, H - 220)]])
cv2.fillPoly(mask, roi, 255)
edges_roi = cv2.bitwise_and(edges, mask)

lines = cv2.HoughLinesP(edges_roi, 1, np.pi / 180, threshold=40,
                         minLineLength=35, maxLineGap=25)

out = img.copy()
if lines is not None:
    for l in lines:
        x1, y1, x2, y2 = l[0]
        cv2.line(out, (x1, y1), (x2, y2), (0, 255, 0), 3)
print(f"Detected {0 if lines is None else len(lines)} line segments")
"""),
        code("""fig, axs = plt.subplots(1, 3, figsize=(16, 5))
axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); axs[0].set_title('Synthetic input')
axs[1].imshow(edges_roi, cmap='gray');               axs[1].set_title('Canny edges (ROI)')
axs[2].imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB)); axs[2].set_title('Hough lanes (green)')
for ax in axs:
    ax.axis('off')
plt.tight_layout()
plt.show()
"""),
    ]
    write("09_perception_lane_detection.ipynb", cells)


# ---------------------------------------------------------------------------
# 10. Occupancy grid
# ---------------------------------------------------------------------------
def nb_10_occupancy_grid():
    cells = [
        md("""# 10 — 2-D Occupancy Grid from Simulated Lidar

**Section:** Mapping · **Mirrors MATLAB:** *Occupancy Grid Utilities*

Build an occupancy grid by ray-casting simulated lidar from a few known robot poses. Each cell stores a **log-odds** value that is incremented along the ray (free) and at the hit cell (occupied):

$$\\ell(c) \\leftarrow \\ell(c) + \\log\\frac{p(z\\mid m_c)}{p(z\\mid \\neg m_c)}$$

Final probability: $p = 1 / (1 + e^{-\\ell})$.
"""),
        code("""import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

world = 20.0
true_obs = [(5, 5, 8, 7), (12, 12, 14, 16), (15, 4, 18, 7), (3, 14, 5, 17)]


def hit_rect(x, y):
    for (x0, y0, x1, y1) in true_obs:
        if x0 <= x <= x1 and y0 <= y <= y1:
            return True
    return False


def lidar(pose, n_beams=180, max_range=12.0, step=0.05):
    angles = np.linspace(-np.pi, np.pi, n_beams, endpoint=False)
    ranges = np.full(n_beams, max_range)
    for i, a in enumerate(angles):
        for r in np.arange(step, max_range, step):
            x = pose[0] + r * np.cos(pose[2] + a)
            y = pose[1] + r * np.sin(pose[2] + a)
            if hit_rect(x, y):
                ranges[i] = r
                break
    return angles, ranges
"""),
        code("""res = 0.1
n = int(world / res)
log_odds = np.zeros((n, n))
l_occ, l_free = 0.7, -0.4

poses = [(3, 10, 0), (10, 3, np.pi / 2), (10, 17, -np.pi / 2), (17, 10, np.pi)]

for pose in poses:
    angles, ranges = lidar(pose, n_beams=120, max_range=12.0)
    for a, r in zip(angles, ranges):
        # Free cells along the ray
        for d in np.arange(0.1, r, res):
            x = pose[0] + d * np.cos(pose[2] + a)
            y = pose[1] + d * np.sin(pose[2] + a)
            if 0 <= x < world and 0 <= y < world:
                log_odds[int(y / res), int(x / res)] += l_free
        # Occupied cell at the hit (skip max-range "no hit" rays)
        if r < 12.0:
            x = pose[0] + r * np.cos(pose[2] + a)
            y = pose[1] + r * np.sin(pose[2] + a)
            if 0 <= x < world and 0 <= y < world:
                log_odds[int(y / res), int(x / res)] += l_occ

prob = 1 / (1 + np.exp(-log_odds))
"""),
        code("""fig, axs = plt.subplots(1, 2, figsize=(14, 6))

ax = axs[0]
for (x0, y0, x1, y1) in true_obs:
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, color='grey'))
for px, py, _ in poses:
    ax.plot(px, py, 'r*', markersize=16)
ax.set_xlim(0, world); ax.set_ylim(0, world)
ax.set_aspect('equal'); ax.grid(); ax.set_title('Ground truth + lidar poses')

axs[1].imshow(prob, origin='lower', cmap='Greys', extent=[0, world, 0, world])
for px, py, _ in poses:
    axs[1].plot(px, py, 'r*', markersize=16)
axs[1].set_title('Estimated occupancy (probability)')
axs[1].set_aspect('equal')
plt.tight_layout()
plt.show()
"""),
    ]
    write("10_mapping_occupancy_grid.ipynb", cells)


def main():
    print(f"Generating notebooks in {NOTEBOOKS_DIR}")
    nb_01_astar()
    nb_02_rrt()
    nb_03_ekf()
    nb_04_particle_filter()
    nb_05_lqr_pendulum()
    nb_06_pure_pursuit()
    nb_07_ik_2link()
    nb_08_quadrotor_pid()
    nb_09_lane_detection()
    nb_10_occupancy_grid()
    print("Done.")


if __name__ == "__main__":
    main()
