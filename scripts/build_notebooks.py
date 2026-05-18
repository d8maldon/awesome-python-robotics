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
        md(r"""## Analytical derivation

**Problem.** Given a graph $G = (V, E)$ with non-negative edge weights $c: E \to \mathbb{R}_{\ge 0}$, a start node $s$, and a goal node $t$, find $\pi = (s = v_0, v_1, \ldots, v_k = t)$ minimizing $\sum_{i=1}^k c(v_{i-1}, v_i)$.

**Best-first search.** For every visited node $n$ keep $g(n)$ = actual cost of the best known path from $s$ to $n$. A heuristic $h: V \to \mathbb{R}_{\ge 0}$ estimates the remaining cost from $n$ to $t$. A\* always expands the node with smallest

$$f(n) = g(n) + h(n).$$

**Admissibility.** $h$ is *admissible* iff $h(n) \le h^*(n)$ for every $n$, where $h^*(n)$ is the true cost-to-go. Euclidean distance is admissible for our 8-connected grid because no path can be shorter than the straight line.

**Consistency.** $h$ is *consistent* iff $h(n) \le c(n, n') + h(n')$ for every edge. Consistency implies admissibility, and Euclidean distance is consistent.

**Optimality theorem.** If $h$ is consistent, A\* expands nodes in order of non-decreasing $f$, and the first time $t$ is popped from the open set its $g$-value is optimal: $g(t) = h^*(s)$.

**Complexity.** Binary-heap priority queue: $O(|E| \log |V|)$ time, $O(|V|)$ space.

### Compatibility check — math ↔ code

| Step | Code |
|---|---|
| $h(n) = \sqrt{(\Delta x)^2 + (\Delta y)^2}$ | `def h(a, b): return np.hypot(a[0]-b[0], a[1]-b[1])` |
| Edge cost = step distance ($1$ or $\sqrt 2$) | `tent = gv + np.hypot(dy, dx)` |
| Always pop smallest $f$ | `heapq.heappop(open_heap)` on `(f, g, node, parent)` |
| Push neighbor with $f = g + h$ | `heapq.heappush(open_heap, (tent + h((ny,nx), goal), tent, (ny,nx), cur))` |
| Skip closed nodes | `if cur in came: continue` |
| Optimal path on first pop of goal | `if cur == goal: return path, ...` |
| Reconstruct path by parent pointers | `while cur is not None: path.append(cur); cur = came[cur]` |
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

RRT incrementally builds a tree by sampling random points in the configuration space, finding the nearest tree node, and growing a small step toward the sample.
"""),
        md(r"""## Analytical derivation

**Algorithm.** Let $\mathcal{X} \subset \mathbb{R}^d$ be the configuration space and $\mathcal{X}_\text{free} \subseteq \mathcal{X}$ the collision-free subset. We grow a tree $\mathcal{T} = (V, E)$ rooted at $x_\text{init}$:

1. Sample $x_\text{rand} \in \mathcal{X}$ (uniform), or with probability $p_g$ set $x_\text{rand} \leftarrow x_\text{goal}$ (goal bias).
2. Find nearest neighbor in tree: $x_\text{near} = \arg\min_{x \in V} \|x - x_\text{rand}\|$.
3. Steer: $x_\text{new} = x_\text{near} + \eta \cdot \frac{x_\text{rand} - x_\text{near}}{\|x_\text{rand} - x_\text{near}\|}$ if distance > step size $\eta$, else $x_\text{new} = x_\text{rand}$.
4. If both $x_\text{new}$ and the line segment $(x_\text{near}, x_\text{new})$ are collision-free, add $x_\text{new}$ to $V$ and edge $(x_\text{near}, x_\text{new})$ to $E$.
5. Stop when $\|x_\text{new} - x_\text{goal}\| < \tau$ (goal tolerance).

**Probabilistic completeness theorem (LaValle, 1998).** If $\mathcal{X}_\text{free}$ has a non-empty interior and a solution path exists with positive clearance, then

$$\lim_{n \to \infty} \Pr\bigl[\text{RRT finds a feasible path in } n \text{ samples}\bigr] = 1.$$

**Not asymptotically optimal.** Unlike RRT\*, plain RRT does *not* converge to the optimal-cost path — its solution is whatever the first feasible sequence of straight-line segments happens to be. RRT\* additionally rewires neighbors within radius $r_n = \gamma (\log n / n)^{1/d}$ on each insertion and provably converges to the optimal path as $n \to \infty$.

**Goal bias.** With $p_g = 0.1$ each sample is the goal with probability $0.1$. This trades off exploration vs. exploitation; $p_g \to 1$ degrades to a greedy planner that gets stuck behind obstacles.

### Compatibility check — math ↔ code

| Step | Code |
|---|---|
| Goal bias sample | `rnd = goal if np.random.random() < 0.1 else np.random.uniform(0, world, 2)` |
| Nearest neighbor (brute force) | `i_n = int(np.argmin([np.linalg.norm(p - rnd) for p in tree]))` |
| Steer by fixed step $\eta$ | `new = near + (dirn / dist) * step if dist > step else rnd` |
| Free-space + edge collision check | `if in_col(new) or edge_col(near, new): continue` |
| Add node + parent edge | `tree.append(new); parent[len(tree)-1] = i_n` |
| Goal tolerance $\tau$ | `if np.linalg.norm(new - goal) < step: ... break` |
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
        md(r"""### Compatibility check — math ↔ code

| Math | Code |
|---|---|
| Motion model $f(x, u)$ | `def motion(x, u, dt): return x + dt*np.array([u[0]*cos(x[2]), u[0]*sin(x[2]), u[1]])` |
| Motion Jacobian $G_x = \partial f / \partial x$ | `def G_x(x, u, dt): return [[1,0,-u[0]*sin(x[2])*dt], [0,1,u[0]*cos(x[2])*dt], [0,0,1]]` |
| Observation $h(x, \ell) = [r, \phi]$ | `def observe(x, lm): return np.array([np.hypot(dx, dy), np.arctan2(dy, dx) - x[2]])` |
| Observation Jacobian $H = \partial h / \partial x$ | `def H_jacobian(x, lm): return [[-dx/r, -dy/r, 0], [dy/q, -dx/q, -1]]` |
| Predict $\mu^- = f(\mu, u)$ | `mu = motion(mu, u_noisy, dt)` |
| Predict $\Sigma^- = G \Sigma G^T + Q$ | `Sigma = G @ Sigma @ G.T + Q` |
| Bearing wrap to $(-\pi, \pi]$ | `def wrap(a): return (a + np.pi) % (2*np.pi) - np.pi` |
| Innovation $\nu = z - h(\mu^-)$ | `innov = z - z_hat; innov[1] = wrap(innov[1])` |
| Kalman gain $K = \Sigma H^T (H\Sigma H^T + R)^{-1}$ | `S = H @ Sigma @ H.T + R; K = Sigma @ H.T @ np.linalg.inv(S)` |
| Update $\mu = \mu + K\nu,\ \Sigma = (I - KH)\Sigma$ | `mu = mu + K @ innov; Sigma = (np.eye(3) - K @ H) @ Sigma` |
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

The particle filter represents the belief over robot pose with a set of weighted samples (particles).
"""),
        md(r"""## Analytical derivation

We want to recursively estimate $p(x_t \mid z_{1:t}, u_{1:t})$ where $x_t$ is the pose, $u_t$ is the control input, and $z_t$ is the observation. Bayes-filter recursion:

$$p(x_t \mid z_{1:t}, u_{1:t}) \;\propto\; p(z_t \mid x_t)\;\int p(x_t \mid x_{t-1}, u_t)\,p(x_{t-1} \mid z_{1:t-1}, u_{1:t-1})\,dx_{t-1}$$

The particle filter approximates this with $N$ weighted samples $\{(x_t^{(i)}, w_t^{(i)})\}_{i=1}^N$:

$$p(x_t \mid z_{1:t}, u_{1:t}) \;\approx\; \sum_{i=1}^N w_t^{(i)}\,\delta(x_t - x_t^{(i)})$$

**Sequential Importance Sampling (with motion-model proposal).** At each step:

1. **Predict:** $\tilde x_t^{(i)} \sim p(x_t \mid x_{t-1}^{(i)}, u_t)$  (sample from the motion model)
2. **Weight:** if we use the *motion model* as proposal, the importance weight simplifies to
$$w_t^{(i)} \;\propto\; w_{t-1}^{(i)}\,p(z_t \mid \tilde x_t^{(i)})$$
For Gaussian observation noise $z = h(x) + v$, $v \sim \mathcal{N}(0, \Sigma)$:
$$p(z_t \mid \tilde x_t^{(i)}) \;\propto\; \exp\!\Bigl(-\tfrac{1}{2}\,(z_t - h(\tilde x_t^{(i)}))^T \Sigma^{-1} (z_t - h(\tilde x_t^{(i)}))\Bigr)$$
Compute in log-space then exponentiate after subtracting the maximum (avoids underflow).
3. **Resample:** if effective sample size $N_\text{eff} = 1 / \sum (w^{(i)})^2$ falls below threshold (or every step). Systematic resampling preserves diversity with $O(N)$ cost:
$$U^{(i)} = \frac{i - 1 + U_0}{N},\quad U_0 \sim \text{Unif}(0,1)$$
and pick particle $j$ such that $\sum_{k=1}^{j-1} w^{(k)} < U^{(i)} \le \sum_{k=1}^{j} w^{(k)}$.

After resampling all weights are reset to $1/N$.

### Compatibility check — math ↔ code

| Step | Code |
|---|---|
| Predict $\tilde x_t^{(i)}$ via motion model + noise | `particles += u + np.random.randn(N, 3) * np.array([0.08, 0.08, 0.04])` |
| $h(x) = \|\ell_j - x\|$ (range to landmark $j$) | `expected = np.linalg.norm(landmarks[:, None, :] - particles[None, :, :2], axis=2)` |
| $(z - h(x))/\sigma$ standardized residual | `err = (z[:, None] - expected) / range_noise` |
| $\log w \propto -\tfrac{1}{2}\sum_j r_j^2$ | `log_w = -0.5 * np.sum(err ** 2, axis=0)` |
| Normalize after max-subtract | `w = np.exp(log_w - log_w.max()); w /= w.sum()` |
| Systematic resampling | `idx = np.searchsorted(np.cumsum(w), (np.arange(N) + np.random.random()) / N)` |
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
# 05. LQR control of a TRIPLE-link inverted pendulum on a cart
# ---------------------------------------------------------------------------
def nb_05_lqr_pendulum():
    cells = [
        md("""# 05 — LQR Control of a **Triple-Link** Inverted Pendulum on a Cart

**Section:** Motion Control · **Mirrors MATLAB:** *Multi-Loop PI Tuning for Robotic Arm* — but harder.

Three pendulum links serially attached to each other on top of a cart. You can only push the cart horizontally. The cart must be moved cleverly so all three links balance simultaneously upright. **One actuator, 4 degrees of freedom (8-dimensional state)** — strongly underactuated.

This is one of the classic benchmark problems in nonlinear control. We:

1. derive the full nonlinear equations of motion **symbolically** with `sympy.physics.mechanics`,
2. linearize around the upright equilibrium,
3. solve the continuous-time algebraic Riccati equation for the LQR gain,
4. integrate the full **nonlinear** closed-loop ODE from a small perturbation.
"""),
        md(r"""## Analytical derivation

We derive the equations of motion **by hand first**, then check that the SymPy code below produces the same thing. Workflow: math → implementation, never the other way around.

### Generalized coordinates

$q = [x,\ \theta_1,\ \theta_2,\ \theta_3]^T$ where $x$ is the cart position along the rail and $\theta_i$ is the angle of link $i$ measured from world vertical (so $\theta_i = 0$ means link $i$ is pointing straight up). Upright equilibrium is therefore $q = 0$.

Each link is treated as a point mass $m_i$ at its midpoint, link length $L_i$, cart mass $M_c$.

### Link kinematics

The base of link 1 is at the cart, so $\mathbf{r}_{J_0} = (x,\ 0)$. The top of link $i$ (= base of link $i+1$) is

$$\mathbf{r}_{J_i} \;=\; \mathbf{r}_{J_{i-1}} \;+\; L_i\,(\sin\theta_i,\ \cos\theta_i),\quad i=1,2,3$$

and the **center of mass** of link $i$ is

$$\mathbf{r}_{C_i} \;=\; \mathbf{r}_{J_{i-1}} \;+\; \tfrac{L_i}{2}\,(\sin\theta_i,\ \cos\theta_i).$$

### Kinetic and potential energy

Differentiating each $\mathbf{r}_{C_i}$ in time gives its CoM velocity $\dot{\mathbf{r}}_{C_i}$, a linear function of $\dot x$ and the $\dot\theta_j$. Total kinetic energy is the sum of point-mass terms (no rotational inertia for point masses):

$$T \;=\; \tfrac{1}{2} M_c\,\dot x^2 \;+\; \sum_{i=1}^{3} \tfrac{1}{2} m_i\,\|\dot{\mathbf{r}}_{C_i}\|^2$$

Total potential energy (only the $y$-component of each CoM matters):

$$V \;=\; \sum_{i=1}^{3} m_i\, g\,(\mathbf{r}_{C_i})_y$$

### Euler-Lagrange

Lagrangian $\mathcal{L} = T - V$. For each generalized coordinate $q_k$:

$$\frac{d}{dt}\frac{\partial \mathcal{L}}{\partial \dot q_k} \;-\; \frac{\partial \mathcal{L}}{\partial q_k} \;=\; Q_k$$

with generalized forces $Q_x = u$ (the cart-input force) and $Q_{\theta_i} = 0$ (joints are passive).

Collecting terms yields the **manipulator equation**

$$\boxed{\;M(q)\,\ddot q + C(q,\dot q)\,\dot q + G(q) \;=\; B\,u,\qquad B = \begin{bmatrix}1\\0\\0\\0\end{bmatrix}\;}$$

where $M(q)\in\mathbb{R}^{4\times 4}$ is symmetric positive-definite, $C(q,\dot q)$ holds Coriolis/centrifugal terms (quadratic in $\dot q$), and $G(q)$ is the gravity vector. The expanded scalar form fills several pages — which is why we let `sympy.physics.mechanics.LagrangesMethod` do the algebra below.

### Linearization at upright

Let $z = [q;\ \dot q]\in\mathbb{R}^8$ be the full state. At equilibrium $z^*=0$, $u^*=0$: $G(0)=0$ (the upright is a critical point of $V$), $C(0,0)=0$ (quadratic in $\dot q$). Linearizing $\ddot q = M^{-1}(-C\dot q - G + Bu)$:

$$\ddot q \;\approx\; -M(0)^{-1}\,\nabla_q G(0)\,q \;+\; M(0)^{-1} B\,u$$

Stacking with $\dot q = \dot q$:

$$\dot z \;=\; \underbrace{\begin{bmatrix} 0 & I_4 \\ -M(0)^{-1}\nabla_q G(0) & 0 \end{bmatrix}}_{A}\,z \;+\; \underbrace{\begin{bmatrix} 0 \\ M(0)^{-1} B \end{bmatrix}}_{B_\ell}\,u$$

The code below computes $A$ and $B_\ell$ by **central finite differences** of `nl_dyn(z, u)` at the origin — numerically equivalent to the formula above.

### LQR

Minimize $J = \int_0^\infty (z^T Q z + u^T R u)\,dt$. The optimal feedback is $u = -K z$ with

$$K = R^{-1} B_\ell^T P, \qquad A^T P + P A \;-\; P B_\ell R^{-1} B_\ell^T P \;+\; Q \;=\; 0$$

(continuous-time algebraic Riccati equation, solved by `scipy.linalg.solve_continuous_are`).

### Compatibility check — math ↔ code

| Step in derivation | Implementation below |
|---|---|
| CoM at $\mathbf{r}_{J_{i-1}} + \tfrac{L_i}{2}(\sin\theta_i,\cos\theta_i)$ | `P_com = joint.locatenew(..., (L_syms[i]/2)*A.y)` with `A` rotated by $\theta_i$ from world |
| $T = \sum \tfrac12 m_i\|\dot{\mathbf{r}}_{C_i}\|^2$ | `KE = sum(b.kinetic_energy(N) for b in bodies)` |
| $V = \sum m_i g(\mathbf{r}_{C_i})_y$ | `PE = sum(b.mass*g_s*b.point.pos_from(O).dot(N.y) for b in bodies)` |
| Euler-Lagrange + manipulator form | `LagrangesMethod.form_lagranges_equations()` → `mass_matrix_full`, `forcing_full` |
| Runtime $\ddot q = M^{-1}(-C\dot q - G + Bu)$ | `np.linalg.solve(M_fn(...), F_fn(...))` in `nl_dyn(z, u)` |
| $A$, $B_\ell$ Jacobians at origin | central finite differences with $\varepsilon = 10^{-6}$ |
| Riccati + LQR feedback | `solve_continuous_are` then $K = R^{-1} B_\ell^T P$ |
"""),
        code("""import numpy as np
import sympy as sp
import sympy.physics.mechanics as me
from scipy.linalg import solve_continuous_are
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

print("Deriving triple-pendulum dynamics symbolically (~10-30 s)...")
"""),
        code("""t = me.dynamicsymbols._t
q  = me.dynamicsymbols('x th1 th2 th3')
qd = [qi.diff(t) for qi in q]
u_sym = me.dynamicsymbols('u')

Mc_s, g_s = sp.symbols('Mc g', positive=True)
m_syms = sp.symbols('m1 m2 m3', positive=True)
L_syms = sp.symbols('L1 L2 L3', positive=True)

N = me.ReferenceFrame('N')
O = me.Point('O'); O.set_vel(N, 0)

Pc = O.locatenew('Pc', q[0]*N.x)
Pc.set_vel(N, qd[0]*N.x)
cart = me.Particle('Cart', Pc, Mc_s)

bodies = [cart]
joint = Pc
for i in range(3):
    A = N.orientnew(f'A{i+1}', 'Axis', [q[i+1], N.z])
    A.set_ang_vel(N, qd[i+1]*N.z)
    P_com = joint.locatenew(f'Pc{i+1}', (L_syms[i]/2)*A.y)
    P_com.v2pt_theory(joint, N, A)
    bodies.append(me.Particle(f'L{i+1}', P_com, m_syms[i]))
    next_j = joint.locatenew(f'J{i+1}', L_syms[i]*A.y)
    next_j.v2pt_theory(joint, N, A)
    joint = next_j

KE = sum(b.kinetic_energy(N) for b in bodies)
PE = sum(b.mass * g_s * b.point.pos_from(O).dot(N.y) for b in bodies)
L_lag = KE - PE

LM = me.LagrangesMethod(L_lag, q, forcelist=[(Pc, u_sym*N.x)], frame=N)
LM.form_lagranges_equations()
M_mat = LM.mass_matrix_full       # 8x8 symbolic
F_vec = LM.forcing_full           # 8x1 symbolic
print(f"M is {M_mat.shape}, F is {F_vec.shape}")
"""),
        code("""# Substitute numerical parameters
params = {Mc_s: 1.0, g_s: 9.81,
          m_syms[0]: 0.1, m_syms[1]: 0.1, m_syms[2]: 0.1,
          L_syms[0]: 0.3, L_syms[1]: 0.3, L_syms[2]: 0.3}

M_num = M_mat.subs(params)
F_num = F_vec.subs(params)

all_vars = list(q) + list(qd) + [u_sym]
M_fn = sp.lambdify(all_vars, M_num, 'numpy')
F_fn = sp.lambdify(all_vars, F_num, 'numpy')


def nl_dyn(state, u_val):
    args = list(state) + [u_val]
    Mm = np.asarray(M_fn(*args), dtype=float)
    Ff = np.asarray(F_fn(*args), dtype=float).flatten()
    return np.linalg.solve(Mm, Ff)
"""),
        code("""# Linearize around upright (z=0, u=0) by central finite differences
eps = 1e-6
z_eq = np.zeros(8)
A_lin = np.zeros((8, 8))
B_lin = np.zeros((8, 1))
for i in range(8):
    zp = z_eq.copy(); zp[i] += eps
    zm = z_eq.copy(); zm[i] -= eps
    A_lin[:, i] = (nl_dyn(zp, 0.0) - nl_dyn(zm, 0.0)) / (2 * eps)
B_lin[:, 0] = (nl_dyn(z_eq, eps) - nl_dyn(z_eq, -eps)) / (2 * eps)

eigs = np.linalg.eigvals(A_lin)
n_unstable = int(np.sum(eigs.real > 1e-6))
print(f"Open-loop eigenvalues (real parts): {np.sort(eigs.real)[::-1].round(2)}")
print(f"Unstable modes: {n_unstable}  (each link is its own inverted pendulum)")
"""),
        code("""# LQR design
Q = np.diag([1.0, 200.0, 200.0, 200.0,
             1.0,   1.0,   1.0,   1.0])
R = np.array([[0.05]])

P = solve_continuous_are(A_lin, B_lin, Q, R)
K = np.linalg.inv(R) @ B_lin.T @ P
print(f"LQR gain K = {K.flatten().round(2)}")

cl_eigs = np.linalg.eigvals(A_lin - B_lin @ K)
print(f"Closed-loop max real part: {cl_eigs.real.max():.3f}  (must be < 0 for stability)")
"""),
        code("""# Simulate nonlinear closed loop from a small perturbation (< 3° per link)
def closed_loop(t_val, z):
    u_val = float(-(K @ z).item())
    return nl_dyn(z, u_val)


z0 = np.array([0.0, 0.04, -0.02, 0.03, 0, 0, 0, 0])
sol = solve_ivp(closed_loop, [0, 4.0], z0, dense_output=True,
                rtol=1e-7, atol=1e-9, max_step=0.005)
print(f"Integration: {'success' if sol.success else 'FAILED'}  ({sol.t.size} steps)")
t_arr = np.linspace(0, 4.0, 500)
states = sol.sol(t_arr).T
"""),
        code("""fig, axs = plt.subplots(3, 1, figsize=(11, 8))
axs[0].plot(t_arr, states[:, 0])
axs[0].set_ylabel('cart x (m)'); axs[0].grid()

for i, c in enumerate(['#1f77b4', '#ff7f0e', '#2ca02c']):
    axs[1].plot(t_arr, np.degrees(states[:, 1 + i]), c=c, label=f'theta{i+1}')
axs[1].axhline(0, color='k', lw=0.5)
axs[1].set_ylabel('link angles (deg)'); axs[1].legend(); axs[1].grid()

u_hist = np.array([-(K @ s).item() for s in states])
axs[2].plot(t_arr, u_hist, color='r')
axs[2].set_ylabel('control u (N)'); axs[2].set_xlabel('t (s)'); axs[2].grid()
plt.tight_layout()
plt.show()
"""),
        code("""# Six snapshots of the triple pendulum as LQR drives it back to upright
def joints(state, L=0.3):
    x = state[0]
    th = state[1:4]
    pts = [(x, 0.0)]
    for i in range(3):
        pts.append((pts[-1][0] + L * np.sin(th[i]),
                    pts[-1][1] + L * np.cos(th[i])))
    return np.array(pts)


fig, axs = plt.subplots(1, 6, figsize=(16, 4), sharey=True)
sample_t = np.linspace(0, 4.0, 6)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
for ax, ts in zip(axs, sample_t):
    s = sol.sol(ts)
    pts = joints(s)
    ax.axhline(0, color='brown', lw=1)
    ax.add_patch(plt.Rectangle((s[0] - 0.15, -0.05), 0.3, 0.1, color='steelblue'))
    for i in range(3):
        ax.plot([pts[i, 0], pts[i + 1, 0]],
                [pts[i, 1], pts[i + 1, 1]],
                color=colors[i], lw=4, marker='o', ms=7)
    ax.set_xlim(-0.8, 0.8); ax.set_ylim(-0.2, 1.2)
    ax.set_aspect('equal'); ax.set_title(f't = {ts:.2f}s')
plt.tight_layout()
plt.show()
"""),
        md("""## Notes

- **Open-loop is triply unstable.** Each link has an unstable mode near $\\sqrt{g/L}$. LQR pushes all three to the left half-plane through a single cart input — the gain captures exactly that coordination via the cross terms.
- **Only the linearization is used to design K.** The closed loop runs the *full nonlinear* dynamics; the LQR is provably optimal only near upright but works well in a region of attraction.
- **From large initial perturbations LQR fails** because the linearization breaks down. Real triple-pendulum balancers use energy-based swing-up (Spong's method) to bring the state into the LQR's region of attraction.
- **Underactuated.** 4 DoFs, 1 input — the cart has to wiggle in a coordinated way to right all three links.
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
"""),
        md(r"""## Analytical derivation

**Geometric setup.** Place the robot at the origin with heading along $+\hat x$. The target point $(x_t, y_t)$ on the path is at distance $L_d$ from the robot:

$$x_t^2 + y_t^2 = L_d^2$$

The robot must trace a *circular arc* from its current pose to the target. Let $R$ be the radius of curvature of that arc. Geometrically, the perpendicular from the robot to the line $\text{(robot} \to \text{target)}$ has the chord property:

$$y_t = \frac{L_d^2}{2R}\quad\Longrightarrow\quad \kappa = \frac{1}{R} = \frac{2 y_t}{L_d^2}$$

Defining $\alpha$ as the angle from the robot heading to the target:

$$y_t = L_d \sin\alpha,\qquad \kappa = \frac{2 \sin\alpha}{L_d}$$

For a unicycle moving at velocity $v$ the relationship between curvature and angular velocity is $\omega = v\kappa$, giving the **pure-pursuit law**:

$$\boxed{\;\omega \;=\; \frac{2 v \sin\alpha}{L_d}\;}$$

**Choosing $L_d$.** Small $L_d$ → aggressive tracking but oscillation. Large $L_d$ → smooth but cuts corners. Common practice: $L_d \propto v$ (longer look-ahead at speed).

**Stability.** Linearizing the closed loop around a straight reference path gives a damped second-order system with damping ratio $\zeta$ increasing in $L_d$. Pure pursuit is globally stable for any $L_d > 0$ on a straight path, locally stable on curved paths if $L_d$ is large compared to the path curvature radius.

### Compatibility check — math ↔ code

| Math | Code |
|---|---|
| $\alpha = \arctan2(y_t - y_r,\ x_t - x_r) - \theta_r$ | `alpha = np.arctan2(tg[1]-x[1], tg[0]-x[0]) - x[2]` |
| Look-ahead point: scan path forward until $\|p_j - p_r\| \ge L_d$ | `while j < len(path)-1 and np.linalg.norm(path[j] - x[:2]) < Ld: j += 1` |
| $\omega = 2 v \sin\alpha / L_d$ | `omega = 2 * v * np.sin(alpha) / Ld` |
| Unicycle integration | `x = x + dt * np.array([v*cos, v*sin, omega])` |
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
        md(r"""### Compatibility check — math ↔ code

| Math | Code |
|---|---|
| $r^2 = x^2 + y^2$ | `r2 = x * x + y * y` |
| $\cos\theta_2 = \dfrac{r^2 - l_1^2 - l_2^2}{2 l_1 l_2}$ (clipped) | `c2 = np.clip((r2 - l1**2 - l2**2) / (2*l1*l2), -1, 1)` |
| $\theta_2 = \pm\arccos(\cos\theta_2)$ | `th2 = np.arccos(c2) if elbow_up else -np.arccos(c2)` |
| $\theta_1 = \arctan2(y, x) - \arctan2(l_2\sin\theta_2,\ l_1 + l_2\cos\theta_2)$ | `th1 = np.arctan2(y, x) - np.arctan2(l2*np.sin(th2), l1 + l2*np.cos(th2))` |
| Forward kinematics for visualization | `p1 = (l1*cos(th1), l1*sin(th1)); p2 = p1 + (l2*cos(th1+th2), l2*sin(th1+th2))` |
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
"""),
        md(r"""## Analytical derivation

Derive the model from Newton-Euler, then design PD controllers analytically, then verify the closed loop is stable — only then write code.

### Planar 2-D quadrotor model

State $\mathbf{x} = [z,\ \dot z,\ \phi,\ \dot\phi]^T$ where $z$ is altitude and $\phi$ is roll angle from vertical (positive = tilted right). Inputs: total thrust $T$ along the body's "up" axis, and roll torque $\tau$ about the CoM.

When the body is rolled by $\phi$, its up-axis in world coordinates is

$$\hat{u}_b \;=\; (-\sin\phi,\ \cos\phi).$$

### Newton-Euler

Translation (Newton's 2nd law in world frame, gravity acts in $-\hat{y}$):

$$m\,\ddot{\mathbf r}_{\text{world}} \;=\; T\hat{u}_b \;+\; m\mathbf{g} \;=\; (-T\sin\phi,\ T\cos\phi - mg)$$

We focus on the altitude channel (the horizontal channel is decoupled at this fidelity), giving

$$\boxed{\;\ddot z \;=\; \frac{T\cos\phi}{m} - g\;}$$

Rotation (scalar in 2-D, since roll is the only rotational DoF):

$$\boxed{\;\ddot\phi \;=\; \frac{\tau}{I}\;}$$

where $I$ is the body's moment of inertia about the roll axis.

### PD control laws

Hover requires $T = mg$ (cancels gravity at $\phi = 0$). We use **gravity-feedforward + PD** on altitude, and pure PD on attitude:

$$T \;=\; m\!\left(g \;+\; K_p^z(z_\text{ref} - z) \;-\; K_d^z\,\dot z\right)$$
$$\tau \;=\; K_p^\phi(\phi_\text{ref} - \phi) \;-\; K_d^\phi\,\dot\phi$$

### Closed-loop analysis (near hover)

Substitute $T$ into the altitude equation and use $\cos\phi \approx 1$ for small angles:

$$\ddot z \;=\; \frac{m\!\left(g + K_p^z(z_\text{ref}-z) - K_d^z \dot z\right)}{m} - g \;=\; K_p^z(z_\text{ref}-z) \;-\; K_d^z\,\dot z$$

That is a standard linear 2nd-order system. With $\tilde z = z - z_\text{ref}$:

$$\ddot{\tilde z} \;+\; K_d^z\,\dot{\tilde z} \;+\; K_p^z\,\tilde z \;=\; 0,\quad \omega_n^z = \sqrt{K_p^z},\quad \zeta^z = \frac{K_d^z}{2\sqrt{K_p^z}}$$

Identically for attitude (with $I$ in the denominator):

$$\ddot\phi \;+\; \frac{K_d^\phi}{I}\,\dot\phi \;+\; \frac{K_p^\phi}{I}\,\phi \;=\; 0,\quad \omega_n^\phi = \sqrt{\tfrac{K_p^\phi}{I}},\quad \zeta^\phi = \frac{K_d^\phi}{2\sqrt{K_p^\phi I}}$$

With our chosen gains ($K_p^z=8$, $K_d^z=4.5$, $K_p^\phi=18$, $K_d^\phi=5.5$, $I=0.01$):

- altitude: $\omega_n^z = 2.83$ rad/s, $\zeta^z = 0.80$ (well-damped)
- attitude: $\omega_n^\phi = 42.4$ rad/s, $\zeta^\phi = 6.48$ (overdamped — sluggish but very robust)

Both loops have all closed-loop poles in the open left-half plane → asymptotically stable around hover.

### Numerical-integration constraint

The attitude loop is **stiff** ($\omega_n^\phi = 42$ rad/s, much faster than altitude). Explicit Euler integration is stable iff $\Delta t \cdot \omega_n^\phi \lesssim 2$. We use $\Delta t = 0.005$ s here ($\Delta t \cdot \omega_n^\phi = 0.21$, safe). The animation in `scripts/build_animations.py` uses $\Delta t = 0.002$ s for margin around disturbances — and we discovered that $\Delta t = 0.01$ pushes the discrete eigenvalues outside the unit disk, blowing the sim up to NaN.

### Compatibility check — math ↔ code

| Derivation step | Code |
|---|---|
| $\ddot z = T\cos\phi/m - g$ | `z_ddot = thrust * np.cos(phi) / m - g` |
| $\ddot\phi = \tau/I$ | `phi_ddot = torque / I` |
| $T = m(g + K_p^z(z_\text{ref}-z) - K_d^z\dot z)$ | `thrust = m * (g + Kp_z*(z_ref - z) - Kd_z*z_dot)` |
| $\tau = K_p^\phi(\phi_\text{ref}-\phi) - K_d^\phi\dot\phi$ | `torque = Kp_phi*(phi_ref - phi) - Kd_phi*phi_dot` |
| $\Delta t < 2/\omega_n^\phi \approx 0.047$ | `dt = 0.005` (well below the bound) |
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

### Analytical underpinning + compatibility

**Canny edges.** For each pixel compute gradient magnitude

$$\|\nabla I\| = \sqrt{I_x^2 + I_y^2},\qquad I_x = \partial I / \partial x,\ I_y = \partial I / \partial y$$

via Sobel kernels, suppress non-maxima along the gradient direction, then hysteresis-threshold (`low`, `high`): keep pixels above `high`, and pixels above `low` that are connected to a `high` pixel.

**Hough line transform.** A line $\rho = x\cos\theta + y\sin\theta$ is parameterized by $(\rho, \theta)$. For each edge pixel $(x_i, y_i)$, vote in $(\rho, \theta)$-space along every curve $\rho = x_i\cos\theta + y_i\sin\theta$. Peaks in the accumulator correspond to lines that pass through many edge pixels.

The *probabilistic* Hough variant samples a random subset of edge pixels (much faster) and returns line *segments* with explicit endpoints rather than the full $(\rho, \theta)$ line.

### Compatibility check — math ↔ code

| Math | Code |
|---|---|
| Sobel-based gradient + hysteresis | `edges = cv2.Canny(gray, 60, 160)` (low=60, high=160) |
| ROI trapezoid mask | `cv2.fillPoly(mask, roi, 255); edges_roi = cv2.bitwise_and(edges, mask)` |
| Probabilistic Hough line segments | `cv2.HoughLinesP(edges_roi, 1, np.pi/180, 40, minLineLength=35, maxLineGap=25)` (resolution: 1 px, 1°; vote threshold 40) |
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

### Compatibility check — math ↔ code

| Math | Code |
|---|---|
| Ray casting per beam $\theta + \alpha$ | `for r in np.arange(step, max_range, step): x = pose[0]+r*cos(pose[2]+a); y = pose[1]+r*sin(pose[2]+a)` |
| Hit test against rectangle obstacles | `def hit_rect(x, y): for (x0,y0,x1,y1) in true_obs: if x0<=x<=x1 and y0<=y<=y1: return True` |
| Free-cell log-odds increment $\ell_\text{free}$ | `log_odds[int(y/res), int(x/res)] += l_free` along ray |
| Occupied-cell log-odds increment $\ell_\text{occ}$ | `log_odds[int(y/res), int(x/res)] += l_occ` at hit |
| Max-range "no hit" rays don't mark occupied | `if r < 12.0: ... log_odds[...] += l_occ` |
| Recover probability $p = \sigma(\ell)$ | `prob = 1 / (1 + np.exp(-log_odds))` |
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


# ---------------------------------------------------------------------------
# 11. ICP scan matching
# ---------------------------------------------------------------------------
def nb_11_icp():
    cells = [
        md("""# 11 — Iterative Closest Point (ICP) Scan Matching

**Section:** SLAM · **Mirrors MATLAB:** *2D Lidar SLAM Implementations* (scan-matching front-end)

ICP aligns two point clouds by alternating between (a) finding nearest-neighbor correspondences and (b) solving for the rigid transform $(R, t)$ that minimizes the sum-of-squared distances over those matches.
"""),
        md(r"""## Analytical derivation

**Problem.** Given source points $\{p_i\}_{i=1}^N$ and target points $\{q_i\}_{i=1}^N$ (already paired), find rotation $R \in SO(d)$ and translation $t \in \mathbb{R}^d$ minimizing

$$E(R, t) \;=\; \sum_{i=1}^{N} \|R\,p_i + t - q_i\|^2$$

**Step 1 — solve for $t$ given $R$.** Setting $\partial E / \partial t = 0$:

$$\sum_i (R\,p_i + t - q_i) = 0 \quad\Longrightarrow\quad t = \bar q - R\,\bar p,\qquad \bar p = \tfrac{1}{N}\sum p_i,\ \bar q = \tfrac{1}{N}\sum q_i$$

So the optimal translation just aligns the centroids. Substituting back, the problem becomes: find $R$ minimizing

$$E'(R) \;=\; \sum_i \|R\,p'_i - q'_i\|^2,\qquad p'_i = p_i - \bar p,\ q'_i = q_i - \bar q$$

**Step 2 — solve for $R$ (Kabsch algorithm).** Expand the norm:

$$E'(R) \;=\; \sum_i (\|p'_i\|^2 + \|q'_i\|^2) - 2 \sum_i {q'_i}^T R\,p'_i$$

The first sum is constant in $R$; minimizing $E'$ is equivalent to *maximizing* $\sum_i {q'_i}^T R\,p'_i = \mathrm{tr}\!\left(R \sum_i p'_i {q'_i}^T\right) = \mathrm{tr}(R H)$ where

$$H \;=\; \sum_i p'_i\,{q'_i}^T \quad \in \mathbb{R}^{d \times d}\quad\text{(cross-covariance)}$$

By the von Neumann trace inequality, $\mathrm{tr}(R H)$ is maximized when the singular values of $R H$ align. With $H = U \Sigma V^T$ (SVD), the optimum is

$$\boxed{\;R^\star \;=\; V\,U^T\;}$$

**Reflection guard.** If $\det(V U^T) = -1$ the answer is an improper rotation (a reflection). Replace by

$$R^\star \;=\; V\,\mathrm{diag}(1, \ldots, 1, -1)\,U^T$$

(equivalent to flipping the sign of the last column of $V$ before forming $R$).

**Iteration.** ICP alternates (1) nearest-neighbor correspondence assignment and (2) optimal rigid transform; each iteration is guaranteed to weakly decrease $E$. Convergence is to a local minimum (initialization matters).

### Compatibility check — math ↔ code

| Step | Code |
|---|---|
| Nearest-neighbor correspondence | `d = np.linalg.norm(s[:, None] - tgt[None], axis=2); m = tgt[d.argmin(axis=1)]` |
| Centroids | `sm, mm = s.mean(0), m.mean(0)` |
| Cross-covariance $H$ | `H = (s - sm).T @ (m - mm)` |
| SVD $H = U \Sigma V^T$ | `U, _, Vt = np.linalg.svd(H)` |
| Optimal $R = V U^T$ | `R = Vt.T @ U.T` |
| Reflection guard | `if np.linalg.det(R) < 0: Vt[-1] *= -1; R = Vt.T @ U.T` |
| Translation $t = \bar q - R \bar p$ | `s = (R @ s.T).T + (mm - R @ sm)` |
"""),
        code("""import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)


def make_shape(n=120):
    t = np.linspace(0, 2 * np.pi, n)
    x = 3 * np.cos(t) + 0.5 * np.sin(3 * t)
    y = 3 * np.sin(t) + 0.5 * np.cos(2 * t)
    return np.column_stack([x, y])


src = make_shape()
true_angle = 0.4
true_R = np.array([[np.cos(true_angle), -np.sin(true_angle)],
                   [np.sin(true_angle),  np.cos(true_angle)]])
true_t = np.array([1.5, -1.0])
tgt = (true_R @ src.T).T + true_t + np.random.randn(*src.shape) * 0.08
print(f"True transform: angle={np.degrees(true_angle):.1f}°, t={true_t}")
"""),
        code("""def icp(src, tgt, n_iter=40, tol=1e-6):
    s = src.copy()
    R_total = np.eye(2); t_total = np.zeros(2)
    prev_err = np.inf
    for _ in range(n_iter):
        # Nearest neighbors (brute force)
        d = np.linalg.norm(s[:, None] - tgt[None], axis=2)
        idx = d.argmin(axis=1)
        matched = tgt[idx]
        # Best-fit rotation via SVD
        s_mean, m_mean = s.mean(axis=0), matched.mean(axis=0)
        H = (s - s_mean).T @ (matched - m_mean)
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1
            R = Vt.T @ U.T
        t = m_mean - R @ s_mean
        s = (R @ s.T).T + t
        R_total = R @ R_total
        t_total = R @ t_total + t
        err = np.mean(np.linalg.norm(s - matched, axis=1) ** 2)
        if abs(prev_err - err) < tol:
            break
        prev_err = err
    return R_total, t_total, s


R_est, t_est, aligned = icp(src.copy(), tgt)
print(f"Recovered: angle={np.degrees(np.arctan2(R_est[1, 0], R_est[0, 0])):.2f}°, t={t_est}")
"""),
        code("""fig, axs = plt.subplots(1, 2, figsize=(13, 5))
axs[0].plot(src[:, 0], src[:, 1], 'b.', label='Source')
axs[0].plot(tgt[:, 0], tgt[:, 1], 'r.', label='Target')
axs[0].set_title('Before ICP'); axs[0].legend(); axs[0].set_aspect('equal'); axs[0].grid()

axs[1].plot(aligned[:, 0], aligned[:, 1], 'g.', label='Aligned source')
axs[1].plot(tgt[:, 0], tgt[:, 1], 'r.', label='Target')
axs[1].set_title('After ICP'); axs[1].legend(); axs[1].set_aspect('equal'); axs[1].grid()
plt.tight_layout()
plt.show()
"""),
    ]
    write("11_slam_icp.ipynb", cells)


# ---------------------------------------------------------------------------
# 12. EKF SLAM
# ---------------------------------------------------------------------------
def nb_12_ekf_slam():
    cells = [
        md("""# 12 — EKF SLAM with Range-Bearing Landmarks

**Section:** SLAM · **Mirrors MATLAB:** *2D Lidar SLAM Implementations* (full SLAM, not just localization)

EKF-SLAM jointly estimates the robot pose **and** the positions of $N$ landmarks. The state vector is:

$$\\mu = [x_r,\\ y_r,\\ \\theta_r,\\ x_{\\ell_1},\\ y_{\\ell_1},\\ \\dots,\\ x_{\\ell_N},\\ y_{\\ell_N}]^T$$

Each time we observe a landmark for the first time, we **initialize** its position from the current robot pose and observation.
"""),
        md(r"""## Analytical derivation

**Augmented state.** Stack robot pose and $N$ landmark positions:

$$\mu = [x_r,\ y_r,\ \theta_r,\ x_{\ell_1},\ y_{\ell_1},\ \dots,\ x_{\ell_N},\ y_{\ell_N}]^T \in \mathbb{R}^{3+2N}$$

Covariance $\Sigma \in \mathbb{R}^{(3+2N)\times(3+2N)}$ is full — the joint posterior couples the robot to every landmark seen.

**Prediction step.** The control $u = (v, \omega)$ moves only the robot:

$$\mu_{t+1}^{1:3} = \mu_t^{1:3} + \Delta t \begin{bmatrix} v\cos\theta_r \\ v\sin\theta_r \\ \omega \end{bmatrix},\qquad \mu_{t+1}^{4:} = \mu_t^{4:}$$

The full-state Jacobian $G$ is identity except for the $3\times 3$ block in the top-left:

$$G = I_{3+2N} + \begin{bmatrix} G_r - I_3 & 0 \\ 0 & 0 \end{bmatrix},\qquad G_r = \begin{bmatrix} 1 & 0 & -v\sin\theta_r\,\Delta t \\ 0 & 1 & v\cos\theta_r\,\Delta t \\ 0 & 0 & 1 \end{bmatrix}$$

Predict covariance: $\Sigma = G \Sigma G^T + F^T Q F$ where $F = [I_3,\ 0]$ injects process noise into only the robot block.

**Landmark initialization.** When landmark $j$ is first observed with $(r, \phi)$:

$$\mu^{\ell_j} = \mu^r + r \begin{bmatrix} \cos(\theta_r + \phi) \\ \sin(\theta_r + \phi) \end{bmatrix}$$

**Observation update.** For range-bearing observation $z = (r, \phi)$ of landmark $j$ at $(\ell_x, \ell_y)$, define $\delta = (\delta_x, \delta_y) = (\ell_x - x_r, \ell_y - y_r)$ and $q = \delta^T\delta$. The expected measurement is

$$\hat z = \begin{bmatrix} \sqrt{q} \\ \arctan2(\delta_y, \delta_x) - \theta_r \end{bmatrix}$$

**Sparse observation Jacobian.** Only the robot-pose columns (1–3) and the columns of landmark $j$ (3+2j-1, 3+2j) are nonzero:

$$H \in \mathbb{R}^{2 \times (3+2N)}:\quad H_{(1:2,\,r)} = \frac{1}{q}\begin{bmatrix} -\sqrt{q}\delta_x & -\sqrt{q}\delta_y & 0 \\ \delta_y & -\delta_x & -q \end{bmatrix},\quad H_{(1:2,\,\ell_j)} = \frac{1}{q}\begin{bmatrix} \sqrt{q}\delta_x & \sqrt{q}\delta_y \\ -\delta_y & \delta_x \end{bmatrix}$$

Innovation: $\nu = z - \hat z$ (wrap bearing to $(-\pi, \pi]$). Innovation covariance: $S = H \Sigma H^T + R$. Kalman gain: $K = \Sigma H^T S^{-1}$. Update:

$$\mu \leftarrow \mu + K\nu,\qquad \Sigma \leftarrow (I - K H)\,\Sigma$$

The cross-correlations in $\Sigma$ are what couple robot-pose corrections to landmark refinements — that's where the "SLAM" magic lives.

### Compatibility check — math ↔ code

| Step | Code |
|---|---|
| Robot motion in augmented state | `mu[:3] = mu[:3] + dt * np.array([u[0]*np.cos(th), u[0]*np.sin(th), u[1]])` |
| Sparse motion Jacobian (identity + robot block) | `G = np.eye(n); G[0,2] = -dt*u[0]*np.sin(th); G[1,2] = dt*u[0]*np.cos(th)` |
| Inject process noise via $F$ | `F = np.zeros((3, n)); F[:3,:3] = np.eye(3); Sigma = G @ Sigma @ G.T + F.T @ Q @ F` |
| First-observation landmark init | `mu[3+2*i] = mu[0] + z[0]*np.cos(z[1]+mu[2]); mu[3+2*i+1] = ...sin(...)` |
| Expected measurement $\hat z$ | `z_hat = np.array([r, wrap(np.arctan2(dy, dx) - mu[2])])` |
| Sparse $H$ (only robot cols + landmark $j$ cols filled) | `H = np.zeros((2, n)); H[:, :3] = ...; H[:, 3+2*i:3+2*i+2] = ...` |
| Kalman update | `K = Sigma @ H.T @ np.linalg.inv(S); mu = mu + K @ innov; Sigma = (I - K @ H) @ Sigma` |
"""),
        code("""import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

T = 200; dt = 0.1
v, omega = 1.0, 0.1
true_robot = np.zeros((T, 3))
for k in range(1, T):
    th = true_robot[k - 1, 2]
    true_robot[k] = true_robot[k - 1] + dt * np.array([v * np.cos(th), v * np.sin(th), omega])

true_lm = np.array([[5, 5], [-5, 5], [0, -8], [7, -2]])
N_LM = len(true_lm)
n = 3 + 2 * N_LM
"""),
        code("""mu = np.zeros(n)
Sigma = np.eye(n) * 1e6
Sigma[:3, :3] = np.eye(3) * 0.01

Q = np.diag([0.05, 0.05, 0.01]) ** 2
R_obs = np.diag([0.2, 0.05]) ** 2
seen = [False] * N_LM
mu_hist = [mu.copy()]


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


for k in range(1, T):
    u = [v + np.random.randn() * 0.05, omega + np.random.randn() * 0.01]
    th = mu[2]
    mu[:3] = mu[:3] + dt * np.array([u[0] * np.cos(th), u[0] * np.sin(th), u[1]])
    G = np.eye(n)
    G[0, 2] = -dt * u[0] * np.sin(th)
    G[1, 2] =  dt * u[0] * np.cos(th)
    F = np.zeros((3, n)); F[:3, :3] = np.eye(3)
    Sigma = G @ Sigma @ G.T + F.T @ Q @ F

    for i, lm in enumerate(true_lm):
        z = np.array([
            np.linalg.norm(lm - true_robot[k, :2]),
            np.arctan2(lm[1] - true_robot[k, 1], lm[0] - true_robot[k, 0]) - true_robot[k, 2],
        ]) + np.random.multivariate_normal([0, 0], R_obs)

        if not seen[i]:
            mu[3 + 2 * i]     = mu[0] + z[0] * np.cos(z[1] + mu[2])
            mu[3 + 2 * i + 1] = mu[1] + z[0] * np.sin(z[1] + mu[2])
            seen[i] = True

        dx = mu[3 + 2 * i] - mu[0]
        dy = mu[3 + 2 * i + 1] - mu[1]
        q = dx * dx + dy * dy
        r = np.sqrt(q)
        z_hat = np.array([r, wrap(np.arctan2(dy, dx) - mu[2])])
        innov = z - z_hat
        innov[1] = wrap(innov[1])

        H = np.zeros((2, n))
        H[0, 0] = -dx / r;  H[0, 1] = -dy / r
        H[1, 0] =  dy / q;  H[1, 1] = -dx / q;  H[1, 2] = -1
        H[0, 3 + 2 * i] =  dx / r;  H[0, 3 + 2 * i + 1] =  dy / r
        H[1, 3 + 2 * i] = -dy / q;  H[1, 3 + 2 * i + 1] =  dx / q

        S = H @ Sigma @ H.T + R_obs
        K = Sigma @ H.T @ np.linalg.inv(S)
        mu = mu + K @ innov
        Sigma = (np.eye(n) - K @ H) @ Sigma

    mu_hist.append(mu.copy())

mu_hist = np.array(mu_hist)
for i in range(N_LM):
    err = np.linalg.norm(mu[3 + 2 * i: 3 + 2 * i + 2] - true_lm[i])
    print(f"LM {i}: true={true_lm[i]}  est=({mu[3+2*i]:+.2f}, {mu[3+2*i+1]:+.2f})  err={err:.3f} m")
"""),
        code("""fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(true_robot[:, 0], true_robot[:, 1], 'b-', lw=2, label='True robot')
ax.plot(mu_hist[:, 0], mu_hist[:, 1], 'r--', lw=1.5, label='SLAM estimate')
ax.scatter(true_lm[:, 0], true_lm[:, 1], c='g', marker='*', s=250, label='True landmarks')
for i in range(N_LM):
    ax.scatter(mu[3 + 2 * i], mu[3 + 2 * i + 1], c='r', marker='x', s=120)
ax.set_aspect('equal'); ax.grid(); ax.legend()
ax.set_title('EKF SLAM — robot trajectory + landmark positions')
plt.tight_layout()
plt.show()
"""),
    ]
    write("12_slam_ekf_slam.ipynb", cells)


# ---------------------------------------------------------------------------
# 13. Dijkstra
# ---------------------------------------------------------------------------
def nb_13_dijkstra():
    cells = [
        md("""# 13 — Dijkstra's Algorithm on an Occupancy Grid

**Section:** Motion Planning · **Mirrors MATLAB:** *Motion Planners (RRT, PRM, Hybrid A\\*)* — Dijkstra baseline

Dijkstra is A\\* with a zero heuristic: it explores cells in pure cost-to-come order. This guarantees optimality but expands more nodes than A\\* (which uses the goal-distance heuristic to bias exploration).

Comparing the two on the same map helps illustrate why an admissible heuristic matters.

### Compatibility check — math ↔ code

| Math | Code |
|---|---|
| $f(n) = g(n) + 0 = g(n)$ (zero heuristic) | priority queue uses `(d, current, parent)` with no $h$ term |
| Edge cost = step distance | `nd = d + np.hypot(dy, dx)` |
| Always pop smallest $g$ | `heapq.heappop(heap)` |
| Relax neighbor: $g(n') = \min(g(n'),\ g(n) + c(n,n'))$ | `if nd < dist.get((ny, nx), np.inf): dist[(ny,nx)] = nd; heapq.heappush(...)` |
| Skip closed nodes | `if cur in came: continue` |
| Path reconstruction | `while cur is not None: path.append(cur); cur = came[cur]` |

Notice that the *only* difference from notebook 01 is the absence of the $h$ term in the priority — Dijkstra is the special case A\\* with $h \equiv 0$.
"""),
        code("""import numpy as np
import matplotlib.pyplot as plt
import heapq

np.random.seed(42)

H, W = 30, 50
grid = np.zeros((H, W), dtype=np.uint8)
for _ in range(40):
    y, x = np.random.randint(2, H - 3), np.random.randint(2, W - 3)
    grid[y - 1:y + 2, x - 1:x + 2] = 1
start, goal = (2, 2), (H - 3, W - 3)
grid[start] = 0; grid[goal] = 0
"""),
        code("""def dijkstra(grid, start, goal):
    H, W = grid.shape
    heap = [(0.0, start, None)]
    came_from, dist = {}, {start: 0.0}
    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)]
    while heap:
        d, current, parent = heapq.heappop(heap)
        if current in came_from:
            continue
        came_from[current] = parent
        if current == goal:
            path = []
            while current is not None:
                path.append(current); current = came_from[current]
            return path[::-1], dist
        for dy, dx in nbrs:
            ny, nx = current[0] + dy, current[1] + dx
            if 0 <= ny < H and 0 <= nx < W and grid[ny, nx] == 0:
                nd = d + np.hypot(dy, dx)
                if nd < dist.get((ny, nx), np.inf):
                    dist[(ny, nx)] = nd
                    heapq.heappush(heap, (nd, (ny, nx), current))
    return None, dist


path, dist = dijkstra(grid, start, goal)
print(f"Path: {len(path)} cells   expanded: {len(dist)} cells")
"""),
        code("""fig, ax = plt.subplots(figsize=(11, 6))
ax.imshow(grid, cmap='Greys', origin='lower')
expanded = np.full_like(grid, np.nan, dtype=float)
for (y, x), c in dist.items():
    expanded[y, x] = c
ax.imshow(expanded, cmap='viridis', alpha=0.4, origin='lower')

ys, xs = zip(*path)
ax.plot(xs, ys, 'r-', lw=2.5, label='Dijkstra path')
ax.plot(start[1], start[0], 'go', ms=14, label='Start')
ax.plot(goal[1], goal[0], 'b*', ms=18, label='Goal')
ax.legend(); ax.set_title("Dijkstra — expanded cells shaded by g-score (uses zero heuristic)")
plt.tight_layout()
plt.show()
"""),
    ]
    write("13_motion_planning_dijkstra.ipynb", cells)


# ---------------------------------------------------------------------------
# 14. Dynamic Window Approach
# ---------------------------------------------------------------------------
def nb_14_dwa():
    cells = [
        md("""# 14 — Dynamic Window Approach (DWA)

**Section:** Motion Planning · **Mirrors MATLAB:** *Path Following with Obstacle Avoidance*

DWA is a **local** planner: at each control step it samples reachable $(v, \omega)$ pairs, simulates each forward for a short horizon, and scores the resulting trajectory.
"""),
        md(r"""## Analytical derivation

**Dynamic window.** For a differential-drive robot with bounded velocity and acceleration, the set of *admissible* controls in the next step $\Delta t$ is

$$V_d = \bigl\{(v, \omega) \;:\; v \in [\max(v_\min, v - a_\max \Delta t),\ \min(v_\max, v + a_\max \Delta t)],\ \omega \in [\max(\omega_\min, \omega - \alpha_\max \Delta t),\ \min(\omega_\max, \omega + \alpha_\max \Delta t)]\bigr\}$$

This is the "dynamic window" — a rectangle in $(v, \omega)$-space centered at the current velocity, sized by acceleration limits.

**Trajectory rollout.** For each $(v, \omega) \in V_d$, simulate forward for horizon $T_p$ using the unicycle model:

$$x(\tau) = x_0 + v \cos\theta_0\,\tau,\quad y(\tau) = y_0 + v \sin\theta_0\,\tau,\quad \theta(\tau) = \theta_0 + \omega \tau$$

(For constant $(v,\omega)$ this is actually an arc, but discretized as a sequence of points.)

**Cost function.** Pick the $(v^*, \omega^*) \in V_d$ minimizing

$$J(v, \omega) \;=\; w_g\,\|\mathbf{p}_T - \mathbf{p}_\text{goal}\|\;+\;w_h\,|\theta_T - \theta_\text{goal-direction}|\;+\;\frac{w_o}{d_\min}\;+\;w_v\,(v_\max - v_T)$$

with $\mathbf{p}_T = (x(T_p), y(T_p))$, $d_\min$ the minimum clearance to any obstacle over the trajectory. Hard constraint: $J = +\infty$ if $d_\min < r_\text{safety}$.

Each term has a clear role:
- **Goal-distance term** ($w_g$) pulls the robot toward the goal.
- **Heading term** ($w_h$) penalizes pointing the wrong way at end-of-horizon.
- **Obstacle clearance term** ($w_o / d_\min$) is large when we get near an obstacle.
- **Velocity reward** ($w_v$) prefers faster motion to avoid getting stuck.

The result is a *receding-horizon, sample-based, single-step MPC* — much cheaper than full MPC but myopic (no look-ahead beyond $T_p$).

### Compatibility check — math ↔ code

| Step | Code |
|---|---|
| Dynamic window in $v$ | `vs = np.linspace(max(v_min, v - a_max*dt), min(v_max, v + a_max*dt), 7)` |
| Dynamic window in $\omega$ | `ws = np.linspace(max(w_min, w - alpha_max*dt), min(w_max, w + alpha_max*dt), 11)` |
| Trajectory rollout for $T_p$ s | `for _ in range(int(predict_t/dt)): s[0]+=v*cos*dt; s[1]+=v*sin*dt; s[2]+=w*dt` |
| Goal-distance term $w_g\|p_T - p_g\|$ | `1.0 * goal_dist` |
| Heading term $w_h\|\theta_T - \angle(p_g - p_T)\|$ | `0.3 * heading_cost` where `heading_cost = abs(np.arctan2(dy,dx) - traj[-1,2])` |
| Clearance term $w_o / d_\min$ | `0.2 / d.min()` |
| Hard clearance constraint | `if d.min() < 0.5: return np.inf` |
| Velocity reward $w_v(v_\max - v_T)$ | `0.05 * (v_max - traj[-1, 3])` |
| Pick best $(v^*,\omega^*)$ | nested loop `if c < best_c: best_c, best_v, best_w = c, v, w` |
"""),
        code("""import numpy as np
import matplotlib.pyplot as plt

robot = np.array([0., 0., 0., 0., 0.])      # x, y, theta, v, omega
goal  = np.array([10., 10.])
obstacles = np.array([[3, 4], [5, 6], [7, 8], [4, 7], [6, 5], [8, 9]])

v_max, v_min = 1.0, 0.0
w_max, w_min = 1.0, -1.0
a_max, alpha_max = 4.0, 6.0      # window must be larger than discrete step
dt, predict_t = 0.1, 2.5
"""),
        code("""def predict(state, v, w):
    s = state.copy()
    traj = [s.copy()]
    for _ in range(int(predict_t / dt)):
        s[0] += v * np.cos(s[2]) * dt
        s[1] += v * np.sin(s[2]) * dt
        s[2] += w * dt
        s[3], s[4] = v, w
        traj.append(s.copy())
    return np.array(traj)


def cost(traj, goal, obstacles):
    dx, dy = goal - traj[-1, :2]
    goal_dist    = np.hypot(dx, dy)
    heading_cost = abs(np.arctan2(dy, dx) - traj[-1, 2])
    d = np.min(np.linalg.norm(traj[:, None, :2] - obstacles[None], axis=2), axis=1)
    if d.min() < 0.5:
        return np.inf
    return 1.0 * goal_dist + 0.3 * heading_cost + 0.2 / d.min() + 0.05 * (v_max - traj[-1, 3])
"""),
        code("""hist = [robot[:2].copy()]
for _ in range(200):
    vs = np.linspace(max(v_min, robot[3] - a_max * dt),
                      min(v_max, robot[3] + a_max * dt), 7)
    ws = np.linspace(max(w_min, robot[4] - alpha_max * dt),
                      min(w_max, robot[4] + alpha_max * dt), 11)
    best_c, best_v, best_w = np.inf, 0.0, 0.0
    for v in vs:
        for w in ws:
            c = cost(predict(robot, v, w), goal, obstacles)
            if c < best_c:
                best_c, best_v, best_w = c, v, w
    robot[0] += best_v * np.cos(robot[2]) * dt
    robot[1] += best_v * np.sin(robot[2]) * dt
    robot[2] += best_w * dt
    robot[3], robot[4] = best_v, best_w
    hist.append(robot[:2].copy())
    if np.linalg.norm(robot[:2] - goal) < 0.3:
        break
hist = np.array(hist)
print(f"Reached goal in {len(hist)} steps. Final distance: {np.linalg.norm(robot[:2] - goal):.3f} m")
"""),
        code("""fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(obstacles[:, 0], obstacles[:, 1], c='grey', s=400, label='Obstacles')
ax.plot(hist[:, 0], hist[:, 1], 'r-', lw=2, label='DWA trajectory')
ax.plot(0, 0, 'go', ms=14, label='Start')
ax.plot(*goal, 'b*', ms=18, label='Goal')
ax.set_aspect('equal'); ax.grid(); ax.legend()
ax.set_xlim(-1, 12); ax.set_ylim(-1, 12)
ax.set_title('Dynamic Window Approach Local Planner')
plt.tight_layout()
plt.show()
"""),
    ]
    write("14_motion_planning_dwa.ipynb", cells)


# ---------------------------------------------------------------------------
# 15. Stanley control
# ---------------------------------------------------------------------------
def nb_15_stanley():
    cells = [
        md("""# 15 — Stanley Path Tracking

**Section:** Path Tracking · **Mirrors MATLAB:** *Path Following with Obstacle Avoidance* (Stanley controller)

Stanley control measures cross-track and heading error at the **front axle** of a bicycle vehicle and chooses steering angle:

$$\\delta = \\psi_e + \\arctan\\!\\left(\\frac{k \\cdot e_{ct}}{v}\\right)$$

where $\\psi_e$ is heading error, $e_{ct}$ is signed cross-track distance, and $k$ is a gain.

### Compatibility check — math ↔ code

| Math | Code |
|---|---|
| Front-axle position $(f_x, f_y) = (x + L\cos\theta, y + L\sin\theta)$ | `fx = x[0] + L*np.cos(x[2]); fy = x[1] + L*np.sin(x[2])` |
| Closest path index $j = \arg\min_k \|p_k - (f_x, f_y)\|$ | `j = int(np.argmin(np.linalg.norm(path - [fx, fy], axis=1)))` |
| Local path tangent angle $\psi_p$ | `path_angle = np.arctan2(path_dir[1], path_dir[0])` |
| Cross-track error (signed, perpendicular to tangent) | `cross = np.dot([fx - path[j,0], fy - path[j,1]], [-sin(path_angle), cos(path_angle)])` |
| Heading error $\psi_e = \psi_p - \theta$ (wrapped) | `heading_err = wrap(path_angle - x[2])` |
| Stanley steering $\delta = \psi_e + \arctan(k \cdot e_{ct} / v)$ | `delta = heading_err + np.arctan2(k * cross, v)` |
| Bicycle kinematics with $\delta$ | `x[0]+=v*cos*dt; x[1]+=v*sin*dt; x[2]+=v*tan(delta)/L*dt` |
"""),
        code("""import numpy as np
import matplotlib.pyplot as plt

xs_ref = np.linspace(0, 30, 600)
ys_ref = 3 * np.sin(xs_ref * 0.3)
path = np.column_stack([xs_ref, ys_ref])

k, L, v = 1.5, 1.5, 1.5
x = np.array([0.0, -2.5, 0.0])
dt, T = 0.05, 25.0
N = int(T / dt)
hist = np.zeros((N, 3))


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi
"""),
        code("""for i in range(N):
    fx = x[0] + L * np.cos(x[2])
    fy = x[1] + L * np.sin(x[2])
    d = np.linalg.norm(path - np.array([fx, fy]), axis=1)
    j = int(np.argmin(d))
    j2 = min(j + 1, len(path) - 1)
    path_dir = path[j2] - path[max(j - 1, 0)]
    path_angle = np.arctan2(path_dir[1], path_dir[0])
    cross = np.dot([fx - path[j, 0], fy - path[j, 1]],
                    [-np.sin(path_angle), np.cos(path_angle)])
    heading_err = wrap(path_angle - x[2])
    delta = heading_err + np.arctan2(k * cross, v)
    x[0] += v * np.cos(x[2]) * dt
    x[1] += v * np.sin(x[2]) * dt
    x[2] += v * np.tan(delta) / L * dt
    hist[i] = x
"""),
        code("""fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(xs_ref, ys_ref, 'b--', label='Reference')
ax.plot(hist[:, 0], hist[:, 1], 'r-', lw=2, label='Vehicle')
ax.plot(0, -2.5, 'go', ms=12, label='Start')
ax.set_aspect('equal'); ax.grid(); ax.legend()
ax.set_title('Stanley Path Tracking (front-axle cross-track error)')
plt.tight_layout()
plt.show()
"""),
    ]
    write("15_path_tracking_stanley.ipynb", cells)


# ---------------------------------------------------------------------------
# 16. Jacobian IK for 3-link arm
# ---------------------------------------------------------------------------
def nb_16_jacobian_ik():
    cells = [
        md("""# 16 — Jacobian-Based Inverse Kinematics for a 3-Link Arm

**Section:** Manipulation · **Mirrors MATLAB:** *Inverse Kinematics with Spatial Constraints*

For arms with more joints than task DOFs (or no closed-form solution), we use **damped least-squares** numerical IK:

$$\\Delta\\theta = J^T (J J^T + \\lambda^2 I)^{-1} \\Delta x$$

The damping $\\lambda$ keeps the update bounded near singularities.

### Compatibility check — math ↔ code

| Math | Code |
|---|---|
| Cumulative angle for link $i$: $\phi_i = \sum_{k \le i} \theta_k$ | accumulated via `a += theta[i]` in `fk_all` |
| End-effector position $p = \sum_i L_i(\cos\phi_i, \sin\phi_i)$ | `pts[-1] + np.array([l[i]*np.cos(a), l[i]*np.sin(a)])` |
| Jacobian column $i$: $z \times (p - p_i)$ where $z = \hat e_z$ | `J[0,i] = -r[1]; J[1,i] = r[0]` with `r = end - pts[i]` |
| Error $\Delta x = x^* - p$ | `err = target - end` |
| Damped least-squares $\Delta\theta = J^T(JJ^T + \lambda^2 I)^{-1}\Delta x$ | `dtheta = J.T @ np.linalg.inv(J @ J.T + lam**2 * np.eye(2)) @ err` |
| Gradient step $\theta \leftarrow \theta + \alpha\Delta\theta$ | `theta = theta + step * dtheta` |
| Convergence test $\|\Delta x\| < \tau$ | `if np.linalg.norm(err) < tol: break` |
"""),
        code("""import numpy as np
import matplotlib.pyplot as plt

l = [1.0, 1.0, 0.8]


def fk_all(theta):
    pts = [np.array([0.0, 0.0])]
    a = 0.0
    for i in range(3):
        a += theta[i]
        pts.append(pts[-1] + np.array([l[i] * np.cos(a), l[i] * np.sin(a)]))
    return np.array(pts)


def jacobian(theta):
    pts = fk_all(theta)
    end = pts[-1]
    J = np.zeros((2, 3))
    for i in range(3):
        r = end - pts[i]
        J[0, i] = -r[1]
        J[1, i] =  r[0]
    return J


def numerical_ik(target, theta_init, max_iter=200, tol=1e-4, lam=0.05, step=0.4):
    theta = theta_init.copy()
    for _ in range(max_iter):
        end = fk_all(theta)[-1]
        err = target - end
        if np.linalg.norm(err) < tol:
            break
        J = jacobian(theta)
        dtheta = J.T @ np.linalg.inv(J @ J.T + lam ** 2 * np.eye(2)) @ err
        theta = theta + step * dtheta
    return theta, np.linalg.norm(err)
"""),
        code("""# Trace a target trajectory
N = 70
phi = np.linspace(0, 2 * np.pi, N)
targets = np.column_stack([1.2 + 0.5 * np.cos(phi), 1.0 + 0.5 * np.sin(phi)])

theta = np.array([0.4, 0.4, 0.4])
arm_log = []
for tgt in targets:
    theta, _ = numerical_ik(tgt, theta)
    arm_log.append(fk_all(theta))

fig, ax = plt.subplots(figsize=(8, 8))
for k, pts in enumerate(arm_log):
    a = 0.15 + 0.7 * (k / N)
    ax.plot(pts[:, 0], pts[:, 1], 'b-', alpha=a, lw=1.2)
    ax.plot(pts[1:-1, 0], pts[1:-1, 1], 'ko', ms=3, alpha=a)
ax.plot(targets[:, 0], targets[:, 1], 'r--', lw=1.5, label='Target')
ax.plot(0, 0, 'gs', ms=12, label='Base')
ax.set_xlim(-1.5, 3); ax.set_ylim(-1, 3)
ax.set_aspect('equal'); ax.grid(); ax.legend()
ax.set_title('3-Link Arm Numerical IK (damped least-squares)')
plt.tight_layout()
plt.show()
"""),
    ]
    write("16_manipulation_jacobian_ik.ipynb", cells)


# ---------------------------------------------------------------------------
# 17. ORB feature matching
# ---------------------------------------------------------------------------
def nb_17_orb():
    cells = [
        md("""# 17 — ORB Feature Detection and Matching

**Section:** Perception · **Mirrors MATLAB:** *Feature Detection, Extraction, and Matching*

ORB (Oriented FAST and Rotated BRIEF) is a fast, rotation-invariant feature detector + descriptor. We detect features in two synthetic images (one is a rotated + translated version of the other) and match them with Hamming distance + cross-check.

### Analytical underpinning + compatibility

**FAST corner test.** A pixel $p$ with intensity $I_p$ is a corner if, among the 16 pixels on a circle of radius 3 around it, at least $N$ contiguous pixels are all brighter than $I_p + t$ or all darker than $I_p - t$ (typically $N = 9$, $t$ = small threshold). The check on $N=9$ contiguous pixels is what makes it both fast and rotationally meaningful.

**Orientation assignment.** Compute the intensity centroid of a patch around the corner; the vector from corner center to centroid defines the orientation angle $\theta$.

**BRIEF descriptor.** Pre-select $n$ (typically 256) pairs of pixel locations $(p_i, q_i)$ within a patch around the corner. The descriptor is the binary string

$$d_i = \begin{cases} 1 & \text{if } I(p_i) < I(q_i) \\ 0 & \text{otherwise} \end{cases},\quad i = 1, \ldots, n$$

For ORB, the pixel pairs are rotated by the orientation $\theta$ before sampling — that's the "Rotated" in "Rotated BRIEF" and is what makes ORB rotation-invariant.

**Matching.** Distance between two BRIEF descriptors is the **Hamming distance** = number of bit positions where they differ = `popcount(d_1 XOR d_2)`. Fast brute-force pairwise comparison is feasible because descriptors are binary.

**Cross-check.** Keep only pairs where $d_a$ is the closest match for $d_b$ *and* vice versa. Greatly reduces false matches.

### Compatibility check — math ↔ code

| Math | Code |
|---|---|
| FAST corner + rotated BRIEF, up to 300 features | `orb = cv2.ORB_create(nfeatures=300)` |
| Detect keypoints + descriptors | `kp, des = orb.detectAndCompute(img, None)` (descriptor is 32-byte binary) |
| Hamming-distance matcher with cross-check | `bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)` |
| Match + sort by descriptor distance | `matches = sorted(bf.match(des1, des2), key=lambda m: m.distance)[:40]` |
"""),
        code("""import cv2
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

H, W = 240, 320
img1 = np.zeros((H, W), dtype=np.uint8)
cv2.rectangle(img1, (50, 50), (100, 100), 200, -1)
cv2.circle(img1, (200, 80), 30, 150, -1)
cv2.line(img1, (60, 150), (260, 200), 250, 3)
cv2.putText(img1, 'PYROBOT', (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 2)
img1 = cv2.GaussianBlur(img1, (3, 3), 0)

M = cv2.getRotationMatrix2D((W / 2, H / 2), 20, 1.0)
M[0, 2] += 25
M[1, 2] += 10
img2 = cv2.warpAffine(img1, M, (W, H))
"""),
        code("""orb = cv2.ORB_create(nfeatures=300)
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = sorted(bf.match(des1, des2), key=lambda m: m.distance)[:40]
print(f"img1 features: {len(kp1)}  img2 features: {len(kp2)}  matches kept: {len(matches)}")

out = cv2.drawMatches(img1, kp1, img2, kp2, matches, None,
                       flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
"""),
        code("""plt.figure(figsize=(14, 6))
plt.imshow(out, cmap='gray')
plt.title('ORB Feature Matches (img1 ↔ rotated/translated img2)')
plt.axis('off')
plt.tight_layout()
plt.show()
"""),
    ]
    write("17_perception_orb_features.ipynb", cells)


# ---------------------------------------------------------------------------
# 18. Kalman tracking
# ---------------------------------------------------------------------------
def nb_18_kalman_tracking():
    cells = [
        md("""# 18 — Kalman Filter Tracking of a Maneuvering Target

**Section:** Perception · **Mirrors MATLAB:** *Object Tracking and Motion Estimation*

We track a 2-D object with a **constant-velocity** Kalman filter using only noisy position measurements.
"""),
        md(r"""## Analytical derivation

**State.** $x = [p_x,\ p_y,\ v_x,\ v_y]^T \in \mathbb{R}^4$ — 2-D position and velocity.

**Linear Gaussian model.** Constant-velocity dynamics over $\Delta t$:

$$x_{t+1} \;=\; F x_t + w_t,\qquad w_t \sim \mathcal{N}(0, Q),\qquad F = \begin{bmatrix} 1 & 0 & \Delta t & 0 \\ 0 & 1 & 0 & \Delta t \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}$$

Process noise $Q$ models the deviation from constant velocity — bigger on the velocity components since accelerations can occur unmodeled.

Observation: we measure position only,

$$z_t \;=\; H x_t + v_t,\qquad v_t \sim \mathcal{N}(0, R),\qquad H = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \end{bmatrix}$$

**Kalman filter recursion.** Maintain Gaussian belief $x \sim \mathcal{N}(\hat x, P)$.

*Predict:*

$$\hat x^- \;=\; F \hat x,\qquad P^- \;=\; F P F^T + Q$$

*Update* (given measurement $z$):

$$\text{innovation:}\quad y = z - H \hat x^-$$
$$\text{innovation covariance:}\quad S = H P^- H^T + R$$
$$\text{Kalman gain:}\quad K = P^- H^T S^{-1}$$
$$\hat x \;=\; \hat x^- + K\,y$$
$$P \;=\; (I - K H)\,P^-$$

**Optimality.** Among all linear estimators, the KF minimizes MSE; for Gaussian noise it is also the *minimum-variance* unbiased estimator.

**Handling unmodeled maneuvers.** When the true target suddenly changes velocity (as at $t = 5\,$s in this notebook), the constant-velocity model is wrong for a few steps. Because $Q$ allows for some unmodeled velocity drift, the filter "catches up" — but transient error appears. Increasing $Q$ on the velocity components trades steady-state smoothness for maneuver responsiveness.

### Compatibility check — math ↔ code

| Math | Code |
|---|---|
| State transition $F$ | `F = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]])` |
| Observation matrix $H$ | `H = np.array([[1,0,0,0],[0,1,0,0]])` |
| Process noise $Q$ | `Q = np.diag([0.01, 0.01, 0.15, 0.15])` |
| Predict $\hat x^- = F \hat x$ | `x_est = F @ x_est` |
| Predict $P^- = F P F^T + Q$ | `P = F @ P @ F.T + Q` |
| Innovation $y = z - H \hat x^-$ | `y = z[k] - H @ x_est` |
| Innovation cov $S = H P H^T + R$ | `S = H @ P @ H.T + R_mat` |
| Kalman gain $K = P H^T S^{-1}$ | `K = P @ H.T @ np.linalg.inv(S)` |
| Update mean | `x_est = x_est + K @ y` |
| Update cov $(I - K H) P$ | `P = (np.eye(4) - K @ H) @ P` |
"""),
        code("""import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

T, dt = 100, 0.1
true = np.zeros((T, 4))           # x, y, vx, vy
true[0] = [0, 0, 1, 0.5]
for k in range(1, T):
    true[k] = true[k - 1].copy()
    true[k, :2] += true[k - 1, 2:] * dt
    if k == 50:
        true[k, 2:] = [-0.5, 1.0]   # sudden maneuver

R_obs = 0.5
z = true[:, :2] + np.random.randn(T, 2) * R_obs
"""),
        code("""F = np.array([[1, 0, dt, 0],
              [0, 1, 0, dt],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])
H = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0]])
Q = np.diag([0.01, 0.01, 0.15, 0.15])
R_mat = np.eye(2) * R_obs ** 2

x_est = np.zeros(4); P = np.eye(4)
est_hist = []
for k in range(T):
    x_est = F @ x_est
    P = F @ P @ F.T + Q
    y = z[k] - H @ x_est
    S = H @ P @ H.T + R_mat
    K = P @ H.T @ np.linalg.inv(S)
    x_est = x_est + K @ y
    P = (np.eye(4) - K @ H) @ P
    est_hist.append(x_est.copy())

est_hist = np.array(est_hist)
rmse = np.sqrt(np.mean((est_hist[:, :2] - true[:, :2]) ** 2))
print(f"Position RMSE: {rmse:.3f} m")
"""),
        code("""fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(true[:, 0], true[:, 1], 'b-', lw=2, label='True')
ax.plot(z[:, 0], z[:, 1], 'k.', alpha=0.4, label='Noisy obs')
ax.plot(est_hist[:, 0], est_hist[:, 1], 'r--', lw=1.5, label='KF estimate')
ax.grid(); ax.legend(); ax.set_aspect('equal')
ax.set_title('Kalman Filter Tracking with Mid-Trajectory Maneuver')
plt.tight_layout()
plt.show()
"""),
    ]
    write("18_perception_kalman_tracking.ipynb", cells)


# ---------------------------------------------------------------------------
# 19. Bicycle kinematic model
# ---------------------------------------------------------------------------
def nb_19_bicycle():
    cells = [
        md("""# 19 — Kinematic Bicycle Model

**Section:** Ground Vehicles and Mobile Robotics · **Mirrors MATLAB:** *Kinematic motion models for simulation*

The bicycle model approximates a car-like vehicle as two wheels on a centerline of length $L$ (wheelbase). With longitudinal speed $v$ and steering angle $\\delta$:

$$\\dot{x} = v\\cos\\theta,\\quad \\dot{y} = v\\sin\\theta,\\quad \\dot\\theta = \\frac{v}{L}\\tan\\delta$$

We sweep three steering schedules to illustrate the resulting trajectories.

### Compatibility check — math ↔ code

| Math | Code |
|---|---|
| $\dot x = v\cos\theta$ | `state[0] += v * np.cos(state[2]) * dt` |
| $\dot y = v\sin\theta$ | `state[1] += v * np.sin(state[2]) * dt` |
| $\dot\theta = \frac{v}{L}\tan\delta$ | `state[2] += v * np.tan(delta) / L * dt` |
| Three steering schedules: constant 0.2, constant 0.4, $0.3\sin(0.4 t)$ | `t1 = simulate(2.0, lambda _: 0.2); t2 = simulate(2.0, lambda _: 0.4); t3 = simulate(2.0, lambda t: 0.3*sin(0.4*t))` |

Note: this is *kinematic* (instantaneous control authority over $v$ and $\delta$), not *dynamic* — there is no notion of tire slip, mass, or lateral force. Good enough for low-speed planning, breaks down at speed where slip matters.
"""),
        code("""import numpy as np
import matplotlib.pyplot as plt

L = 2.5
dt, T = 0.05, 30.0
N = int(T / dt)


def simulate(v, delta_schedule):
    state = np.zeros(3)
    hist = np.zeros((N, 3))
    for i in range(N):
        delta = delta_schedule(i * dt)
        state[0] += v * np.cos(state[2]) * dt
        state[1] += v * np.sin(state[2]) * dt
        state[2] += v * np.tan(delta) / L * dt
        hist[i] = state
    return hist


t1 = simulate(2.0, lambda _: 0.2)
t2 = simulate(2.0, lambda _: 0.4)
t3 = simulate(2.0, lambda t: 0.3 * np.sin(0.4 * t))
"""),
        code("""fig, ax = plt.subplots(figsize=(9, 8))
ax.plot(t1[:, 0], t1[:, 1], 'b-', lw=2, label='δ = 0.2 rad (gentle turn)')
ax.plot(t2[:, 0], t2[:, 1], 'r-', lw=2, label='δ = 0.4 rad (tight turn)')
ax.plot(t3[:, 0], t3[:, 1], 'g-', lw=2, label='δ = 0.3 sin(0.4 t) (slalom)')
ax.set_aspect('equal'); ax.grid(); ax.legend()
ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
ax.set_title(f'Kinematic Bicycle Model — L = {L} m, v = 2 m/s')
plt.tight_layout()
plt.show()
"""),
    ]
    write("19_ground_vehicles_bicycle.ipynb", cells)


# ---------------------------------------------------------------------------
# 20. Symbolic pendulum dynamics with SymPy
# ---------------------------------------------------------------------------
def nb_20_symbolic_dynamics():
    cells = [
        md("""# 20 — Symbolic Lagrangian Dynamics with SymPy

**Section:** Robot Modeling · **Mirrors MATLAB:** *Simscape Tools for Modeling and Simulation of Physical Systems*

We derive the equation of motion for a simple pendulum **symbolically** using SymPy, then convert the result into a fast numerical function with `lambdify` and integrate it.

This mirrors MATLAB's Simscape / Symbolic Math Toolbox workflow: model in symbols, derive analytically, simulate numerically.

### Analytical setup + compatibility

Single pendulum hanging from a fixed pivot. Generalized coordinate $\\theta$ (angle from straight-down). Bob position $\\mathbf{r}(\\theta) = (L\\sin\\theta,\\ -L\\cos\\theta)$; velocity $\\dot{\\mathbf{r}} = L\\dot\\theta(\\cos\\theta,\\ \\sin\\theta)$.

$$T = \\tfrac{1}{2} m\\|\\dot{\\mathbf{r}}\\|^2 = \\tfrac{1}{2} m L^2 \\dot\\theta^2,\\qquad V = m g y_\\text{bob} = -m g L \\cos\\theta$$

Lagrangian $\\mathcal{L} = T - V$. Euler-Lagrange gives the closed-form

$$\\boxed{\\;\\ddot\\theta = -\\frac{g}{L}\\sin\\theta\\;}$$

| Math | Code |
|---|---|
| Bob position $(L\\sin\\theta,\\ -L\\cos\\theta)$ | `x_p = L * sp.sin(theta); y_p = -L * sp.cos(theta)` |
| $T = \\tfrac12 m(\\dot x^2 + \\dot y^2)$ | `KE = sp.Rational(1,2) * m * (sp.diff(x_p, t)**2 + sp.diff(y_p, t)**2)` |
| $V = m g\\,y_\\text{bob}$ | `PE = m * g * y_p` |
| Lagrangian $\\mathcal{L} = T - V$ | `Lagrangian = sp.simplify(KE - PE)` |
| Euler-Lagrange operator | `EL = sp.diff(sp.diff(L, theta_dot), t) - sp.diff(L, theta)` |
| Solve for $\\ddot\\theta$ → $-(g/L)\\sin\\theta$ | `sol = sp.solve(EL, theta_ddot)[0]` |
| Lambdify for numerics | `f = sp.lambdify((theta, m, L, g), sol, 'numpy')` |
| RK4 integration | manual 4-stage Runge-Kutta in the integration loop |
"""),
        code("""import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

t = sp.symbols('t')
theta = sp.Function('theta')(t)
m, L, g = sp.symbols('m L g', positive=True)

# Position of the pendulum bob (theta measured from straight-down)
x_p =  L * sp.sin(theta)
y_p = -L * sp.cos(theta)

KE = sp.Rational(1, 2) * m * (sp.diff(x_p, t) ** 2 + sp.diff(y_p, t) ** 2)
PE = m * g * y_p

Lagrangian = sp.simplify(KE - PE)
print("Lagrangian L = T - V:")
sp.pprint(sp.simplify(Lagrangian))
"""),
        code("""theta_dot  = sp.diff(theta, t)
theta_ddot = sp.diff(theta, t, 2)

EL = sp.diff(sp.diff(Lagrangian, theta_dot), t) - sp.diff(Lagrangian, theta)
EL = sp.simplify(EL)
print("Euler-Lagrange equation:")
sp.pprint(EL)

sol = sp.solve(EL, theta_ddot)[0]
print("\\ntheta'' =")
sp.pprint(sp.simplify(sol))
"""),
        code("""# Numerical integration
f = sp.lambdify((theta, m, L, g), sol, 'numpy')


def rhs(state, _t):
    th, om = state
    return np.array([om, f(th, 1.0, 1.0, 9.81)])


dt_sim, T_sim = 0.01, 10.0
N = int(T_sim / dt_sim)
state = np.array([np.pi / 3, 0.0])  # initial 60° angle, zero rate
hist = np.zeros((N, 2))
for i in range(N):
    hist[i] = state
    k1 = rhs(state, i * dt_sim)
    k2 = rhs(state + dt_sim * k1 / 2, i * dt_sim + dt_sim / 2)
    k3 = rhs(state + dt_sim * k2 / 2, i * dt_sim + dt_sim / 2)
    k4 = rhs(state + dt_sim * k3, i * dt_sim + dt_sim)
    state = state + dt_sim * (k1 + 2 * k2 + 2 * k3 + k4) / 6
"""),
        code("""t_arr = np.arange(N) * dt_sim
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(t_arr, np.degrees(hist[:, 0]), 'b-', label='angle (deg)')
ax.plot(t_arr, np.degrees(hist[:, 1]), 'r--', lw=1, label='angular vel (deg/s)')
ax.set_xlabel('time (s)'); ax.grid(); ax.legend()
ax.set_title('Pendulum Dynamics — Lagrangian Derived with SymPy, Integrated with RK4')
plt.tight_layout()
plt.show()
"""),
    ]
    write("20_modeling_symbolic_pendulum.ipynb", cells)


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
    nb_11_icp()
    nb_12_ekf_slam()
    nb_13_dijkstra()
    nb_14_dwa()
    nb_15_stanley()
    nb_16_jacobian_ik()
    nb_17_orb()
    nb_18_kalman_tracking()
    nb_19_bicycle()
    nb_20_symbolic_dynamics()
    print("Done.")


if __name__ == "__main__":
    main()
