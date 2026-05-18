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
        md("""## Generalized coordinates

$q = [x,\\ \\theta_1,\\ \\theta_2,\\ \\theta_3]^T$ — cart position plus 3 link angles measured from world vertical. Upright equilibrium is $q = 0$.

Each link is treated as a point mass at its midpoint (mass $m_i$, length $L_i$).
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


# ---------------------------------------------------------------------------
# 11. ICP scan matching
# ---------------------------------------------------------------------------
def nb_11_icp():
    cells = [
        md("""# 11 — Iterative Closest Point (ICP) Scan Matching

**Section:** SLAM · **Mirrors MATLAB:** *2D Lidar SLAM Implementations* (scan-matching front-end)

ICP aligns two point clouds by alternating between (a) finding nearest-neighbor correspondences and (b) solving for the rigid transform (R, t) that minimizes the sum-of-squared distances over those matches.

The rigid transform step has a closed-form solution via SVD on the cross-covariance matrix.
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

Each time we observe a landmark for the first time, we **initialize** its position from the current robot pose and observation. Subsequent observations refine both the robot pose and the landmark estimate (and their covariance).
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

DWA is a **local** planner: at each control step it samples reachable (v, ω) pairs in the dynamic window (limited by current velocity and acceleration), simulates each forward for a short horizon, and scores the resulting trajectory by heading toward the goal, clearance to obstacles, and forward velocity.
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

We track a 2-D object with a **constant-velocity** Kalman filter using only noisy position measurements. The filter recovers a smooth trajectory and stable velocity estimate even when the true target suddenly changes direction.
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
