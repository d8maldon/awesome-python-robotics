"""Build animated GIFs for the README's Featured Demos section.

Run: python scripts/build_animations.py

Produces ~8 GIFs in media/ — small (< 2 MB each), looping, GitHub-renderable.
"""
from pathlib import Path
import heapq
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.linalg import solve_continuous_are

MEDIA_DIR = Path(__file__).resolve().parent.parent / "media"
MEDIA_DIR.mkdir(exist_ok=True)


def save(anim, name, fps=20, dpi=72):
    path = MEDIA_DIR / f"{name}.gif"
    anim.save(str(path), writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close("all")
    sz = path.stat().st_size / 1024
    print(f"  wrote {path.name}  ({sz:.0f} KB)")


# ----- 1. A* expansion -----
def anim_astar():
    np.random.seed(42)
    H, W = 25, 40
    grid = np.zeros((H, W), dtype=np.uint8)
    for _ in range(30):
        y, x = np.random.randint(2, H - 3), np.random.randint(2, W - 3)
        grid[y - 1:y + 2, x - 1:x + 2] = 1
    start, goal = (2, 2), (H - 3, W - 3)
    grid[start] = 0; grid[goal] = 0

    def h(a, b): return np.hypot(a[0] - b[0], a[1] - b[1])

    heap = [(h(start, goal), 0.0, start, None)]
    came, g = {}, {start: 0.0}
    order = []
    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    while heap:
        f, gv, cur, par = heapq.heappop(heap)
        if cur in came:
            continue
        came[cur] = par
        order.append(cur)
        if cur == goal:
            break
        for dy, dx in nbrs:
            ny, nx = cur[0] + dy, cur[1] + dx
            if 0 <= ny < H and 0 <= nx < W and not grid[ny, nx]:
                tent = gv + np.hypot(dy, dx)
                if tent < g.get((ny, nx), np.inf):
                    g[(ny, nx)] = tent
                    heapq.heappush(heap, (tent + h((ny, nx), goal), tent, (ny, nx), cur))

    path, cur = [], goal
    while cur is not None:
        path.append(cur)
        cur = came[cur]
    path = path[::-1]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.imshow(grid, cmap="Greys", origin="lower")
    heat = np.full_like(grid, np.nan, dtype=float)
    im = ax.imshow(heat, cmap="viridis", alpha=0.55, origin="lower",
                   vmin=0, vmax=len(order))
    line, = ax.plot([], [], "r-", lw=3)
    ax.plot(start[1], start[0], "go", ms=12)
    ax.plot(goal[1], goal[0], "b*", ms=16)
    ax.set_title("A* Path Planning")
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()

    expand_frames = len(order)
    hold = 25
    step = max(1, expand_frames // 80)  # subsample to ~80 frames

    def update(frame):
        if frame < expand_frames // step:
            end = (frame + 1) * step
            for y, x in order[:end]:
                heat[y, x] = order.index((y, x))
            im.set_array(heat)
            line.set_data([], [])
        else:
            ys, xs = zip(*path)
            line.set_data(xs, ys)
        return [im, line]

    anim = FuncAnimation(fig, update, frames=expand_frames // step + hold,
                         interval=50, blit=False)
    save(anim, "astar_expansion", fps=18)


# ----- 2. RRT tree growth -----
def anim_rrt():
    np.random.seed(0)
    world = 100.0
    obs = [(30, 30, 15), (60, 70, 12), (75, 30, 10), (20, 70, 8), (50, 45, 8)]
    start, goal = np.array([5.0, 5.0]), np.array([95.0, 95.0])

    def in_col(p):
        return any((p[0] - cx) ** 2 + (p[1] - cy) ** 2 < r * r for cx, cy, r in obs)

    def edge_col(p1, p2, n=20):
        return any(in_col(p1 * (1 - t) + p2 * t) for t in np.linspace(0, 1, n))

    tree = [start]
    parent = {0: -1}
    step = 4.0
    for _ in range(2000):
        rnd = goal if np.random.random() < 0.1 else np.random.uniform(0, world, 2)
        d = [np.linalg.norm(p - rnd) for p in tree]
        i_n = int(np.argmin(d))
        near = tree[i_n]
        dirn = rnd - near
        dist = np.linalg.norm(dirn)
        new = near + (dirn / dist) * step if dist > step else rnd
        if in_col(new) or edge_col(near, new):
            continue
        tree.append(new)
        parent[len(tree) - 1] = i_n
        if np.linalg.norm(new - goal) < step:
            tree.append(goal)
            parent[len(tree) - 1] = len(tree) - 2
            break

    fig, ax = plt.subplots(figsize=(7, 7))
    for cx, cy, r in obs:
        ax.add_patch(plt.Circle((cx, cy), r, color="lightgrey"))
    ax.plot(*start, "go", ms=14)
    ax.plot(*goal, "b*", ms=18)
    ax.set_xlim(0, world); ax.set_ylim(0, world); ax.set_aspect("equal")
    ax.set_title("RRT Tree Growth")
    ax.set_xticks([]); ax.set_yticks([])

    seg_lines = []
    path_line, = ax.plot([], [], "r-", lw=3)

    n_nodes = len(tree)
    sub = max(1, n_nodes // 80)
    hold = 20

    # Reconstruct path
    path_idx, i = [], len(tree) - 1
    while i != -1:
        path_idx.append(i)
        i = parent[i]
    path = np.array([tree[i] for i in path_idx[::-1]])

    def update(frame):
        if frame * sub < n_nodes:
            n_shown = min((frame + 1) * sub, n_nodes)
            for line in seg_lines:
                line.remove()
            seg_lines.clear()
            for k in range(1, n_shown):
                p1 = tree[k]; p2 = tree[parent[k]]
                ln, = ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "g-",
                              lw=0.5, alpha=0.6)
                seg_lines.append(ln)
            path_line.set_data([], [])
        else:
            path_line.set_data(path[:, 0], path[:, 1])
        return seg_lines + [path_line]

    anim = FuncAnimation(fig, update, frames=n_nodes // sub + hold,
                         interval=60, blit=False)
    save(anim, "rrt_growth", fps=15)


# ----- 3. Particle filter collapse -----
def anim_pf():
    np.random.seed(0)
    T, dt = 100, 0.1
    true_x = np.zeros((T, 3))
    for t in range(T):
        a = t * dt
        true_x[t] = [5 * np.sin(a), 3 * np.sin(2 * a), a]
    landmarks = np.array([[6, 0], [-6, 0], [0, 4], [0, -4], [4, 4], [-4, -4]])

    N = 400
    particles = np.zeros((N, 3))
    particles[:, :2] = np.random.uniform(-4, 4, (N, 2))
    particles[:, 2] = np.random.uniform(-np.pi, np.pi, N)
    range_noise = 0.25

    snaps = []
    for t in range(T):
        u = (true_x[t] - true_x[t - 1]) if t > 0 else np.zeros(3)
        particles += u + np.random.randn(N, 3) * np.array([0.08, 0.08, 0.04])
        z = np.linalg.norm(landmarks - true_x[t, :2], axis=1) + np.random.randn(len(landmarks)) * range_noise
        expected = np.linalg.norm(landmarks[:, None, :] - particles[None, :, :2], axis=2)
        err = (z[:, None] - expected) / range_noise
        log_w = -0.5 * np.sum(err ** 2, axis=0)
        w = np.exp(log_w - log_w.max()); w /= w.sum()
        idx = np.searchsorted(np.cumsum(w), (np.arange(N) + np.random.random()) / N)
        particles = particles[idx]
        snaps.append(particles.copy())

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(landmarks[:, 0], landmarks[:, 1], c="k", marker="*", s=180, zorder=5)
    truth_line, = ax.plot([], [], "b-", lw=2)
    scat = ax.scatter([], [], c="g", s=6, alpha=0.5)
    cur_dot, = ax.plot([], [], "ro", ms=10)
    ax.set_xlim(-7, 7); ax.set_ylim(-5, 5); ax.set_aspect("equal")
    ax.set_title("Particle Filter Localization")
    ax.set_xticks([]); ax.set_yticks([])

    def update(t):
        truth_line.set_data(true_x[:t + 1, 0], true_x[:t + 1, 1])
        scat.set_offsets(snaps[t][:, :2])
        cur_dot.set_data([true_x[t, 0]], [true_x[t, 1]])
        return [truth_line, scat, cur_dot]

    anim = FuncAnimation(fig, update, frames=T, interval=80, blit=False)
    save(anim, "particle_filter", fps=15)


# ----- 4. LQR triple-link inverted pendulum -----
def anim_pendulum():
    """Cart-pole swing-up + LQR catch.

    Starts pendulum hanging straight down, pumps mechanical energy via cart
    motion alone (the only actuator) until the pendulum enters the LQR basin,
    then catches with a CARE-based LQR. Matches notebook 05 exactly.
    """
    M, m, l, g = 1.0, 0.2, 0.5, 9.81

    def nl_dyn(z, u):
        pos, vel, th, om = z
        s, c = np.sin(th), np.cos(th)
        den = M + m * s ** 2
        acc = (u + m * l * s * om ** 2 - m * g * s * c) / den
        a_th = ((M + m) * g * s - u * c - m * l * s * c * om ** 2) / (l * den)
        return np.array([vel, acc, om, a_th])

    def E_pend(z):
        pos, vel, th, om = z
        bob_vx = vel + l * np.cos(th) * om
        bob_vy = -l * np.sin(th) * om
        return 0.5 * m * (bob_vx ** 2 + bob_vy ** 2) + m * g * l * np.cos(th)

    E_star = m * g * l

    A = np.array([[0, 1, 0, 0],
                  [0, 0, -m * g / M, 0],
                  [0, 0, 0, 1],
                  [0, 0, (M + m) * g / (M * l), 0]])
    B = np.array([[0], [1 / M], [0], [-1 / (M * l)]])
    Q = np.diag([1.0, 1.0, 20.0, 1.0])
    R_lqr = np.array([[0.1]])
    P_riccati = solve_continuous_are(A, B, Q, R_lqr)
    K_lqr = np.linalg.inv(R_lqr) @ B.T @ P_riccati

    K_x, K_xd = 1.0, 0.5
    u_max, theta_LQR, c_ROA = 15.0, 0.5, 20.0

    def wrap(angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def control(z):
        th_w = wrap(z[2])
        zl = np.array([z[0], z[1], th_w, z[3]])
        if abs(th_w) < theta_LQR and (zl @ P_riccati @ zl) < c_ROA:
            return float(np.clip(-(K_lqr @ zl).item(), -u_max, u_max)), "LQR"
        Ep = E_pend(z)
        sv = (Ep - E_star) * np.cos(z[2]) * z[3]
        u_pump = +u_max * np.sign(sv) if abs(sv) > 1e-6 else 0.0
        u_cart = -K_x * z[0] - K_xd * z[1]
        return float(np.clip(u_pump + u_cart, -u_max, u_max)), "swing"

    dt = 0.002
    T_sim = 6.0
    N = int(T_sim / dt)
    z = np.array([0.0, 0.0, np.pi, 0.1])
    states = np.zeros((N, 4))
    modes = []
    for i in range(N):
        u, mode = control(z)
        states[i] = z
        modes.append(mode)
        z = z + dt * nl_dyn(z, u)

    # Sub-sample to ~120 frames for a snappy GIF.
    stride = max(1, N // 120)
    states_a = states[::stride]
    modes_a = modes[::stride]

    fig, ax = plt.subplots(figsize=(7, 5))
    xmax = max(2.0, float(abs(states_a[:, 0]).max()) + 0.3)
    ax.set_xlim(-xmax, xmax); ax.set_ylim(-0.8, 0.8); ax.set_aspect("equal")
    ax.axhline(0, color="brown", lw=2)
    cart_patch = plt.Rectangle((-0.15, -0.05), 0.3, 0.1, color="steelblue")
    ax.add_patch(cart_patch)
    (link_line,) = ax.plot([], [], "k-", lw=3)
    (bob,) = ax.plot([], [], "ro", ms=14)
    title = ax.set_title("Cart-Pole Swing-Up + LQR Catch")
    ax.set_xticks([]); ax.set_yticks([])

    def update(i):
        s = states_a[i]
        bx, by = s[0] + l * np.sin(s[2]), l * np.cos(s[2])
        cart_patch.set_xy((s[0] - 0.15, -0.05))
        link_line.set_data([s[0], bx], [0, by])
        bob.set_data([bx], [by])
        bob.set_color("limegreen" if modes_a[i] == "LQR" else "red")
        title.set_text(f"Cart-Pole Swing-Up + LQR Catch  |  mode: {modes_a[i]}")
        return [cart_patch, link_line, bob, title]

    anim = FuncAnimation(fig, update, frames=len(states_a), interval=50, blit=False)
    save(anim, "lqr_pendulum", fps=20)


# ----- 5. Pure pursuit -----
def anim_pure_pursuit():
    xs_ref = np.linspace(0, 30, 400)
    ys_ref = 3 * np.sin(xs_ref * 0.3)
    path = np.column_stack([xs_ref, ys_ref])

    x = np.array([0.0, -2.5, 0.0])
    v, Ld, dt = 1.5, 1.8, 0.05
    N = int(25 / dt)
    hist = np.zeros((N, 3))
    for i in range(N):
        d = np.linalg.norm(path - x[:2], axis=1)
        j = int(np.argmin(d))
        while j < len(path) - 1 and np.linalg.norm(path[j] - x[:2]) < Ld:
            j += 1
        tg = path[j]
        alpha = np.arctan2(tg[1] - x[1], tg[0] - x[0]) - x[2]
        omega = 2 * v * np.sin(alpha) / Ld
        x = x + dt * np.array([v * np.cos(x[2]), v * np.sin(x[2]), omega])
        hist[i] = x

    sub = 4
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(xs_ref, ys_ref, "b--", lw=1.5)
    trail, = ax.plot([], [], "r-", lw=2)
    car_dot, = ax.plot([], [], "ko", ms=10)
    car_dir, = ax.plot([], [], "k-", lw=2)
    ax.set_xlim(-1, 31); ax.set_ylim(-5, 5); ax.set_aspect("equal")
    ax.set_title("Pure Pursuit Path Tracking")
    ax.set_xticks([]); ax.set_yticks([])

    def update(i):
        end = i * sub
        trail.set_data(hist[:end + 1, 0], hist[:end + 1, 1])
        car_dot.set_data([hist[end, 0]], [hist[end, 1]])
        L = 1.2
        car_dir.set_data([hist[end, 0], hist[end, 0] + L * np.cos(hist[end, 2])],
                         [hist[end, 1], hist[end, 1] + L * np.sin(hist[end, 2])])
        return [trail, car_dot, car_dir]

    anim = FuncAnimation(fig, update, frames=N // sub, interval=40, blit=False)
    save(anim, "pure_pursuit", fps=20)


# ----- 7. 2-link IK tracing circle -----
def anim_ik():
    l1, l2 = 1.0, 1.0

    def ik(target):
        x, y = target
        c2 = np.clip((x * x + y * y - l1 ** 2 - l2 ** 2) / (2 * l1 * l2), -1, 1)
        th2 = np.arccos(c2)
        th1 = np.arctan2(y, x) - np.arctan2(l2 * np.sin(th2), l1 + l2 * np.cos(th2))
        return th1, th2

    N = 120
    phi = np.linspace(0, 2 * np.pi, N)
    targets = np.column_stack([1.0 + 0.4 * np.cos(phi), 0.7 + 0.4 * np.sin(phi)])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-0.5, 2.5); ax.set_ylim(-0.5, 2)
    ax.set_aspect("equal")
    ax.plot(targets[:, 0], targets[:, 1], "r--", lw=1.5, alpha=0.4)
    arm, = ax.plot([], [], "b-o", lw=3, markersize=8)
    tip, = ax.plot([], [], "g*", ms=14)
    ax.plot(0, 0, "ks", ms=10)
    ax.set_title("Inverse Kinematics: 2-link arm tracing circle")
    ax.set_xticks([]); ax.set_yticks([])

    def update(i):
        th1, th2 = ik(targets[i])
        p1 = np.array([l1 * np.cos(th1), l1 * np.sin(th1)])
        p2 = p1 + np.array([l2 * np.cos(th1 + th2), l2 * np.sin(th1 + th2)])
        arm.set_data([0, p1[0], p2[0]], [0, p1[1], p2[1]])
        tip.set_data([p2[0]], [p2[1]])
        return [arm, tip]

    anim = FuncAnimation(fig, update, frames=N, interval=50, blit=False)
    save(anim, "ik_2link", fps=20)


# ----- 8. ICP alignment iteration -----
def anim_icp():
    np.random.seed(0)
    t = np.linspace(0, 2 * np.pi, 120)
    src = np.column_stack([3 * np.cos(t) + 0.5 * np.sin(3 * t),
                            3 * np.sin(t) + 0.5 * np.cos(2 * t)])
    true_R = np.array([[np.cos(0.5), -np.sin(0.5)], [np.sin(0.5), np.cos(0.5)]])
    true_t = np.array([1.5, -1.0])
    tgt = (true_R @ src.T).T + true_t + np.random.randn(*src.shape) * 0.05

    snaps = [src.copy()]
    s = src.copy()
    for _ in range(20):
        d = np.linalg.norm(s[:, None] - tgt[None], axis=2)
        m = tgt[d.argmin(axis=1)]
        sm, mm = s.mean(0), m.mean(0)
        H = (s - sm).T @ (m - mm)
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1
            R = Vt.T @ U.T
        s = (R @ s.T).T + (mm - R @ sm)
        snaps.append(s.copy())

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(tgt[:, 0], tgt[:, 1], c="r", s=15, label="Target")
    moving = ax.scatter(snaps[0][:, 0], snaps[0][:, 1], c="b", s=15, label="Source")
    ax.set_aspect("equal"); ax.legend()
    title = ax.set_title("ICP Alignment — iter 0")
    ax.set_xticks([]); ax.set_yticks([])

    def update(i):
        moving.set_offsets(snaps[i])
        title.set_text(f"ICP Alignment — iter {i}")
        return [moving, title]

    anim = FuncAnimation(fig, update, frames=len(snaps), interval=300, blit=False)
    save(anim, "icp_alignment", fps=4)


# ----- 9. CBF safety filter — point robot dodging disk -----
def anim_cbf():
    # Obstacle OFFSET from the start->goal line so the CBF can find a
    # nonzero tangent direction at the boundary. With obs exactly on the
    # line, u_nom is radial and projection -> zero -> robot stalls.
    obs = np.array([3.0, 0.7]); r_obs = 1.0
    goal = np.array([6.0, 3.0]); start = np.array([0.0, 0.0])
    K_p, u_max, gamma, dt = 1.0, 1.0, 5.0, 0.05
    N_sim = int(12 / dt)

    def run(use_cbf):
        x = start.copy()
        hist = [x.copy()]
        for _ in range(N_sim):
            u_nom = K_p * (goal - x)
            n = np.linalg.norm(u_nom)
            if n > u_max:
                u_nom = u_nom * (u_max / n)
            if use_cbf:
                lgh = 2 * (x - obs)
                h = np.linalg.norm(x - obs) ** 2 - r_obs ** 2
                slack = max(0.0, -lgh @ u_nom - gamma * h)
                if lgh @ lgh > 1e-10:
                    u_nom = u_nom + slack * lgh / (lgh @ lgh)
                n = np.linalg.norm(u_nom)
                if n > u_max:
                    u_nom = u_nom * (u_max / n)
            x = x + dt * u_nom
            hist.append(x.copy())
        return np.array(hist)

    traj_u = run(False)
    traj_s = run(True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.add_patch(plt.Circle(obs, r_obs, color="lightgray"))
    ax.plot(*start, "go", ms=12)
    ax.plot(*goal, "g*", ms=18)
    ax.set_xlim(-1, 7); ax.set_ylim(-1, 4.5); ax.set_aspect("equal")
    ax.set_title("CBF Safety Filter: Nominal (red) vs Filtered (blue)")
    line_u, = ax.plot([], [], "r-", lw=2, label="Nominal P-controller")
    line_s, = ax.plot([], [], "b-", lw=2, label="CBF safety filter")
    dot_u, = ax.plot([], [], "ro", ms=8)
    dot_s, = ax.plot([], [], "bo", ms=8)
    ax.legend(loc="upper left")
    ax.set_xticks([]); ax.set_yticks([])

    sub = 2
    n_frames = len(traj_u) // sub

    def update(frame):
        k = min(frame * sub, len(traj_u) - 1)
        line_u.set_data(traj_u[:k + 1, 0], traj_u[:k + 1, 1])
        line_s.set_data(traj_s[:k + 1, 0], traj_s[:k + 1, 1])
        dot_u.set_data([traj_u[k, 0]], [traj_u[k, 1]])
        dot_s.set_data([traj_s[k, 0]], [traj_s[k, 1]])
        return [line_u, line_s, dot_u, dot_s]

    anim = FuncAnimation(fig, update, frames=n_frames, interval=50, blit=False)
    save(anim, "cbf_safety_filter", fps=20)


# ----- 10. MPC cart-pole -----
def anim_mpc_cartpole():
    """LQR-flavoured MPC visualization. We reuse LQR as a proxy for the
    constrained MPC because the qualitative behaviour (settle from a
    perturbation under constraint-respecting input) is essentially the
    same near the linearization."""
    M, m, l, g = 1.0, 0.1, 0.5, 9.81
    A = np.array([[0, 1, 0, 0],
                  [0, 0, -m * g / M, 0],
                  [0, 0, 0, 1],
                  [0, 0, (M + m) * g / (M * l), 0]])
    B = np.array([[0], [1 / M], [0], [-1 / (M * l)]])
    Q = np.diag([1, 0.1, 10, 1]); R = np.array([[0.1]])
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P

    def dyn(x, u):
        pos, v, th, om = x
        s, c = np.sin(th), np.cos(th)
        den = M + m * s ** 2
        acc = (u + m * l * s * om ** 2 - m * g * s * c) / den
        a_th = ((M + m) * g * s - u * c - m * l * s * c * om ** 2) / (l * den)
        return np.array([v, acc, om, a_th])

    x = np.array([0.0, 0.0, 0.15, 0.0])  # 9 deg initial
    dt = 0.005; N = 800
    u_max = 5.0
    xs = np.zeros((N, 4)); us = np.zeros(N)
    for i in range(N):
        u = float(-(K @ x).item())
        u = np.clip(u, -u_max, u_max)              # input constraint (MPC's job)
        xs[i] = x; us[i] = u
        x = x + dt * dyn(x, u)

    sub = 5
    fig = plt.figure(figsize=(10, 5))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])
    ax = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-0.4, 1.0); ax.set_aspect("equal")
    ax.axhline(0, color="k", lw=1)
    cart = plt.Rectangle((-0.15, -0.075), 0.3, 0.15, color="steelblue")
    ax.add_patch(cart)
    rod, = ax.plot([], [], "k-", lw=3)
    bob, = ax.plot([], [], "ro", ms=12)
    ax.set_title("MPC Cart-Pole — input constrained to ±5 N")
    ax.set_xticks([]); ax.set_yticks([])

    ax2.axhline(u_max, color="r", ls="--", alpha=0.5)
    ax2.axhline(-u_max, color="r", ls="--", alpha=0.5)
    ax2.set_ylim(-u_max * 1.3, u_max * 1.3)
    ax2.set_xlim(0, N * dt)
    u_line, = ax2.plot([], [], "r-")
    ax2.set_ylabel("u (N)"); ax2.set_xlabel("t (s)"); ax2.grid()

    def update(frame):
        i = min(frame * sub, N - 1)
        pos = xs[i, 0]; th = xs[i, 2]
        cart.set_xy((pos - 0.15, -0.075))
        bx, by = pos + l * np.sin(th), l * np.cos(th)
        rod.set_data([pos, bx], [0, by])
        bob.set_data([bx], [by])
        t_arr = np.arange(i + 1) * dt
        u_line.set_data(t_arr, us[:i + 1])
        return [cart, rod, bob, u_line]

    anim = FuncAnimation(fig, update, frames=N // sub, interval=40, blit=False)
    save(anim, "mpc_cartpole", fps=20)


# ----- 11. EKF-SLAM with covariance ellipses -----
def anim_ekf_slam():
    np.random.seed(1)
    T = 200; dt = 0.1
    v, omega = 1.0, 0.1
    true_robot = np.zeros((T, 3))
    for k in range(1, T):
        th = true_robot[k - 1, 2]
        true_robot[k] = true_robot[k - 1] + dt * np.array([v * np.cos(th), v * np.sin(th), omega])
    true_lm = np.array([[5, 5], [-5, 5], [0, -8], [7, -2]])
    N_LM = len(true_lm); n = 3 + 2 * N_LM

    mu = np.zeros(n)
    Sigma = np.eye(n) * 1e6; Sigma[:3, :3] = np.eye(3) * 0.01
    Q = np.diag([0.05, 0.05, 0.01]) ** 2
    R_obs = np.diag([0.2, 0.05]) ** 2
    seen = [False] * N_LM
    mu_snaps = [mu.copy()]; Sigma_snaps = [Sigma.copy()]

    def wrap(a):
        return (a + np.pi) % (2 * np.pi) - np.pi

    for k in range(1, T):
        u = [v + np.random.randn() * 0.05, omega + np.random.randn() * 0.01]
        th = mu[2]
        mu[:3] = mu[:3] + dt * np.array([u[0] * np.cos(th), u[0] * np.sin(th), u[1]])
        G = np.eye(n)
        G[0, 2] = -dt * u[0] * np.sin(th); G[1, 2] = dt * u[0] * np.cos(th)
        F = np.zeros((3, n)); F[:3, :3] = np.eye(3)
        Sigma = G @ Sigma @ G.T + F.T @ Q @ F
        for i, lm in enumerate(true_lm):
            z = np.array([np.linalg.norm(lm - true_robot[k, :2]),
                          np.arctan2(lm[1] - true_robot[k, 1], lm[0] - true_robot[k, 0]) - true_robot[k, 2]]) \
                + np.random.multivariate_normal([0, 0], R_obs)
            if not seen[i]:
                mu[3 + 2 * i] = mu[0] + z[0] * np.cos(z[1] + mu[2])
                mu[3 + 2 * i + 1] = mu[1] + z[0] * np.sin(z[1] + mu[2])
                seen[i] = True
            dx = mu[3 + 2 * i] - mu[0]; dy = mu[3 + 2 * i + 1] - mu[1]
            q = dx * dx + dy * dy; r = np.sqrt(q)
            z_hat = np.array([r, wrap(np.arctan2(dy, dx) - mu[2])])
            innov = z - z_hat; innov[1] = wrap(innov[1])
            H = np.zeros((2, n))
            H[0, 0] = -dx / r;  H[0, 1] = -dy / r
            H[1, 0] =  dy / q;  H[1, 1] = -dx / q;  H[1, 2] = -1
            H[0, 3 + 2 * i] =  dx / r;  H[0, 3 + 2 * i + 1] =  dy / r
            H[1, 3 + 2 * i] = -dy / q;  H[1, 3 + 2 * i + 1] =  dx / q
            S = H @ Sigma @ H.T + R_obs
            K_g = Sigma @ H.T @ np.linalg.inv(S)
            mu = mu + K_g @ innov
            Sigma = (np.eye(n) - K_g @ H) @ Sigma
        mu_snaps.append(mu.copy()); Sigma_snaps.append(Sigma.copy())

    def ellipse_pts(center, cov2, scale=2.0, n=40):
        """2-sigma ellipse for visualization."""
        vals, vecs = np.linalg.eigh(cov2)
        vals = np.maximum(vals, 1e-8)
        ang = np.linspace(0, 2 * np.pi, n)
        circle = np.stack([np.cos(ang), np.sin(ang)])
        ell = vecs @ np.diag(np.sqrt(vals) * scale) @ circle
        return ell[0] + center[0], ell[1] + center[1]

    sub = 4
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(true_lm[:, 0], true_lm[:, 1], c="g", marker="*", s=200, zorder=5)
    ax.plot(true_robot[:, 0], true_robot[:, 1], "b-", lw=1, alpha=0.3)
    truth_line, = ax.plot([], [], "b-", lw=2, label="true robot")
    est_line, = ax.plot([], [], "r--", lw=1.5, label="SLAM estimate")
    lm_dots = [ax.plot([], [], "rx", ms=14)[0] for _ in range(N_LM)]
    lm_ellipses = [ax.plot([], [], "r-", lw=0.8, alpha=0.6)[0] for _ in range(N_LM)]
    ax.set_xlim(-10, 12); ax.set_ylim(-12, 10); ax.set_aspect("equal")
    ax.set_title("EKF SLAM — landmark uncertainty ellipses shrink as robot moves")
    ax.legend(loc="upper right")
    ax.set_xticks([]); ax.set_yticks([])

    def update(frame):
        t = min(frame * sub, len(mu_snaps) - 1)
        truth_line.set_data(true_robot[:t + 1, 0], true_robot[:t + 1, 1])
        traj_est = np.array([m[:2] for m in mu_snaps[:t + 1]])
        est_line.set_data(traj_est[:, 0], traj_est[:, 1])
        mu_t = mu_snaps[t]; S_t = Sigma_snaps[t]
        for i in range(N_LM):
            lm_x, lm_y = mu_t[3 + 2 * i], mu_t[3 + 2 * i + 1]
            cov = S_t[3 + 2 * i:3 + 2 * i + 2, 3 + 2 * i:3 + 2 * i + 2]
            if cov[0, 0] < 1e5:  # only show seen landmarks
                lm_dots[i].set_data([lm_x], [lm_y])
                ex, ey = ellipse_pts(np.array([lm_x, lm_y]), cov)
                lm_ellipses[i].set_data(ex, ey)
        return [truth_line, est_line] + lm_dots + lm_ellipses

    anim = FuncAnimation(fig, update, frames=T // sub, interval=60, blit=False)
    save(anim, "ekf_slam", fps=15)


# ----- 13. Occupancy grid building scan-by-scan -----
def anim_occupancy():
    np.random.seed(0)
    world = 20.0
    true_obs = [(5, 5, 8, 7), (12, 12, 14, 16), (15, 4, 18, 7), (3, 14, 5, 17)]

    def hit_rect(x, y):
        for (x0, y0, x1, y1) in true_obs:
            if x0 <= x <= x1 and y0 <= y <= y1:
                return True
        return False

    def lidar(pose, n_beams=120, max_range=12.0, step=0.1):
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

    res = 0.2
    nn = int(world / res)
    log_odds = np.zeros((nn, nn))
    l_occ, l_free = 0.7, -0.4
    poses = [(3, 10, 0), (10, 3, np.pi / 2), (10, 17, -np.pi / 2),
             (17, 10, np.pi), (10, 10, 0)]

    snaps = []
    for pose in poses:
        angles, ranges = lidar(pose)
        for a, r in zip(angles, ranges):
            for d in np.arange(0.2, r, res):
                x = pose[0] + d * np.cos(pose[2] + a)
                y = pose[1] + d * np.sin(pose[2] + a)
                if 0 <= x < world and 0 <= y < world:
                    log_odds[int(y / res), int(x / res)] += l_free
            if r < 12.0:
                x = pose[0] + r * np.cos(pose[2] + a)
                y = pose[1] + r * np.sin(pose[2] + a)
                if 0 <= x < world and 0 <= y < world:
                    log_odds[int(y / res), int(x / res)] += l_occ
        snaps.append((1 / (1 + np.exp(-log_odds.copy())), pose))

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(snaps[0][0], origin="lower", cmap="Greys",
                   extent=[0, world, 0, world], vmin=0, vmax=1)
    star, = ax.plot([], [], "r*", ms=20)
    title = ax.set_title("Occupancy grid: scan 1 / 5")
    # Tiny per-frame counter forces frame uniqueness so PillowWriter
    # doesn't dedupe — otherwise each held scan collapses to one frame.
    counter = ax.text(0.01, 0.99, "", transform=ax.transAxes,
                     fontsize=6, color="gray", va="top")
    ax.set_xticks([]); ax.set_yticks([])

    hold = 12
    total = len(snaps) * hold

    def update(frame):
        k = min(frame // hold, len(snaps) - 1)
        prob, pose = snaps[k]
        im.set_data(prob)
        star.set_data([pose[0]], [pose[1]])
        title.set_text(f"Occupancy grid: scan {k + 1} / {len(snaps)}")
        counter.set_text(f"f{frame:03d}")
        return [im, star, title, counter]

    anim = FuncAnimation(fig, update, frames=total, interval=70, blit=False)
    save(anim, "occupancy_grid_building", fps=14)


# ----- 14. SymPy pendulum swinging -----
def anim_sympy_pendulum():
    L_p, g_p = 1.0, 9.81
    dt = 0.01; N = 1000

    def rhs(state):
        th, om = state
        return np.array([om, -(g_p / L_p) * np.sin(th)])

    state = np.array([np.pi / 3, 0.0])
    hist = np.zeros((N, 2))
    for i in range(N):
        hist[i] = state
        k1 = rhs(state)
        k2 = rhs(state + dt * k1 / 2)
        k3 = rhs(state + dt * k2 / 2)
        k4 = rhs(state + dt * k3)
        state = state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    sub = 5
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3); ax.set_aspect("equal")
    ax.plot(0, 0, "ks", ms=10)             # pivot
    rod, = ax.plot([], [], "k-", lw=2)
    bob, = ax.plot([], [], "ro", ms=18)
    trail, = ax.plot([], [], "r-", alpha=0.3, lw=1)
    title = ax.set_title("SymPy Pendulum — Lagrangian derivation + RK4")
    ax.set_xticks([]); ax.set_yticks([])
    ax.axis("off")

    trail_x, trail_y = [], []

    def update(frame):
        i = min(frame * sub, N - 1)
        th = hist[i, 0]
        bx, by = L_p * np.sin(th), -L_p * np.cos(th)
        rod.set_data([0, bx], [0, by])
        bob.set_data([bx], [by])
        trail_x.append(bx); trail_y.append(by)
        if len(trail_x) > 50:
            trail_x.pop(0); trail_y.pop(0)
        trail.set_data(trail_x, trail_y)
        return [rod, bob, trail, title]

    anim = FuncAnimation(fig, update, frames=N // sub, interval=40, blit=False)
    save(anim, "sympy_pendulum_swing", fps=25)


def main():
    print(f"Generating animations in {MEDIA_DIR}")
    anim_astar()
    anim_rrt()
    anim_pf()
    anim_pendulum()
    anim_pure_pursuit()
    anim_ik()
    anim_icp()
    anim_cbf()
    anim_mpc_cartpole()
    anim_ekf_slam()
    anim_occupancy()
    anim_sympy_pendulum()
    print("Done.")


if __name__ == "__main__":
    main()
