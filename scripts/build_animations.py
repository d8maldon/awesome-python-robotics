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
    """Triple-link inverted pendulum balanced by LQR.

    Re-derives the dynamics symbolically with sympy.physics.mechanics
    (takes ~10-30 s) then animates the nonlinear closed-loop simulation.
    """
    import sympy as sp
    import sympy.physics.mechanics as me
    from scipy.integrate import solve_ivp

    print("    (deriving triple-pendulum dynamics — this takes ~10-30 s)")
    t_sym = me.dynamicsymbols._t
    q  = me.dynamicsymbols('x th1 th2 th3')
    qd = [qi.diff(t_sym) for qi in q]
    u_sym = me.dynamicsymbols('u')

    Mc_s, g_s = sp.symbols('Mc g', positive=True)
    m_syms = sp.symbols('m1 m2 m3', positive=True)
    L_syms = sp.symbols('L1 L2 L3', positive=True)

    Nf = me.ReferenceFrame('Nf')
    O = me.Point('O'); O.set_vel(Nf, 0)
    Pc = O.locatenew('Pc', q[0]*Nf.x)
    Pc.set_vel(Nf, qd[0]*Nf.x)
    cart = me.Particle('Cart', Pc, Mc_s)

    bodies = [cart]
    joint = Pc
    for i in range(3):
        A = Nf.orientnew(f'A{i+1}', 'Axis', [q[i+1], Nf.z])
        A.set_ang_vel(Nf, qd[i+1]*Nf.z)
        P_com = joint.locatenew(f'Pc{i+1}', (L_syms[i]/2)*A.y)
        P_com.v2pt_theory(joint, Nf, A)
        bodies.append(me.Particle(f'L{i+1}', P_com, m_syms[i]))
        next_j = joint.locatenew(f'J{i+1}', L_syms[i]*A.y)
        next_j.v2pt_theory(joint, Nf, A)
        joint = next_j

    KE = sum(b.kinetic_energy(Nf) for b in bodies)
    PE = sum(b.mass * g_s * b.point.pos_from(O).dot(Nf.y) for b in bodies)
    L_lag = KE - PE

    LM = me.LagrangesMethod(L_lag, q, forcelist=[(Pc, u_sym*Nf.x)], frame=Nf)
    LM.form_lagranges_equations()

    Lk = 0.3
    params = {Mc_s: 1.0, g_s: 9.81,
              m_syms[0]: 0.1, m_syms[1]: 0.1, m_syms[2]: 0.1,
              L_syms[0]: Lk, L_syms[1]: Lk, L_syms[2]: Lk}
    M_fn = sp.lambdify(list(q) + list(qd) + [u_sym], LM.mass_matrix_full.subs(params), 'numpy')
    F_fn = sp.lambdify(list(q) + list(qd) + [u_sym], LM.forcing_full.subs(params), 'numpy')

    def nl_dyn(z, u_val):
        args = list(z) + [u_val]
        Mm = np.asarray(M_fn(*args), dtype=float)
        Ff = np.asarray(F_fn(*args), dtype=float).flatten()
        return np.linalg.solve(Mm, Ff)

    eps = 1e-6
    Ax = np.zeros((8, 8)); Bx = np.zeros((8, 1))
    z_eq = np.zeros(8)
    for i in range(8):
        zp = z_eq.copy(); zp[i] += eps
        zm = z_eq.copy(); zm[i] -= eps
        Ax[:, i] = (nl_dyn(zp, 0.0) - nl_dyn(zm, 0.0)) / (2 * eps)
    Bx[:, 0] = (nl_dyn(z_eq, eps) - nl_dyn(z_eq, -eps)) / (2 * eps)

    Q = np.diag([1.0, 200.0, 200.0, 200.0, 1.0, 1.0, 1.0, 1.0])
    R_lqr = np.array([[0.05]])
    P_riccati = solve_continuous_are(Ax, Bx, Q, R_lqr)
    K = np.linalg.inv(R_lqr) @ Bx.T @ P_riccati

    def closed_loop(_, z):
        return nl_dyn(z, float(-(K @ z).item()))

    z0 = np.array([0.0, 0.04, -0.02, 0.03, 0, 0, 0, 0])
    sol = solve_ivp(closed_loop, [0, 4.0], z0, dense_output=True,
                    rtol=1e-7, atol=1e-9, max_step=0.005)
    t_arr = np.linspace(0, 4.0, 200)
    states = sol.sol(t_arr).T

    def joints(state):
        x = state[0]
        th = state[1:4]
        pts = [(x, 0.0)]
        for i in range(3):
            pts.append((pts[-1][0] + Lk * np.sin(th[i]),
                        pts[-1][1] + Lk * np.cos(th[i])))
        return np.array(pts)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_xlim(-0.9, 0.9); ax.set_ylim(-0.15, 1.1); ax.set_aspect("equal")
    ax.axhline(0, color="brown", lw=2)
    cart = plt.Rectangle((-0.15, -0.05), 0.3, 0.1, color="steelblue")
    ax.add_patch(cart)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    link_lines = [ax.plot([], [], color=c, lw=4, marker="o", ms=6)[0] for c in colors]
    ax.set_title("LQR — Triple-Link Inverted Pendulum")
    ax.set_xticks([]); ax.set_yticks([])

    def update(i):
        s = states[i]
        pts = joints(s)
        cart.set_xy((s[0] - 0.15, -0.05))
        for k in range(3):
            link_lines[k].set_data([pts[k, 0], pts[k+1, 0]],
                                    [pts[k, 1], pts[k+1, 1]])
        return [cart] + link_lines

    anim = FuncAnimation(fig, update, frames=len(states), interval=50, blit=False)
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


# ----- 6. Quadrotor PID with moving setpoint -----
def anim_quadrotor():
    np.random.seed(0)
    g, m, I_inertia = 9.81, 0.5, 0.01
    Kp_z, Kd_z = 8, 4.5
    Kp_phi, Kd_phi = 18, 5.5
    x = np.array([0.0, 0.0, 0.3, 0.0])
    dt = 0.002                       # smaller dt — was 0.01, too coarse for stiff phi loop (Kp/I=1800)
    N = 2500                         # 5 s of sim
    hist = np.zeros((N, 4))
    setpts = np.zeros(N)
    for i in range(N):
        z_set = 1.6 + 0.5 * np.sin(0.005 * i)
        setpts[i] = z_set
        z, zd, phi, pd = x
        T = m * (g + Kp_z * (z_set - z) - Kd_z * zd)
        tau = Kp_phi * (0 - phi) - Kd_phi * pd
        if i in (600, 1400, 2100):        # periodic roll disturbance
            x[3] += 0.8
        x = x + dt * np.array([zd, T * np.cos(phi) / m - g, pd, tau / I_inertia])
        hist[i] = x

    sub = 20                                # 125 frames
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-0.8, 0.8); ax.set_ylim(0, 3)
    ax.axhline(0, color="brown", lw=2)
    setpt_line, = ax.plot([], [], "r--", alpha=0.6, label="setpoint")
    body, = ax.plot([], [], "k-", lw=4)
    r_left, = ax.plot([], [], "bo", ms=10)
    r_right, = ax.plot([], [], "bo", ms=10)
    ax.set_title("Quadrotor PID — moving setpoint + roll kicks")
    ax.set_xticks([]); ax.legend(loc="upper right")
    L = 0.3

    def update(frame):
        idx = frame * sub
        z, _, phi, _ = hist[idx]
        dx = L * np.cos(phi); dy = L * np.sin(phi)
        body.set_data([-dx, dx], [z - dy, z + dy])
        r_left.set_data([-dx], [z - dy])
        r_right.set_data([dx], [z + dy])
        setpt_line.set_data([-0.7, 0.7], [setpts[idx], setpts[idx]])
        return [body, r_left, r_right, setpt_line]

    anim = FuncAnimation(fig, update, frames=N // sub, interval=40, blit=False)
    save(anim, "quadrotor_pid", fps=25)


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


def main():
    print(f"Generating animations in {MEDIA_DIR}")
    anim_astar()
    anim_rrt()
    anim_pf()
    anim_pendulum()
    anim_pure_pursuit()
    anim_quadrotor()
    anim_ik()
    anim_icp()
    print("Done.")


if __name__ == "__main__":
    main()
