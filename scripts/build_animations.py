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


# ----- 4. 2-link cart-pendulum swing-up via iLQR + LQR catch -----
def anim_pendulum():
    """Cart + double pendulum swing-up via iLQR + LQR catch.

    Matches notebook 05. Derives dynamics symbolically (sympy.physics.mechanics),
    runs ~60 iterations of iLQR (~20 s), then animates the closed-loop trajectory.
    Bob colors: orange = iLQR open-loop swing-up, lime = LQR catch.
    """
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    import sympy as sp
    import sympy.physics.mechanics as me

    print("    (deriving cart-double-pendulum dynamics + running iLQR — ~25 s)")

    t_sym = me.dynamicsymbols._t
    q = me.dynamicsymbols('x th1 th2')
    qd = [qi.diff(t_sym) for qi in q]
    u_sym = me.dynamicsymbols('u')
    Mc_s, g_s = sp.symbols('Mc g', positive=True)
    m1_s, m2_s = sp.symbols('m1 m2', positive=True)
    L1_s, L2_s = sp.symbols('L1 L2', positive=True)
    Nf = me.ReferenceFrame('Nf')
    O = me.Point('O'); O.set_vel(Nf, 0)
    Pc = O.locatenew('Pc', q[0] * Nf.x)
    Pc.set_vel(Nf, qd[0] * Nf.x)
    cart = me.Particle('Cart', Pc, Mc_s)
    A1 = Nf.orientnew('A1', 'Axis', [q[1], Nf.z])
    A1.set_ang_vel(Nf, qd[1] * Nf.z)
    P1 = Pc.locatenew('P1', (L1_s / 2) * A1.y); P1.v2pt_theory(Pc, Nf, A1)
    link1 = me.Particle('L1', P1, m1_s)
    J1 = Pc.locatenew('J1', L1_s * A1.y); J1.v2pt_theory(Pc, Nf, A1)
    A2 = Nf.orientnew('A2', 'Axis', [q[2], Nf.z])
    A2.set_ang_vel(Nf, qd[2] * Nf.z)
    P2 = J1.locatenew('P2', (L2_s / 2) * A2.y); P2.v2pt_theory(J1, Nf, A2)
    link2 = me.Particle('L2', P2, m2_s)
    KE = sum(b.kinetic_energy(Nf) for b in [cart, link1, link2])
    PE = sum(b.mass * g_s * b.point.pos_from(O).dot(Nf.y) for b in [cart, link1, link2])
    LM = me.LagrangesMethod(KE - PE, q, forcelist=[(Pc, u_sym * Nf.x)], frame=Nf)
    LM.form_lagranges_equations()
    params = {Mc_s: 1.0, g_s: 9.81, m1_s: 0.3, m2_s: 0.3, L1_s: 0.4, L2_s: 0.4}
    M_fn = sp.lambdify(list(q) + list(qd) + [u_sym], LM.mass_matrix_full.subs(params), 'numpy')
    F_fn = sp.lambdify(list(q) + list(qd) + [u_sym], LM.forcing_full.subs(params), 'numpy')

    def nl_dyn(z, u):
        args = list(z) + [u]
        Mm = np.asarray(M_fn(*args), dtype=float)
        Ff = np.asarray(F_fn(*args), dtype=float).flatten()
        return np.linalg.solve(Mm, Ff)

    def step(z, u, dt):
        k1 = nl_dyn(z, u); k2 = nl_dyn(z + 0.5 * dt * k1, u)
        k3 = nl_dyn(z + 0.5 * dt * k2, u); k4 = nl_dyn(z + dt * k3, u)
        return z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def wrap(a):
        return (a + np.pi) % (2 * np.pi) - np.pi

    # LQR around upright
    eps = 1e-6; z_eq = np.zeros(6)
    A_lin = np.zeros((6, 6)); B_lin = np.zeros((6, 1))
    for i in range(6):
        zp = z_eq.copy(); zp[i] += eps; zm = z_eq.copy(); zm[i] -= eps
        A_lin[:, i] = (nl_dyn(zp, 0.0) - nl_dyn(zm, 0.0)) / (2 * eps)
    B_lin[:, 0] = (nl_dyn(z_eq, eps) - nl_dyn(z_eq, -eps)) / (2 * eps)
    Q_lqr = np.diag([10.0, 100.0, 100.0, 1.0, 1.0, 1.0])
    R_lqr_m = np.array([[0.1]])
    P_riccati = solve_continuous_are(A_lin, B_lin, Q_lqr, R_lqr_m)
    K_lqr = np.linalg.inv(R_lqr_m) @ B_lin.T @ P_riccati

    def diff_to_goal(z, zg):
        zd = z.copy()
        zd[1] = wrap(z[1] - zg[1]); zd[2] = wrap(z[2] - zg[2])
        zd[0] -= zg[0]; zd[3:] -= zg[3:]
        return zd

    def linearize(z, u, dt):
        f0 = step(z, u, dt)
        fx = np.zeros((6, 6)); fu = np.zeros((6, 1)); e = 1e-5
        for i in range(6):
            zp = z.copy(); zp[i] += e
            fx[:, i] = (step(zp, u, dt) - f0) / e
        fu[:, 0] = (step(z, u + e, dt) - f0) / e
        return fx, fu

    def total_cost(zs, us, zg, Q, R, Qf):
        J = 0.0
        for tt in range(len(us)):
            zd = diff_to_goal(zs[tt], zg)
            J += zd @ Q @ zd + R[0, 0] * us[tt] ** 2
        zd = diff_to_goal(zs[-1], zg)
        return J + zd @ Qf @ zd

    def ilqr_run(z0, zg, U_init, dt, N, Q, R, Qf, u_max=30.0, max_iter=60):
        U = U_init.copy()
        Z = np.zeros((N + 1, 6)); Z[0] = z0
        for tt in range(N):
            Z[tt + 1] = step(Z[tt], float(U[tt]), dt)
        J_prev = total_cost(Z, U, zg, Q, R, Qf); mu = 1.0
        for it in range(max_iter):
            fx_arr = np.zeros((N, 6, 6)); fu_arr = np.zeros((N, 6, 1))
            for tt in range(N):
                fx_arr[tt], fu_arr[tt] = linearize(Z[tt], float(U[tt]), dt)
            zd_f = diff_to_goal(Z[-1], zg); Vx = 2 * Qf @ zd_f; Vxx = 2 * Qf
            K_gain = np.zeros((N, 1, 6)); k_gain = np.zeros((N, 1)); ok = True
            for tt in range(N - 1, -1, -1):
                zd = diff_to_goal(Z[tt], zg)
                lx = 2 * Q @ zd; lu = np.array([2 * R[0, 0] * float(U[tt])])
                lxx = 2 * Q; luu = 2 * R
                fx, fu = fx_arr[tt], fu_arr[tt]
                Qx = lx + fx.T @ Vx; Qu = lu + fu.T @ Vx
                Qxx = lxx + fx.T @ Vxx @ fx; Qux = fu.T @ Vxx @ fx
                Quu = luu + fu.T @ Vxx @ fu
                try:
                    Quu_inv = np.linalg.inv(Quu + mu * np.eye(1))
                except np.linalg.LinAlgError:
                    ok = False; break
                k_t = -Quu_inv @ Qu; K_t = -Quu_inv @ Qux
                K_gain[tt] = K_t; k_gain[tt] = k_t.flatten()
                Vx = Qx + K_t.T @ Quu @ k_t.flatten() + K_t.T @ Qu + Qux.T @ k_t.flatten()
                Vxx = Qxx + K_t.T @ Quu @ K_t + K_t.T @ Qux + Qux.T @ K_t
                Vxx = 0.5 * (Vxx + Vxx.T)
            if not ok:
                mu *= 10; continue
            best_J = J_prev; best_Z = None; best_U = None
            for alpha in [1.0, 0.5, 0.25, 0.1, 0.05, 0.01]:
                U_new = np.zeros(N); Z_new = np.zeros((N + 1, 6)); Z_new[0] = z0
                for tt in range(N):
                    dz = Z_new[tt] - Z[tt]
                    dz[1] = wrap(dz[1]); dz[2] = wrap(dz[2])
                    u_new = U[tt] + alpha * k_gain[tt, 0] + (K_gain[tt] @ dz).item()
                    u_new = float(np.clip(u_new, -u_max, u_max))
                    U_new[tt] = u_new
                    Z_new[tt + 1] = step(Z_new[tt], u_new, dt)
                J_new = total_cost(Z_new, U_new, zg, Q, R, Qf)
                if J_new < J_prev:
                    best_J = J_new; best_Z = Z_new; best_U = U_new; break
            if best_Z is None:
                mu *= 2
                if mu > 1e6: break
                continue
            improvement = J_prev - best_J
            Z = best_Z; U = best_U; J_prev = best_J
            mu = max(0.1, mu / 2)
            if improvement < 1e-3 and it > 15: break
        return Z, U

    # ---- Optimize trajectory ----
    N = 200; dt = 0.02
    z0 = np.array([0.0, np.pi, np.pi, 0.0, 0.0, 0.0])
    zg = np.zeros(6)
    Q_run = np.diag([0.1, 1.0, 1.0, 0.01, 0.01, 0.01])
    Qf = np.diag([10.0, 100.0, 100.0, 10.0, 10.0, 10.0])
    R = np.array([[0.01]])
    omega_n = np.sqrt(9.81 / 0.4)
    ts = np.arange(N) * dt
    U_init = 8.0 * np.sin(2 * omega_n * ts)
    Z_opt, U_opt = ilqr_run(z0, zg, U_init, dt, N, Q_run, R, Qf, u_max=30.0, max_iter=60)

    # ---- Hybrid simulation ----
    theta_LQR = 0.4; c_ROA = 10.0
    def hybrid(z, t_idx):
        zl = z.copy(); zl[1] = wrap(zl[1]); zl[2] = wrap(zl[2])
        inside = (abs(zl[1]) < theta_LQR and abs(zl[2]) < theta_LQR
                  and (zl @ P_riccati @ zl) < c_ROA)
        if inside:
            return float(np.clip(-(K_lqr @ zl).item(), -30, 30)), "LQR"
        if t_idx < len(U_opt):
            return float(U_opt[t_idx]), "iLQR"
        return float(np.clip(-(K_lqr @ zl).item(), -30, 30)), "LQR"

    T_sim = 6.0; N_total = int(T_sim / dt)
    z = z0.copy()
    states = np.zeros((N_total, 6)); modes = []
    for i in range(N_total):
        u, mode = hybrid(z, i)
        states[i] = z; modes.append(mode)
        z = step(z, u, dt)

    # ---- Animation ----
    L1, L2 = 0.4, 0.4
    stride = max(1, N_total // 120)
    states_a = states[::stride]; modes_a = modes[::stride]

    fig, ax = plt.subplots(figsize=(7, 6))
    xmax = max(1.2, float(abs(states_a[:, 0]).max()) + 0.3)
    ax.set_xlim(-xmax, xmax); ax.set_ylim(-1.0, 1.0); ax.set_aspect("equal")
    ax.axhline(0, color="brown", lw=2)
    cart_patch = plt.Rectangle((-0.12, -0.04), 0.24, 0.08, color="steelblue")
    ax.add_patch(cart_patch)
    (link1_line,) = ax.plot([], [], color="#1f77b4", lw=4, marker="o", ms=7)
    (link2_line,) = ax.plot([], [], color="#ff7f0e", lw=4, marker="o", ms=7)
    (tip,) = ax.plot([], [], "o", ms=11)
    title = ax.set_title("2-Link Cart-Pendulum: iLQR Swing-Up + LQR Catch")
    ax.set_xticks([]); ax.set_yticks([])

    def update(i):
        s = states_a[i]
        x = s[0]; th1 = s[1]; th2 = s[2]
        j1 = np.array([x + L1 * np.sin(th1), L1 * np.cos(th1)])
        j2 = j1 + np.array([L2 * np.sin(th2), L2 * np.cos(th2)])
        cart_patch.set_xy((x - 0.12, -0.04))
        link1_line.set_data([x, j1[0]], [0, j1[1]])
        link2_line.set_data([j1[0], j2[0]], [j1[1], j2[1]])
        tip.set_data([j2[0]], [j2[1]])
        tip.set_color("limegreen" if modes_a[i] == "LQR" else "red")
        title.set_text(f"2-Link Cart-Pendulum: mode = {modes_a[i]}")
        return [cart_patch, link1_line, link2_line, tip, title]

    anim = FuncAnimation(fig, update, frames=len(states_a), interval=50, blit=False)
    save(anim, "lqr_pendulum", fps=20)




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
    anim_pendulum()
    anim_cbf()
    anim_mpc_cartpole()
    anim_ekf_slam()
    anim_occupancy()
    anim_sympy_pendulum()
    print("Done.")


if __name__ == "__main__":
    main()
