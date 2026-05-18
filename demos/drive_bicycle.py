"""Drive the kinematic bicycle model with arrow keys.

    Up / Down    accelerate / brake
    Left / Right steer
    R            reset
    Q / Esc      quit

Run:  pip install pygame  &&  python demos/drive_bicycle.py
"""
import sys
import math
import pygame

# --- World <-> screen ---
W, H = 1000, 700
PX_PER_M = 18  # 18 pixels per meter

def w2s(x, y):
    return int(W / 2 + x * PX_PER_M), int(H / 2 - y * PX_PER_M)


# --- Bicycle params ---
L = 2.5            # wheelbase (m)
V_MAX = 12.0       # m/s
A_MAX = 4.0        # m/s^2
DELTA_MAX = math.radians(35)
FRICTION = 1.5     # m/s^2 passive decel


def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Drive the Bicycle")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 18)
    big_font = pygame.font.SysFont("consolas", 24, bold=True)

    state = reset_state()
    trail = []

    running = True
    while running:
        dt = clock.tick(60) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key in (pygame.K_q, pygame.K_ESCAPE):
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                state = reset_state(); trail = []

        keys = pygame.key.get_pressed()
        accel = 0.0
        if keys[pygame.K_UP]:    accel += A_MAX
        if keys[pygame.K_DOWN]:  accel -= A_MAX
        if abs(accel) < 0.1:
            accel = -FRICTION * (1 if state["v"] > 0 else -1 if state["v"] < 0 else 0)
        delta = 0.0
        if keys[pygame.K_LEFT]:  delta = +DELTA_MAX
        if keys[pygame.K_RIGHT]: delta = -DELTA_MAX

        # Integrate
        state["v"] = max(-V_MAX / 2, min(V_MAX, state["v"] + accel * dt))
        state["x"] += state["v"] * math.cos(state["theta"]) * dt
        state["y"] += state["v"] * math.sin(state["theta"]) * dt
        state["theta"] += state["v"] / L * math.tan(delta) * dt

        trail.append((state["x"], state["y"]))
        if len(trail) > 1200:
            trail.pop(0)

        # Camera follows car
        cam_x, cam_y = state["x"], state["y"]
        screen.fill((35, 35, 40))
        # Draw ground grid
        for gx in range(-200, 201, 5):
            sx, _ = w2s(gx - cam_x, 0)
            pygame.draw.line(screen, (55, 55, 60), (sx, 0), (sx, H))
        for gy in range(-200, 201, 5):
            _, sy = w2s(0, gy - cam_y)
            pygame.draw.line(screen, (55, 55, 60), (0, sy), (W, sy))

        # Draw trail
        if len(trail) > 1:
            pts = [w2s(tx - cam_x, ty - cam_y) for tx, ty in trail]
            pygame.draw.lines(screen, (200, 80, 80), False, pts, 2)

        # Draw car
        cx, cy = w2s(0, 0)
        car_len, car_wid = L * PX_PER_M, 1.2 * PX_PER_M
        car_pts = []
        for dx, dy in [(-car_len/2, -car_wid/2), (car_len/2, -car_wid/2),
                        (car_len/2, car_wid/2), (-car_len/2, car_wid/2)]:
            rx = dx * math.cos(state["theta"]) - dy * math.sin(state["theta"])
            ry = dx * math.sin(state["theta"]) + dy * math.cos(state["theta"])
            car_pts.append((cx + rx, cy - ry))
        pygame.draw.polygon(screen, (100, 180, 255), car_pts)
        # Front wheel direction indicator
        fx = cx + (L / 2 * PX_PER_M) * math.cos(state["theta"])
        fy = cy - (L / 2 * PX_PER_M) * math.sin(state["theta"])
        front_th = state["theta"] + delta
        pygame.draw.line(screen, (255, 220, 80),
                         (fx, fy),
                         (fx + 25 * math.cos(front_th), fy - 25 * math.sin(front_th)), 4)

        # HUD
        hud = [
            f"x:     {state['x']:+7.2f} m",
            f"y:     {state['y']:+7.2f} m",
            f"theta: {math.degrees(state['theta']):+7.1f}°",
            f"v:     {state['v']:+7.2f} m/s",
            f"delta: {math.degrees(delta):+7.1f}°",
        ]
        for i, line in enumerate(hud):
            screen.blit(font.render(line, True, (220, 220, 220)), (12, 12 + 22 * i))
        screen.blit(big_font.render("Drive the Bicycle  ·  Arrows / R reset / Q quit",
                                     True, (180, 180, 200)), (12, H - 36))

        pygame.display.flip()

    pygame.quit()


def reset_state():
    return {"x": 0.0, "y": 0.0, "theta": 0.0, "v": 0.0}


if __name__ == "__main__":
    main()
