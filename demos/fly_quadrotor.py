"""Fly the planar quadrotor with arrow keys.

    Up           thrust (overrides hover)
    Left/Right   roll torque
    Space        re-arm / reset
    Q / Esc      quit

The model is a 2-D planar quad (state = [x, z, vx, vz, phi, phi_dot]).
Gravity always pulls down. Without any input the quad hovers at ~80% throttle.

Run:  pip install pygame  &&  python demos/fly_quadrotor.py
"""
import sys
import math
import pygame

W, H = 1000, 700
GROUND_Y = H - 60
PX_PER_M = 50

G = 9.81
M = 0.5
I = 0.01
HOVER_THRUST = M * G
EXTRA_THRUST = HOVER_THRUST * 1.5
TORQUE = 0.10


def w2s(x, z):
    return int(W / 2 + x * PX_PER_M), int(GROUND_Y - z * PX_PER_M)


def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Fly the Quadrotor")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 18)
    big_font = pygame.font.SysFont("consolas", 22, bold=True)

    state = reset()
    trail = []
    crashed = False

    running = True
    while running:
        dt = min(clock.tick(60) / 1000.0, 0.02)
        for e in pygame.event.get():
            if e.type == pygame.QUIT: running = False
            elif e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_q, pygame.K_ESCAPE): running = False
                if e.key == pygame.K_SPACE:
                    state = reset(); trail = []; crashed = False

        if not crashed:
            keys = pygame.key.get_pressed()
            T = HOVER_THRUST
            if keys[pygame.K_UP]:    T = EXTRA_THRUST
            if keys[pygame.K_DOWN]:  T = HOVER_THRUST * 0.4
            tau = 0.0
            if keys[pygame.K_LEFT]:  tau = +TORQUE
            if keys[pygame.K_RIGHT]: tau = -TORQUE

            ax = -T * math.sin(state["phi"]) / M
            az = T * math.cos(state["phi"]) / M - G
            state["vx"] += ax * dt
            state["vz"] += az * dt
            state["x"] += state["vx"] * dt
            state["z"] += state["vz"] * dt
            state["phi_dot"] += tau / I * dt
            state["phi"] += state["phi_dot"] * dt

            if state["z"] < 0.05:
                state["z"] = 0.05
                if state["vz"] < -3 or abs(state["phi"]) > math.radians(45):
                    crashed = True
                else:
                    state["vz"] = 0
                    state["vx"] *= 0.7
                    state["phi_dot"] *= 0.5

            trail.append((state["x"], state["z"]))
            if len(trail) > 800: trail.pop(0)

        # Camera follows quad horizontally
        cam_x = state["x"]
        screen.fill((30, 35, 55))
        for gx in range(-200, 201, 2):
            sx, _ = w2s(gx - cam_x, 0)
            pygame.draw.line(screen, (45, 50, 70), (sx, 0), (sx, GROUND_Y))
        pygame.draw.rect(screen, (90, 65, 45), (0, GROUND_Y, W, H - GROUND_Y))

        if len(trail) > 1:
            pts = [w2s(tx - cam_x, tz) for tx, tz in trail]
            pygame.draw.lines(screen, (200, 200, 60), False, pts, 2)

        qx, qz = w2s(0, state["z"])
        L = 0.4
        dx = L * math.cos(state["phi"]) * PX_PER_M
        dy = L * math.sin(state["phi"]) * PX_PER_M
        left  = (qx - dx, qz + dy)
        right = (qx + dx, qz - dy)
        pygame.draw.line(screen, (180, 180, 200), left, right, 6)
        pygame.draw.circle(screen, (255, 100, 80), (int(left[0]), int(left[1])), 8)
        pygame.draw.circle(screen, (255, 100, 80), (int(right[0]), int(right[1])), 8)

        hud = [
            f"x:    {state['x']:+7.2f} m",
            f"z:    {state['z']:+7.2f} m",
            f"vx:   {state['vx']:+7.2f} m/s",
            f"vz:   {state['vz']:+7.2f} m/s",
            f"phi:  {math.degrees(state['phi']):+7.1f}°",
        ]
        for i, line in enumerate(hud):
            screen.blit(font.render(line, True, (220, 220, 220)), (12, 12 + 22 * i))
        screen.blit(big_font.render(
            "Fly the Quadrotor  ·  Up=thrust  Left/Right=roll  Space=reset  Q=quit",
            True, (180, 180, 200)), (12, H - 32))
        if crashed:
            crash = big_font.render("CRASHED — press Space to re-arm", True, (255, 100, 80))
            screen.blit(crash, (W // 2 - crash.get_width() // 2, H // 2))

        pygame.display.flip()
    pygame.quit()


def reset():
    return {"x": 0.0, "z": 0.5, "vx": 0.0, "vz": 0.0, "phi": 0.0, "phi_dot": 0.0}


if __name__ == "__main__":
    main()
