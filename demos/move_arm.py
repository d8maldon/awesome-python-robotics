"""Move a 2-link planar arm with the mouse — analytical inverse kinematics.

The end-effector tries to track your cursor. If the cursor is outside the
reachable workspace, the arm fully extends toward it.

    Mouse        target end-effector position
    E            toggle elbow-up / elbow-down
    Q / Esc      quit

Run:  pip install pygame  &&  python demos/move_arm.py
"""
import sys
import math
import pygame

W, H = 900, 700
ORIGIN = (W // 2, H // 2 + 100)
PX_PER_M = 140

L1, L2 = 1.2, 1.0  # link lengths in meters


def w2s(x, y):
    return int(ORIGIN[0] + x * PX_PER_M), int(ORIGIN[1] - y * PX_PER_M)


def s2w(sx, sy):
    return (sx - ORIGIN[0]) / PX_PER_M, (ORIGIN[1] - sy) / PX_PER_M


def ik(target, elbow_up=True):
    x, y = target
    r2 = x * x + y * y
    c2 = max(-1.0, min(1.0, (r2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)))
    th2 = math.acos(c2) if elbow_up else -math.acos(c2)
    th1 = math.atan2(y, x) - math.atan2(L2 * math.sin(th2), L1 + L2 * math.cos(th2))
    return th1, th2, math.sqrt(r2) <= L1 + L2


def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Mouse-controlled 2-link IK")
    font = pygame.font.SysFont("consolas", 16)
    big_font = pygame.font.SysFont("consolas", 20, bold=True)
    clock = pygame.time.Clock()

    elbow_up = True
    running = True
    while running:
        clock.tick(60)
        for e in pygame.event.get():
            if e.type == pygame.QUIT: running = False
            elif e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_q, pygame.K_ESCAPE): running = False
                if e.key == pygame.K_e: elbow_up = not elbow_up

        screen.fill((25, 30, 40))

        # Draw workspace annulus
        inner = abs(L1 - L2) * PX_PER_M
        outer = (L1 + L2) * PX_PER_M
        pygame.draw.circle(screen, (45, 55, 75), ORIGIN, int(outer), 1)
        if inner > 1:
            pygame.draw.circle(screen, (45, 55, 75), ORIGIN, int(inner), 1)

        mx, my = pygame.mouse.get_pos()
        tx, ty = s2w(mx, my)
        r = math.hypot(tx, ty)
        # Clamp target to workspace
        if r > L1 + L2:
            scale = (L1 + L2 - 1e-3) / r
            tx_c, ty_c = tx * scale, ty * scale
            reachable = False
        elif r < abs(L1 - L2):
            scale = (abs(L1 - L2) + 1e-3) / max(r, 1e-3)
            tx_c, ty_c = tx * scale, ty * scale
            reachable = False
        else:
            tx_c, ty_c = tx, ty
            reachable = True

        th1, th2, _ = ik((tx_c, ty_c), elbow_up)
        p1 = (L1 * math.cos(th1), L1 * math.sin(th1))
        p2 = (p1[0] + L2 * math.cos(th1 + th2), p1[1] + L2 * math.sin(th1 + th2))

        s_base = w2s(0, 0)
        s_p1 = w2s(*p1)
        s_p2 = w2s(*p2)

        # Cursor & target marker
        col = (120, 220, 120) if reachable else (240, 140, 80)
        pygame.draw.circle(screen, col, (mx, my), 8, 2)
        pygame.draw.line(screen, col, (mx - 12, my), (mx + 12, my), 1)
        pygame.draw.line(screen, col, (mx, my - 12), (mx, my + 12), 1)

        # Arm
        pygame.draw.line(screen, (180, 180, 200), s_base, s_p1, 8)
        pygame.draw.line(screen, (180, 180, 200), s_p1, s_p2, 8)
        pygame.draw.circle(screen, (120, 130, 145), s_base, 10)
        pygame.draw.circle(screen, (120, 130, 145), s_p1, 8)
        pygame.draw.circle(screen, (220, 80, 80), s_p2, 9)

        hud = [
            f"target: ({tx:+.2f}, {ty:+.2f}) m",
            f"theta1: {math.degrees(th1):+.1f}°",
            f"theta2: {math.degrees(th2):+.1f}°",
            f"elbow:  {'up' if elbow_up else 'down'}  (E to toggle)",
            f"reach:  {'OK' if reachable else 'OUT OF WORKSPACE'}",
        ]
        for i, line in enumerate(hud):
            screen.blit(font.render(line, True, (220, 220, 220)), (14, 14 + 22 * i))
        screen.blit(big_font.render(
            "Mouse-controlled 2-link IK  ·  Move cursor  ·  E elbow  ·  Q quit",
            True, (180, 180, 200)), (14, H - 32))

        pygame.display.flip()
    pygame.quit()


if __name__ == "__main__":
    main()
