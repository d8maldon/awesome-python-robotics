"""Interactive A* path planner on a 2-D occupancy grid.

    Left click       paint obstacle
    Right click      erase obstacle
    S + click        set START
    G + click        set GOAL
    Space            run A* (auto-runs after any change)
    C                clear obstacles
    R                reset start/goal
    Q / Esc          quit

Run:  pip install pygame  &&  python demos/click_to_plan.py
"""
import sys
import math
import heapq
import pygame

GRID_W, GRID_H = 60, 40
CELL = 16
W, H = GRID_W * CELL, GRID_H * CELL + 32  # extra row for HUD


def astar(grid, start, goal):
    H_, W_ = len(grid), len(grid[0])

    def h(a, b): return math.hypot(a[0] - b[0], a[1] - b[1])

    heap = [(h(start, goal), 0.0, start, None)]
    came = {}
    g = {start: 0.0}
    expanded = set()
    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    while heap:
        f, gv, cur, par = heapq.heappop(heap)
        if cur in came: continue
        came[cur] = par
        expanded.add(cur)
        if cur == goal:
            path = []
            while cur is not None:
                path.append(cur); cur = came[cur]
            return path[::-1], expanded
        for dy, dx in nbrs:
            ny, nx = cur[0] + dy, cur[1] + dx
            if 0 <= ny < H_ and 0 <= nx < W_ and not grid[ny][nx]:
                tent = gv + math.hypot(dy, dx)
                if tent < g.get((ny, nx), math.inf):
                    g[(ny, nx)] = tent
                    heapq.heappush(heap, (tent + h((ny, nx), goal), tent, (ny, nx), cur))
    return None, expanded


def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Click-to-plan A*")
    font = pygame.font.SysFont("consolas", 14)

    grid = [[0] * GRID_W for _ in range(GRID_H)]
    start = (5, 5)
    goal = (GRID_H - 6, GRID_W - 6)
    path, expanded = astar(grid, start, goal)

    drag = None  # "paint" / "erase"
    running = True
    clock = pygame.time.Clock()

    while running:
        clock.tick(60)
        keys = pygame.key.get_pressed()
        for e in pygame.event.get():
            if e.type == pygame.QUIT: running = False
            elif e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_q, pygame.K_ESCAPE): running = False
                elif e.key == pygame.K_c:
                    grid = [[0] * GRID_W for _ in range(GRID_H)]
                    path, expanded = astar(grid, start, goal)
                elif e.key == pygame.K_r:
                    start, goal = (5, 5), (GRID_H - 6, GRID_W - 6)
                    path, expanded = astar(grid, start, goal)
                elif e.key == pygame.K_SPACE:
                    path, expanded = astar(grid, start, goal)
            elif e.type == pygame.MOUSEBUTTONDOWN:
                mx, my = e.pos
                if my >= GRID_H * CELL: continue
                gx, gy = mx // CELL, my // CELL
                if not (0 <= gx < GRID_W and 0 <= gy < GRID_H): continue
                if keys[pygame.K_s]:
                    if not grid[gy][gx]:
                        start = (gy, gx)
                        path, expanded = astar(grid, start, goal)
                    continue
                if keys[pygame.K_g]:
                    if not grid[gy][gx]:
                        goal = (gy, gx)
                        path, expanded = astar(grid, start, goal)
                    continue
                drag = "paint" if e.button == 1 else "erase"
                _click(grid, gy, gx, drag, start, goal)
                path, expanded = astar(grid, start, goal)
            elif e.type == pygame.MOUSEMOTION and drag is not None:
                mx, my = e.pos
                if my >= GRID_H * CELL: continue
                gx, gy = mx // CELL, my // CELL
                if not (0 <= gx < GRID_W and 0 <= gy < GRID_H): continue
                if _click(grid, gy, gx, drag, start, goal):
                    path, expanded = astar(grid, start, goal)
            elif e.type == pygame.MOUSEBUTTONUP:
                drag = None

        screen.fill((25, 25, 30))
        # Grid
        for y in range(GRID_H):
            for x in range(GRID_W):
                rect = (x * CELL, y * CELL, CELL - 1, CELL - 1)
                if grid[y][x]:
                    pygame.draw.rect(screen, (200, 200, 210), rect)
                elif (y, x) in expanded:
                    pygame.draw.rect(screen, (60, 90, 60), rect)
                else:
                    pygame.draw.rect(screen, (45, 45, 55), rect)
        # Path
        if path:
            for (y, x) in path:
                rect = (x * CELL + 2, y * CELL + 2, CELL - 5, CELL - 5)
                pygame.draw.rect(screen, (255, 80, 80), rect)
        # Start / goal
        pygame.draw.circle(screen, (80, 220, 80),
                            (start[1] * CELL + CELL // 2, start[0] * CELL + CELL // 2),
                            CELL // 2 - 2)
        pygame.draw.circle(screen, (80, 140, 255),
                            (goal[1] * CELL + CELL // 2, goal[0] * CELL + CELL // 2),
                            CELL // 2 - 2)
        # HUD
        pygame.draw.rect(screen, (15, 15, 20), (0, GRID_H * CELL, W, 32))
        hint = ("LMB obstacle  ·  RMB erase  ·  S+click set START  ·  G+click set GOAL  "
                "·  C clear  ·  R reset  ·  Q quit")
        screen.blit(font.render(hint, True, (200, 200, 210)), (10, GRID_H * CELL + 8))
        msg = f"expanded {len(expanded)}  ·  path {len(path) if path else 'NONE'}"
        m = font.render(msg, True, (255, 220, 120))
        screen.blit(m, (W - m.get_width() - 10, GRID_H * CELL + 8))

        pygame.display.flip()
    pygame.quit()


def _click(grid, gy, gx, mode, start, goal):
    if (gy, gx) in (start, goal): return False
    if mode == "paint":
        if grid[gy][gx]: return False
        grid[gy][gx] = 1
    else:
        if not grid[gy][gx]: return False
        grid[gy][gx] = 0
    return True


if __name__ == "__main__":
    main()
