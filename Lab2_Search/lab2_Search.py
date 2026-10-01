"""
AI Laboratory: Search and A* -- Warehouse Robot Navigation
=============================================================
Core deliverable, matching the laboratory's required tests exactly
(Task 3: Tests 1-4 on the original warehouse map and the maps the
lab sheet specifies). BFS and A* share the same grid utilities so
they can be compared fairly under Task 5.

------------------------------------------------------------------
Search problem formulation  P = (S, A, T, s0, G, c)
------------------------------------------------------------------
State (S)      : a robot position (row, col) on the grid.
Actions (A)    : {Up, Down, Left, Right}.
Transition (T) : moving one cell in the chosen direction, provided
                 the destination is inside the grid and free ('.').
Initial state  : the cell marked 'S'.
Goal (G)       : the cell marked 'G' (goal test: state == goal).
Cost (c)       : every move costs 1 (uniform-cost grid).
"""

import heapq
from collections import deque


# ---------------------------------------------------------------------
# Grid utilities (shared by both search algorithms)
# ---------------------------------------------------------------------
ACTIONS = {
    "Up":    (-1, 0),
    "Down":  (1, 0),
    "Left":  (0, -1),
    "Right": (0, 1),
}


def parse_grid(grid_lines):
    """Locate S and G in an ASCII map; return (grid, start, goal)."""
    grid = grid_lines
    start = goal = None
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            if cell == "S":
                start = (r, c)
            elif cell == "G":
                goal = (r, c)
    if start is None or goal is None:
        raise ValueError("Grid must contain both 'S' and 'G'.")
    return grid, start, goal


def is_free(grid, pos):
    """A state is valid if it is inside the grid and not an obstacle."""
    r, c = pos
    if r < 0 or r >= len(grid):
        return False
    if c < 0 or c >= len(grid[r]):
        return False
    return grid[r][c] != "#"


def neighbours(grid, pos):
    """The transition function T: yield (action, next_state) pairs."""
    for action, (dr, dc) in ACTIONS.items():
        nxt = (pos[0] + dr, pos[1] + dc)
        if is_free(grid, nxt):
            yield action, nxt


def reconstruct_path(came_from, start, goal):
    """Path reconstruction: walk the came_from chain backwards."""
    path = []
    node = goal
    while node != start:
        prev, action = came_from[node]
        path.append((action, node))
        node = prev
    path.reverse()
    return path


# ---------------------------------------------------------------------
# Heuristic for A* (Manhattan distance, as specified in the lab sheet)
# ---------------------------------------------------------------------
def manhattan(a, b):
    """h(n) = |x - xG| + |y - yG| -- admissible for 4-directional grids."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def euclidean(a, b):
    """Straight-line distance -- used in Task 6's heuristic investigation."""
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def zero_heuristic(a, b):
    """h(n) = 0 for all n -- used in Task 6's heuristic investigation."""
    return 0


# ---------------------------------------------------------------------
# Breadth-First Search (blind search)
# ---------------------------------------------------------------------
def bfs(grid, start, goal):
    """FIFO frontier; no cost/heuristic information is used at all."""
    frontier = deque([start])          # <-- THE FRONTIER
    came_from = {start: None}
    visited = {start}                  # <-- VISITED STATES
    expanded = 0

    while frontier:
        current = frontier.popleft()   # next state to expand: oldest first
        expanded += 1

        if current == goal:            # <-- GOAL TEST
            return reconstruct_path(came_from, start, goal), expanded

        for action, nxt in neighbours(grid, current):   # <-- ACTION / TRANSITION
            if nxt not in visited:
                visited.add(nxt)
                came_from[nxt] = (current, action)
                frontier.append(nxt)

    return None, expanded


# ---------------------------------------------------------------------
# A* Search: f(n) = g(n) + h(n)
# ---------------------------------------------------------------------
def astar(grid, start, goal, heuristic=manhattan, weight=1.0):
    """Priority-queue frontier ordered by f(n) = g(n) + h(n).

    `weight` multiplies h(n) -- used only in the Task 6 heuristic
    experiments (weight=1 is standard A*; weight>1 is "greedier").
    """
    counter = 0  # tie-breaker so the heap never compares tuples' states
    frontier = [(heuristic(start, goal) * weight, counter, start)]  # <-- FRONTIER (min-heap by f)
    came_from = {start: None}
    g_score = {start: 0}               # <-- g(n)
    expanded = 0
    closed = set()                     # <-- VISITED / CLOSED SET

    while frontier:
        f, _, current = heapq.heappop(frontier)  # state with smallest f(n)

        if current in closed:
            continue
        closed.add(current)
        expanded += 1

        if current == goal:            # <-- GOAL TEST
            return reconstruct_path(came_from, start, goal), expanded, g_score[goal]

        for action, nxt in neighbours(grid, current):   # <-- ACTION / TRANSITION
            tentative_g = g_score[current] + 1  # every move costs 1
            if nxt not in g_score or tentative_g < g_score[nxt]:
                g_score[nxt] = tentative_g
                came_from[nxt] = (current, action)
                h = heuristic(nxt, goal) * weight        # <-- h(n)
                f_score = tentative_g + h                # <-- f(n) = g(n) + h(n)
                counter += 1
                heapq.heappush(frontier, (f_score, counter, nxt))

    return None, expanded, None


# ---------------------------------------------------------------------
# Reporting helper
# ---------------------------------------------------------------------
def report(name, path, expanded, cost=None):
    print(f"--- {name} ---")
    if path is None:
        print("Solution found: NO")
    else:
        actions = [a for a, _ in path]
        print("Solution found: YES")
        print(f"Path length   : {len(path)}")
        if cost is not None:
            print(f"Path cost     : {cost}")
        print(f"Actions       : {' -> '.join(actions)}")
    print(f"States expanded: {expanded}")
    print()


# ---------------------------------------------------------------------
# Test maps (Task 3 of the lab sheet)
# ---------------------------------------------------------------------
MAIN_MAP = [
    "#################",
    "#S....#.........#",
    "#.###.#.#######.#",
    "#...#.#.......#.#",
    "###.#.#######.#.#",
    "#...#.........#.#",
    "#.###########.#.#",
    "#.............#G#",
    "#################",
]

TRIVIAL_MAP = [
    "#####",
    "#SG##",
    "#####",
]

NO_SOLUTION_MAP = [
    "#######",
    "#S....#",
    "###.###",
    "#...#G#",
    "#######",
]

ALT_PATHS_MAP = [
    "###########",
    "#S.........#",
    "#.#########",
    "#.........G",
    "###########",
]


def run_comparison(name, grid_lines):
    print(f"========== {name} ==========\n")
    grid, start, goal = parse_grid(grid_lines)
    bfs_path, bfs_expanded = bfs(grid, start, goal)
    report("BFS", bfs_path, bfs_expanded)
    astar_path, astar_expanded, astar_cost = astar(grid, start, goal)
    report("A* (Manhattan)", astar_path, astar_expanded, astar_cost)


if __name__ == "__main__":
    run_comparison("Test 1: Original warehouse map", MAIN_MAP)
    run_comparison("Test 2: Trivial case (goal adjacent to start)", TRIVIAL_MAP)
    run_comparison("Test 3: No solution (goal unreachable)", NO_SOLUTION_MAP)
    run_comparison("Test 4: Alternative paths", ALT_PATHS_MAP)

    # ------------------------------------------------------------------
    # Task 6: heuristic investigation, run on the original warehouse map
    # ------------------------------------------------------------------
    print("========== Task 6: Heuristic investigation (original map) ==========\n")
    grid, start, goal = parse_grid(MAIN_MAP)

    path, expanded, cost = astar(grid, start, goal, heuristic=zero_heuristic)
    report("h(n) = 0  (uniform-cost search)", path, expanded, cost)

    path, expanded, cost = astar(grid, start, goal, heuristic=manhattan)
    report("h(n) = Manhattan distance", path, expanded, cost)

    path, expanded, cost = astar(grid, start, goal, heuristic=euclidean)
    report("h(n) = Euclidean distance", path, expanded, cost)

    path, expanded, cost = astar(grid, start, goal, heuristic=manhattan, weight=2.0)
    report("h(n) = 2 x Manhattan distance (inadmissible)", path, expanded, cost)