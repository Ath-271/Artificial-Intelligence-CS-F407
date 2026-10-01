"""
Warehouse Navigation Agent
===========================

This module implements a GOAL-BASED AGENT that solves the warehouse
navigation problem described in the AI Agents laboratory exercise.

Problem
-------
An autonomous warehouse vehicle must travel from a starting position (S)
to a goal position (G) on a 2D grid, without crossing any shelving units
(obstacles, marked '#'). Free space is marked '.'.

Why this is a GOAL-BASED agent (and not a simple reflex agent)
----------------------------------------------------------------
A simple reflex agent chooses its next action based only on the CURRENT
percept (e.g. "if there is a wall to my right, turn left"). It has no
memory of where it has been and no explicit notion of "success".

This agent, by contrast:
  * maintains an explicit internal model of the environment (the grid),
  * has an explicit GOAL state (the dispatch area G),
  * considers the CONSEQUENCES of sequences of actions (a full path),
  * selects the action sequence that is predicted to achieve the goal.

This is precisely the definition of a goal-based agent: it doesn't just
react to what it currently perceives, it plans ahead using a model of
the world in order to satisfy a goal condition.

Search algorithm chosen: Breadth-First Search (BFS)
-----------------------------------------------------
BFS is appropriate here because:
  * All moves (Up/Down/Left/Right) have the SAME cost (one grid square),
    so the path found by BFS is guaranteed to be a SHORTEST path.
  * The state space (grid cells) is small enough that BFS's memory
    requirements are not a concern.
  * BFS is complete: if a path exists, BFS is guaranteed to find it.
  * It is simpler to implement and reason about than informed search
    algorithms (e.g. A*), while still guaranteeing optimality here
    because the environment is unweighted.

If the warehouse were much larger, or if we had a good heuristic
(e.g. Manhattan distance to G), A* search would likely be preferred,
since it uses that heuristic to explore fewer nodes than BFS while
still finding a shortest path.
"""

from collections import deque

# ---------------------------------------------------------------------
# 1. THE ENVIRONMENT
# ---------------------------------------------------------------------
WAREHOUSE_MAP = [
    "#####################",
    "#S....#............G#",
    "#.##....##########..#",
    "#....##.............#",
    "#.######.###.#.###..#",
    "#........#..........#",
    "#####################",
]


def parse_grid(grid_lines):
    """Convert the list-of-strings map into a grid, and locate S and G.

    Returns
    -------
    grid : list[str]          the raw grid (rows of characters)
    start : tuple(row, col)   coordinates of 'S'
    goal : tuple(row, col)    coordinates of 'G'
    """
    grid = grid_lines
    start = goal = None
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            if cell == "S":
                start = (r, c)
            elif cell == "G":
                goal = (r, c)
    if start is None or goal is None:
        raise ValueError("Grid must contain both a start 'S' and a goal 'G'.")
    return grid, start, goal


def is_free(grid, pos):
    """An agent may occupy a cell if it is inside the grid and not a wall."""
    r, c = pos
    if r < 0 or r >= len(grid):
        return False
    if c < 0 or c >= len(grid[r]):
        return False
    return grid[r][c] != "#"


# ---------------------------------------------------------------------
# 2. THE AGENT'S AVAILABLE ACTIONS
# ---------------------------------------------------------------------
# Each action maps to a (delta_row, delta_col) change in position.
ACTIONS = {
    "Up":    (-1, 0),
    "Down":  (1, 0),
    "Left":  (0, -1),
    "Right": (0, 1),
}


# ---------------------------------------------------------------------
# 3. THE DECISION-MAKING COMPONENT (goal-based search)
# ---------------------------------------------------------------------
def find_path(grid, start, goal):
    """Breadth-First Search for a collision-free path from start to goal.

    Returns a list of (action, position) pairs representing the path,
    or None if no path exists.
    """
    frontier = deque([start])
    came_from = {start: None}   # position -> (previous_position, action_taken)

    while frontier:
        current = frontier.popleft()

        if current == goal:
            return _reconstruct_path(came_from, start, goal)

        for action, (dr, dc) in ACTIONS.items():
            next_pos = (current[0] + dr, current[1] + dc)
            if is_free(grid, next_pos) and next_pos not in came_from:
                came_from[next_pos] = (current, action)
                frontier.append(next_pos)

    return None  # goal is unreachable


def _reconstruct_path(came_from, start, goal):
    """Walk the came_from chain backwards from goal to start."""
    path = []
    node = goal
    while node != start:
        prev, action = came_from[node]
        path.append((action, node))
        node = prev
    path.reverse()
    return path


# ---------------------------------------------------------------------
# 4. AGENT EXECUTION / REPORTING
# ---------------------------------------------------------------------
def print_solution(grid, start, goal, path):
    print("Warehouse map:")
    for row in grid:
        print(row)
    print()
    print(f"Start position: {start}")
    print(f"Goal position : {goal}")
    print()

    if path is None:
        print("No collision-free path exists between S and G.")
        return

    actions = [action for action, _ in path]
    print(f"Path found! Length: {len(path)} moves")
    print(f"Action sequence: {' -> '.join(actions)}")
    print()

    # Draw the path onto the grid for a visual check.
    grid_chars = [list(row) for row in grid]
    for _, (r, c) in path:
        if grid_chars[r][c] == ".":
            grid_chars[r][c] = "*"
    print("Path visualised ('*' marks the route taken):")
    for row in grid_chars:
        print("".join(row))


if __name__ == "__main__":
    grid, start, goal = parse_grid(WAREHOUSE_MAP)
    solution = find_path(grid, start, goal)
    print_solution(grid, start, goal, solution)