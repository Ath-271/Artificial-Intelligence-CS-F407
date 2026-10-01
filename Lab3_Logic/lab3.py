"""
Grid-navigation planner, built the same way as the warehouse planner in
Task 2 of the Logical Planning lab (logic_lab_ex.pdf):

  - A state is a set of logical propositions.
  - Each Action has a name, positive preconditions, negative preconditions,
    positive effects, and negative effects.
  - An action is applicable iff all its preconditions hold in the state:
        S |= Preconditions(a)
  - Applying an action removes its negative effects, then adds its
    positive effects:  S' = Apply(S, a)
  - Breadth-first search explores sequences of applicable actions until
    the goal proposition set is satisfied.

Here the only proposition that matters is the vehicle's location,
At(Robot, (row, col)). One Move action is generated for every pair of
orthogonally-adjacent FREE cells in the maze -- this plays exactly the
role that Move(A,B) / Move(B,C) etc. played for the three named
locations A, B, C in Task 2; we just have many more locations, so the
actions are generated from the grid instead of typed out by hand.
"""

from collections import deque

# ============================================================
# EDIT THIS to change the input maze.
# S = start, G = goal, # = obstacle, . = free space.
# All rows must be the same length.
# ============================================================
MAZE = """\
#####################
#S....#............G#
#.##....##########..#
#....##.............#
#.######.###.#.###..#
#........#..........#
#####################
"""

DIRECTIONS = [
    ("Up",    -1,  0),
    ("Down",   1,  0),
    ("Left",   0, -1),
    ("Right",  0,  1),
]


# ---------------------------------------------------------------
# Action representation (identical shape to Task 2's planner)
# ---------------------------------------------------------------
class Action:
    def __init__(self, name, pos_pre, neg_pre, pos_eff, neg_eff):
        self.name = name
        self.pos_pre = frozenset(pos_pre)   # must be TRUE in state
        self.neg_pre = frozenset(neg_pre)   # must be FALSE in state
        self.pos_eff = frozenset(pos_eff)   # become TRUE
        self.neg_eff = frozenset(neg_eff)   # become FALSE

    def is_applicable(self, state):
        # S |= Preconditions(a)
        return self.pos_pre <= state and self.neg_pre.isdisjoint(state)

    def apply(self, state):
        new_state = set(state)
        new_state -= self.neg_eff   # remove negative effects first
        new_state |= self.pos_eff   # then add positive effects
        return frozenset(new_state)

    def __repr__(self):
        return self.name


def bfs_plan(initial_state, goal, actions):
    initial_state = frozenset(initial_state)
    goal = frozenset(goal)

    if goal <= initial_state:
        return [], [initial_state]

    frontier = deque([(initial_state, [])])
    visited = {initial_state}
    state_history = {initial_state: [initial_state]}

    while frontier:
        state, path = frontier.popleft()
        for action in actions:
            if action.is_applicable(state):
                new_state = action.apply(state)
                if new_state not in visited:
                    new_path = path + [action]
                    if goal <= new_state:
                        return new_path, state_history[state] + [new_state]
                    visited.add(new_state)
                    state_history[new_state] = state_history[state] + [new_state]
                    frontier.append((new_state, new_path))

    return None, None  # no plan found


# ---------------------------------------------------------------
# Build the planning problem (I, A, G) from the maze
# ---------------------------------------------------------------
def parse_maze(text):
    rows = [r for r in text.splitlines() if r.strip() != ""]
    width = len(rows[0])
    grid = [list(r.ljust(width, "#")) for r in rows]

    start = goal = None
    for r, row in enumerate(grid):
        for c, ch in enumerate(row):
            if ch == "S":
                start = (r, c)
            elif ch == "G":
                goal = (r, c)
    if start is None or goal is None:
        raise ValueError("Maze must contain exactly one 'S' and one 'G'.")
    return grid, start, goal


def is_free(grid, r, c):
    rows, cols = len(grid), len(grid[0])
    if 0 <= r < rows and 0 <= c < cols:
        return grid[r][c] != "#"
    return False


def build_actions(grid):
    """
    One Action per (free_cell -> adjacent free_cell) transition, e.g.:

        Move_Up((2,4)->(1,4))
            preconditions: At(Robot,(2,4))
            effects:        remove At(Robot,(2,4)), add At(Robot,(1,4))

    This is the grid analogue of Move(A,B) / Move(B,C) / ... in the
    warehouse problem from the lab.
    """
    actions = []
    rows, cols = len(grid), len(grid[0])
    for r in range(rows):
        for c in range(cols):
            if not is_free(grid, r, c):
                continue
            here = (r, c)
            for name, dr, dc in DIRECTIONS:
                there = (r + dr, c + dc)
                if is_free(grid, *there):
                    actions.append(Action(
                        name=f"{name}({here}->{there})",
                        pos_pre={f"At(Robot,{here})"},
                        neg_pre=set(),
                        pos_eff={f"At(Robot,{there})"},
                        neg_eff={f"At(Robot,{here})"},
                    ))
    return actions


def render(grid, path_cells=None):
    grid_copy = [row[:] for row in grid]
    if path_cells:
        for (r, c) in path_cells:
            if grid_copy[r][c] not in ("S", "G"):
                grid_copy[r][c] = "*"
    return "\n".join("".join(row) for row in grid_copy)


def cell_of(proposition):
    """Extract the (row,col) tuple out of an 'At(Robot,(r,c))' proposition."""
    inside = proposition[len("At(Robot,"):-1]
    return eval(inside)  # "(1, 1)" -> (1, 1)


# ---------------------------------------------------------------
# Run the planner, in the same reporting style as Task 2
# ---------------------------------------------------------------
def main():
    grid, start, goal = parse_maze(MAZE)
    actions = build_actions(grid)

    I = {f"At(Robot,{start})"}     # initial state
    G = {f"At(Robot,{goal})"}      # goal

    print("Input maze:")
    print(render(grid))
    print(f"\nInitial state I = {I}")
    print(f"Goal G          = {G}\n")

    plan, states = bfs_plan(I, G, actions)

    if plan is None:
        print("Result: No plan found")
        return

    print(f"Plan found ({len(plan)} steps):")
    for i, a in enumerate(plan, start=1):
        print(f"  Step {i}: {a.name}")

    print("\nState trace:")
    print(f"  S0: {set(states[0])}")
    for i, (a, s) in enumerate(zip(plan, states[1:]), start=1):
        print(f"  --{a.name}--> S{i}: {set(s)}")

    path_cells = [cell_of(next(iter(s))) for s in states]
    print("\nMaze with path marked ('*'):")
    print(render(grid, path_cells))


if __name__ == "__main__":
    main()