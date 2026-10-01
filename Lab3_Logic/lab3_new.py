from collections import deque

class Action:
    def __init__(self, name, pos_pre=None, neg_pre=None, pos_eff=None, neg_eff=None):
        """
        Represents an action in a classical planning domain.
        pos_pre: Set of propositions that MUST be present in the state.
        neg_pre: Set of propositions that MUST NOT be present in the state.
        pos_eff: Set of propositions added after execution.
        neg_eff: Set of propositions removed after execution.
        """
        self.name = name
        self.pos_pre = set(pos_pre) if pos_pre else set()
        self.neg_pre = set(neg_pre) if neg_pre else set()
        self.pos_eff = set(pos_eff) if pos_eff else set()
        self.neg_eff = set(neg_eff) if neg_eff else set()

    def is_applicable(self, state):
        """An action is applicable if positive preconditions are satisfied and negative ones are absent."""
        return self.pos_pre.issubset(state) and self.neg_pre.isdisjoint(state)

    def apply(self, state):
        """Applies the STRIPS effects to return a brand new state."""
        new_state = set(state)
        new_state.difference_update(self.neg_eff)  # Remove negative effects
        new_state.update(self.pos_eff)             # Add positive effects
        return new_state


def breadth_first_search(initial_state, goal_state, actions):
    """Explores alternative plans using Breath-First Search (BFS)."""
    # Queue stores: (current_state, path_of_actions_taken)
    queue = deque([(initial_state, [])])
    visited = {frozenset(initial_state)}

    while queue:
        current_state, path = queue.popleft()

        # Goal check: verifying if target conditions are met in the state
        if goal_state.issubset(current_state):
            return path

        # Explore every valid action choice
        for action in actions:
            if action.is_applicable(current_state):
                next_state = action.apply(current_state)
                state_key = frozenset(next_state)

                if state_key not in visited:
                    visited.add(state_key)
                    queue.append((next_state, path + [action]))

    return None  # Triggers when the state frontier runs dry without a match


def run_planner(initial_state, goal_state, actions, test_label="BASE CASE"):
    """Runs the engine over the setup and outputs clear execution tracking."""
    print(f"\n==================== RUNNING TEST: {test_label} ====================")
    print(f"Initial State: {sorted(list(initial_state))}")
    print(f"Goal Target:   {sorted(list(goal_state))}\n")

    path = breadth_first_search(initial_state, goal_state, actions)

    if path is None:
        print("RESULT -> No plan found!")
    else:
        print("RESULT -> Shortest path plan found!\n")
        print("Step 0: Start")
        print(f"  Facts: {sorted(list(initial_state))}")
        
        current_state = initial_state
        for i, action in enumerate(path, 1):
            current_state = action.apply(current_state)
            print(f"\nStep {i}: Action -> {action.name}")
            print(f"  Facts: {sorted(list(current_state))}")
    print("=" * 60)


# ==========================================
# SCENARIO DECLARATION (LABORATORY TASK 2)
# ==========================================

# Base problem parameters
initial_state = {"At(Robot, A)", "At(Package, A)"}
goal_state = {"At(Package, C)"}

# Complete environment structural dictionary mappings
standard_actions = [
    # Move transitions
    Action("Move(A, B)", pos_pre={"At(Robot, A)"}, pos_eff={"At(Robot, B)"}, neg_eff={"At(Robot, A)"}),
    Action("Move(B, A)", pos_pre={"At(Robot, B)"}, pos_eff={"At(Robot, A)"}, neg_eff={"At(Robot, B)"}),
    Action("Move(B, C)", pos_pre={"At(Robot, B)"}, pos_eff={"At(Robot, C)"}, neg_eff={"At(Robot, B)"}),
    Action("Move(C, B)", pos_pre={"At(Robot, C)"}, pos_eff={"At(Robot, B)"}, neg_eff={"At(Robot, C)"}),
    
    # PickUp transitions
    Action("PickUp(Package, A)", pos_pre={"At(Robot, A)", "At(Package, A)"}, pos_eff={"Holding(Package)"}, neg_eff={"At(Package, A)"}),
    Action("PickUp(Package, B)", pos_pre={"At(Robot, B)", "At(Package, B)"}, pos_eff={"Holding(Package)"}, neg_eff={"At(Package, B)"}),
    Action("PickUp(Package, C)", pos_pre={"At(Robot, C)", "At(Package, C)"}, pos_eff={"Holding(Package)"}, neg_eff={"At(Package, C)"}),
    
    # Drop transitions
    Action("Drop(Package, A)", pos_pre={"At(Robot, A)", "Holding(Package)"}, pos_eff={"At(Package, A)"}, neg_eff={"Holding(Package)"}),
    Action("Drop(Package, B)", pos_pre={"At(Robot, B)", "Holding(Package)"}, pos_eff={"At(Package, B)"}, neg_eff={"Holding(Package)"}),
    Action("Drop(Package, C)", pos_pre={"At(Robot, C)", "Holding(Package)"}, pos_eff={"At(Package, C)"}, neg_eff={"Holding(Package)"})
]

if __name__ == "__main__":
    # Task 2 baseline execution block
    run_planner(initial_state, goal_state, standard_actions, "Warehouse Base Solver")
