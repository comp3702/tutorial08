import random

UP = 'U'
DOWN = 'D'
LEFT = 'L'
RIGHT = 'R'

ACTIONS = [UP, DOWN, LEFT, RIGHT]

ACTION_DELTAS = {
    UP: (-1, 0),
    DOWN: (1, 0),
    LEFT: (0, -1),
    RIGHT: (0, 1)
}

class GridWorld:
    def __init__(
            self,
            num_rows: int,
            num_columns: int,
            obstacles: list[tuple[int, int]],
            terminal_states: list[tuple[int, int]],
            rewards: dict[tuple[int, int], float],
            transition_probability: float,
            discount: float
    ):
        self.num_rows = num_rows
        self.num_cols = num_columns
        self.obstacles = obstacles
        self.states = [(x, y) for x in range(self.num_rows) for y in range(self.num_cols) if (x, y) not in obstacles]
        self.discount = discount
        self.terminal_states = terminal_states
        self.rewards = rewards
        self.p = transition_probability

    def attempt_move(self, state: tuple[int, int], action: str) -> tuple[int, int]:
        if state in self.terminal_states:
            return state

        x, y = state
        dx, dy = ACTION_DELTAS.get(action)
        new_state = max(min(x + dx, self.num_rows - 1), 0), max(min(y + dy, self.num_cols - 1), 0)

        if new_state in self.obstacles:
            return state

        return new_state

    def stoch_action(self, a: str) -> dict[str, float]:
        # Stochastic actions probability distributions
        if a == RIGHT:
            return {RIGHT: 0.8, UP: 0.1, DOWN: 0.1}
        elif a == UP:
            return {UP: 0.8, LEFT: 0.1, RIGHT: 0.1}
        elif a == LEFT:
            return {LEFT: 0.8, UP: 0.1, DOWN: 0.1}
        elif a == DOWN:
            return {DOWN: 0.8, LEFT: 0.1, RIGHT: 0.1}

    def perform_action(self, state: tuple[int, int], action: str) -> tuple[int, int]:
        actions = list(self.stoch_action(action).items())
        action_chosen = random.choices([item[0] for item in actions], weights=[item[1] for item in actions])[0]
        next_state = self.attempt_move(state, action_chosen)
        return next_state

    def get_reward(self, state: tuple[int, int]) -> float:
        return self.rewards.get(state, 0.0)
