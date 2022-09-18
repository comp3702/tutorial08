from envs.GridWorld import GridWorld
from solvers.mcts import MCTS
from visualizer.views import MCTSView


class MCTSController:
    def __init__(self, master):
        num_rows = 3
        num_cols = 4
        obstacles = [(1, 1)]
        rewards = {
            (1, 3): -1,
            (0, 3): 1,
        }
        self.current_state = (2, 0)
        terminal_states = list(rewards.keys())
        self.grid = GridWorld(num_rows, num_cols, obstacles, terminal_states, rewards, 0.8, 0.9)

        INITIAL_STATE = (0, 0)
        self.mcts = MCTS(self.grid)
        self.mcts.plan_online(self.current_state, iters=1)

        mcts = self.mcts
        self._num_sims = 0
        self.view = MCTSView(master, num_rows, num_cols, self.move, self.sim)

        self.rewards = rewards
        self.view.redraw_grid(self.rewards, self.mcts.q_sa, self.mcts.n_sa, self.mcts.n_s, mcts.extract_policy(),
                              self.current_state)
        self.view.update_labels(self.current_state, self._num_sims)

    def move(self, action):
        self.current_state = self.grid.perform_action(self.current_state, action)
        self.view.update_labels(self.current_state, self._num_sims)
        self.view.redraw_grid(self.rewards, self.mcts.q_sa, self.mcts.n_sa, self.mcts.n_s, self.mcts.extract_policy(),
                              self.current_state)

    def sim(self, num_iter):
        self._num_sims += num_iter
        self.mcts.plan_online(self.current_state, iters=num_iter)
        self.view.redraw_grid(self.rewards, self.mcts.q_sa, self.mcts.n_sa, self.mcts.n_s, self.mcts.extract_policy(),
                              self.current_state)
        self.view.update_labels(self.current_state, self._num_sims)
