from envs.GridWorld import ACTIONS
import random
import math


class MCTS:
    # This class is adapted from the A2 sample solution with credit to Nick Collins
    VISITS_PER_SIM = 1
    MAX_ROLLOUT_DEPTH = 200
    TRIALS_PER_ROLLOUT = 1
    EXP_BIAS = 4000  # This is set very high for demo so can easily predict next selected

    def __init__(self, env):
        self.env = env
        self.q_sa = {}
        self.n_s = {}
        self.n_sa = {}

    def selection(self, state):
        """ Given a state, selects the next action based on UCB1. """
        unvisited = []
        for a in ACTIONS:
            if (state, a) not in self.n_sa:
                unvisited.append(a)
        if unvisited:
            # If there's an unvisited, go there to see what it's like
            return random.choice(unvisited)

        # They've all been visited, so pick which one to try again based on UCB1
        best_u = -float('inf')
        best_a = None
        for a in ACTIONS:
            u = self.q_sa.get((state, a), 0) + (
                        self.EXP_BIAS * math.sqrt(math.log(self.n_s.get(state, 0)) / self.n_sa.get((state, a), 1)))
            if u > best_u:
                best_u = u
                best_a = a
        return best_a if best_a is not None else random.choice(ACTIONS)

    def simulate(self, initial_state):
        # self.initial_state = initial_state
        visited = {}
        return self.mcts_search(initial_state, 0, visited)

    def plan_online(self, state, iters=10000):
        max_iter = iters
        for i in range(max_iter):
            self.simulate(state)
        return self.mcts_select_action(state)

    def mcts_search(self, state, depth, visited):
        # Check for non-visit conditions
        if (state in visited and visited[state] >= self.VISITS_PER_SIM) or (depth > self.MAX_ROLLOUT_DEPTH):
            # Choose the best Q-value if one exists
            best_q = -float('inf')
            best_a = None
            for a in ACTIONS:
                if (state, a) in self.q_sa and self.q_sa[(state, a)] > best_q:
                    best_q = self.q_sa[(state, a)]
                    best_a = a
            if best_a is not None:
                return best_q
            else:
                return self.mcts_random_rollout(state, self.MAX_ROLLOUT_DEPTH - depth, self.TRIALS_PER_ROLLOUT)
        else:
            visited[state] = visited.get(state, 0) + 1

        # Check for terminal state
        if state in self.env.terminal_states:
            self.n_s[state] = 1
            return self.env.get_reward(state)

        # Check for leaf node:
        if state not in self.n_s:
            # Reached an unexpanded state (i.e. simulation time) so perform rollout from here
            self.n_s[state] = 0
            return self.mcts_random_rollout(state, self.MAX_ROLLOUT_DEPTH - depth, self.TRIALS_PER_ROLLOUT)
        else:
            action = self.selection(state)

            # Update counts
            self.n_sa[(state, action)] = self.n_sa.get((state, action), 0) + 1
            self.n_s[state] += 1

            # Execute the selected action and recurse
            new_state = self.env.perform_action(state, action)
            r = self.env.get_reward(new_state) + self.env.discount * self.mcts_search(new_state, depth + 1, visited)

            # update node statistics
            if (state, action) not in self.q_sa:
                self.q_sa[(state, action)] = r
            else:
                self.q_sa[(state, action)] = ((self.q_sa[(state, action)] * self.n_sa[(state, action)]) + r) / (
                            self.n_sa[(state, action)] + 1)

            return r

    def mcts_random_rollout(self, state, max_depth, trials):
        total = 0
        s = state
        for i in range(trials):
            d = 0
            while d < max_depth and not s in self.env.terminal_states:
                action = random.choice(ACTIONS)
                new_state = self.env.perform_action(s, action)
                reward = self.env.get_reward(new_state)
                total += (self.env.discount ** (d + 1)) * (reward)
                s = new_state
                d += 1
        return total / trials

    def mcts_select_action(self, state):
        best_q = -float('inf')
        best_a = None
        for a in ACTIONS:
            if (state, a) in self.q_sa and self.q_sa[(state, a)] > best_q:
                best_q = self.q_sa[(state, a)]
                best_a = a
        return best_a

    def extract_policy(self):
        policy = {}
        for row in range(self.env.num_rows):
            for col in range(self.env.num_cols):
                state = (row, col)
                action = self.mcts_select_action(state)
                policy[state] = action
        return policy

    def __str__(self):
        return str(self.q_sa) + ':' + str(self.n_s) + ':' + str(self.n_sa)

    def __repr__(self):
        return str(self)
