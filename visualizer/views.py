import tkinter as tk

from envs.GridWorld import UP, DOWN, LEFT, RIGHT

VALUE_FONT = ('Arial', 24)
POLICY_FONT = ('Arial', 16)

POLICY_CHARS = {
    UP: '↑',
    DOWN: '↓',
    LEFT: '←',
    RIGHT: '→'
}


class GridWorldView(tk.Canvas):
    BOARD_WIDTH = 1000
    BOARD_HEIGHT = 750
    POLICY_CHARS = {
        UP: '↑',
        DOWN: '↓',
        LEFT: '←',
        RIGHT: '→'
    }

    def __init__(self, master, num_rows, num_cols):
        super().__init__(master, width=self.BOARD_WIDTH, height=self.BOARD_HEIGHT)
        self._master = master
        self.num_rows = num_rows
        self.num_cols = num_cols

    def _get_cell_size(self) -> tuple[int, int]:
        return self.BOARD_WIDTH // self.num_cols, self.BOARD_HEIGHT // self.num_rows

    def _get_bbox(self, cell: tuple[int, int]) -> tuple[int, int, int, int]:
        row, col = cell
        cell_width, cell_height = self._get_cell_size()
        x_min, y_min = col * cell_width, row * cell_height
        x_max, y_max = x_min + cell_width, y_min + cell_height
        return x_min, y_min, x_max, y_max

    def _get_midpoint(self, cell: tuple[int, int]) -> tuple[int, int]:
        x_min, y_min, x_max, y_max = self._get_bbox(cell)
        return (x_min + x_max) // 2, (y_min + y_max) // 2

    def redraw(self, rewards, q_sa, n_sa, n_s, policy) -> None:
        self.clear()
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                state = (row, col)
                self.draw_cell(state, rewards.get(state, 0), q_sa, n_sa, n_s, policy)

    def draw_cell(self, state, reward, q_sa, n_sa, n_s, policy):
        if n_s.get(state) is None:
            return

        colour = 'light grey'
        if reward:
            colour = 'light green' if reward > 0 else 'orange'
        self.create_rectangle(self._get_bbox(state), fill=colour)

        if reward:
            self.create_text(self._get_midpoint(state), text=str(reward), font=VALUE_FONT)
            return  # In grid world all rewards are terminal states so don't need more info drawn

        # Draw lines to segment cell into 4 sections
        x_min, y_min, x_max, y_max = self._get_bbox(state)
        self.create_line((x_min, y_min), (x_max, y_max))
        self.create_line((x_max, y_min), (x_min, y_max))

        # Draw (state, action) info: TODO refactor; this is pretty bad duplication
        x, y = self._get_midpoint(state)
        BUFFER = self._get_cell_size()[0] // 3

        up_q = q_sa.get((state, UP))
        if up_q:
            up_q = round(up_q, 2)
            self.create_text(x, y - BUFFER, text=f'Q: {up_q}', font=POLICY_FONT)

        down_q = q_sa.get((state, DOWN))
        if down_q:
            down_q = round(down_q, 2)
            self.create_text(x, y + BUFFER, text=f'Q: {down_q}', font=POLICY_FONT)

        right_q = q_sa.get((state, RIGHT))
        if right_q:
            right_q = round(right_q, 2)
            self.create_text(x + BUFFER, y, text=f'Q: {right_q}', font=POLICY_FONT)

        left_q = q_sa.get((state, LEFT))
        if left_q:
            left_q = round(left_q, 2)
            self.create_text(x - BUFFER, y, text=f'Q: {left_q}', font=POLICY_FONT)

        BUFFER2 = 15

        up_n = n_sa.get((state, UP))
        if up_n:
            self.create_text(x, y - BUFFER - BUFFER2, text=f'N: {up_n}', font=POLICY_FONT)

        down_n = n_sa.get((state, DOWN))
        if down_n:
            self.create_text(x, y + BUFFER - BUFFER2, text=f'N:{down_n}', font=POLICY_FONT)

        right_n = n_sa.get((state, RIGHT))
        if right_n:
            self.create_text(x + BUFFER, y - BUFFER2, text=f'N: {right_n}', font=POLICY_FONT)

        left_n = n_sa.get((state, LEFT))
        if left_n:
            self.create_text(x - BUFFER, y - BUFFER2, text=f'N:{left_n}', font=POLICY_FONT)

        # Draw N(s) and the current policy
        self.create_rectangle((x - 40, y - 20), (x + 40, y + 20), fill='pink')
        self.create_text(x - 5, y, text=str(n_s.get(state, '')) + POLICY_CHARS.get(policy.get(state), ''),
                         font=POLICY_FONT)

    def draw_current_cell(self, cell):
        # This method just puts a blue box around the cell the agent is in
        x_min, y_min, x_max, y_max = self._get_bbox(cell)
        self.create_line((x_min, y_min), (x_min, y_max), width=4, fill='blue')
        self.create_line((x_max, y_min), (x_max, y_max), width=4, fill='blue')
        self.create_line((x_min, y_min), (x_max, y_min), width=4, fill='blue')
        self.create_line((x_min, y_max), (x_max, y_max), width=4, fill='blue')

    def clear(self) -> None:
        """ Clears all items off this canvas. """
        self.delete(tk.ALL)


class MCTSView:
    def __init__(self, master, num_rows, num_cols, move_command, simulate_command):
        self.grid = GridWorldView(master, num_rows, num_cols)
        self.grid.pack(side=tk.LEFT)

        controls = tk.Frame(master)
        controls.pack(side=tk.LEFT)

        self._num_sims_label = tk.Label(controls, text='# Simulations')
        self._num_sims_label.pack()

        self._pos_label = tk.Label(controls, text='Current position:')
        self._pos_label.pack()

        self._current_move_label = tk.Label(controls, text='Current recommended move:')
        self._current_move_label.pack()

        btn_frame = tk.Frame(controls)
        btn_frame.pack()
        entry = tk.Entry(btn_frame)
        entry.pack(side=tk.LEFT)
        tk.Button(btn_frame, text='Simulate', command=lambda: simulate_command(int(entry.get()))).pack(side=tk.LEFT)

        move_frame = tk.Frame(controls)
        move_frame.pack()

        tk.Button(move_frame, text='UP', command=lambda: move_command(UP)).pack(side=tk.LEFT)
        tk.Button(move_frame, text='DOWN', command=lambda: move_command(DOWN)).pack(side=tk.LEFT)
        tk.Button(move_frame, text='LEFT', command=lambda: move_command(LEFT)).pack(side=tk.LEFT)
        tk.Button(move_frame, text='RIGHT', command=lambda: move_command(RIGHT)).pack(side=tk.LEFT)

    def redraw_grid(self, rewards, q_sa, n_sa, n_s, policy, current_state):
        self.grid.redraw(rewards, q_sa, n_sa, n_s, policy)
        self.grid.draw_current_cell(current_state)

    def update_labels(self, position, num_sims):
        self._pos_label.config(text=f'Current position: {position}')
        self._num_sims_label.config(text=f'# Simulations: {num_sims}')
