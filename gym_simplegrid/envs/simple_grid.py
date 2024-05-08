from __future__ import annotations
import logging
import numpy as np
from gymnasium import spaces, Env
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys

MAPS = {
    "4x4": ["0000", "0101", "0001", "1000"],
    "8x8": [
        "00000000",
        "00000000",
        "00010000",
        "00000100",
        "00010000",
        "01100010",
        "01001010",
        "00010000",
    ],
}


N_QAGENT = 50  # iterations of the basic value iteration agent
N_IRLAGENT = 1  # number of iterations of IRLplus to perform
EXPLORE = 0.3  # the explore proportion: (1-EXPLORE) for exloit
MANUAL_FEEDBACK = 0.1  # reward feedback from human: + and -
NEUTRAL_FEEDBACK = 0.05  # if no feedback, this reward applied (+)
LOGGING = False  # set full logging to terminal or not...
# maze setup - leave alone for now
# BOARD_ROWS = 3
# BOARD_COLS = 4
# WIN_STATE = (0, 3)
# LOSE_STATE = (1, 3)
# START = (2, 0)          #third row, first column


class SimpleGridEnv(Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 7}
    FREE: int = 0
    OBSTACLE: int = 1
    MOVES: dict[int, tuple] = {
        0: (-1, 0),  # UP
        1: (1, 0),  # DOWN
        2: (0, -1),  # LEFT
        3: (0, 1),  # RIGHT
    }
    MOVES_TEXT = ["U", "D", "L", "R"]

    def __init__(
        self,
        obstacle_map: str | list[str],
        render_mode: str | None = None,
        human_feedback=1,
        retrospective_feedback=True,
        reward_type="q_learning",
        decision_map=[],
    ):
        """
        Initialise the environment.

        Parameters
        ----------
        agent_color: str
            Color of the agent. The available colors are: red, green, blue, purple, yellow, grey and black. Note that the goal cell will have the same color.
        obstacle_map: str | list[str]
            Map to be loaded. If a string is passed, the map is loaded from a set of pre-existing maps. The names of the available pre-existing maps are "4x4" and "8x8". If a list of strings is passed, the map provided by the user is parsed and loaded. The map must be a list of strings, where each string denotes a row of the grid and is a sequence of 0s and 1s, where 0 denotes a free cell and 1 denotes a wall cell.
            An example of a 4x4 map is the following:
            ["0000",
             "0101",
             "0001",
             "1000"]
        """

        # Env confinguration
        self.obstacles = self.parse_obstacle_map(obstacle_map)  # walls
        self.decision_map = decision_map
        self.nrow, self.ncol = self.obstacles.shape

        self.action_space = spaces.Discrete(len(self.MOVES))
        self.observation_space = spaces.Discrete(n=self.nrow * self.ncol)
        self.state_values = np.zeros((self.nrow, self.ncol))  # state values

        # Rendering configuration
        self.fig = None

        self.render_mode = render_mode
        self.fps = self.metadata["render_fps"]
        self.human_feedback = human_feedback
        self.reward_type = reward_type
        self.retrospective_feedback = retrospective_feedback
        self.lr = 0.2
        self.exp_rate = EXPLORE
        self.q_table = {}
        self.initialise_q_table()
        # initial Q values

        self.Q_values = {}
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                self.Q_values[(i, j)] = {}
                for a in self.actions:
                    self.Q_values[(i, j)][a] = 0  # Q value is a dict of dict

    def chooseAction(self):
        # choose action with most expected value
        mx_nxt_reward = 0
        action = ""

        if np.random.uniform(0, 1) <= self.exp_rate:
            action = np.random.choice(self.actions)
        else:
            # greedy action
            for a in self.actions:
                current_position = self.State.state
                nxt_reward = self.Q_values[current_position][a]
                if nxt_reward >= mx_nxt_reward:
                    action = a
                    mx_nxt_reward = nxt_reward
            # print("current pos: {}, greedy aciton: {}".format(self.State.state, action))
        return action

    def initialise_q_table(self):
        # For each position in the grid
        for i in range(self.nrow):
            for j in range(self.ncol):
                # For each possible move
                for move in self.MOVES:
                    # Initialize the Q-value for this state-action pair to 0
                    self.q_table[((i, j), move)] = 0

    # def initialise_state_values(self):
    #     # For each position in the grid
    #     for i in range(self.nrow):
    #         for j in range(self.ncol):
    #             # Initialize the state value to 0
    #             self.state_values[i, j] = 0

    def reset(self, seed: int | None = None, options: dict = dict()) -> tuple:
        """
        Reset the environment.

        Parameters
        ----------
        seed: int | None
            Random seed.
        options: dict
            Optional dict that allows you to define the start (`start_loc` key) and goal (`goal_loc`key) position when resetting the env. By default options={}, i.e. no preference is expressed for the start and goal states and they are randomly sampled.
        """

        # Set seed
        super().reset(seed=seed)

        # parse options
        self.start_xy = self.parse_state_option("start_loc", options)
        self.goal_xy = self.parse_state_option("goal_loc", options)

        # initialise internal vars
        self.agent_xy = self.start_xy
        self.reward = self.get_reward(*self.agent_xy)
        self.done = self.on_goal()
        self.n_iter = 0

        # Check integrity
        self.integrity_checks()

        # if self.render_mode == "human":
        self.render()

        return self.get_obs(), self.get_info()

    def chooseAction(self, rand=False, actAvoid=0):
        # choose action with most expected value,
        #  unless we want an explicitly different action...
        #  this can be changed significantly to improve the performance
        mx_nxt_reward = (
            -99999
        )  # needs to initially be large and negative, because of possible negative state values
        action = ""

        if rand == True:  # this part intentionally left uncommented: what does it do?
            flag = True
            while flag:
                action = np.random.choice(self.MOVES)
                if action == actAvoid:
                    flag = True
                else:
                    flag = False
            return action

        if np.random.uniform(0, 1) <= self.exp_rate:
            action = np.random.choice(self.actions)
        else:
            # greedy action
            for a in self.actions:
                nxt_reward = self.state_values[self.State.nxtPosition(a)]
                if nxt_reward >= mx_nxt_reward:
                    action = a
                    mx_nxt_reward = nxt_reward
        return action

    def get_action(self, action=None):
        """
        Take a step in the environment.
        """
        if action == None:
            action = self.action_space.sample()

        # f.write(
        #     f"{t},{info['agent_xy'][0]},{info['agent_xy'][1]},{rew},{done},{action}\n"
        # )

        # Get the current position of the agent
        row, col = self.agent_xy
        dx, dy = self.MOVES[action]
        target_row = row + dx
        target_col = col + dy

        return target_row, target_col, action

    def automated_future_response(self, move):
        pos_x = self.agent_xy[0]
        pos_y = self.agent_xy[1]

        if move in self.decision_map[pos_x][pos_y]:
            return True
        else:
            return False

    def automated_response(self, action, position):
        if self.MOVES_TEXT[action] in self.decision_map[position[0]][position[1]]:
            return 1
        else:
            return 2

    def giveReward(self):
        if self.state == WIN_STATE:
            return 1
        elif self.state == LOSE_STATE:
            return -1
        else:
            return 0

    def step(self, action):
        target_row, target_col, action = self.get_action(action)
        u_reward = 0

        # if done - return
        self.step_count += 1

        if self.human_feedback == 1:
            print("Next action chosen: ", self.MOVES_TEXT[action])
            print("action", action)
            print("agent_xy", self.agent_xy)
            good_feedback = self.automated_response(action, self.agent_xy)
            # feedback = input("      *will* this action be g(ood) or b(ad): ")
            if good_feedback == 1:
                action = action
                # print("Good Action")
            else:
                action = self.action_space.sample()
                target_row, target_col, action = self.get_action()
                # print("Bad Action - new selected")

        prev_agent_xy = self.agent_xy
        prev_action = action

        # TODO: should i be setting q_learning here?
        # Check if the move is valid
        if self.is_in_bounds(target_row, target_col) and self.is_free(
            target_row, target_col
        ):
            self.agent_xy = (target_row, target_col)
            self.done = self.on_goal()
        self.n_iter += 1

        # if self.render_mode == "human":
        self.render()

        if self.retrospective_feedback == 1:
            ret_response = self.automated_response(prev_action, prev_agent_xy)
            if ret_response == 1:
                # feedback = input("      /was/ this action g(ood) or b(ad): ")
                # if feedback == "g":
                u_reward = MANUAL_FEEDBACK
            elif ret_response == 2:
                u_reward = -MANUAL_FEEDBACK
            else:
                u_reward = NEUTRAL_FEEDBACK

        # Compute the reward
        block_reward = self.get_reward(*self.agent_xy)
        reward_value = self.state_values[target_row][target_col] + u_reward
        discount = 0.9
        max_reward

        if self.State.isEnd:
            # back propagate
            reward = self.State.giveReward()
            for a in self.actions:
                self.Q_values[self.State.state][a] = reward
            print("Game End Reward", reward)
            for s in reversed(self.states):
                current_q_value = self.Q_values[s[0]][s[1]]
                reward = current_q_value + self.lr * (
                    self.decay_gamma * reward - current_q_value
                )
                self.Q_values[s[0]][s[1]] = round(reward, 3)
            self.reset()
            i += 1

        if self.reward_type == "q_learning":
            self.reward += self.lr * (
                (u_reward + block_reward)
                # + discount*()
                - self.state_values[target_row][target_col]
            )
            self.state_values[target_row][target_col] = round(reward, 3)
        else:
            self.reward = reward_value

        return self.get_obs(), self.reward, self.done, False, self.get_info()

    def parse_obstacle_map(self, obstacle_map) -> np.ndarray:
        """
        Initialise the grid.

        The grid is described by a map, i.e. a list of strings where each string denotes a row of the grid and is a sequence of 0s and 1s, where 0 denotes a free cell and 1 denotes a wall cell.

        The grid can be initialised by passing a map name or a custom map.
        If a map name is passed, the map is loaded from a set of pre-existing maps. If a custom map is passed, the map provided by the user is parsed and loaded.

        Examples
        --------
        >>> my_map = ["001", "010", "011]
        >>> SimpleGridEnv.parse_obstacle_map(my_map)
        array([[0, 0, 1],
               [0, 1, 0],
               [0, 1, 1]])
        """
        if isinstance(obstacle_map, list):
            map_str = np.asarray(obstacle_map, dtype="c")
            map_int = np.asarray(map_str, dtype=int)
            return map_int
        elif isinstance(obstacle_map, str):
            map_str = MAPS[obstacle_map]
            map_str = np.asarray(map_str, dtype="c")
            map_int = np.asarray(map_str, dtype=int)
            return map_int
        else:
            raise ValueError(
                f"You must provide either a map of obstacles or the name of an existing map. Available existing maps are {', '.join(MAPS.keys())}."
            )

    def parse_state_option(self, state_name: str, options: dict) -> tuple:
        """
        parse the value of an option of type state from the dictionary of options usually passed to the reset method. Such value denotes a position on the map and it must be an int or a tuple.
        """
        try:
            state = options[state_name]
            if isinstance(state, int):
                return self.to_xy(state)
            elif isinstance(state, tuple):
                return state
            else:
                raise TypeError(f"Allowed types for `{state_name}` are int or tuple.")
        except KeyError:
            state = self.sample_valid_state_xy()
            logger = logging.getLogger()
            logger.info(
                f"Key `{state_name}` not found in `options`. Random sampling a valid value for it:"
            )
            logger.info(f"...`{state_name}` has value: {state}")
            return state

    def sample_valid_state_xy(self) -> tuple:
        state = self.observation_space.sample()
        pos_xy = self.to_xy(state)
        while not self.is_free(*pos_xy):
            state = self.observation_space.sample()
            pos_xy = self.to_xy(state)
        return pos_xy

    def integrity_checks(self) -> None:
        # check that goals do not overlap with walls
        assert (
            self.obstacles[self.start_xy] == self.FREE
        ), f"Start position {self.start_xy} overlaps with a wall."
        assert (
            self.obstacles[self.goal_xy] == self.FREE
        ), f"Goal position {self.goal_xy} overlaps with a wall."
        assert self.is_in_bounds(
            *self.start_xy
        ), f"Start position {self.start_xy} is out of bounds."
        assert self.is_in_bounds(
            *self.goal_xy
        ), f"Goal position {self.goal_xy} is out of bounds."

    def to_s(self, row: int, col: int) -> int:
        """
        Transform a (row, col) point to a state in the observation space.
        """
        return row * self.ncol + col

    def to_xy(self, s: int) -> tuple[int, int]:
        """
        Transform a state in the observation space to a (row, col) point.
        """
        return (s // self.ncol, s % self.ncol)

    def on_goal(self) -> bool:
        """
        Check if the agent is on its own goal.
        """
        return self.agent_xy == self.goal_xy

    def is_free(self, row: int, col: int) -> bool:
        """
        Check if a cell is free.
        """
        return self.obstacles[row, col] == self.FREE

    def is_in_bounds(self, row: int, col: int) -> bool:
        """
        Check if a target cell is in the grid bounds.
        """
        return 0 <= row < self.nrow and 0 <= col < self.ncol

    def get_reward(self, x: int, y: int) -> float:
        """
        Get the reward of a given cell.
        """
        if not self.is_in_bounds(x, y):
            return -1.0
        elif not self.is_free(x, y):
            return -1.0
        elif (x, y) == self.goal_xy:
            return 1.0
        else:
            return 0.0

    def get_obs(self) -> int:
        return self.to_s(*self.agent_xy)

    def get_info(self) -> dict:
        return {
            "agent_xy": self.agent_xy,
            "n_iter": self.n_iter,
        }

    def render(self):
        """
        Render the environment.
        """
        if self.render_mode == "human":
            if self.fig is None:
                self.render_initial_frame()
                self.fig.canvas.mpl_connect("close_event", self.close)
            else:
                self.update_agent_patch()

            self.ax.set_title(f"Step: {self.n_iter}, Reward: {self.reward}")
            # self.fig.canvas.draw()
            # self.fig.canvas.flush_events()
            plt.pause(1 / self.fps)
            # plt.show(block=False)
            return None

        elif self.render_mode == "none":
            return None

        else:
            raise ValueError(f"Unsupported rendering mode {self.render_mode}")

    def create_agent_patch(self):
        """
        Create a Circle patch for the agent.

        @NOTE: If agent position is (x,y) then, to properly render it, we have to pass (y,x) as center to the Circle patch.
        """
        return mpl.patches.Circle(
            (self.agent_xy[1] + 0.5, self.agent_xy[0] + 0.5),
            0.3,
            facecolor="orange",
            fill=True,
            edgecolor="black",
            linewidth=1.5,
            zorder=100,
        )

    def update_agent_patch(self):
        """
        @NOTE: If agent position is (x,y) then, to properly
        render it, we have to pass (y,x) as center to the Circle patch.
        """
        self.agent_patch.center = (self.agent_xy[1] + 0.5, self.agent_xy[0] + 0.5)
        return None

    def render_initial_frame(self):
        """
        Render the initial frame.

        @NOTE: 0: free cell (white), 1: obstacle (black), 2: start (red), 3: goal (green)
        """
        data = self.obstacles.copy()
        data[self.start_xy] = 2
        data[self.goal_xy] = 3

        colors = ["white", "black", "red", "green"]
        bounds = [i - 0.1 for i in [0, 1, 2, 3, 4]]

        # create discrete colormap
        cmap = mpl.colors.ListedColormap(colors)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

        plt.ion()
        fig, ax = plt.subplots()
        self.fig = fig
        self.ax = ax

        # ax.grid(axis='both', color='#D3D3D3', linewidth=2)
        ax.grid(axis="both", color="k", linewidth=1.3)
        ax.set_xticks(np.arange(0, data.shape[1], 1))  # correct grid sizes
        ax.set_yticks(np.arange(0, data.shape[0], 1))
        ax.tick_params(
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False,
        )

        # draw the grid
        ax.imshow(
            data,
            cmap=cmap,
            norm=norm,
            extent=[0, data.shape[1], data.shape[0], 0],
            interpolation="none",
        )

        # Create white holes on start and goal positions
        for pos in [self.start_xy, self.goal_xy]:
            wp = self.create_white_patch(*pos)
            ax.add_patch(wp)

        # Create agent patch in start position
        self.agent_patch = self.create_agent_patch()
        ax.add_patch(self.agent_patch)

        return None

    def create_white_patch(self, x, y):
        """
        Render a white patch in the given position.
        """
        return mpl.patches.Circle(
            (y + 0.5, x + 0.5),
            0.4,
            color="white",
            fill=True,
            zorder=99,
        )

    def close(self, *args):
        """
        Close the environment.
        """
        plt.close(self.fig)
        sys.exit()


# calculate action
#
# mean reward
#
# number of steps
