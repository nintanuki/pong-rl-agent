"""Tabular Q-learning agent.

This is the primary agent. The idea behind Q-learning is simple to say: for
every situation the agent can be in, keep a running estimate of how good each
possible move is, and keep nudging those estimates toward what actually
happened. The "Q-table" is just that big lookup of situation to move-scores.

Because a lookup table needs exact, repeatable keys, the agent first rounds the
environment's continuous observation into a discrete bucketed state (see
`_discretize`). Everything else here is the textbook Q-learning loop: pick a
move, watch the result, and update the estimate.

Pure NumPy and the standard library; no PyTorch needed.
"""

import pickle
from collections import defaultdict

import numpy as np

from settings import Discretization


class QLearningAgent:
    """Epsilon-greedy tabular Q-learning over a bucketed observation."""

    def __init__(
        self,
        action_count: int,
        learning_rate: float,
        discount_factor: float,
        epsilon_start: float,
        epsilon_end: float,
        epsilon_decay: float,
    ) -> None:
        """Configure the learning knobs and start with an empty table.

        Args:
            action_count: How many moves exist (three for Pong).
            learning_rate: How hard each new experience pulls the old estimate.
                Bigger learns faster but is jumpier.
            discount_factor: How much the agent cares about future reward versus
                reward right now. Closer to 1 is more far-sighted.
            epsilon_start: Starting chance of ignoring the table and trying a
                random move, so the agent explores before it trusts itself.
            epsilon_end: The floor that exploration chance decays toward.
            epsilon_decay: The multiplier applied to epsilon each episode.
        """
        self.action_count = action_count
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Unseen states start with all-zero move-scores. defaultdict means we
        # never have to check "have I been here before?" before reading.
        self.q_table: dict[tuple, np.ndarray] = defaultdict(
            lambda: np.zeros(action_count, dtype=np.float64)
        )

    def select_action(self, observation: np.ndarray) -> int:
        """Choose a move: usually the best one known, occasionally a random one.

        Args:
            observation: The env's six-number snapshot, fresh from `reset` or
                `step`.

        Returns:
            An action in [0, action_count). Passed straight to `env.step`, whose
            result later comes back to `update`.
        """
        # The occasional random move is exploration: it's how the agent finds
        # tactics its current table would never point it toward.
        if np.random.random() < self.epsilon:
            return int(np.random.randint(self.action_count))

        state = self._discretize(observation)
        return int(np.argmax(self.q_table[state]))

    def update(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        terminated: bool,
    ) -> None:
        """Nudge one move-score toward the reward we just saw.

        This is the heart of Q-learning. The "target" is what the score should
        have been: the reward we got, plus how good the next state looks (unless
        the match just ended). We move the stored estimate a fraction of the way
        toward that target, that fraction being the learning rate.

        Args:
            observation: The snapshot before the move.
            action: The move taken.
            reward: What `env.step` paid out for it.
            next_observation: The snapshot after the move.
            terminated: True if the match ended on this step, in which case there
                is no "next state" worth looking ahead to.
        """
        state = self._discretize(observation)
        next_state = self._discretize(next_observation)

        best_next_value = 0.0 if terminated else float(np.max(self.q_table[next_state]))
        target = reward + self.discount_factor * best_next_value

        current_estimate = self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * (target - current_estimate)

    def decay_epsilon(self) -> None:
        """Take one step of the exploration schedule, never below the floor.

        Called once per episode by the Trainer, so the agent gradually shifts
        from trying things out to trusting what it has learned.
        """
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def _discretize(self, observation: np.ndarray) -> tuple:
        """Round the continuous snapshot into an exact, repeatable table key.

        Positions get bucketed into a handful of bands; the two direction
        numbers are already just sign (left/right, up/down) so they pass through
        as -1 or 1.

        Args:
            observation: The env's six-number snapshot, each value in [-1, 1].

        Returns:
            A small tuple used as the lookup key in `q_table` by both
            `select_action` and `update`.
        """
        ball_x, ball_y, velocity_x, velocity_y, agent_paddle_y, _opponent = observation

        return (
            self._bucket(ball_x, Discretization.BALL_X_BINS),
            self._bucket(ball_y, Discretization.BALL_Y_BINS),
            int(np.sign(velocity_x)),
            int(np.sign(velocity_y)),
            self._bucket(agent_paddle_y, Discretization.AGENT_PADDLE_Y_BINS),
        )

    def _bucket(self, value: float, bin_count: int) -> int:
        """Map one [-1, 1] number to a band index in [0, bin_count).

        Args:
            value: A single normalized observation component.
            bin_count: How many bands to split the range into.

        Returns:
            The band index this value falls in. Used to build the table key in
            `_discretize`.
        """
        # Shift [-1, 1] to [0, 1], scale up to the band count, and clamp so the
        # exact value 1.0 lands in the last band instead of one past the end.
        scaled = (value + 1) / 2 * bin_count
        return int(min(bin_count - 1, max(0, int(scaled))))

    def save(self, path) -> None:
        """Write the learned table to disk so it can be played back later.

        Args:
            path: Destination file. `load` reads the same format back.
        """
        # defaultdict doesn't pickle cleanly because of its factory, so store a
        # plain dict; load rebuilds the defaultdict around it.
        with open(path, "wb") as file:
            pickle.dump(dict(self.q_table), file)

    def load(self, path) -> None:
        """Reload a table written by `save`, ready for evaluation.

        Args:
            path: A file previously written by `save`.
        """
        with open(path, "rb") as file:
            stored = pickle.load(file)
        self.q_table = defaultdict(
            lambda: np.zeros(self.action_count, dtype=np.float64), stored
        )
