"""Tabular Q-Learning agent.

Stores a Q-table keyed by the discretized state vector, picks actions with
an epsilon-greedy policy, and updates entries with the standard one-step
Bellman update. Pure NumPy / Python; no PyTorch dependency.
"""


class QLearningAgent:
    """Epsilon-greedy tabular Q-learning."""

    def __init__(
        self,
        action_count: int,
        learning_rate: float,
        discount_factor: float,
        epsilon_start: float,
        epsilon_end: float,
        epsilon_decay: float,
    ) -> None:
        """Configure hyperparameters and initialise the Q-table.

        Args:
            action_count: Size of the discrete action space.
            learning_rate: Alpha. Step size for the Bellman update.
            discount_factor: Gamma. Future-reward weighting.
            epsilon_start: Initial exploration probability.
            epsilon_end: Floor exploration probability.
            epsilon_decay: Multiplicative decay applied per episode.
        """
        # TODO: store; build Q-table as defaultdict(lambda: np.zeros(action_count))
        raise NotImplementedError

    def select_action(self, state) -> int:
        """Pick an action using the epsilon-greedy policy.

        Args:
            state: A hashable representation of the current observation.

        Returns:
            An integer action in [0, action_count).
        """
        # TODO: with prob epsilon explore, otherwise argmax(Q[state])
        raise NotImplementedError

    def update(self, state, action: int, reward: float, next_state, terminated: bool) -> None:
        """Apply one Bellman update to the Q-table.

        Args:
            state: Hashable observation before the action.
            action: Action taken.
            reward: Reward received.
            next_state: Hashable observation after the action.
            terminated: True if the episode ended on this step.
        """
        # TODO: target = reward + gamma * max(Q[next_state]) * (not terminated)
        # TODO: Q[state][action] += alpha * (target - Q[state][action])
        raise NotImplementedError

    def decay_epsilon(self) -> None:
        """Reduce epsilon by one step of the decay schedule."""
        # TODO: epsilon = max(epsilon_end, epsilon * epsilon_decay)
        raise NotImplementedError

    def save(self, path) -> None:
        """Persist the Q-table to disk.

        Args:
            path: Filesystem path to write the serialized table to.
        """
        # TODO: pickle the dict of state -> action-value array
        raise NotImplementedError

    def load(self, path) -> None:
        """Load a previously persisted Q-table from disk.

        Args:
            path: Filesystem path to the serialized table.
        """
        # TODO: unpickle into self.q_table
        raise NotImplementedError
