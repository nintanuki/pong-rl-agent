"""Deep Q-Network agent.

A small PyTorch MLP that approximates the Q-function instead of storing it
in a table. Uses experience replay and a target network for stability, both
standard tricks from the original DQN paper. Reads its hyperparameters from
`Settings.DQN`.
"""


class DQNAgent:
    """Deep Q-learning over the same 11-boolean state vector as the tabular agent."""

    def __init__(
        self,
        state_size: int,
        action_count: int,
        hidden_layer_sizes: tuple[int, ...],
        learning_rate: float,
        discount_factor: float,
        epsilon_start: float,
        epsilon_end: float,
        epsilon_decay: float,
        batch_size: int,
        replay_buffer_size: int,
        target_update_steps: int,
    ) -> None:
        """Configure the agent, build the online and target networks.

        Args:
            state_size: Length of the observation vector.
            action_count: Size of the discrete action space.
            hidden_layer_sizes: Sizes of the MLP hidden layers, in order.
            learning_rate: Optimizer step size.
            discount_factor: Gamma.
            epsilon_start: Initial exploration probability.
            epsilon_end: Floor exploration probability.
            epsilon_decay: Multiplicative decay applied per episode.
            batch_size: Number of transitions sampled per learning step.
            replay_buffer_size: Maximum number of transitions retained.
            target_update_steps: How many environment steps between target-net copies.
        """
        # TODO: store config; build torch.nn.Sequential online + target networks
        # TODO: optimizer = torch.optim.Adam(online.parameters(), lr=learning_rate)
        # TODO: replay buffer = collections.deque(maxlen=replay_buffer_size)
        raise NotImplementedError

    def select_action(self, state) -> int:
        """Pick an action using the epsilon-greedy policy.

        Args:
            state: A length-`state_size` observation vector.

        Returns:
            An integer action in [0, action_count).
        """
        # TODO: explore with prob epsilon; otherwise argmax(online(state))
        raise NotImplementedError

    def remember(self, state, action: int, reward: float, next_state, terminated: bool) -> None:
        """Append one transition to the replay buffer.

        Args:
            state: Observation before the action.
            action: Action taken.
            reward: Reward received.
            next_state: Observation after the action.
            terminated: True if the episode ended on this step.
        """
        # TODO: replay.append((state, action, reward, next_state, terminated))
        raise NotImplementedError

    def learn(self) -> None:
        """Sample a batch and apply one DQN update."""
        # TODO: skip if len(replay) < batch_size
        # TODO: compute targets via target network; MSE loss against online network
        # TODO: optimizer.zero_grad(); loss.backward(); optimizer.step()
        # TODO: every `target_update_steps`, copy online weights into target
        raise NotImplementedError

    def decay_epsilon(self) -> None:
        """Reduce epsilon by one step of the decay schedule."""
        # TODO: epsilon = max(epsilon_end, epsilon * epsilon_decay)
        raise NotImplementedError

    def save(self, path) -> None:
        """Persist the online network's state dict.

        Args:
            path: Filesystem path to write the serialized weights to.
        """
        # TODO: torch.save(self.online.state_dict(), path)
        raise NotImplementedError

    def load(self, path) -> None:
        """Load a previously persisted state dict into the online network.

        Args:
            path: Filesystem path to the serialized weights.
        """
        # TODO: self.online.load_state_dict(torch.load(path))
        raise NotImplementedError
