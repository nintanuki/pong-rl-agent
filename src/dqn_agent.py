"""Deep Q-Network (DQN) agent.

This is the comparison agent. It chases the exact same goal as the tabular
Q-learning agent, scoring each move in each situation, but instead of a lookup
table it uses a small neural network to estimate those scores. The upside is
that a network can generalize across similar situations it has never seen
exactly, so it doesn't need the observation rounded into buckets and can read
the six raw numbers directly.

Two well-known tricks from the original DQN paper keep the training steady:
  - Experience replay: remember past moves and learn from a random mix of them,
    rather than only the latest, which would be lopsided.
  - A target network: a slow-moving copy of the network used to set the
    learning goal, so the agent isn't chasing a target that shifts every step.

PyTorch is only imported here, so the tabular agent can run without it.
"""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn


class DQNAgent:
    """Deep Q-learning over the raw six-number observation."""

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
        """Build the two networks and the memory, and store the learning knobs.

        Args:
            state_size: Length of the observation vector (six for Pong).
            action_count: How many moves exist (three).
            hidden_layer_sizes: Width of each hidden layer of the network.
            learning_rate: Step size for the optimizer.
            discount_factor: How much future reward counts versus immediate.
            epsilon_start: Starting chance of a random (exploring) move.
            epsilon_end: Floor that the exploration chance decays toward.
            epsilon_decay: Per-episode multiplier on the exploration chance.
            batch_size: How many remembered moves each learning step samples.
            replay_buffer_size: How many past moves to keep around.
            target_update_steps: How often (in learning steps) to refresh the
                slow target network from the live one.
        """
        self.action_count = action_count
        self.discount_factor = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_steps = target_update_steps

        # The live network learns every step; the target network is its
        # occasional snapshot, used to set a stable learning goal.
        self.online_network = self._build_network(
            state_size, action_count, hidden_layer_sizes
        )
        self.target_network = self._build_network(
            state_size, action_count, hidden_layer_sizes
        )
        self.target_network.load_state_dict(self.online_network.state_dict())

        self.optimizer = torch.optim.Adam(
            self.online_network.parameters(), lr=learning_rate
        )
        self.loss_function = nn.MSELoss()

        # The replay memory of past moves, oldest dropped first once full.
        self.replay_buffer: deque = deque(maxlen=replay_buffer_size)
        self._learn_step_counter = 0

    def _build_network(
        self, state_size: int, action_count: int, hidden_layer_sizes: tuple[int, ...]
    ) -> nn.Module:
        """Assemble a plain feed-forward network: state in, move-scores out.

        Args:
            state_size: Number of inputs (the observation length).
            action_count: Number of outputs (one score per move).
            hidden_layer_sizes: Width of each hidden layer.

        Returns:
            The network module. Built twice in `__init__`, once for the live
            network and once for the target.
        """
        layers: list[nn.Module] = []
        input_size = state_size

        # Each hidden layer is a linear map followed by a ReLU, which is what
        # lets the network represent non-straight-line relationships.
        for layer_size in hidden_layer_sizes:
            layers.append(nn.Linear(input_size, layer_size))
            layers.append(nn.ReLU())
            input_size = layer_size

        # Final layer has one output per move and no activation: these are the
        # raw predicted scores.
        layers.append(nn.Linear(input_size, action_count))
        return nn.Sequential(*layers)

    def select_action(self, observation: np.ndarray) -> int:
        """Choose a move: usually the network's favorite, sometimes random.

        Args:
            observation: The env's six-number snapshot.

        Returns:
            An action in [0, action_count), passed to `env.step`.
        """
        if random.random() < self.epsilon:
            return random.randrange(self.action_count)

        # No gradient bookkeeping needed when we're only reading a prediction.
        with torch.no_grad():
            state_tensor = torch.as_tensor(observation, dtype=torch.float32)
            move_scores = self.online_network(state_tensor)
        return int(torch.argmax(move_scores).item())

    def remember(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        terminated: bool,
    ) -> None:
        """File one move away in replay memory for later learning.

        Args:
            observation: Snapshot before the move.
            action: Move taken.
            reward: Reward `env.step` paid for it.
            next_observation: Snapshot after the move.
            terminated: Whether the match ended on this step.
        """
        self.replay_buffer.append(
            (observation, action, reward, next_observation, terminated)
        )

    def learn(self) -> None:
        """Improve the network from one random batch of remembered moves.

        Does nothing until enough moves have piled up to fill a batch. Otherwise
        it nudges the network's predicted scores toward the reward-plus-lookahead
        target, and every so often refreshes the target network.
        """
        if len(self.replay_buffer) < self.batch_size:
            return

        # A random sample breaks up the strong correlation between back-to-back
        # moves, which makes the learning far more stable.
        batch = random.sample(self.replay_buffer, self.batch_size)
        observations, actions, rewards, next_observations, terminateds = zip(*batch)

        observations = torch.as_tensor(np.array(observations), dtype=torch.float32)
        actions = torch.as_tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards = torch.as_tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_observations = torch.as_tensor(
            np.array(next_observations), dtype=torch.float32
        )
        terminateds = torch.as_tensor(terminateds, dtype=torch.float32).unsqueeze(1)

        # What the network currently predicts for the moves we actually took.
        predicted_scores = self.online_network(observations).gather(1, actions)

        # The goal: the reward, plus the best score the target network sees in
        # the next state (zeroed out when the match ended there).
        with torch.no_grad():
            best_next_scores = self.target_network(next_observations).max(
                1, keepdim=True
            )[0]
            target_scores = rewards + self.discount_factor * best_next_scores * (
                1 - terminateds
            )

        loss = self.loss_function(predicted_scores, target_scores)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Periodically copy the live network into the slow target network.
        self._learn_step_counter += 1
        if self._learn_step_counter % self.target_update_steps == 0:
            self.target_network.load_state_dict(self.online_network.state_dict())

    def decay_epsilon(self) -> None:
        """Take one step of the exploration schedule, never below the floor."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, path) -> None:
        """Write the live network's weights to disk for later play.

        Args:
            path: Destination file. `load` reads the same format back.
        """
        torch.save(self.online_network.state_dict(), path)

    def load(self, path) -> None:
        """Reload weights written by `save` into both networks.

        Args:
            path: A file previously written by `save`.
        """
        weights = torch.load(path)
        self.online_network.load_state_dict(weights)
        self.target_network.load_state_dict(weights)
