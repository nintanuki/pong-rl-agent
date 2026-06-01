"""Gymnasium environment wrapper around GameEngine.

Exposes the standard `gymnasium.Env` API so any RL library can train against
this game without knowing anything about Pygame or the internal state
machine. Observation is the 11-boolean state vector commonly used for
tabular Q-learning on Snake; action space is Discrete(4).
"""


class SnakeEnv:
    """Gymnasium-compatible environment that wraps the GameEngine."""

    def __init__(self, game_engine, reward_config) -> None:
        """Store collaborators and declare spaces.

        Args:
            game_engine: A GameEngine instance.
            reward_config: A namespace exposing Reward.* values.
        """
        # TODO: store; build observation_space and action_space from gymnasium.spaces
        raise NotImplementedError

    def reset(self, seed=None, options=None) -> tuple:
        """Reset the environment to a starting state.

        Args:
            seed: Optional RNG seed.
            options: Optional gymnasium options dict (unused).

        Returns:
            A tuple (observation, info_dict).
        """
        # TODO: engine.reset(); return self._observe(), {}
        raise NotImplementedError

    def step(self, action: int) -> tuple:
        """Apply one action and return the gymnasium step tuple.

        Args:
            action: One of 0, 1, 2, 3.

        Returns:
            A tuple (observation, reward, terminated, truncated, info).
        """
        # TODO: engine.step(action); compute reward via reward_config; build tuple
        raise NotImplementedError

    def render(self) -> None:
        """Render the current frame using Pygame. Optional for training."""
        # TODO: defer Pygame imports to here; draw snake, food, score
        raise NotImplementedError

    def _observe(self):
        """Build the 11-boolean state vector from the engine state.

        Returns:
            A NumPy array of dtype int8 with shape (11,).
        """
        # TODO: danger straight/left/right, direction one-hot, food direction
        raise NotImplementedError
