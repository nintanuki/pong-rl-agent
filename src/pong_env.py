"""Gymnasium environment wrapper around GameEngine.

Exposes the standard `gymnasium.Env` API so any RL library can train against
this game without knowing anything about Pygame or the internal state machine.
The observation is a 6-value continuous vector (ball x, ball y, ball vx, ball
vy, agent paddle y, opponent paddle y), normalized to roughly [-1, 1]; the
action space is Discrete(3): 0 = up, 1 = stay, 2 = down. The DQN agent consumes
the raw vector while the tabular agent discretizes it into bins.
"""


class PongEnv:
    """Gymnasium-compatible environment that wraps the GameEngine."""

    def __init__(self, game_engine, reward_config) -> None:
        """Store collaborators and declare spaces.

        Args:
            game_engine: A GameEngine instance.
            reward_config: A namespace exposing Reward.* values.
        """
        # TODO: store; build observation_space (Box, shape (6,)) and
        # action_space (Discrete(3)) from gymnasium.spaces
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
            action: One of 0 (up), 1 (stay), 2 (down).

        Returns:
            A tuple (observation, reward, terminated, truncated, info).
        """
        # TODO: result = engine.step(action); map result.scored / result.agent_hit
        # to a reward via reward_config; terminated = result.done; build tuple
        raise NotImplementedError

    def render(self) -> None:
        """Render the current frame using Pygame. Optional for training."""
        # TODO: defer Pygame imports to here; draw paddles, ball, center line, score
        raise NotImplementedError

    def _observe(self):
        """Build the normalized 6-value observation vector from engine state.

        Returns:
            A NumPy array of dtype float32 with shape (6,): ball x, ball y,
            ball vx, ball vy, agent paddle y, opponent paddle y.
        """
        # TODO: pull ball_position / ball_velocity / paddle ys from the engine
        # and normalize each component against the screen / speed extents
        raise NotImplementedError
