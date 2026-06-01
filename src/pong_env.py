"""Gymnasium environment wrapper around GameEngine.

This is the translator that sits between the raw Pong game and any learning
agent. The game thinks in pixels; a learning algorithm wants tidy numbers, a
single reward, and a signal for when a match is over. This wrapper provides
exactly that through the standard `gymnasium.Env` interface, so the same agent
code could later be pointed at a different game without changes.

The agent never touches the game directly. It calls `reset` to start a match
and `step` to make a move, and in return it gets an observation (what the court
looks like right now), a reward (how good the last move turned out), and a flag
for whether the match has ended.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from game_engine import GameEngine


# The agent has exactly three choices each tick. Naming them keeps the rest of
# the file readable instead of sprinkling bare 0/1/2 around.
ACTION_COUNT = 3

# Length of the observation vector the agent sees (see `_observe`).
OBSERVATION_SIZE = 6


class PongEnv(gym.Env):
    """Wrap a `GameEngine` in the gymnasium API the learning agents expect."""

    def __init__(self, game_engine: GameEngine, reward_config) -> None:
        """Store the game and the reward rulebook, and declare the spaces.

        Args:
            game_engine: The pure-logic Pong game this env drives. Its `step`
                return value is what `step` below turns into a reward.
            reward_config: The `Reward` settings namespace (points and the hit
                bonus). Read by `_reward_for` on every step.
        """
        super().__init__()
        self._game = game_engine
        self._reward = reward_config

        # The action space is the three moves; the observation space is six
        # numbers already scaled into [-1, 1] by `_observe`, which keeps the
        # learning math well-behaved.
        self.action_space = spaces.Discrete(ACTION_COUNT)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(OBSERVATION_SIZE,), dtype=np.float32
        )

    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict]:
        """Start a fresh match and hand back the opening observation.

        Args:
            seed: Optional RNG seed, forwarded to gymnasium's bookkeeping.
            options: Unused; present to match the gymnasium signature.

        Returns:
            A pair (observation, info). The observation feeds straight into the
            agent's `select_action`; info is an empty dict here.
        """
        super().reset(seed=seed)
        self._game.reset()
        return self._observe(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Apply one move and report what came of it.

        Args:
            action: The agent's choice from `select_action`: 0 up, 1 stay,
                2 down.

        Returns:
            The standard gymnasium five-tuple (observation, reward, terminated,
            truncated, info). The observation and reward are what the agent's
            `update`/`learn` step trains on; terminated says the match was won.
            `truncated` is always False here because the cap on match length is
            the Trainer's job, not the env's.
        """
        result = self._game.step(action)
        reward = self._reward_for(result)
        observation = self._observe()
        info = {"scored": result.scored, "agent_hit": result.agent_hit}
        return observation, reward, result.done, False, info

    def _reward_for(self, result) -> float:
        """Turn the engine's step outcome into a single number to learn from.

        Args:
            result: The `StepResult` from `GameEngine.step` (who scored, whether
                the agent's paddle hit the ball, whether the match ended).

        Returns:
            The reward for this step, summed from the configured pieces. Used by
            `step` and, through it, by the agent's learning update.
        """
        reward = self._reward.PER_STEP

        # Winning a point is good; losing one is bad. These are the signals that
        # ultimately teach the agent to actually win matches.
        if result.scored == "agent":
            reward += self._reward.SCORE_POINT
        elif result.scored == "opponent":
            reward += self._reward.CONCEDE_POINT

        # A small bonus just for returning the ball. Scoring is rare early on,
        # so this nudges the agent toward the paddle in the meantime.
        if result.agent_hit and self._reward.USE_HIT_SHAPING:
            reward += self._reward.PADDLE_HIT

        return reward

    def _observe(self) -> np.ndarray:
        """Snapshot the court as six numbers the agent can reason about.

        The six are ball x, ball y, ball horizontal direction, ball vertical
        direction, the agent's paddle height, and the opponent's paddle height.
        Each is scaled to roughly [-1, 1] so no single number dominates the
        learning simply because it happens to use larger units.

        Returns:
            A float32 array of length six. Handed to the agent's
            `select_action`, and stored as the "before" and "after" snapshots
            its learning update compares.
        """
        game = self._game
        ball_x, ball_y = game.ball_position
        ball_velocity_x, ball_velocity_y = game.ball_velocity

        # Positions: divide by the screen size to land in [0, 1], then shift to
        # [-1, 1] so the middle of the court reads as 0.
        normalized_ball_x = ball_x / game.screen_width * 2 - 1
        normalized_ball_y = ball_y / game.screen_height * 2 - 1
        normalized_agent_paddle = game.agent_paddle_y / game.screen_height * 2 - 1
        normalized_opponent_paddle = (
            game.opponent_paddle_y / game.screen_height * 2 - 1
        )

        # Velocities only ever swing between plus and minus their top speed, so
        # dividing by that speed gives a clean -1 (one way) or +1 (the other).
        normalized_velocity_x = ball_velocity_x / game.ball_speed_x
        normalized_velocity_y = ball_velocity_y / game.ball_speed_y

        return np.array(
            [
                normalized_ball_x,
                normalized_ball_y,
                normalized_velocity_x,
                normalized_velocity_y,
                normalized_agent_paddle,
                normalized_opponent_paddle,
            ],
            dtype=np.float32,
        )

    def render(self) -> None:
        """Draw the current frame with pygame, for watching a trained agent.

        Rendering is deliberately optional and imported lazily so that headless
        training never pays for pygame. Paddles and ball are plain rectangles;
        there are no sprites, sound, or effects.
        """
        import pygame

        from settings import Colors

        game = self._game

        # First call sets up the window and clock; later calls reuse them.
        if not hasattr(self, "_screen"):
            pygame.init()
            self._screen = pygame.display.set_mode(
                (game.screen_width, game.screen_height)
            )
            pygame.display.set_caption("Pong (trained agent)")
            self._clock = pygame.time.Clock()
            self._font = pygame.font.SysFont(None, 64)

        # Draining the event queue keeps the OS from treating the window as
        # frozen while the agent plays.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        self._screen.fill(Colors.BACKGROUND)
        half_paddle = game.paddle_height / 2

        agent_rect = pygame.Rect(
            game.agent_paddle_x,
            game.agent_paddle_y - half_paddle,
            game.paddle_width,
            game.paddle_height,
        )
        opponent_rect = pygame.Rect(
            game.opponent_paddle_x,
            game.opponent_paddle_y - half_paddle,
            game.paddle_width,
            game.paddle_height,
        )
        pygame.draw.rect(self._screen, Colors.PADDLE, agent_rect)
        pygame.draw.rect(self._screen, Colors.PADDLE, opponent_rect)

        ball_x, ball_y = game.ball_position
        half_ball = game.ball_size / 2
        ball_rect = pygame.Rect(
            ball_x - half_ball, ball_y - half_ball, game.ball_size, game.ball_size
        )
        pygame.draw.rect(self._screen, Colors.BALL, ball_rect)

        pygame.display.flip()
        self._clock.tick(60)
