"""Pong gameplay logic.

A self-contained, dependency-free implementation of Pong. Distilled from the
ClearCode Pong build by stripping out rendering, audio, music, controller
input, fullscreen handling, the pause overlay, the CRT effect, the inter-round
countdown, and all sprite assets, so the engine can run headless during
training. The only thing this module knows about is the state machine: two
paddles, a ball with velocity, wall and paddle collisions, and the score.

Coordinates are screen-space pixels where x grows rightward and y grows
downward. The agent controls the right-hand paddle; a simple scripted AI
controls the left one. The engine exposes only what the gymnasium env wrapper
and the renderer need to compute observations, rewards, and termination.

Lineage
-------
The original gameplay logic was learned from ClearCode's tutorial
"Learning pygame by making Pong" (https://www.youtube.com/watch?v=Qf3-aDXG8q4).
That code was extended with arcade-cabinet bells and whistles (joystick input,
pause overlay, music, a CRT overlay, custom sprites) for a separate project and
then distilled back down here for reinforcement learning, where rendering and
audio are noise.
"""

import random
from typing import NamedTuple, Optional


# Discrete actions for the agent paddle.
ACTION_UP: int = 0
ACTION_STAY: int = 1
ACTION_DOWN: int = 2


class StepResult(NamedTuple):
    """Outcome of advancing the engine by one tick.

    Attributes:
        scored: "agent" if the agent scored this tick, "opponent" if the agent
            conceded, or None if no point was scored.
        agent_hit: True if the agent's paddle returned the ball this tick.
        done: True if either side has reached the points-to-win threshold.
    """

    scored: Optional[str]
    agent_hit: bool
    done: bool


class GameEngine:
    """Pure-logic Pong game suitable for both headless and rendered use."""

    # ------------------------------------------------------------------
    # INIT
    # ------------------------------------------------------------------

    def __init__(
        self,
        screen_width: int,
        screen_height: int,
        paddle_width: int,
        paddle_height: int,
        paddle_margin: int,
        paddle_speed: int,
        opponent_speed: int,
        ball_size: int,
        ball_speed_x: int,
        ball_speed_y: int,
        points_to_win: int,
        seed: Optional[int] = None,
    ) -> None:
        """Configure the playfield, paddles, and ball.

        Args:
            screen_width: Playfield width in pixels.
            screen_height: Playfield height in pixels.
            paddle_width: Paddle width in pixels.
            paddle_height: Paddle height in pixels.
            paddle_margin: Gap in pixels between each paddle and its wall.
            paddle_speed: Per-tick travel of the agent paddle in pixels.
            opponent_speed: Per-tick travel of the scripted opponent in pixels.
            ball_size: Side length of the (square) ball in pixels.
            ball_speed_x: Horizontal velocity magnitude in pixels per tick.
            ball_speed_y: Vertical velocity magnitude in pixels per tick.
            points_to_win: Score at which an episode ends.
            seed: Optional RNG seed for reproducible serves.
        """
        self._screen_width = screen_width
        self._screen_height = screen_height
        self._paddle_width = paddle_width
        self._paddle_height = paddle_height
        self._paddle_margin = paddle_margin
        self._paddle_speed = paddle_speed
        self._opponent_speed = opponent_speed
        self._ball_size = ball_size
        self._ball_speed_x = ball_speed_x
        self._ball_speed_y = ball_speed_y
        self._points_to_win = points_to_win
        self._rng = random.Random(seed)

        # Left edge (x) of each paddle is fixed; only the y-center moves.
        self._opponent_paddle_x = paddle_margin
        self._agent_paddle_x = screen_width - paddle_margin - paddle_width

        # Mutable state, populated by reset().
        self._agent_paddle_y: float = 0.0
        self._opponent_paddle_y: float = 0.0
        self._ball_x: float = 0.0
        self._ball_y: float = 0.0
        self._ball_vx: float = 0.0
        self._ball_vy: float = 0.0
        self._agent_score: int = 0
        self._opponent_score: int = 0
        self.reset()

    # ------------------------------------------------------------------
    # RESET AND SERVE
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset paddles to center, zero the score, and serve a fresh ball."""
        center_y = self._screen_height / 2
        self._agent_paddle_y = center_y
        self._opponent_paddle_y = center_y
        self._agent_score = 0
        self._opponent_score = 0
        self._serve_ball()

    def _serve_ball(self) -> None:
        """Center the ball and randomize its direction (no inter-round delay)."""
        self._ball_x = self._screen_width / 2
        self._ball_y = self._screen_height / 2
        self._ball_vx = self._ball_speed_x * self._rng.choice((-1, 1))
        self._ball_vy = self._ball_speed_y * self._rng.choice((-1, 1))

    # ------------------------------------------------------------------
    # PER-STEP UPDATE
    # ------------------------------------------------------------------

    def step(self, action: int) -> StepResult:
        """Advance the game by one tick.

        Args:
            action: One of ACTION_UP, ACTION_STAY, ACTION_DOWN. Any other value
                is treated as ACTION_STAY.

        Returns:
            A StepResult describing scoring, paddle contact, and termination.
        """
        self._move_agent_paddle(action)
        self._move_opponent_paddle()

        self._ball_x += self._ball_vx
        self._ball_y += self._ball_vy

        self._bounce_off_walls()
        agent_hit = self._bounce_off_paddles()
        scored = self._check_scoring()

        done = (
            self._agent_score >= self._points_to_win
            or self._opponent_score >= self._points_to_win
        )
        return StepResult(scored=scored, agent_hit=agent_hit, done=done)

    def _move_agent_paddle(self, action: int) -> None:
        """Move the agent paddle per the action and clamp it on screen.

        Args:
            action: One of ACTION_UP, ACTION_STAY, ACTION_DOWN.
        """
        if action == ACTION_UP:
            self._agent_paddle_y -= self._paddle_speed
        elif action == ACTION_DOWN:
            self._agent_paddle_y += self._paddle_speed
        self._agent_paddle_y = self._clamp_paddle(self._agent_paddle_y)

    def _move_opponent_paddle(self) -> None:
        """Track the ball vertically, capped at the opponent's speed."""
        if self._opponent_paddle_y < self._ball_y:
            self._opponent_paddle_y += min(
                self._opponent_speed, self._ball_y - self._opponent_paddle_y
            )
        elif self._opponent_paddle_y > self._ball_y:
            self._opponent_paddle_y -= min(
                self._opponent_speed, self._opponent_paddle_y - self._ball_y
            )
        self._opponent_paddle_y = self._clamp_paddle(self._opponent_paddle_y)

    def _clamp_paddle(self, center_y: float) -> float:
        """Keep a paddle's center within the visible playfield.

        Args:
            center_y: Proposed paddle center y-coordinate.

        Returns:
            The center y-coordinate clamped so the paddle stays on screen.
        """
        half = self._paddle_height / 2
        return max(half, min(self._screen_height - half, center_y))

    # ------------------------------------------------------------------
    # COLLISIONS
    # ------------------------------------------------------------------

    def _bounce_off_walls(self) -> None:
        """Reflect the ball off the top and bottom walls."""
        half = self._ball_size / 2
        if self._ball_y - half <= 0:
            self._ball_y = half
            self._ball_vy = abs(self._ball_vy)
        elif self._ball_y + half >= self._screen_height:
            self._ball_y = self._screen_height - half
            self._ball_vy = -abs(self._ball_vy)

    def _bounce_off_paddles(self) -> bool:
        """Reflect the ball off either paddle when they overlap.

        Returns:
            True if the agent's paddle returned the ball this tick.
        """
        agent_hit = False

        # Agent paddle (right): only relevant while the ball travels rightward.
        if self._ball_vx > 0 and self._overlaps_paddle(
            self._agent_paddle_x, self._agent_paddle_y
        ):
            self._ball_x = self._agent_paddle_x - self._ball_size / 2
            self._ball_vx = -abs(self._ball_vx)
            agent_hit = True

        # Opponent paddle (left): only relevant while the ball travels leftward.
        elif self._ball_vx < 0 and self._overlaps_paddle(
            self._opponent_paddle_x, self._opponent_paddle_y
        ):
            self._ball_x = (
                self._opponent_paddle_x + self._paddle_width + self._ball_size / 2
            )
            self._ball_vx = abs(self._ball_vx)

        return agent_hit

    def _overlaps_paddle(self, paddle_left_x: float, paddle_center_y: float) -> bool:
        """Axis-aligned overlap test between the ball and a paddle.

        Args:
            paddle_left_x: Left edge x-coordinate of the paddle.
            paddle_center_y: Center y-coordinate of the paddle.

        Returns:
            True when the ball's bounding box intersects the paddle's.
        """
        half_ball = self._ball_size / 2
        half_paddle = self._paddle_height / 2
        ball_left = self._ball_x - half_ball
        ball_right = self._ball_x + half_ball
        ball_top = self._ball_y - half_ball
        ball_bottom = self._ball_y + half_ball

        paddle_right_x = paddle_left_x + self._paddle_width
        paddle_top = paddle_center_y - half_paddle
        paddle_bottom = paddle_center_y + half_paddle

        return (
            ball_right >= paddle_left_x
            and ball_left <= paddle_right_x
            and ball_bottom >= paddle_top
            and ball_top <= paddle_bottom
        )

    # ------------------------------------------------------------------
    # SCORING
    # ------------------------------------------------------------------

    def _check_scoring(self) -> Optional[str]:
        """Award a point and re-serve when the ball leaves through a goal.

        The agent defends the right wall and attacks the left, so a ball that
        passes the left edge scores for the agent and one that passes the right
        edge scores for the opponent.

        Returns:
            "agent", "opponent", or None.
        """
        half = self._ball_size / 2
        if self._ball_x - half <= 0:
            self._agent_score += 1
            self._serve_ball()
            return "agent"
        if self._ball_x + half >= self._screen_width:
            self._opponent_score += 1
            self._serve_ball()
            return "opponent"
        return None

    # ------------------------------------------------------------------
    # READ-ONLY STATE
    # ------------------------------------------------------------------

    @property
    def ball_position(self) -> tuple[float, float]:
        """Ball center as an (x, y) tuple."""
        return self._ball_x, self._ball_y

    @property
    def ball_velocity(self) -> tuple[float, float]:
        """Ball velocity as a (vx, vy) tuple."""
        return self._ball_vx, self._ball_vy

    @property
    def agent_paddle_y(self) -> float:
        """Center y-coordinate of the agent (right) paddle."""
        return self._agent_paddle_y

    @property
    def opponent_paddle_y(self) -> float:
        """Center y-coordinate of the opponent (left) paddle."""
        return self._opponent_paddle_y

    @property
    def agent_paddle_x(self) -> int:
        """Left edge x-coordinate of the agent (right) paddle."""
        return self._agent_paddle_x

    @property
    def opponent_paddle_x(self) -> int:
        """Left edge x-coordinate of the opponent (left) paddle."""
        return self._opponent_paddle_x

    @property
    def agent_score(self) -> int:
        """Points the agent has scored this episode."""
        return self._agent_score

    @property
    def opponent_score(self) -> int:
        """Points the opponent has scored this episode."""
        return self._opponent_score

    @property
    def screen_width(self) -> int:
        """Playfield width in pixels."""
        return self._screen_width

    @property
    def screen_height(self) -> int:
        """Playfield height in pixels."""
        return self._screen_height

    @property
    def paddle_width(self) -> int:
        """Paddle width in pixels."""
        return self._paddle_width

    @property
    def paddle_height(self) -> int:
        """Paddle height in pixels."""
        return self._paddle_height

    @property
    def ball_size(self) -> int:
        """Ball side length in pixels."""
        return self._ball_size

    @property
    def points_to_win(self) -> int:
        """Score at which an episode ends."""
        return self._points_to_win

    @property
    def ball_speed_x(self) -> int:
        """Horizontal speed magnitude. Lets the env scale velocity to [-1, 1]."""
        return self._ball_speed_x

    @property
    def ball_speed_y(self) -> int:
        """Vertical speed magnitude. Lets the env scale velocity to [-1, 1]."""
        return self._ball_speed_y
