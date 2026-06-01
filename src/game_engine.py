"""Snake gameplay logic.

A self-contained, dependency-free implementation of Snake. Distilled from the
arcade-cabinet Snake game by stripping out rendering, audio, controller input,
fullscreen handling, pause overlay, and sprite assets so the engine can run
headless during training. The only thing this module knows about is the state
machine: snake body, direction, food placement, collision detection, score.

Coordinates are (x, y) tuples where x grows rightward and y grows downward
(screen convention). The engine exposes only what the gymnasium env wrapper
needs to compute observations, rewards, and termination.

Lineage
-------
The original gameplay logic was learned from Clear Code's tutorial
"Learning pygame by creating Snake" (https://www.youtube.com/watch?v=QFvqStqPCRU).
That code was extended with arcade-cabinet bells and whistles (joystick input,
pause overlay, music, custom sprites) for a separate project and then distilled
back down here for reinforcement learning, where rendering and audio are noise.
"""

import random
from typing import Optional


Point = tuple[int, int]
Direction = tuple[int, int]


# Direction constants chosen so adjacent directions never reverse the snake.
UP: Direction = (0, -1)
DOWN: Direction = (0, 1)
LEFT: Direction = (-1, 0)
RIGHT: Direction = (1, 0)


class GameEngine:
    """Pure-logic Snake game suitable for both headless and rendered use."""

    def __init__(
        self,
        grid_width: int,
        grid_height: int,
        initial_length: int,
        seed: Optional[int] = None,
    ) -> None:
        """Configure the playfield and the starting snake length.

        Args:
            grid_width: Width of the play grid in cells.
            grid_height: Height of the play grid in cells.
            initial_length: Number of cells the snake starts with.
            seed: Optional RNG seed for reproducible food placement.
        """
        self._grid_width = grid_width
        self._grid_height = grid_height
        self._initial_length = initial_length
        self._rng = random.Random(seed)
        self._body: list[Point] = []
        self._direction: Direction = (0, 0)
        self._food: Point = (0, 0)
        self._score: int = 0
        self._pending_growth: bool = False
        self.reset()

    def reset(self) -> None:
        """Reset the snake to its starting position and spawn fresh food."""
        center_x = self._grid_width // 2
        center_y = self._grid_height // 2
        # Lay the snake horizontally with its head to the right of the body
        # so the first frame has an unambiguous heading once a direction is set.
        self._body = [
            (center_x - offset, center_y) for offset in range(self._initial_length)
        ]
        self._direction = (0, 0)
        self._score = 0
        self._pending_growth = False
        self._food = self._random_empty_cell()

    def set_direction(self, direction: Direction) -> None:
        """Update the snake's heading, rejecting 180-degree reversals.

        Args:
            direction: One of UP, DOWN, LEFT, RIGHT.
        """
        dx, dy = direction
        cur_dx, cur_dy = self._direction
        # Block reversal: x flipping while y is zero, or y flipping while x is zero.
        if dx and dx == -cur_dx:
            return
        if dy and dy == -cur_dy:
            return
        self._direction = direction

    def step(self) -> tuple[bool, bool]:
        """Advance the game by one tick.

        Returns:
            A tuple (ate_food, game_over).
        """
        if self._direction == (0, 0):
            # No move yet; nothing changes and the game cannot end.
            return False, False

        head_x, head_y = self._body[0]
        dx, dy = self._direction
        new_head: Point = (head_x + dx, head_y + dy)

        # Wall collision.
        if not self._in_bounds(new_head):
            return False, True

        # Self collision. Compare against every block except the tail because
        # the tail is about to vacate its cell unless food was eaten.
        body_to_check = self._body if self._pending_growth else self._body[:-1]
        if new_head in body_to_check:
            return False, True

        ate_food = new_head == self._food
        if ate_food:
            self._body.insert(0, new_head)
            self._score += 1
            self._pending_growth = False
            self._food = self._random_empty_cell()
            return True, False

        # Normal move: insert new head, drop tail unless we owe a growth.
        self._body.insert(0, new_head)
        if self._pending_growth:
            self._pending_growth = False
        else:
            self._body.pop()
        return False, False

    def _in_bounds(self, point: Point) -> bool:
        """Check whether a cell is inside the playfield.

        Args:
            point: An (x, y) cell coordinate.

        Returns:
            True when the cell is within the grid.
        """
        x, y = point
        return 0 <= x < self._grid_width and 0 <= y < self._grid_height

    def _random_empty_cell(self) -> Point:
        """Pick a random cell that is not occupied by the snake.

        Returns:
            A fresh (x, y) coordinate suitable for food placement.
        """
        body_set = set(self._body)
        # Build candidate list lazily; the grid is small so this is cheap.
        candidates = [
            (x, y)
            for x in range(self._grid_width)
            for y in range(self._grid_height)
            if (x, y) not in body_set
        ]
        if not candidates:
            # Snake has filled the board; return the head so the env can
            # treat this as a terminal "perfect game" state if it wants.
            return self._body[0]
        return self._rng.choice(candidates)

    @property
    def head(self) -> Point:
        """Current position of the snake's head as an (x, y) tuple."""
        return self._body[0]

    @property
    def body(self) -> list[Point]:
        """Read-only view of the snake body (head first, tail last)."""
        return list(self._body)

    @property
    def direction(self) -> Direction:
        """Current heading as an (dx, dy) tuple."""
        return self._direction

    @property
    def food(self) -> Point:
        """Current food position as an (x, y) tuple."""
        return self._food

    @property
    def score(self) -> int:
        """Number of food items eaten this episode."""
        return self._score

    @property
    def grid_width(self) -> int:
        """Playfield width in cells."""
        return self._grid_width

    @property
    def grid_height(self) -> int:
        """Playfield height in cells."""
        return self._grid_height
