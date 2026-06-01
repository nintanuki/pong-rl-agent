"""Human-playable Snake driver.

A small standalone script that wraps `GameEngine` with a keyboard input
handler, simple pygame rendering, and an optional crunch sound. Lets you
manually verify that the gameplay logic feels correct before plugging the
engine into the RL environment.

Run from the `src/` directory:

    python play_human.py

Controls: arrow keys to steer, window close (or X) to quit. There is no
pause and no fullscreen toggle by design; the script is a sanity test, not
a polished game build.
"""

from pathlib import Path

import pygame

from game_engine import DOWN, GameEngine, LEFT, RIGHT, UP
from settings import Game


ASSETS_DIR = Path(__file__).resolve().parent / "assets"
CRUNCH_SOUND_FILE = ASSETS_DIR / "crunch.wav"

# Colors for the minimal-rectangle render.
COLOR_BACKGROUND = (175, 215, 70)
COLOR_GRID_DARK = (167, 209, 61)
COLOR_SNAKE = (52, 90, 32)
COLOR_FOOD = (200, 38, 0)
COLOR_SCORE = (56, 74, 12)


class HumanPlayer:
    """Pygame loop that lets a human pilot the Snake engine."""

    def __init__(self) -> None:
        """Initialize pygame, the engine, the window, and the crunch sound."""
        pygame.init()
        self._engine = GameEngine(
            grid_width=Game.GRID_WIDTH_CELLS,
            grid_height=Game.GRID_HEIGHT_CELLS,
            initial_length=Game.INITIAL_SNAKE_LENGTH,
        )
        window_width = Game.GRID_WIDTH_CELLS * Game.CELL_SIZE_PIXELS
        window_height = Game.GRID_HEIGHT_CELLS * Game.CELL_SIZE_PIXELS
        self._screen = pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption("Snake (manual test mode)")
        self._clock = pygame.time.Clock()
        self._font = pygame.font.SysFont(None, 32)
        self._crunch = self._load_crunch_sound()

    def _load_crunch_sound(self) -> pygame.mixer.Sound | None:
        """Load the crunch sound effect if the asset is present.

        Returns:
            A loaded pygame Sound, or None when the asset is missing.
        """
        if not CRUNCH_SOUND_FILE.exists():
            return None
        try:
            return pygame.mixer.Sound(str(CRUNCH_SOUND_FILE))
        except pygame.error:
            return None

    def _handle_input(self) -> bool:
        """Process the pygame event queue for one frame.

        Returns:
            True when the player wants to keep playing, False to quit.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    self._engine.set_direction(UP)
                elif event.key == pygame.K_DOWN:
                    self._engine.set_direction(DOWN)
                elif event.key == pygame.K_LEFT:
                    self._engine.set_direction(LEFT)
                elif event.key == pygame.K_RIGHT:
                    self._engine.set_direction(RIGHT)
        return True

    def _draw(self) -> None:
        """Render the current frame: background, food, snake, score."""
        self._screen.fill(COLOR_BACKGROUND)
        cell = Game.CELL_SIZE_PIXELS

        # Checkerboard background for visual readability.
        for row in range(Game.GRID_HEIGHT_CELLS):
            for col in range(Game.GRID_WIDTH_CELLS):
                if (row + col) % 2 == 0:
                    rect = pygame.Rect(col * cell, row * cell, cell, cell)
                    pygame.draw.rect(self._screen, COLOR_GRID_DARK, rect)

        # Food.
        food_x, food_y = self._engine.food
        food_rect = pygame.Rect(food_x * cell, food_y * cell, cell, cell)
        pygame.draw.rect(self._screen, COLOR_FOOD, food_rect)

        # Snake body, head distinguished by a subtle inset.
        for index, (x, y) in enumerate(self._engine.body):
            rect = pygame.Rect(x * cell, y * cell, cell, cell)
            pygame.draw.rect(self._screen, COLOR_SNAKE, rect)
            if index == 0:
                inner = rect.inflate(-cell // 3, -cell // 3)
                pygame.draw.rect(self._screen, COLOR_BACKGROUND, inner, width=2)

        # Score in the upper-left corner.
        score_surface = self._font.render(
            f"Score: {self._engine.score}", True, COLOR_SCORE
        )
        self._screen.blit(score_surface, (10, 10))

        pygame.display.flip()

    def run(self) -> None:
        """Run the human-play loop until the window is closed."""
        running = True
        while running:
            running = self._handle_input()
            ate_food, game_over = self._engine.step()
            if ate_food and self._crunch is not None:
                self._crunch.play()
            if game_over:
                self._engine.reset()
            self._draw()
            # Tick at the human step rate so the snake feels playable; the
            # render-fps setting is for the trained-agent demo, not for here.
            self._clock.tick(Game.HUMAN_STEPS_PER_SECOND)
        pygame.quit()


def main() -> None:
    """Console entry point. Builds the player and runs the loop."""
    player = HumanPlayer()
    player.run()


if __name__ == "__main__":
    main()
