"""Human-playable Pong driver.

A small standalone script that wraps `GameEngine` with a keyboard input
handler and simple pygame rendering. Lets you manually verify that the
gameplay logic feels correct before plugging the engine into the RL
environment.

Run from the `src/` directory:

    python play_human.py

Controls: Up / Down arrows move the right-hand paddle; the left paddle is the
scripted AI opponent. Close the window (or press Esc) to quit. There is no
sound, no pause, no countdown, and no fullscreen toggle by design; the script
is a sanity test, not a polished game build.
"""

import pygame

from game_engine import ACTION_DOWN, ACTION_STAY, ACTION_UP, GameEngine
from settings import ColorSettings, GameSettings


class HumanPlayer:
    """Pygame loop that lets a human pilot the agent paddle."""

    # ------------------------------------------------------------------
    # INIT
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        """Initialize pygame, the engine, and the window."""
        pygame.init()
        self._engine = GameEngine(
            screen_width=GameSettings.SCREEN_WIDTH_PIXELS,
            screen_height=GameSettings.SCREEN_HEIGHT_PIXELS,
            paddle_width=GameSettings.PADDLE_WIDTH_PIXELS,
            paddle_height=GameSettings.PADDLE_HEIGHT_PIXELS,
            paddle_margin=GameSettings.PADDLE_MARGIN_PIXELS,
            paddle_speed=GameSettings.PADDLE_SPEED_PIXELS,
            opponent_speed=GameSettings.OPPONENT_SPEED_PIXELS,
            ball_size=GameSettings.BALL_SIZE_PIXELS,
            ball_speed_x=GameSettings.BALL_SPEED_X_PIXELS,
            ball_speed_y=GameSettings.BALL_SPEED_Y_PIXELS,
            points_to_win=GameSettings.POINTS_TO_WIN,
        )
        self._screen = pygame.display.set_mode(
            (GameSettings.SCREEN_WIDTH_PIXELS, GameSettings.SCREEN_HEIGHT_PIXELS)
        )
        pygame.display.set_caption("Pong (manual test mode)")
        self._clock = pygame.time.Clock()
        self._font = pygame.font.SysFont(None, 64)

    # ------------------------------------------------------------------
    # INPUT
    # ------------------------------------------------------------------

    def _handle_quit_events(self) -> bool:
        """Drain the event queue, watching only for quit requests.

        Returns:
            True to keep playing, False when the player wants to quit.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False
        return True

    def _read_action(self) -> int:
        """Translate the currently held arrow keys into an engine action.

        Returns:
            One of ACTION_UP, ACTION_DOWN, or ACTION_STAY.
        """
        keys = pygame.key.get_pressed()
        up_held = keys[pygame.K_UP]
        down_held = keys[pygame.K_DOWN]
        if up_held and not down_held:
            return ACTION_UP
        if down_held and not up_held:
            return ACTION_DOWN
        return ACTION_STAY

    # ------------------------------------------------------------------
    # RENDER
    # ------------------------------------------------------------------

    def _draw(self) -> None:
        """Render one frame: background, center line, paddles, ball, score."""
        engine = self._engine
        self._screen.fill(ColorSettings.BACKGROUND)

        # Dashed center line.
        dash_height = 16
        gap = 12
        x = engine.screen_width // 2 - 2
        for y in range(0, engine.screen_height, dash_height + gap):
            pygame.draw.rect(self._screen, ColorSettings.CENTER_LINE, (x, y, 4, dash_height))

        # Paddles.
        half_paddle = engine.paddle_height / 2
        agent_rect = pygame.Rect(
            engine.agent_paddle_x,
            engine.agent_paddle_y - half_paddle,
            engine.paddle_width,
            engine.paddle_height,
        )
        opponent_rect = pygame.Rect(
            engine.opponent_paddle_x,
            engine.opponent_paddle_y - half_paddle,
            engine.paddle_width,
            engine.paddle_height,
        )
        pygame.draw.rect(self._screen, ColorSettings.PADDLE, agent_rect)
        pygame.draw.rect(self._screen, ColorSettings.PADDLE, opponent_rect)

        # Ball.
        ball_x, ball_y = engine.ball_position
        half_ball = engine.ball_size / 2
        ball_rect = pygame.Rect(
            ball_x - half_ball, ball_y - half_ball, engine.ball_size, engine.ball_size
        )
        pygame.draw.rect(self._screen, ColorSettings.BALL, ball_rect)

        # Score: opponent on the left of center, agent on the right.
        opponent_surface = self._font.render(
            str(engine.opponent_score), True, ColorSettings.SCORE
        )
        agent_surface = self._font.render(str(engine.agent_score), True, ColorSettings.SCORE)
        center_x = engine.screen_width // 2
        self._screen.blit(
            opponent_surface, opponent_surface.get_rect(midright=(center_x - 40, 50))
        )
        self._screen.blit(
            agent_surface, agent_surface.get_rect(midleft=(center_x + 40, 50))
        )

        pygame.display.flip()

    # ------------------------------------------------------------------
    # RUN
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Run the human-play loop until the window is closed."""
        running = True
        while running:
            running = self._handle_quit_events()
            result = self._engine.step(self._read_action())
            if result.done:
                self._engine.reset()
            self._draw()
            self._clock.tick(GameSettings.HUMAN_FRAMES_PER_SECOND)
        pygame.quit()


def main() -> None:
    """Console entry point. Builds the player and runs the loop."""
    player = HumanPlayer()
    player.run()


if __name__ == "__main__":
    main()
