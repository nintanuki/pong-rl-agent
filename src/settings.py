"""Application-wide constants and toggleable values.

Each top-level class acts as a namespace for one concern. There are no magic
numbers elsewhere in the project; if a value can be tuned, it lives here.
"""

from pathlib import Path


class Paths:
    """Filesystem paths used by the application."""

    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
    OUTPUT_DIR: Path = PROJECT_ROOT / "src" / "output"
    CHECKPOINT_DIR: Path = OUTPUT_DIR / "checkpoints"
    PLOT_DIR: Path = OUTPUT_DIR / "plots"


class Game:
    """Snake gameplay geometry."""

    GRID_WIDTH_CELLS: int = 20
    GRID_HEIGHT_CELLS: int = 20
    CELL_SIZE_PIXELS: int = 24
    INITIAL_SNAKE_LENGTH: int = 3
    # Frame rate used while rendering. Decouple from step rate so the agent's
    # visuals stay smooth even when the snake itself only moves a few times
    # per second.
    RENDER_FRAMES_PER_SECOND: int = 30
    # Snake tick rate for the manual test mode in play_human.py. Matches the
    # ~150 ms timer the original arcade-cabinet build used.
    HUMAN_STEPS_PER_SECOND: int = 7


class Reward:
    """Reward shaping for the Snake environment."""

    EAT_FOOD: float = 10.0
    GAME_OVER: float = -10.0
    PER_STEP: float = -0.01
    CLOSER_TO_FOOD: float = 0.1
    FARTHER_FROM_FOOD: float = -0.1
    USE_DISTANCE_SHAPING: bool = True


class Training:
    """Training-loop hyperparameters."""

    EPISODES: int = 5000
    MAX_STEPS_PER_EPISODE: int = 1000
    LEARNING_RATE: float = 0.1
    DISCOUNT_FACTOR: float = 0.9
    EPSILON_START: float = 1.0
    EPSILON_END: float = 0.05
    EPSILON_DECAY: float = 0.995
    CHECKPOINT_EVERY_EPISODES: int = 500
    RANDOM_SEED: int = 42


class DQN:
    """Deep Q-Network hyperparameters (only used if DQN is enabled)."""

    HIDDEN_LAYER_SIZES: tuple[int, ...] = (128, 128)
    BATCH_SIZE: int = 64
    REPLAY_BUFFER_SIZE: int = 50_000
    TARGET_NETWORK_UPDATE_EVERY_STEPS: int = 1000
    LEARNING_RATE: float = 1e-3


class Agents:
    """Which agents to run."""

    RUN_Q_LEARNING: bool = True
    RUN_DQN: bool = True


class Visualization:
    """Plot styling and output sizes."""

    FIGURE_WIDTH_INCHES: float = 10.0
    FIGURE_HEIGHT_INCHES: float = 6.0
    DPI: int = 120


class Logging:
    """Console logging levels and prefixes."""

    LEVEL: str = "INFO"
    PREFIX: str = "[snake-rl]"
