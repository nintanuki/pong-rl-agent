"""Application-wide constants and toggleable values.

Each top-level class acts as a namespace for one concern. There are no magic
numbers elsewhere in the project; if a value can be tuned, it lives here.
"""

from pathlib import Path


# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------

class Paths:
    """Filesystem paths used by the application."""

    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
    OUTPUT_DIR: Path = PROJECT_ROOT / "src" / "output"
    CHECKPOINT_DIR: Path = OUTPUT_DIR / "checkpoints"
    PLOT_DIR: Path = OUTPUT_DIR / "plots"


# ---------------------------------------------------------------------------
# GAME
# ---------------------------------------------------------------------------

class GameSettings:
    """Pong playfield geometry and motion.

    The agent controls the right-hand paddle; a scripted AI controls the left
    one. Coordinates are screen-space pixels (x grows rightward, y downward).
    """

    SCREEN_WIDTH_PIXELS: int = 800
    SCREEN_HEIGHT_PIXELS: int = 600

    # Paddle dimensions and the gap between each paddle face and its wall.
    PADDLE_WIDTH_PIXELS: int = 12
    PADDLE_HEIGHT_PIXELS: int = 90
    PADDLE_MARGIN_PIXELS: int = 24

    # Per-step paddle travel. The agent paddle and the scripted opponent are
    # tuned separately so the opponent can be made beatable later.
    PADDLE_SPEED_PIXELS: int = 6
    OPPONENT_SPEED_PIXELS: int = 4

    # Ball size and the per-step velocity magnitude on each axis. The sign of
    # each component is randomized on every serve.
    BALL_SIZE_PIXELS: int = 14
    BALL_SPEED_X_PIXELS: int = 5
    BALL_SPEED_Y_PIXELS: int = 5

    # First side to reach this many points ends an episode.
    POINTS_TO_WIN: int = 11

    # Frame rate used while rendering a trained agent. Decoupled from the
    # human-play rate below so each can be tuned independently.
    RENDER_FRAMES_PER_SECOND: int = 60
    # Frame rate for the manual test mode in play_human.py.
    HUMAN_FRAMES_PER_SECOND: int = 60


class ColorSettings:
    """RGB colors for the minimal, asset-free render."""

    BACKGROUND: tuple[int, int, int] = (20, 20, 20)
    PADDLE: tuple[int, int, int] = (220, 220, 220)
    BALL: tuple[int, int, int] = (220, 220, 220)
    CENTER_LINE: tuple[int, int, int] = (80, 80, 80)
    SCORE: tuple[int, int, int] = (200, 200, 200)


# ---------------------------------------------------------------------------
# REWARDS
# ---------------------------------------------------------------------------

class RewardSettings:
    """Reward shaping for the Pong environment.

    Rewards are framed from the agent's (right paddle) point of view: scoring
    on the opponent is positive, conceding is negative. The optional hit bonus
    densifies the signal so the agent learns to return the ball before it ever
    learns to win a full rally.
    """

    SCORE_POINT: float = 1.0
    CONCEDE_POINT: float = -1.0
    PADDLE_HIT: float = 0.1
    PER_STEP: float = 0.0
    USE_HIT_SHAPING: bool = True


# ---------------------------------------------------------------------------
# LEARNING
# ---------------------------------------------------------------------------

class DiscretizationSettings:
    """How the tabular Q-learning agent buckets the court into discrete states.

    The environment hands the agent six continuous numbers, but a Q-table can
    only look things up by exact, repeatable keys. So we round each number into
    one of a small set of buckets. Fewer buckets means the agent generalizes
    faster but sees the world more coarsely; more buckets is the reverse. Ball
    direction is already just left/right and up/down, so it needs no buckets.
    """

    BALL_X_BINS: int = 12
    BALL_Y_BINS: int = 12
    AGENT_PADDLE_Y_BINS: int = 12


class TrainingSettings:
    """Training-loop hyperparameters."""

    EPISODES: int = 2000
    MAX_STEPS_PER_EPISODE: int = 2000
    LEARNING_RATE: float = 0.1
    DISCOUNT_FACTOR: float = 0.9
    EPSILON_START: float = 1.0
    EPSILON_END: float = 0.05
    EPSILON_DECAY: float = 0.995
    CHECKPOINT_EVERY_EPISODES: int = 200
    # Matches played per agent when measuring final performance.
    EVALUATION_EPISODES: int = 20
    RANDOM_SEED: int = 42


class DQNSettings:
    """Deep Q-Network hyperparameters (only used if DQN is enabled)."""

    HIDDEN_LAYER_SIZES: tuple[int, ...] = (128, 128)
    BATCH_SIZE: int = 64
    REPLAY_BUFFER_SIZE: int = 50_000
    TARGET_NETWORK_UPDATE_EVERY_STEPS: int = 1000
    LEARNING_RATE: float = 1e-3

    # The DQN trains a neural network on every step, so it is far slower per
    # step than the tabular agent, and its games lengthen as it learns to rally.
    # A smaller episode budget and a tighter per-game step cap keep its training
    # to a few minutes (see the report's Discussion). The tabular agent keeps
    # the larger budget in TrainingSettings.
    EPISODES: int = 600
    MAX_STEPS_PER_EPISODE: int = 800


class AgentSettings:
    """Which agents to run."""

    RUN_Q_LEARNING: bool = True
    RUN_DQN: bool = True


# ---------------------------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------------------------

class VisualizationSettings:
    """Plot styling and output sizes."""

    FIGURE_WIDTH_INCHES: float = 10.0
    FIGURE_HEIGHT_INCHES: float = 6.0
    DPI: int = 120


class LoggingSettings:
    """Console logging levels and prefixes."""

    LEVEL: str = "INFO"
    PREFIX: str = "[pong-rl]"
