"""Application orchestration.

The `Application` class is the conductor: it builds the game, the environment,
and the agents from settings.py, and runs either training or testing. main.py
keeps only the command-line entry point and hands the parsed flags here.

The build order is always the same: a GameEngine holds the rules, a PongEnv
wraps it for learning, an agent learns or plays through that env, and the
Trainer or Evaluator drives the loop.
"""

import random

import numpy as np

from settings import (
    AgentSettings,
    DQNSettings,
    GameSettings,
    Paths,
    RewardSettings,
    TrainingSettings,
)
from game_engine import GameEngine
from pong_env import ACTION_COUNT, OBSERVATION_SIZE, PongEnv
from q_learning_agent import QLearningAgent
from trainer import Trainer
from evaluator import Evaluator


# Labels used in checkpoint filenames and logs, kept in one place so training
# and testing always agree on them.
Q_LEARNING_NAME = "q_learning"
DQN_NAME = "dqn"


class Application:
    """Orchestrates training and testing runs."""

    # ------------------------------------------------------------------
    # INIT
    # ------------------------------------------------------------------

    def __init__(self, mode: str, headless: bool = False) -> None:
        """Remember which job to run and seed the random number generators.

        Args:
            mode: Either "train" or "test", from the command line.
            headless: In test mode, skip watching and just print the scores.
        """
        self.mode = mode
        self.headless = headless

        # One seed for every source of randomness makes a run repeatable, which
        # matters when comparing agents or debugging.
        random.seed(TrainingSettings.RANDOM_SEED)
        np.random.seed(TrainingSettings.RANDOM_SEED)

    # ------------------------------------------------------------------
    # BUILD HELPERS
    # ------------------------------------------------------------------

    def _build_env(self) -> PongEnv:
        """Build a fresh game wrapped in the learning environment.

        Returns:
            A `PongEnv` ready for an agent. Each agent gets its own so their runs
            never interfere.
        """
        game = GameEngine(
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
            seed=TrainingSettings.RANDOM_SEED,
        )
        return PongEnv(game, RewardSettings)

    def _build_q_learning_agent(self) -> QLearningAgent:
        """Build the tabular Q-learning agent from its settings.

        Returns:
            An untrained `QLearningAgent`.
        """
        return QLearningAgent(
            action_count=ACTION_COUNT,
            learning_rate=TrainingSettings.LEARNING_RATE,
            discount_factor=TrainingSettings.DISCOUNT_FACTOR,
            epsilon_start=TrainingSettings.EPSILON_START,
            epsilon_end=TrainingSettings.EPSILON_END,
            epsilon_decay=TrainingSettings.EPSILON_DECAY,
        )

    def _build_dqn_agent(self):
        """Build the DQN agent, importing PyTorch only when it is actually used.

        Keeping the import inside this method means the tabular agent still runs
        on a machine without PyTorch installed.

        Returns:
            An untrained `DQNAgent`.
        """
        from dqn_agent import DQNAgent

        return DQNAgent(
            state_size=OBSERVATION_SIZE,
            action_count=ACTION_COUNT,
            hidden_layer_sizes=DQNSettings.HIDDEN_LAYER_SIZES,
            learning_rate=DQNSettings.LEARNING_RATE,
            discount_factor=TrainingSettings.DISCOUNT_FACTOR,
            epsilon_start=TrainingSettings.EPSILON_START,
            epsilon_end=TrainingSettings.EPSILON_END,
            epsilon_decay=TrainingSettings.EPSILON_DECAY,
            batch_size=DQNSettings.BATCH_SIZE,
            replay_buffer_size=DQNSettings.REPLAY_BUFFER_SIZE,
            target_update_steps=DQNSettings.TARGET_NETWORK_UPDATE_EVERY_STEPS,
        )

    def _latest_checkpoint(self, name: str):
        """Find the most recent checkpoint file for an agent, if any.

        Args:
            name: The agent's label.

        Returns:
            The newest matching checkpoint path, or None when none exist yet.
        """
        candidates = sorted(Paths.CHECKPOINT_DIR.glob(f"{name}_episode_*.ckpt"))
        return candidates[-1] if candidates else None

    # ------------------------------------------------------------------
    # TRAIN
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Train every enabled agent and save its checkpoints and reward curve.

        The two agents get different run lengths: the tabular agent uses the
        full TrainingSettings budget, while the DQN uses its own smaller budget
        (it is much slower per step, so this keeps its training to minutes).
        """
        if AgentSettings.RUN_Q_LEARNING:
            self._train_one(
                self._build_q_learning_agent(),
                Q_LEARNING_NAME,
                TrainingSettings.EPISODES,
                TrainingSettings.MAX_STEPS_PER_EPISODE,
            )

        if AgentSettings.RUN_DQN:
            self._train_one(
                self._build_dqn_agent(),
                DQN_NAME,
                DQNSettings.EPISODES,
                DQNSettings.MAX_STEPS_PER_EPISODE,
            )

    def _train_one(self, agent, name: str, episodes: int, max_steps: int) -> None:
        """Run the full training loop for a single agent.

        Args:
            agent: A freshly built agent to train.
            name: Its label, e.g. "q_learning", used for its files and logs.
            episodes: How many matches to train this agent for.
            max_steps: The per-match step cap for this agent.
        """
        trainer = Trainer(
            env=self._build_env(),
            agent=agent,
            episodes=episodes,
            max_steps=max_steps,
            name=name,
            checkpoint_every=TrainingSettings.CHECKPOINT_EVERY_EPISODES,
        )
        trainer.train()

    # ------------------------------------------------------------------
    # TEST
    # ------------------------------------------------------------------

    def test(self) -> None:
        """Watch and score every enabled agent's latest checkpoint."""
        if AgentSettings.RUN_Q_LEARNING:
            self._test_one(self._build_q_learning_agent(), Q_LEARNING_NAME)

        if AgentSettings.RUN_DQN:
            self._test_one(self._build_dqn_agent(), DQN_NAME)

    def _test_one(self, agent, name: str) -> None:
        """Load an agent's latest checkpoint, optionally watch it, then score it.

        Args:
            agent: A fresh agent of the right type to load weights into.
            name: Its label, used to find the matching checkpoint file.
        """
        checkpoint = self._latest_checkpoint(name)
        if checkpoint is None:
            print(f"No checkpoint found for {name}; train it first.")
            return
        agent.load(checkpoint)

        # Watch first (unless headless): a separate window run that ends when the
        # viewer closes the window. It never feeds into the scores below.
        if not self.headless:
            print(f"Watching {name}. Close the window when you're done.")
            Evaluator(
                self._build_env(), episodes=TrainingSettings.EVALUATION_EPISODES
            ).watch(agent)

        # Then score on a fresh, headless run so the numbers are always the same
        # regardless of how long you watched.
        evaluator = Evaluator(
            self._build_env(), episodes=TrainingSettings.EVALUATION_EPISODES
        )
        trained = evaluator.run_agent(agent)
        baseline = evaluator.run_agent(None)
        self._report(name, trained, baseline)

    # ------------------------------------------------------------------
    # REPORTING
    # ------------------------------------------------------------------

    def _report(self, name: str, trained: dict, baseline: dict) -> None:
        """Print the trained-versus-random comparison for one agent.

        Args:
            name: The agent's label.
            trained: Summary dict from evaluating the trained agent.
            baseline: Summary dict from the random baseline.
        """
        print(f"\n=== {name} ===")
        print(
            f"  trained : mean {trained['mean_reward']:+.2f} "
            f"| max {trained['max_reward']:+.2f} | std {trained['std_reward']:.2f}"
        )
        print(
            f"  random  : mean {baseline['mean_reward']:+.2f} "
            f"| max {baseline['max_reward']:+.2f} | std {baseline['std_reward']:.2f}"
        )

    # ------------------------------------------------------------------
    # RUN
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Dispatch to the training or test job chosen on the command line."""
        if self.mode == "train":
            self.train()
        else:
            self.test()
