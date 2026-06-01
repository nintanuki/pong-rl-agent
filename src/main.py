"""Entry point.

Wires the pieces together and runs one of two jobs chosen by a command-line
flag: train a fresh agent, or evaluate a previously trained one against a random
baseline. Every tunable lives in settings.py, so this file only decides what
runs in what order, not with what numbers.

The build order is always the same: a GameEngine holds the rules, a PongEnv
wraps it for learning, an agent learns through that env, and the Trainer or
Evaluator drives the loop.
"""

import argparse
import random

import numpy as np

from settings import Agents, DQN, Game, Paths, Reward, Training
from game_engine import GameEngine
from pong_env import ACTION_COUNT, OBSERVATION_SIZE, PongEnv
from q_learning_agent import QLearningAgent
from trainer import Trainer
from evaluator import Evaluator


# Labels used in checkpoint filenames and logs, kept in one place so training
# and evaluation always agree on them.
Q_LEARNING_NAME = "q_learning"
DQN_NAME = "dqn"


class Application:
    """Orchestrates training and evaluation runs."""

    def __init__(self, mode: str) -> None:
        """Remember which job to run and seed the random number generators.

        Args:
            mode: Either "train" or "evaluate", from the command line.
        """
        self.mode = mode

        # One seed for every source of randomness makes a run repeatable, which
        # matters when comparing agents or debugging.
        random.seed(Training.RANDOM_SEED)
        np.random.seed(Training.RANDOM_SEED)

    def run(self) -> None:
        """Dispatch to the training or evaluation job."""
        if self.mode == "train":
            self.train()
        else:
            self.evaluate()

    def train(self) -> None:
        """Train every enabled agent and save its checkpoints and reward curve."""
        if Agents.RUN_Q_LEARNING:
            agent = self._build_q_learning_agent()
            self._train_one(agent, Q_LEARNING_NAME)

        if Agents.RUN_DQN:
            agent = self._build_dqn_agent()
            self._train_one(agent, DQN_NAME)

    def _train_one(self, agent, name: str) -> None:
        """Run the full training loop for a single agent.

        Args:
            agent: A freshly built agent to train.
            name: Its label, e.g. "q_learning", used for its files and logs.
        """
        env = self._build_env()
        trainer = Trainer(
            env=env,
            agent=agent,
            episodes=Training.EPISODES,
            max_steps=Training.MAX_STEPS_PER_EPISODE,
            name=name,
            checkpoint_every=Training.CHECKPOINT_EVERY_EPISODES,
        )
        trainer.train()

    def evaluate(self) -> None:
        """Score every enabled agent's latest checkpoint against random play."""
        if Agents.RUN_Q_LEARNING:
            self._evaluate_one(self._build_q_learning_agent(), Q_LEARNING_NAME)

        if Agents.RUN_DQN:
            self._evaluate_one(self._build_dqn_agent(), DQN_NAME)

    def _evaluate_one(self, agent, name: str) -> None:
        """Load an agent's latest checkpoint and compare it to the baseline.

        Args:
            agent: A fresh agent of the right type to load weights into.
            name: Its label, used to find the matching checkpoint file.
        """
        checkpoint = self._latest_checkpoint(name)
        if checkpoint is None:
            print(f"No checkpoint found for {name}; train it first.")
            return

        agent.load(checkpoint)
        env = self._build_env()
        evaluator = Evaluator(env, episodes=Training.EVALUATION_EPISODES)

        trained = evaluator.run_agent(agent)
        baseline = evaluator.run_agent(None)
        self._report(name, trained, baseline)

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

    def _build_env(self) -> PongEnv:
        """Build a fresh game wrapped in the learning environment.

        Returns:
            A `PongEnv` ready for an agent. Each agent gets its own so their runs
            never interfere.
        """
        game = GameEngine(
            screen_width=Game.SCREEN_WIDTH_PIXELS,
            screen_height=Game.SCREEN_HEIGHT_PIXELS,
            paddle_width=Game.PADDLE_WIDTH_PIXELS,
            paddle_height=Game.PADDLE_HEIGHT_PIXELS,
            paddle_margin=Game.PADDLE_MARGIN_PIXELS,
            paddle_speed=Game.PADDLE_SPEED_PIXELS,
            opponent_speed=Game.OPPONENT_SPEED_PIXELS,
            ball_size=Game.BALL_SIZE_PIXELS,
            ball_speed_x=Game.BALL_SPEED_X_PIXELS,
            ball_speed_y=Game.BALL_SPEED_Y_PIXELS,
            points_to_win=Game.POINTS_TO_WIN,
            seed=Training.RANDOM_SEED,
        )
        return PongEnv(game, Reward)

    def _build_q_learning_agent(self) -> QLearningAgent:
        """Build the tabular Q-learning agent from its settings.

        Returns:
            An untrained `QLearningAgent`.
        """
        return QLearningAgent(
            action_count=ACTION_COUNT,
            learning_rate=Training.LEARNING_RATE,
            discount_factor=Training.DISCOUNT_FACTOR,
            epsilon_start=Training.EPSILON_START,
            epsilon_end=Training.EPSILON_END,
            epsilon_decay=Training.EPSILON_DECAY,
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
            hidden_layer_sizes=DQN.HIDDEN_LAYER_SIZES,
            learning_rate=DQN.LEARNING_RATE,
            discount_factor=Training.DISCOUNT_FACTOR,
            epsilon_start=Training.EPSILON_START,
            epsilon_end=Training.EPSILON_END,
            epsilon_decay=Training.EPSILON_DECAY,
            batch_size=DQN.BATCH_SIZE,
            replay_buffer_size=DQN.REPLAY_BUFFER_SIZE,
            target_update_steps=DQN.TARGET_NETWORK_UPDATE_EVERY_STEPS,
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


def main() -> None:
    """Console entry point. Parses flags and runs the Application."""
    parser = argparse.ArgumentParser(description="Pong RL Agent")
    parser.add_argument(
        "--mode",
        choices=("train", "evaluate"),
        default="train",
        help="Whether to train a new agent or evaluate an existing checkpoint.",
    )
    args = parser.parse_args()
    app = Application(mode=args.mode)
    app.run()


if __name__ == "__main__":
    main()
