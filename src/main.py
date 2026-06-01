"""Entry point.

Dispatches between training and evaluation based on a CLI flag. Reads all
hyperparameters from settings.py; nothing is hardcoded here.
"""

import argparse

from settings import Agents, Game, Paths, Reward, Training


class Application:
    """Orchestrates training and evaluation runs."""

    def __init__(self, mode: str) -> None:
        """Store the run mode and prepare collaborators lazily.

        Args:
            mode: Either "train" or "evaluate".
        """
        # TODO: store mode; instantiate GameEngine / SnakeEnv / agents inside run
        raise NotImplementedError

    def train(self) -> None:
        """Build env + agent and run the Trainer."""
        # TODO: GameEngine -> SnakeEnv -> QLearningAgent -> Trainer.train()
        raise NotImplementedError

    def evaluate(self) -> None:
        """Build env + agent from a checkpoint and run the Evaluator."""
        # TODO: load latest checkpoint; run trained agent + random baseline
        raise NotImplementedError

    def run(self) -> None:
        """Dispatch to train() or evaluate() based on configured mode."""
        # TODO: dispatch
        raise NotImplementedError


def main() -> None:
    """Console entry point. Parses flags and runs the Application."""
    parser = argparse.ArgumentParser(description="Snake RL Agent")
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
