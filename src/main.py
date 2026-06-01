"""Entry point.

The only job of this file is to read the command-line flags and hand them to
the `Application` in app.py, which does the actual work. `--mode train` trains a
fresh agent and saves it; `--mode test` loads the most recently trained agent,
lets you watch it play until you close the window, and then prints how it scored
against a random baseline; adding `--headless` skips the window and goes
straight to those numbers.
"""

import argparse

from app import Application


def main() -> None:
    """Parse the command-line flags and run the Application."""
    parser = argparse.ArgumentParser(description="Pong RL Agent")
    parser.add_argument(
        "--mode",
        choices=("train", "test"),
        default="train",
        help="Train a new agent, or load the latest and test (watch + score) it.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="In test mode, skip the watch window and just print the scores.",
    )
    args = parser.parse_args()
    app = Application(mode=args.mode, headless=args.headless)
    app.run()


if __name__ == "__main__":
    main()
