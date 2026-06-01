"""Training loop.

Drives the agent through a configurable number of episodes, tracks rolling
mean reward, persists checkpoints, and writes a training-reward curve at
the end. Stays renderer-agnostic so training can run headless.
"""


class Trainer:
    """Runs the episode loop for a single agent."""

    def __init__(self, env, agent, episodes: int, max_steps: int) -> None:
        """Store collaborators and run length.

        Args:
            env: A SnakeEnv-compatible environment.
            agent: An agent exposing select_action / update / decay_epsilon.
            episodes: Total number of episodes to train for.
            max_steps: Hard cap on steps per episode.
        """
        # TODO: store; initialise reward-history list
        raise NotImplementedError

    def train(self) -> list:
        """Run the training loop and return the per-episode reward history.

        Returns:
            A list of total rewards, one entry per episode.
        """
        # TODO: nested loops; periodic logging; periodic checkpoint
        raise NotImplementedError

    def save_checkpoint(self, episode: int) -> None:
        """Persist the current agent state.

        Args:
            episode: Episode index, used in the checkpoint filename.
        """
        # TODO: agent.save(checkpoint_dir / f'q_table_{episode}.pkl')
        raise NotImplementedError

    def plot_rewards(self, rewards: list) -> None:
        """Save a line plot of total reward per episode.

        Args:
            rewards: Per-episode reward history.
        """
        # TODO: matplotlib; raw curve + rolling mean overlay
        raise NotImplementedError
