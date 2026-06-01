"""Training loop.

Plays the agent through many matches and lets it learn from each one. The
Trainer is deliberately agnostic about *which* agent it is driving: it only
needs something that can choose a move, learn from a move, and ease off
exploring over time. That way the exact same loop trains both the tabular
Q-learning agent and the DQN agent.

It also keeps a record of how much reward the agent earned per match, saves the
agent to disk every so often, and draws a reward curve at the end so you can see
the learning happen.
"""

import matplotlib

# Use a non-interactive backend so the curve saves to a file during headless
# training without needing a display.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from settings import Logging, Paths, Visualization


class Trainer:
    """Runs the episode loop for a single agent and records its progress."""

    def __init__(
        self,
        env,
        agent,
        episodes: int,
        max_steps: int,
        name: str,
        checkpoint_every: int,
    ) -> None:
        """Store the pieces the loop needs and prepare the output folders.

        Args:
            env: A `PongEnv`-style environment.
            agent: Anything exposing `select_action`, a per-move learning hook,
                `decay_epsilon`, and `save` (the Q-learning or DQN agent).
            episodes: How many matches to train for.
            max_steps: A hard cap on ticks per match, so a stalemate can't run
                forever.
            name: Short label used in checkpoint filenames and the plot, e.g.
                "q_learning" or "dqn".
            checkpoint_every: Save the agent (and log progress) this often.
        """
        self.env = env
        self.agent = agent
        self.episodes = episodes
        self.max_steps = max_steps
        self.name = name
        self.checkpoint_every = checkpoint_every

        # One reward total per match; filled in by `train` and handed to
        # `plot_rewards`.
        self.reward_history: list[float] = []

        Paths.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        Paths.PLOT_DIR.mkdir(parents=True, exist_ok=True)

    def train(self) -> list[float]:
        """Play and learn from `episodes` matches, returning the reward history.

        Returns:
            The per-match reward totals, also stored on `self.reward_history`
            and drawn by `plot_rewards`.
        """
        for episode in range(1, self.episodes + 1):
            observation, _info = self.env.reset()
            total_reward = 0.0

            for _step in range(self.max_steps):
                action = self.agent.select_action(observation)
                next_observation, reward, terminated, truncated, _info = self.env.step(
                    action
                )

                self._learn_from_step(
                    observation, action, reward, next_observation, terminated
                )

                observation = next_observation
                total_reward += reward
                if terminated or truncated:
                    break

            # Each match, explore a little less and trust the learning a little
            # more.
            self.agent.decay_epsilon()
            self.reward_history.append(total_reward)

            if episode % self.checkpoint_every == 0:
                self.save_checkpoint(episode)
                self._log_progress(episode)

        self.plot_rewards(self.reward_history)
        return self.reward_history

    def _learn_from_step(
        self, observation, action, reward, next_observation, terminated
    ) -> None:
        """Hand one move to the agent in whatever form it learns from.

        The two agents learn differently: the tabular agent updates its table on
        the spot, while the DQN agent files the move into replay memory and then
        trains on a batch. We pick the right path by what the agent supports.

        Args:
            observation: Snapshot before the move.
            action: Move taken.
            reward: Reward earned.
            next_observation: Snapshot after the move.
            terminated: Whether the match ended on this step.
        """
        if hasattr(self.agent, "update"):
            # Tabular Q-learning: adjust the estimate immediately.
            self.agent.update(
                observation, action, reward, next_observation, terminated
            )
        else:
            # Replay-based agent (DQN): remember now, learn from a batch.
            self.agent.remember(
                observation, action, reward, next_observation, terminated
            )
            self.agent.learn()

    def save_checkpoint(self, episode: int) -> None:
        """Write the current agent to disk, tagged with the episode number.

        Args:
            episode: Which match we just finished, used in the filename.
        """
        checkpoint_path = (
            Paths.CHECKPOINT_DIR / f"{self.name}_episode_{episode:05d}.ckpt"
        )
        self.agent.save(checkpoint_path)

    def plot_rewards(self, rewards: list[float]) -> None:
        """Save a reward-per-match line chart with a smoothed overlay.

        The raw line is noisy match to match, so a rolling average is drawn on
        top to make the overall trend (hopefully upward) easy to read.

        Args:
            rewards: The per-match reward totals from `train`.
        """
        figure, axes = plt.subplots(
            figsize=(Visualization.FIGURE_WIDTH_INCHES, Visualization.FIGURE_HEIGHT_INCHES)
        )
        axes.plot(rewards, color="#bbbbbb", linewidth=1, label="Reward per match")

        rolling = self._rolling_mean(rewards, window=50)
        if rolling:
            axes.plot(rolling, color="#1f77b4", linewidth=2, label="50-match average")

        axes.set_title(f"Training reward: {self.name}")
        axes.set_xlabel("Match")
        axes.set_ylabel("Total reward")
        axes.legend()

        plot_path = Paths.PLOT_DIR / f"{self.name}_reward_curve.png"
        figure.savefig(plot_path, dpi=Visualization.DPI, bbox_inches="tight")
        plt.close(figure)

    def _rolling_mean(self, values: list[float], window: int) -> list[float]:
        """Average each point with its recent neighbors to smooth the curve.

        Args:
            values: The raw per-match rewards.
            window: How many recent matches to average over.

        Returns:
            The smoothed series, one shorter-window value per input point. Used
            only by `plot_rewards`.
        """
        if len(values) < window:
            return []

        smoothed: list[float] = []
        running_sum = 0.0
        for index, value in enumerate(values):
            running_sum += value
            if index >= window:
                running_sum -= values[index - window]
            if index >= window - 1:
                smoothed.append(running_sum / window)
        return smoothed

    def _log_progress(self, episode: int) -> None:
        """Print a one-line progress update for the most recent stretch.

        Args:
            episode: The match number just reached.
        """
        recent = self.reward_history[-self.checkpoint_every:]
        average_recent = sum(recent) / len(recent)
        epsilon = getattr(self.agent, "epsilon", 0.0)
        print(
            f"{Logging.PREFIX} {self.name} | episode {episode}/{self.episodes} "
            f"| avg reward {average_recent:+.2f} | explore rate {epsilon:.3f}"
        )
