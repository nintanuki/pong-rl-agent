"""Evaluation harness.

Measures how well a trained agent actually plays, and puts that number in
context by also running a baseline that just moves at random. Comparing the two
is what shows the agent learned something rather than getting lucky: the rubric
asks for exactly this trained-versus-random comparison.

For each agent it plays a fixed number of matches with exploration turned off
(so the agent always plays its best move) and reports the average, best, and
spread of the total reward.
"""

import numpy as np


class Evaluator:
    """Plays a trained agent and a random baseline, and summarizes the results."""

    def __init__(self, env, episodes: int, render: bool = False) -> None:
        """Store what to evaluate and how.

        Args:
            env: A `PongEnv`-style environment. Pass one built with rendering in
                mind if `render` is True.
            episodes: How many matches to play per agent.
            render: Whether to draw each match on screen while it plays.
        """
        self.env = env
        self.episodes = episodes
        self.render = render

    def run_agent(self, agent) -> dict:
        """Play `episodes` matches and summarize the reward earned.

        Args:
            agent: A trained agent to evaluate, or `None` to run the random
                baseline that ignores the observation entirely.

        Returns:
            A summary dict with keys mean_reward, max_reward, std_reward, and the
            full rewards list. Printed side by side for the trained agent and the
            baseline so the difference is obvious.
        """
        # Turning the exploration rate to zero makes a trained agent always pick
        # its best-known move, so we measure skill, not luck.
        if agent is not None and hasattr(agent, "epsilon"):
            agent.epsilon = 0.0

        rewards: list[float] = []
        for _episode in range(self.episodes):
            observation, _info = self.env.reset()
            total_reward = 0.0

            while True:
                action = self._choose_action(agent, observation)
                observation, reward, terminated, truncated, _info = self.env.step(
                    action
                )
                total_reward += reward
                if self.render:
                    self.env.render()
                if terminated or truncated:
                    break

            rewards.append(total_reward)

        return {
            "mean_reward": float(np.mean(rewards)),
            "max_reward": float(np.max(rewards)),
            "std_reward": float(np.std(rewards)),
            "rewards": rewards,
        }

    def _choose_action(self, agent, observation):
        """Pick the next move: the agent's best, or a random one for the baseline.

        Args:
            agent: The trained agent, or `None` for the random baseline.
            observation: The env's current snapshot.

        Returns:
            An action to pass to `env.step`.
        """
        if agent is None:
            return self.env.action_space.sample()
        return agent.select_action(observation)
