"""Evaluation harness.

Loads a trained agent from a checkpoint, runs it for a fixed number of
episodes with rendering enabled, and reports mean, max, and standard
deviation of the total reward. Also runs a random-action baseline for the
comparison required by the rubric.
"""


class Evaluator:
    """Evaluates a trained agent and a random-action baseline."""

    def __init__(self, env, episodes: int) -> None:
        """Store the environment and number of evaluation episodes.

        Args:
            env: A SnakeEnv-compatible environment with rendering enabled.
            episodes: Number of evaluation episodes per agent.
        """
        # TODO: store
        raise NotImplementedError

    def run_agent(self, agent) -> dict:
        """Run the given agent for the configured number of episodes.

        Args:
            agent: An agent with a select_action method. Pass `None` for the
                random-action baseline.

        Returns:
            A dict with keys: mean_reward, max_reward, std_reward, rewards.
        """
        # TODO: nested loop; force epsilon=0 for the trained agent; render each step
        raise NotImplementedError
