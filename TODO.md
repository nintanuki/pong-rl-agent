# TODO

## Decisions

- [x] Game: Pong (pivoted from Snake for simpler controls and a denser reward signal)
- [x] Game engine: distilled from ClearCode's Pong tutorial, stripped of audio / music / sprites / controller / fullscreen / pause / CRT / inter-round countdown
- [x] Human test driver (`play_human.py`) so the engine can be sanity-checked before plugging in the env wrapper
- [x] Primary algorithm: tabular Q-learning
- [x] DQN comparator: included
- [x] Reward shaping: paddle-hit bonus ON (`Reward.USE_HIT_SHAPING = True`)

## Environment

- [x] Implement `PongEnv.__init__()` (declare `observation_space` as Box(6,) and `action_space` as Discrete(3))
- [x] Implement `PongEnv.reset()`
- [x] Implement `PongEnv.step()` (delegate to `GameEngine.step`, map scored / agent_hit to reward, build gym tuple)
- [x] Implement `PongEnv._observe()` (6-value vector: ball x/y, ball vx/vy, agent paddle y, opponent paddle y; normalized)
- [x] Implement `PongEnv.render()` (pygame; rectangles only)

## Agent

- [x] Implement `QLearningAgent.select_action()` (epsilon-greedy)
- [x] Implement `QLearningAgent.update()` (Bellman update)
- [x] Implement `QLearningAgent.decay_epsilon()`
- [x] Implement `QLearningAgent.save()` / `load()` (pickle the Q-table)
- [x] Decide and implement the state discretization used to key the Q-table (position bins + velocity sign, see `Discretization` in settings)
- [x] Implement `DQNAgent` (replay buffer + target network)

## Training and evaluation

- [x] Implement `Trainer.train()` (episode loop, logging, periodic checkpoints)
- [x] Implement `Trainer.plot_rewards()`
- [x] Implement `Evaluator.run_agent()` (load checkpoint, render, report mean / max / std)
- [x] Implement the random-action baseline path
- [x] Implement `Application.train()` and `Application.evaluate()` in `main.py`
- [ ] Run a full-length training run on the chosen settings and tune `OPPONENT_SPEED_PIXELS` from the curve

## Report

- [x] Fill the descriptive report sections (Name/Purpose, Algorithms, Environment, Libraries, Application Design, Instructions)
- [ ] Fill the remaining italicized TODOs in `docs/REPORT.docx` (Results, Discussion)
- [ ] Capture training-reward curve and gameplay screenshots for the Results section
- [x] Cite ClearCode's Pong tutorial (https://www.youtube.com/watch?v=Qf3-aDXG8q4) as the upstream lineage of the engine — counts as one of the 3 outside sources
- [ ] Add at least 2 more outside sources to references (suggested: Gymnasium docs + DQN Nature paper)
- [ ] Run a final APA formatting + grammar pass

## Submission

- [ ] Freeze `requirements.txt` with `pip freeze`
- [ ] Verify README setup steps work from a fresh clone
- [ ] Push to public GitHub repo; submit the link
