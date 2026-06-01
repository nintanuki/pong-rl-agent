# TODO

## Decisions

- [x] Game: Pong (pivoted from Snake for simpler controls and a denser reward signal)
- [x] Game engine: distilled from ClearCode's Pong tutorial, stripped of audio / music / sprites / controller / fullscreen / pause / CRT / inter-round countdown
- [x] Human test driver (`play_human.py`) so the engine can be sanity-checked before plugging in the env wrapper
- [x] Primary algorithm: tabular Q-learning
- [x] DQN comparator: included
- [x] Reward shaping: paddle-hit bonus ON (`Reward.USE_HIT_SHAPING = True`)

## Environment

- [ ] Implement `PongEnv.__init__()` (declare `observation_space` as Box(6,) and `action_space` as Discrete(3))
- [ ] Implement `PongEnv.reset()`
- [ ] Implement `PongEnv.step()` (delegate to `GameEngine.step`, map scored / agent_hit to reward, build gym tuple)
- [ ] Implement `PongEnv._observe()` (6-value vector: ball x/y, ball vx/vy, agent paddle y, opponent paddle y; normalized)
- [ ] Implement `PongEnv.render()` (pygame; rectangles only)

## Agent

- [ ] Implement `QLearningAgent.select_action()` (epsilon-greedy)
- [ ] Implement `QLearningAgent.update()` (Bellman update)
- [ ] Implement `QLearningAgent.decay_epsilon()`
- [ ] Implement `QLearningAgent.save()` / `load()` (pickle the Q-table)
- [ ] Decide and implement the state discretization used to key the Q-table
- [ ] Implement `DQNAgent` (replay buffer + target network)

## Training and evaluation

- [ ] Implement `Trainer.train()` (episode loop, logging, periodic checkpoints)
- [ ] Implement `Trainer.plot_rewards()`
- [ ] Implement `Evaluator.run_agent()` (load checkpoint, render, report mean / max / std)
- [ ] Implement the random-action baseline path
- [ ] Implement `Application.train()` and `Application.evaluate()` in `main.py`

## Report

- [ ] Fill out every italicized TODO placeholder in `docs/REPORT.docx`
- [ ] Capture training-reward curve and gameplay screenshots for the Results section
- [x] Cite ClearCode's Pong tutorial (https://www.youtube.com/watch?v=Qf3-aDXG8q4) as the upstream lineage of the engine — counts as one of the 3 outside sources
- [ ] Add at least 2 more outside sources to references (suggested: Gymnasium docs + DQN Nature paper)
- [ ] Run a final APA formatting + grammar pass

## Submission

- [ ] Freeze `requirements.txt` with `pip freeze`
- [ ] Verify README setup steps work from a fresh clone
- [ ] Push to public GitHub repo; submit the link
