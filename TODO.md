# TODO

## Decisions

- [x] Game: Snake
- [x] Game engine: ported from arcade-cabinet, stripped of audio / sprites / input / pause / menu
- [x] Human test driver (`play_human.py`) so the engine can be sanity-checked before plugging in the env wrapper
- [x] Primary algorithm: tabular Q-learning
- [x] DQN comparator: included
- [x] Reward shaping: distance bonus ON (`Settings.Reward.USE_DISTANCE_SHAPING = True`)

## Environment

- [ ] Implement `SnakeEnv.__init__()` (declare `observation_space` and `action_space`)
- [ ] Implement `SnakeEnv.reset()`
- [ ] Implement `SnakeEnv.step()` (delegate to `GameEngine.step`, compute reward, build gym tuple)
- [ ] Implement `SnakeEnv._observe()` (11-boolean state vector: danger flags + direction + food direction)
- [ ] Implement `SnakeEnv.render()` (pygame; rectangles only)

## Agent

- [ ] Implement `QLearningAgent.select_action()` (epsilon-greedy)
- [ ] Implement `QLearningAgent.update()` (Bellman update)
- [ ] Implement `QLearningAgent.decay_epsilon()`
- [ ] Implement `QLearningAgent.save()` / `load()` (pickle the Q-table)
- [ ] Implement `DQNAgent` (replay buffer + target network)

## Training and evaluation

- [ ] Implement `Trainer.train()` (episode loop, logging, periodic checkpoints)
- [ ] Implement `Trainer.plot_rewards()`
- [ ] Implement `Evaluator.run_agent()` (load checkpoint, render, report mean / max / std)
- [ ] Implement the random-action baseline path
- [ ] Implement `Application.train()` and `Application.evaluate()` in `main.py`

## Report

- [ ] Fill out every `[TODO: ...]` placeholder in `docs/REPORT.docx`
- [ ] Capture training-reward curve and gameplay screenshots for the Results section
- [x] Cite Clear Code's Snake tutorial (https://www.youtube.com/watch?v=QFvqStqPCRU) as the upstream lineage of the engine — counts as one of the 3 outside sources
- [ ] Add at least 2 more outside sources to references (suggested: Gymnasium docs + DQN Nature paper)
- [ ] Run Gemini APA formatting + grammar pass

## Submission

- [ ] Freeze `requirements.txt` with `pip freeze`
- [ ] Verify README setup steps work from a fresh clone
- [ ] Push to public GitHub repo; submit the link
