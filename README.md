# Pong Reinforcement Learning Agent

A reinforcement learning agent that learns to play Pong through trial and error in a simulated environment.

## Project layout

```
.
├── docs/           Project report and AI use documentation
├── src/
│   ├── output/              Training checkpoints and reward-curve plots
│   ├── main.py              Entry point with `--mode {train, evaluate}`
│   ├── settings.py          Namespaced hyperparameters
│   ├── game_engine.py       Pure-logic Pong state machine
│   ├── pong_env.py          Gymnasium-compatible environment wrapper
│   ├── play_human.py        Manual keyboard driver for sanity-checking the engine
│   ├── q_learning_agent.py  Tabular Q-learning agent (primary)
│   ├── dqn_agent.py         Deep Q-Network agent (comparator)
│   ├── trainer.py           Episode loop, checkpoints, reward curve
│   └── evaluator.py         Trained agent vs. random baseline
├── README.md
├── TODO.md
├── requirements.txt
└── .gitignore
```

## Architecture

`GameEngine` is the source of truth for Pong gameplay and has zero external dependencies. It models two paddles, a ball with velocity, wall and paddle collisions, and the score; the agent controls the right-hand paddle while a simple scripted AI controls the left one. `PongEnv` wraps the engine in the `gymnasium.Env` interface so any RL library can drive it; it also owns the optional pygame render layer used during evaluation. The action space is `Discrete(3)` (up, stay, down) and the observation is a 6-value vector (ball x, ball y, ball velocity x/y, agent paddle y, opponent paddle y). Two agents are trained and compared: `QLearningAgent` learns a Q-table over a discretized version of that observation, and `DQNAgent` approximates the same Q-function with a small PyTorch MLP using experience replay and a target network over the raw vector. `Trainer` runs the episode loop headless during training and writes checkpoints plus a reward curve. `Evaluator` loads a checkpoint, renders the trained agent at play, and compares its performance against a random-action baseline.

The Pong gameplay logic was distilled from ClearCode's tutorial [Learning pygame by making Pong](https://www.youtube.com/watch?v=Qf3-aDXG8q4), stripped of audio, music, controller input, fullscreen, pause, the CRT overlay, the inter-round countdown, and sprite assets so it can run headless.

## Setup

```powershell
# from the project root
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On macOS / Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
cd src
# Train a new agent
python main.py --mode train

# Watch a previously trained agent play
python main.py --mode evaluate

# Play Pong yourself to sanity-check the engine (Up / Down arrows)
python play_human.py
```

Checkpoints land in `src/output/checkpoints/` and the training reward curve in `src/output/plots/`. Both directories are gitignored.
