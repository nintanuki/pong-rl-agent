# Snake Reinforcement Learning Agent

A reinforcement learning agent that learns to play Snake through trial and error in a simulated environment.

## Project layout

```
.
├── docs/           Project report and AI use documentation
├── references/     Reference materials
├── rubric/         Source assignment requirements
├── src/
│   ├── output/              Training checkpoints and reward-curve plots
│   ├── main.py              Entry point with `--mode {train, evaluate}`
│   ├── settings.py          Namespaced hyperparameters
│   ├── game_engine.py       Pure-logic Snake state machine
│   ├── snake_env.py         Gymnasium-compatible environment wrapper
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

`GameEngine` is the source of truth for Snake gameplay and has zero external dependencies. `SnakeEnv` wraps the engine in the `gymnasium.Env` interface so any RL library can drive it; it also owns the optional pygame render layer used during evaluation. Two agents are trained and compared: `QLearningAgent` learns a Q-table over the 11-boolean state vector (danger detection, current direction, food direction), and `DQNAgent` approximates the same Q-function with a small PyTorch MLP using experience replay and a target network. `Trainer` runs the episode loop headless during training and writes checkpoints plus a reward curve. `Evaluator` loads a checkpoint, renders the trained agent at play, and compares its performance against a random-action baseline.

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
```

Checkpoints land in `src/output/checkpoints/` and the training reward curve in `src/output/plots/`. Both directories are gitignored.
