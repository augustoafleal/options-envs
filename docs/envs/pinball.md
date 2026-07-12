# Pinball

## Gymnasium ID

`OptionsEnv/Pinball-v0`

## Minimal usage

```python
import gymnasium as gym
import options_envs

env = gym.make("OptionsEnv/Pinball-v0")
obs, info = env.reset(seed=0)
```

## Available task

- `default-v0`

## Runnable examples

```bash
python examples/pinball/random_agent.py
python examples/pinball/render.py
python examples/pinball/render.py outputs/my_run.mp4
```

## Code location

`options_envs/envs/pinball/`

## Layouts and assets

`options_envs/envs/pinball/assets/layouts/`

## Main files

```text
env.py          implementation of the Gymnasium Pinball environment
tasks.py        versioned task definitions
registration.py registration of the OptionsEnv/Pinball-v0 ID
assets/layouts/ environment layouts/configurations
```

## Background and references

The Pinball environment in this project is based on the classic Pinball domain used in reinforcement learning and option-learning research.

Relevant references:

- Konidaris / Brown IRL Pinball domain: http://irl.cs.brown.edu/pinball/
- Pierre-Luc Bacon's Python RL implementation/reference: https://github.com/amarack/python-rl/tree/master
