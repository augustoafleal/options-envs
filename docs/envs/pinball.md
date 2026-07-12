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

| File | Description |
|------|-------------|
| `env.py` | Implementation of the Gymnasium Pinball environment |
| `tasks.py` | Versioned task definitions |
| `registration.py` | Registration of the `OptionsEnv/Pinball-v0` ID |
| `assets/layouts/` | Environment layouts and configurations |

## Default configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `task` | `default-v0` | Task identifier |
| `max_steps` | `500` | Episode horizon |
| `step_penalty` | `-1.0` | Reward per step without thrust |
| `thrust_penalty` | `-5.0` | Reward per step with thrust |
| `success_reward` | `10000.0` | Reward upon reaching the target |
| `action_space` | `Discrete(5)` | 0=right, 1=down, 2=left, 3=up, 4=no-op |
| `observation_space` | `Box(4,)` | `[x, y, xdot, ydot]` — positions in `[0,1]`, velocities in `[-1,1]` |

## Render modes

| Mode | Description |
|------|-------------|
| `None` | No rendering (fastest) |
| `"rgb_array"` | Returns an `(H, W, 3)` numpy array — suitable for recording or headless environments |
| `"human"` | Opens a Pygame window displaying the pinball table in real time |

## Background and references

The Pinball environment in this project is based on the classic Pinball domain used in reinforcement learning and option-learning research.

Relevant references:

- Konidaris / Brown IRL Pinball domain: http://irl.cs.brown.edu/pinball/
- Pierre-Luc Bacon's Python RL implementation/reference: https://github.com/amarack/python-rl/tree/master
