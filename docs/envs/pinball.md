# Pinball

## Gymnasium ID

`OptionsEnv/Pinball-v0`

## Observation

By default, `OptionsEnv/Pinball-v0` returns a continuous **state vector** as observation (`obs_type="state"`). RGB imagery is available exclusively through `render()` with `render_mode="rgb_array"` — it is not the default observation.

### State vector

The observation is a `numpy.ndarray` of shape `(4,)` and dtype `np.float32`:

```
[x, y, vx, vy]
```

| Index | Field | Range |
|-------|-------|-------|
| `0` | `x` — horizontal position | `[0, 1]` |
| `1` | `y` — vertical position | `[0, 1]` |
| `2` | `vx` — horizontal velocity | `[-1, 1]` |
| `3` | `vy` — vertical velocity | `[-1, 1]` |

```python
import gymnasium as gym
import options_envs

# Default: state observation
env = gym.make("OptionsEnv/Pinball-v0")
obs, info = env.reset(seed=0)

print(obs.shape) # (4,)
print(obs.dtype) # float32
```

Explicit `obs_type` parameter:

```python
env = gym.make("OptionsEnv/Pinball-v0", obs_type="state")
obs, info = env.reset(seed=0)
```

### RGB rendering (not observation)

To obtain RGB frames for recording or visualization, use `render_mode="rgb_array"`. The observation **remains** the state vector; only `render()` returns an image.

```python
env = gym.make("OptionsEnv/Pinball-v0", render_mode="rgb_array")
obs, info = env.reset(seed=0)
frame = env.render() # (600, 800, 3) uint8

# obs is still the state vector
print(obs.shape) # (4,)
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
| `obs_type` | `"state"` | Observation type (currently only `"state"`) |
| `step_penalty` | `-1.0` | Reward per step without thrust |
| `thrust_penalty` | `-5.0` | Reward per step with thrust |
| `success_reward` | `10000.0` | Reward upon reaching the target |
| `action_space` | `Discrete(5)` | 0=right, 1=down, 2=left, 3=up, 4=no-op |
| `observation_space` | `Box(4,)` | `[x, y, vx, vy]` — positions in `[0,1]`, velocities in `[-1,1]` |

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
