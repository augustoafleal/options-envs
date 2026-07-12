import gymnasium as gym
import numpy as np
import options_envs


def test_pinball_contract():
    env = gym.make("OptionsEnv/Pinball-v0")

    try:
        obs, info = env.reset(seed=0)

        assert isinstance(info, dict)
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (4,)
        assert obs.dtype == np.float32
        assert env.observation_space.contains(obs)

        action = env.action_space.sample()
        result = env.step(action)

        assert len(result) == 5

        obs, reward, terminated, truncated, info = result

        assert isinstance(obs, np.ndarray)
        assert obs.shape == (4,)
        assert obs.dtype == np.float32
        assert env.observation_space.contains(obs)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
    finally:
        env.close()


def test_pinball_state_obs_type():
    env = gym.make("OptionsEnv/Pinball-v0", obs_type="state")

    try:
        obs, info = env.reset(seed=0)

        assert isinstance(obs, np.ndarray)
        assert obs.shape == (4,)
        assert obs.dtype == np.float32
        assert env.observation_space.contains(obs)

        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        assert obs.shape == (4,)
        assert obs.dtype == np.float32
        assert env.observation_space.contains(obs)
    finally:
        env.close()


def test_pinball_render_rgb_array():
    env = gym.make("OptionsEnv/Pinball-v0", render_mode="rgb_array")

    try:
        obs, info = env.reset(seed=0)

        assert isinstance(obs, np.ndarray)
        assert obs.shape == (4,)
        assert obs.dtype == np.float32

        frame = env.render()
        assert isinstance(frame, np.ndarray)
        assert frame.dtype == np.uint8
        assert frame.shape == (600, 800, 3)

        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        assert obs.shape == (4,)
        assert obs.dtype == np.float32
    finally:
        env.close()
