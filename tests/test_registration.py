import gymnasium as gym
import options_envs


def test_make_pinball_env():
    env = gym.make("OptionsEnv/Pinball-v0")
    try:
        assert env is not None
    finally:
        env.close()
