import gymnasium as gym
import options_envs


def test_pinball_contract():
    env = gym.make("OptionsEnv/Pinball-v0")

    try:
        obs, info = env.reset(seed=0)

        assert isinstance(info, dict)
        assert env.observation_space.contains(obs)

        action = env.action_space.sample()
        result = env.step(action)

        assert len(result) == 5

        obs, reward, terminated, truncated, info = result

        assert env.observation_space.contains(obs)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
    finally:
        env.close()
