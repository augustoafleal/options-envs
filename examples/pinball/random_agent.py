import gymnasium as gym
import options_envs


def main():
    env = gym.make("OptionsEnv/Pinball-v0")

    try:
        for episode in range(3):
            obs, info = env.reset(seed=episode)
            del obs, info

            total_reward = 0.0
            done = False
            while not done:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                del obs, info

                total_reward += reward
                done = terminated or truncated

            print(f"episode={episode} total_reward={total_reward:.2f}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
