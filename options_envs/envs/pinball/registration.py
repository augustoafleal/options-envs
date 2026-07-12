from gymnasium.envs.registration import register, registry


def register_pinball_envs():
    env_id = "OptionsEnv/Pinball-v0"
    if env_id not in registry:
        register(
            id=env_id,
            entry_point="options_envs.envs.pinball.env:PinballEnv",
            kwargs={"task": "default-v0"},
        )
