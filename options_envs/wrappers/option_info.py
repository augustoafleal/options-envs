import gymnasium as gym


class OptionInfoWrapper(gym.Wrapper):
    DEFAULT_KEYS = (
        "position",
        "goal",
        "is_success",
        "terminal_reason",
        "available_options",
        "option_mask",
        "subgoal",
        "room_id",
        "landmark_id",
    )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, self._normalize_info(info)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, self._normalize_info(info)

    def _normalize_info(self, info):
        info = dict(info)
        for key in self.DEFAULT_KEYS:
            info.setdefault(key, None)
        return info
