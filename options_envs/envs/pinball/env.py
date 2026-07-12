from importlib.resources import files

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

from options_envs.envs.pinball.config import load_configuration
from options_envs.envs.pinball.render import render_rgb
from options_envs.envs.pinball.tasks import TASKS


class PinballEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, task="default-v0", render_mode=None, max_steps=None, obs_type="state"):
        super().__init__()

        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render_mode: {render_mode}")
        if task not in TASKS:
            raise ValueError(f"Unknown Pinball task: {task}")
        if obs_type not in ("state",):
            raise ValueError(f"Unsupported obs_type: {obs_type}")

        task_config = TASKS[task]
        layout_name = task_config["layout"]

        self.task = task
        self.render_mode = render_mode
        self.obs_type = obs_type
        self.layout_path = files("options_envs.envs.pinball").joinpath("assets", "layouts", layout_name)
        self.max_episode_steps = int(max_steps if max_steps is not None else task_config["max_steps"])

        self.step_penalty = -1.0
        self.thrust_penalty = -5.0
        self.success_reward = 10000.0

        self.steps = 0
        self.ball = None
        self.obstacles = []
        self.target_pos = None
        self.target_rad = 0.0
        self._window = None

        self.action_effects = {
            0: (1, 0),
            1: (0, 1),
            2: (-1, 0),
            3: (0, -1),
            4: (0, 0),
        }

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        del options
        super().reset(seed=seed)

        self.ball, self.obstacles, self.target_pos, self.target_rad = load_configuration(self.layout_path, self.np_random)
        self.steps = 0

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return observation, info

    def step(self, action):
        if self.ball is None:
            raise RuntimeError("PinballEnv.step() called before reset().")

        self.steps += 1
        reward = float(self._take_action(int(action)))
        terminated = bool(self._reached_target())
        truncated = bool(self.steps >= self.max_episode_steps and not terminated)
        observation = self._get_obs()
        info = self._get_info(terminated=terminated, truncated=truncated)

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.ball is None:
            return None

        pygame.init()
        image = render_rgb(self.ball, self.obstacles, self.target_pos, self.target_rad)

        if self.render_mode == "rgb_array":
            return image

        if self.render_mode == "human":
            if self._window is None:
                self._window = pygame.display.set_mode((800, 600))
                pygame.display.set_caption("Pinball")

            surface = pygame.surfarray.make_surface(np.transpose(image, (1, 0, 2)))
            self._window.blit(surface, (0, 0))
            pygame.display.flip()

        return None

    def close(self):
        if self._window is not None:
            pygame.display.quit()
            self._window = None
        pygame.quit()

    def _take_action(self, action):
        for i in range(20):
            if i == 0:
                self.ball.add_impulse(*self.action_effects[action])

            self.ball.step()

            delta_velocity = np.zeros(2)
            hits = 0
            for obstacle in self.obstacles:
                if obstacle.collision(self.ball):
                    delta_velocity += obstacle.collision_effect(self.ball)
                    hits += 1

            if hits == 1:
                self.ball.xdot, self.ball.ydot = delta_velocity
                if i == 19:
                    self.ball.step()
            elif hits > 1:
                self.ball.xdot = -1
                self.ball.ydot = -1

            if self._reached_target():
                return self.success_reward

        self.ball.add_drag()
        self._check_bounds()
        return self.step_penalty if action == 4 else self.thrust_penalty

    def _get_obs(self):
        return np.array(
            [
                self.ball.position[0],
                self.ball.position[1],
                self.ball.xdot,
                self.ball.ydot,
            ],
            dtype=np.float32,
        )

    def _get_info(self, terminated=False, truncated=False):
        terminal_reason = None
        if terminated:
            terminal_reason = "goal"
        elif truncated:
            terminal_reason = "max_steps"

        return {
            "position": self._to_native_pair(self.ball.position),
            "goal": self._to_native_pair(self.target_pos),
            "is_success": terminated,
            "terminal_reason": terminal_reason,
        }

    def _reached_target(self):
        distance = np.linalg.norm(np.array(self.ball.position) - np.array(self.target_pos))
        return distance < self.target_rad

    def _check_bounds(self):
        self.ball.position[0] = np.clip(self.ball.position[0], 0.01, 1.0)
        self.ball.position[1] = np.clip(self.ball.position[1], 0.01, 1.0)

    @staticmethod
    def _to_native_pair(values):
        if values is None:
            return None
        return (float(values[0]), float(values[1]))
