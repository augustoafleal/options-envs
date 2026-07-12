import numpy as np
import pygame

SCREEN_W = 800
SCREEN_H = 600


def render_rgb(ball, obstacles, target_pos, target_rad):
    surface = pygame.Surface((SCREEN_W, SCREEN_H))
    surface.fill((255, 255, 255))

    for obs in obstacles:
        points = [(x * SCREEN_W, y * SCREEN_H) for x, y in obs.get_points()]
        pygame.draw.polygon(surface, (80, 80, 80), points)

    pygame.draw.circle(
        surface,
        (0, 0, 255),
        (int(ball.position[0] * SCREEN_W), int(ball.position[1] * SCREEN_H)),
        int(ball.radius * SCREEN_W),
    )

    pygame.draw.circle(
        surface,
        (255, 0, 0),
        (int(target_pos[0] * SCREEN_W), int(target_pos[1] * SCREEN_H)),
        int(target_rad * SCREEN_W),
    )

    image = pygame.surfarray.array3d(surface)
    return np.transpose(image, (1, 0, 2))
