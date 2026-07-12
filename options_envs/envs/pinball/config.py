from pathlib import Path

from options_envs.envs.pinball.ball import BallModel
from options_envs.envs.pinball.obstacle import PinballObstacle


def load_configuration(path, rng):
    path = Path(path)
    obstacles = []
    target_pos = None
    target_rad = 0.01
    ball_rad = 0.01
    start_positions = []

    with path.open(encoding="utf-8") as file:
        for line in file:
            tokens = line.strip().split()
            if not tokens:
                continue

            if tokens[0] == "polygon":
                pts = list(zip(*[iter(map(float, tokens[1:]))] * 2))
                obstacles.append(PinballObstacle(pts))
            elif tokens[0] == "target":
                target_pos = [float(tokens[1]), float(tokens[2])]
                target_rad = float(tokens[3])
            elif tokens[0] == "start":
                start_positions = list(zip(*[iter(map(float, tokens[1:]))] * 2))
            elif tokens[0] == "ball":
                ball_rad = float(tokens[1])

    if target_pos is None:
        raise ValueError(f"Missing target definition in layout: {path}")
    if not start_positions:
        raise ValueError(f"Missing start definition in layout: {path}")

    start_index = int(rng.integers(len(start_positions)))
    ball = BallModel(start_positions[start_index], ball_rad)
    return ball, obstacles, target_pos, target_rad
