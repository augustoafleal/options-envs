import argparse
from pathlib import Path
import sys

import gymnasium as gym
import numpy as np
import options_envs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", nargs="?", default="outputs/pinball.mp4")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=10)
    return parser.parse_args()


def open_video_writer(output_path, fps):
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise RuntimeError(
            "Could not generate the render video because `imageio` is not installed. "
            'Install the dev dependencies with `pip install -e ".[dev]"`.'
        ) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        return imageio.get_writer(
            output_path,
            fps=fps,
            codec="libx264",
            format="FFMPEG",
            macro_block_size=1,
        )
    except Exception as exc:
        raise RuntimeError(f"Could not open the render video output at `{output_path}`: {exc}") from exc


def main():
    args = parse_args()
    output_path = Path(args.output)
    env = gym.make("OptionsEnv/Pinball-v0", render_mode="rgb_array")
    writer = None
    frame_count = 0

    try:
        obs, info = env.reset(seed=args.seed)
        del obs, info

        for step in range(args.steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            del obs, reward, info

            frame = None
            try:
                frame = env.render()
            except Exception:
                frame = None

            if frame is not None:
                frame = np.asarray(frame, dtype=np.uint8)
                if writer is None:
                    writer = open_video_writer(output_path, args.fps)
                writer.append_data(frame)
                frame_count += 1
                print(f"step={step} frame_shape={frame.shape}")

            if terminated or truncated:
                obs, info = env.reset()
                del obs, info
    finally:
        if writer is not None:
            writer.close()
        env.close()

    if frame_count == 0:
        raise RuntimeError(
            "Could not generate the render video because the environment did not return any frames."
        )

    print(f"saved_video={output_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
