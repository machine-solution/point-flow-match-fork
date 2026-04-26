#!/usr/bin/env python3
"""
Render simple RGB videos for recorded episodes (headless).

This avoids GUI playback issues by running RLBench headless and saving camera frames.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from pfp import set_seeds
from pfp.envs.rlbench_env import RLBenchEnv


def _ensure_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-file", type=Path, required=True)
    ap.add_argument("--episodes", type=str, required=True, help="Comma-separated episode indices, e.g. 1,4,8")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--camera", type=str, default="front", choices=["right", "left", "overhead", "front", "wrist"])
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--max-frames", type=int, default=250, help="Hard cap to avoid huge videos")
    args = ap.parse_args()

    cam_to_idx = {"right": 0, "left": 1, "overhead": 2, "front": 3, "wrist": 4}
    cam_idx = cam_to_idx[args.camera]

    data = json.loads(args.input_file.read_text())
    env_cfg = data.get("env_config", {})

    # Force rgb mode for rendering
    env = RLBenchEnv(
        task_name=env_cfg.get("task_name", "open_fridge"),
        voxel_size=float(env_cfg.get("voxel_size", 0.01)),
        n_points=int(env_cfg.get("n_points", 4096)),
        use_pc_color=bool(env_cfg.get("use_pc_color", False)),
        headless=True,
        vis=False,
        obs_mode="rgb",
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    episode_indices = [int(x.strip()) for x in args.episodes.split(",") if x.strip()]
    for ep_i in episode_indices:
        ep = data["episodes"][ep_i]
        actions = ep["actions"]
        episode_seed = ep.get("episode_seed", None)

        if episode_seed is not None:
            set_seeds(int(episode_seed))
        env.reset()
        frames: list[np.ndarray] = []
        actual_success = False

        for step, action in enumerate(actions):
            if step >= int(args.max_frames):
                break
            _, obs = env.get_obs()
            frame = _ensure_uint8(obs[cam_idx])
            frames.append(frame)

            reward, terminate = env.step(np.array(action, dtype=np.float32))
            if bool(reward):
                actual_success = True
            if bool(reward) or bool(terminate):
                # capture one more frame after termination for context
                try:
                    _, obs2 = env.get_obs()
                    frames.append(_ensure_uint8(obs2[cam_idx]))
                except Exception:
                    pass
                break

        recorded_success = bool(ep.get("success", False))
        suffix = "success" if actual_success else "fail"
        rec_tag = "recOK" if recorded_success else "recFAIL"
        out_path = args.out_dir / f"{args.input_file.stem}_ep{ep_i:03d}_{suffix}_{rec_tag}_{args.camera}.mp4"
        iio.imwrite(out_path, np.stack(frames, axis=0), fps=int(args.fps))
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

