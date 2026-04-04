#!/usr/bin/env python3
"""
Разбивает zarr-датасет демонстраций (RobotReplayBuffer / diffusion_policy ReplayBuffer)
на два датасета по первому захвату (гриппер закрыт).

  • pre (до и включая первый кадр захвата): кадры [0 .. t]  → срез [: t + 1]
  • post (от первого захвата): кадры [t .. T-1]           → срез [t :]

Кадр с индексом t входит в ОБА датасета.

Эпизод без закрытия гриппера: целиком попадает только в pre; в post не добавляется.

Пример:
  python two_layers_planning/split_dataset_at_first_grasp.py \\
    --input demos/sim/open_fridge/train \\
    --output-pre demos/sim/open_fridge/train_pre_grasp \\
    --output-post demos/sim/open_fridge/train_post_grasp
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# diffusion_policy (ReplayBuffer) — как в demo_format_tools/zarr_to_json_ee.py
_DP = _REPO.parent / "diffusion_policy"
if _DP.is_dir() and str(_DP) not in sys.path:
    sys.path.insert(0, str(_DP))

from pfp.data.replay_buffer import RobotReplayBuffer


def first_grasp_index(robot_state: np.ndarray, *, thr: float) -> int | None:
    """Первый шаг, где gripper < thr (индекс 9 в robot_state). Если закрытия нет — None."""
    g = np.asarray(robot_state, dtype=np.float64)[:, 9].ravel()
    closed = g < thr
    if not np.any(closed):
        return None
    return int(np.argmax(closed))


def _episode_slice(ep: dict[str, np.ndarray], start: int, stop: int) -> dict[str, np.ndarray]:
    sl = slice(start, stop)
    out: dict[str, np.ndarray] = {}
    for key, arr in ep.items():
        x = np.asarray(arr[sl])
        # независимая копия для записи в новый буфер
        if not x.flags.writeable or isinstance(x, np.memmap):
            x = x.copy()
        out[key] = x
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Каталог исходного zarr (data/meta), как у collect_demos",
    )
    ap.add_argument(
        "--output-pre",
        type=Path,
        required=True,
        help="Каталог для датасета до и включая кадр первого захвата",
    )
    ap.add_argument(
        "--output-post",
        type=Path,
        required=True,
        help="Каталог для датасета от первого захвата (тот же кадр t — первый в post)",
    )
    ap.add_argument(
        "--gripper-thr",
        type=float,
        default=0.5,
        help="Порог: значение gripper (канал 9) < thr считается закрытым",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Удалить output-pre / output-post, если они уже существуют",
    )
    args = ap.parse_args()

    inp = args.input.expanduser().resolve()
    out_pre = args.output_pre.expanduser().resolve()
    out_post = args.output_post.expanduser().resolve()
    thr = float(args.gripper_thr)

    if not inp.is_dir():
        raise FileNotFoundError(f"Нет каталога: {inp}")

    for p, name in [(out_pre, "output-pre"), (out_post, "output-post")]:
        if p.exists():
            if not args.overwrite:
                raise FileExistsError(
                    f"{name} уже существует: {p}. Удалите вручную или укажите --overwrite"
                )
            shutil.rmtree(p)

    src = RobotReplayBuffer.create_from_path(str(inp), mode="r")
    n_ep = src.n_episodes
    if n_ep == 0:
        raise ValueError(f"В источнике нет эпизодов: {inp}")

    rb_pre = RobotReplayBuffer.create_from_path(str(out_pre), mode="w")
    rb_post = RobotReplayBuffer.create_from_path(str(out_post), mode="w")

    stats = {
        "source": str(inp),
        "output_pre": str(out_pre),
        "output_post": str(out_post),
        "n_source_episodes": n_ep,
        "gripper_thr": thr,
        "pre_episodes": 0,
        "post_episodes": 0,
        "no_grasp_pre_only": 0,
    }

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None  # type: ignore

    it = range(n_ep)
    if tqdm is not None:
        it = tqdm(it, desc="episodes", total=n_ep)

    for ep_idx in it:
        ep = src.get_episode(ep_idx, copy=True)
        if "robot_state" not in ep:
            raise KeyError(f"Эпизод {ep_idx}: нет ключа robot_state")

        rs = ep["robot_state"]
        T = int(rs.shape[0])
        t = first_grasp_index(rs, thr=thr)

        if t is None:
            rb_pre.add_episode(_episode_slice(ep, 0, T))
            stats["pre_episodes"] += 1
            stats["no_grasp_pre_only"] += 1
            continue

        # pre: [0 .. t] включительно
        pre = _episode_slice(ep, 0, t + 1)
        rb_pre.add_episode(pre)
        stats["pre_episodes"] += 1

        # post: [t .. T-1], кадр t дублируется из pre
        post = _episode_slice(ep, t, T)
        assert post["robot_state"].shape[0] >= 1
        rb_post.add_episode(post)
        stats["post_episodes"] += 1

    manifest_path = out_pre.parent / f"{out_pre.name}_split_manifest.json"
    manifest_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(
        f"Готово: pre {stats['pre_episodes']} эпизодов → {out_pre}\n"
        f"        post {stats['post_episodes']} эпизодов → {out_post}\n"
        f"        без захвата (только pre): {stats['no_grasp_pre_only']}\n"
        f"Манифест: {manifest_path}"
    )


if __name__ == "__main__":
    main()
