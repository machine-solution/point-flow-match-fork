#!/usr/bin/env python3
"""
Разбивает zarr-датасет демонстраций (RobotReplayBuffer / diffusion_policy ReplayBuffer)
на два датасета по захвату (гриппер закрыт), с опцией "устойчивого закрытия".

Определение "кадр захвата":
  - находим первый индекс t, где gripper < thr
  - если указать --closed-steps N>1, то ищем первый индекс t,
    где gripper < thr сохраняется N шагов подряд, и берём t как
    *последний* кадр этой устойчивой последовательности (т.е. в окно входит t-N+1..t).

  • pre (до и включая кадр захвата): кадры [0 .. t]  → срез [: t + 1]
  • post (от кадра захвата): кадры [t .. T-1]        → срез [t :]

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


def first_grasp_index(
    robot_state: np.ndarray, *, thr: float, closed_steps: int = 1
) -> int | None:
    """
    Первый индекс t, который считается "захватом".

    - closed_steps=1: первый шаг, где gripper < thr.
    - closed_steps=N>1: первый шаг, где gripper < thr держится N шагов подряд.
      Возвращаем t = (start + N - 1), т.е. индекс ПОСЛЕДНЕГО кадра устойчивого окна.
    """
    closed_steps = int(closed_steps)
    if closed_steps < 1:
        raise ValueError(f"closed_steps must be >= 1, got {closed_steps}")
    g = np.asarray(robot_state, dtype=np.float64)[:, 9].ravel()
    closed = g < thr
    T = int(closed.shape[0])
    if T == 0:
        return None
    if closed_steps == 1:
        if not np.any(closed):
            return None
        return int(np.argmax(closed))

    # Find first window of length N with all True, return last index of that window.
    # This is O(T*N) but T is small (episode length) so it's fine and explicit.
    N = closed_steps
    for start in range(0, T - N + 1):
        if bool(np.all(closed[start : start + N])):
            return int(start + N - 1)
    return None


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
        "--closed-steps",
        type=int,
        default=1,
        help=(
            "Сколько шагов подряд gripper должен быть < thr, чтобы считать захват устойчивым. "
            "1 = как раньше (первое закрытие). 3 = 'устойчивое закрытие' как в TwoPhasePolicy."
        ),
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
    closed_steps = int(args.closed_steps)

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
        "closed_steps": closed_steps,
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
        t = first_grasp_index(rs, thr=thr, closed_steps=closed_steps)

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

    suffix = f"_stable{closed_steps}" if closed_steps != 1 else ""
    manifest_path = out_pre.parent / f"{out_pre.name}{suffix}_split_manifest.json"
    manifest_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(
        f"Готово: pre {stats['pre_episodes']} эпизодов → {out_pre}\n"
        f"        post {stats['post_episodes']} эпизодов → {out_post}\n"
        f"        без захвата (только pre): {stats['no_grasp_pre_only']}\n"
        f"Манифест: {manifest_path}"
    )


if __name__ == "__main__":
    main()
