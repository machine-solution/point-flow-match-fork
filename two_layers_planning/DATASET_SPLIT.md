## Dataset split: pre-grasp / post-grasp

В этом репозитории датасет демонстраций (zarr / `RobotReplayBuffer`) можно разбить на две части:

- **pre-grasp**: до захвата (и включая кадр захвата)
- **post-grasp**: после захвата (начиная с того же кадра захвата)

Идея: отдельная модель учится *подвести руку и закрыть гриппер*, а вторая — *делать то, что нужно после захвата* (для `open_fridge`: тянуть дверь).

### Формат данных

Скрипт читает эпизоды из zarr-буфера и использует массив `robot_state` формы `(T, 10)`.

- `robot_state[:, 9]` — скаляр **gripper_open** (меньше = более закрыт).

### Как определяется “кадр захвата”

Порог задаётся `--gripper-thr` (по умолчанию `0.5`).

Есть два режима:

- **Первое закрытие** (`--closed-steps 1`):
  - берём первый индекс \(t\), где `robot_state[t, 9] < thr`.
- **Устойчивое закрытие** (`--closed-steps N`, где `N > 1`):
  - ищем первое окно длины `N`, где `robot_state[k:k+N, 9] < thr` для всех шагов окна
  - считаем “кадром захвата” **последний кадр окна**: \(t = k + N - 1\)

Почему это важно: “первое закрытие” может случиться **до** устойчивого контакта/фиксации на ручке (особенно в симуляции), из-за чего `post`-датасет стартует слишком рано и содержит много микро-движений/стабилизации.

### Как режется эпизод

Кадр `t` входит в оба датасета:

- **pre**: кадры `[0 .. t]` (включительно) → Python slice `[:t+1]`
- **post**: кадры `[t .. T-1]` → Python slice `[t:]`

Эпизод, где захвата нет:

- целиком попадает только в **pre**
- в **post** не добавляется

### Команды

Пример: “устойчивое закрытие 3 шага подряд” (аналогично `closed_steps_to_switch=3` в `TwoPhasePolicy`):

```bash
python3 two_layers_planning/split_dataset_at_first_grasp.py \
  --input demos/sim/open_fridge/train \
  --output-pre demos/sim/open_fridge/train_pre_grasp_stable3 \
  --output-post demos/sim/open_fridge/train_post_grasp_stable3 \
  --gripper-thr 0.5 \
  --closed-steps 3
```

Манифест с параметрами сплита и статистикой сохраняется рядом с `output-pre`:

- `<output-pre-name>_stable<closed_steps>_split_manifest.json`

### Готовые архивы (stable3)

- **pre**: `https://disk.yandex.ru/d/koQoDdaJ-t4b8A`
- **post**: `https://disk.yandex.ru/d/U4bBHq7LZuBg9A`

