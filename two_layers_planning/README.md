# Двухфазное обучение `open_fridge` (до / после первого захвата)

Здесь — разрез исходного zarr по моменту первого закрытия гриппера и запуск обучения **двух отдельных политик**: pre-grasp и post-grasp. Отдельный **валидационный** датасет для этих конфигов не нужен: в `train_open_fridge_pre_grasp.yaml` / `train_open_fridge_post_grasp.yaml` включено `use_validation: false` (Composer не гоняет eval по эпохам).

## Какие датасеты вообще бывают для `open_fridge`

| Назначение | Каталоги / архивы |
|------------|-------------------|
| Классическое обучение одной политики (как в `dexter/run_pointflowmatch_open_fridge.sbatch`) | **`train`** + **`valid`** под `demos/sim/open_fridge/`. Скачивание одним архивом: `bash dexter/download_dataset.sh` из корня репо; публичная папка: [Я.Диск — полный сим-датасет](https://disk.yandex.ru/d/Ssr_BffZItISOg). |
| Двухфазное обучение (этот документ) | **`train_pre_grasp`** и **`train_post_grasp`** — скачать скриптом или разрезать `train`. **`valid` не используется** для конфигов `train_open_fridge_*_grasp`. |

**Скачать готовые pre/post с Яндекс.Диска (с кластера / виртуалки, из корня репо):**

```bash
cd /path/to/PointFlowMatch
bash dexter/download_open_fridge_two_phase.sh
```

Перекачать с нуля: `bash dexter/download_open_fridge_two_phase.sh --force`. Подробности и интерактив на Dexter: **`dexter/README_pointflowmatch_dexter.md`** §2.

## Что нужно в окружении

- Установленный PointFlowMatch (`pip install -e .` из корня репозитория).
- Рядом с репозиторием клонированный **`diffusion_policy`** (ветка `develop/eugenio`, как в корневом `README.md`), он нужен импортам и replay buffer.
- В `PYTHONPATH`: корень PointFlowMatch и `../diffusion_policy` (в Slurm-скриптах это уже сделано).

## 1. Разрезать датасет по первому захвату

Скрипт читает zarr в формате `RobotReplayBuffer` (как после `collect_demos.py`: в каталоге есть `data/` и `meta/`), находит первый шаг, где gripper (канал 9 в `robot_state`) ниже порога, и пишет два новых zarr:

| Выход | Смысл |
|--------|--------|
| **pre** | Кадры `0 … t` включительно (`[:t+1]`), где `t` — первый «захват». |
| **post** | Кадры `t … T-1` (`[t:]`); кадр `t` входит и в pre, и в post. |

Эпизоды **без** закрытия гриппера попадают только в pre; в post не добавляются.

Пример из корня репозитория:

```bash
python two_layers_planning/split_dataset_at_first_grasp.py \
  --input demos/sim/open_fridge/train \
  --output-pre demos/sim/open_fridge/train_pre_grasp \
  --output-post demos/sim/open_fridge/train_post_grasp
```

Опции:

- `--gripper-thr` — порог «закрыт» (по умолчанию `0.5`).
- `--overwrite` — перезаписать каталоги pre/post, если уже существуют.

Рядом с `train_pre_grasp` создаётся манифест `train_pre_grasp_split_manifest.json` со статистикой разреза.

## 2. Готовые архивы pre / post (без локального разреза)

Не входят в `demos_open_fridge_sim.tar.gz` из `dexter/download_dataset.sh` — отдельные шары, но скачиваются **одной командой**:

```bash
bash dexter/download_open_fridge_two_phase.sh
```

Внутри — два файла `train_pre_grasp.tar.gz` и `train_post_grasp.tar.gz` (ссылки: [pre](https://disk.yandex.ru/d/E81_4UbQiwAYpw), [post](https://disk.yandex.ru/d/OmzMhzSy0lGTMw)); распаковка в `demos/sim/open_fridge/train_pre_grasp` и `train_post_grasp`.

## 3. Запуск обучения вручную (не Slurm)

Из корня репозитория, с выставленным `PYTHONPATH`:

```bash
export PYTHONPATH="${PWD}:${PWD}/../diffusion_policy${PYTHONPATH:+:$PYTHONPATH}"

# Pre-grasp
python scripts/train.py --config-name=train_open_fridge_pre_grasp \
  dataset_path_train=/abs/path/to/train_pre_grasp \
  run_name=my_pre_run \
  dataloader.num_workers=8

# Post-grasp
python scripts/train.py --config-name=train_open_fridge_post_grasp \
  dataset_path_train=/abs/path/to/train_post_grasp \
  run_name=my_post_run \
  dataloader.num_workers=8
```

Если **`dataset_path_train` не передавать** (остаётся `null` в yaml), `scripts/train.py` подставляет обычный путь **`demos/sim/open_fridge/train`** — это исходный train, не разрезанный pre/post. Для двухфазного обучения **обязательно** укажите `dataset_path_train=...` на каталог с `train_pre_grasp` / `train_post_grasp` (или положите zarr в те пути и задайте их явно).

В этих конфигах уже выставлено `launch_eval_after_train: false` — после `fit` автоматически не запускается `bash/start_eval.sh`.

## 4. Запуск на кластере (Slurm)

**Dexter / conda-окружение в репо (`./pfp-train-env`)** — используйте скрипты в **`dexter/`** (как `run_pointflowmatch_open_fridge.sbatch`): там уже `conda activate`, загрузка данных и бэкап чекпоинтов.

```bash
cd ~/point_flow_match/PointFlowMatch
bash dexter/download_open_fridge_two_phase.sh

PRE=$(sbatch --parsable dexter/run_open_fridge_pre_grasp.sbatch)
sbatch --dependency=afterok:"${PRE}" dexter/run_open_fridge_post_grasp.sbatch
# одной задачей подряд: sbatch dexter/run_open_fridge_two_phase_chain.sbatch
```

Подробно: **`dexter/README_pointflowmatch_dexter.md`** §3.

**Локально / `.venv` в корне репо** — без conda:

```bash
sbatch two_layers_planning/sbatch/train_open_fridge_pre_grasp.sbatch
sbatch two_layers_planning/sbatch/train_open_fridge_post_grasp.sbatch
```

Там вызывается **`.venv/bin/python`**. Переопределение путей:

```bash
export PFP_TRAIN_PRE=/path/to/train_pre_grasp
sbatch two_layers_planning/sbatch/train_open_fridge_pre_grasp.sbatch

export PFP_TRAIN_POST=/path/to/train_post_grasp
sbatch two_layers_planning/sbatch/train_open_fridge_post_grasp.sbatch
```

Проверяется наличие `$PFP_TRAIN_*/data`. Имя рана в `two_layers/.../sbatch`: `open_fridge_pre_<SLURM_JOB_ID>` / `open_fridge_post_<SLURM_JOB_ID>`.

## 5. Чекпоинты и общие настройки

Общая логика задаётся в `conf/train.yaml`: эпохи, EMA, `checkpoint_milestones` (копии вида `milestone_ep500.pt`, `milestone_ep1000.pt` в `ckpt/<run_name>/`), опционально `use_validation` и пути `dataset_path_train` / `dataset_path_valid`.

Для обычного однофазного обучения (другие задачи) снова включайте валидацию: `use_validation: true` и положите `train`/`valid` или укажите пути в конфиге.

## Связанные файлы

| Файл | Назначение |
|------|------------|
| `dexter/download_open_fridge_two_phase.sh` | Скачать с Я.Диска `train_pre_grasp` + `train_post_grasp` |
| `dexter/run_open_fridge_pre_grasp.sbatch` / `post` / `two_phase_chain` | Slurm на Dexter (conda) |
| `two_layers_planning/split_dataset_at_first_grasp.py` | Разрез zarr по первому захвату |
| `conf/train_open_fridge_pre_grasp.yaml` | Конфиг Hydra: pre, без valid |
| `conf/train_open_fridge_post_grasp.yaml` | Конфиг Hydra: post, без valid |
| `two_layers_planning/sbatch/train_open_fridge_pre_grasp.sbatch` | Slurm: pre |
| `two_layers_planning/sbatch/train_open_fridge_post_grasp.sbatch` | Slurm: post |
