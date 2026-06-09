### Настройка PointFlowMatch на Dexter (DGX A100)

Этот файл описывает полный путь от пустой директории до запуска обучения `open_fridge` на Dexter.

#### 0. Подготовка директории и репозиториев

На Dexter:

```bash
ssh <user>@dexter

mkdir -p ~/point_flow_match
cd ~/point_flow_match

# Если репозитории ещё не склонированы:
# PointFlowMatch — твой форк (или оригинал).
# diffusion_policy — форк автора PointFlowMatch (ветка develop/eugenio), иначе будет ошибка use_dropout в ConditionalUnet1D.
git clone https://github.com/machine-solution/point-flow-match-fork.git PointFlowMatch
git clone https://github.com/chisarie/diffusion_policy.git diffusion_policy
cd diffusion_policy && git checkout develop/eugenio && cd ..

# Если уже клонировал раньше — обнови и проверь ветку diffusion_policy:
cd ~/point_flow_match/PointFlowMatch
git pull
cd ../diffusion_policy
git fetch origin && git checkout develop/eugenio && git pull
cd ../PointFlowMatch
```

Структура в итоге должна быть такой:

```text
~/point_flow_match/
  PointFlowMatch/          # этот репозиторий
  diffusion_policy/        # соседний репозиторий
```

#### 1. Создание Conda‑окружения

**Важно:** команду `conda env create` нужно запускать **из корня репозитория** `PointFlowMatch`, а не из папки `dexter/`.

В репозитории уже есть готовый файл окружения:

```bash
cd ~/point_flow_match/PointFlowMatch

conda env create -f dexter/pfp_train_env.yml -p ./pfp-train-env
conda activate ./pfp-train-env
```

Если при создании окружения pip выдаёт ошибку вида  
`file:///.../PointFlowMatch/dexter does not appear to be a Python project` — значит использована старая версия yml или команда была запущена из папки `dexter/`. Тогда удалите окружение, обновите репозиторий и создайте заново из корня:

```bash
cd ~/point_flow_match/PointFlowMatch
conda env remove -p ./pfp-train-env
git pull
conda env create -f dexter/pfp_train_env.yml -p ./pfp-train-env
conda activate ./pfp-train-env
```

Если при `conda env create` появляется длинная ошибка про `imagecodecs` и `python_abi`,
это из‑за того, что свежие сборки `imagecodecs` требуют Python 3.11+, а в базовом
`torch-env-dexter` используется 3.10. В актуальной версии `pfp_train_env.yml`
`imagecodecs` убран из conda‑зависимостей (он подтянется транзитивно через
`diffusion_policy`), так что после `git pull` и повторного запуска команда должна
отработать без этой ошибки.

В основном окружении (`pfp_train_env.yml`) стоит **PyTorch 2.5.1**, чтобы при дообучении работал полный Composer autoresume (эпоха, LR, оптимизатор); в 2.6+ загрузка чекпоинтов ломается. Для продолжения с чекпоинта задай в sbatch только `run_name=<папка_с_чекпоинтом>` (без `resume_from_ckpt_*`).

Дальше ставим соседний репозиторий и сам проект. Обе команды запускай из корня `PointFlowMatch`, с уже активированным окружением (`conda activate ./pfp-train-env`). Путь `../diffusion_policy` — соседняя папка рядом с `PointFlowMatch`. Используй `python -m pip`, чтобы пакеты точно попали в тот же Python, с которым потом запускаешь обучение.

```bash
cd ~/point_flow_match/PointFlowMatch
conda activate ./pfp-train-env

python -m pip install -e ../diffusion_policy
python -m pip install -e . --no-deps
```

Проверка, что всё на месте (сначала задай `PYTHONPATH`, чтобы Python видел соседний репозиторий):

```bash
export PYTHONPATH=~/point_flow_match/diffusion_policy:${PYTHONPATH:-}
python -c "import diffusion_policy; import pfp; print('OK')"
```

Если выводится `OK`, окружение готово. Если появляется `ModuleNotFoundError: No module named 'diffusion_policy'`, задай `PYTHONPATH` как выше — на некоторых системах editable install не добавляет путь, и явный `PYTHONPATH` это обходит. В Slurm-задаче этот путь уже прописан в `run_pointflowmatch_open_fridge.sbatch`.

> Примечание: если при создании окружения возникнет ошибка на пакете `pytorch3d`,
> можно закомментировать его строку в `dexter/pfp_train_env.yml` и установить отдельно:
>
> ```bash
> conda install -c pytorch3d pytorch3d
> ```

#### 2. Загрузка датасетов `open_fridge`

Всё качается **командами из корня репозитория** (нужны `python3`, `tar`, сеть). На Dexter делай это на **login-ноде или в интерактивной сессии** (`srun --pty bash` / выделенная задача), а не внутри короткого `sbatch` без интерактива — иначе неудобно следить за прогрессом.

**Baseline — один архив `train` + `valid` (~4.3 ГБ):**

```bash
cd ~/point_flow_match/PointFlowMatch
bash dexter/download_dataset.sh
# повторно: bash dexter/download_dataset.sh --force
```

**Двухфазное обучение — два архива `train_pre_grasp` + `train_post_grasp` (~2.5 + ~1.5 ГБ, суммарно больше места чем только baseline):**

```bash
cd ~/point_flow_match/PointFlowMatch
bash dexter/download_open_fridge_two_phase.sh
# повторно: bash dexter/download_open_fridge_two_phase.sh --force
```

**Двухфазное обучение (stable3, устойчивое закрытие) — новые архивы `train_pre_grasp_stable3` + `train_post_grasp_stable3`:**

```bash
cd ~/point_flow_match/PointFlowMatch
bash dexter/download_open_fridge_two_phase.sh --stable3
# повторно: bash dexter/download_open_fridge_two_phase.sh --stable3 --force
```

Скрипт `dexter/download_open_fridge_two_phase.sh` по очереди: API Яндекс.Диска → скачивание `train_pre_grasp.tar.gz` / `train_post_grasp.tar.gz` → распаковка в `demos/sim/open_fridge/train_pre_grasp` и `train_post_grasp` (через `dexter/extract_zarr_tarball.py`). Если каталог уже есть и в нём есть `data/` и `meta/`, шаг пропускается (кроме `--force`).

Если не хочешь скачивать готовые pre/post, можно **разрезать** локально полный `train` после `download_dataset.sh`: `python two_layers_planning/split_dataset_at_first_grasp.py ...` (см. **`two_layers_planning/README.md`**).

| Задача | Куда ложится zarr | Отдельный `valid` |
|--------|-------------------|---------------------|
| `run_pointflowmatch_open_fridge*.sbatch` | `demos/sim/open_fridge/train`, `valid` | нужен |
| `run_pointflowmatch_open_fridge_phase_prediction.sbatch` | то же; **FMPolicy + phase_head** (`phase_prediction=enabled`) | нужен |
| `two_layers_planning/sbatch/train_open_fridge_*_grasp.sbatch` | `train_pre_grasp`, `train_post_grasp` | не нужен (`use_validation: false`) |

Ссылки на публичные шары (те же, что вшиты в скрипты): [полный сим-набор](https://disk.yandex.ru/d/Ssr_BffZItISOg), [pre](https://disk.yandex.ru/d/E81_4UbQiwAYpw), [post](https://disk.yandex.ru/d/OmzMhzSy0lGTMw).

#### 3. Запуск обучения через Slurm

Из корня `PointFlowMatch`. В sbatch-скрипте уже прописан `PYTHONPATH` на соседний `diffusion_policy`, при ручном запуске его нужно выставить самому (см. проверку выше).

На Dexter в sbatch используется системный conda: `source /opt/miniconda3/etc/profile.d/conda.sh` (как в примере из инструкции кластера).

**Базовая модель:**

```bash
cd ~/point_flow_match/PointFlowMatch
conda activate ./pfp-train-env
sbatch dexter/run_pointflowmatch_open_fridge.sbatch
```

**Вариант с весами вокруг смены гриппера:**

```bash
cd ~/point_flow_match/PointFlowMatch
conda activate ./pfp-train-env
sbatch dexter/run_pointflowmatch_open_fridge_gripper_weighted.sbatch
```

**Phase‑Conditioned Single Model (одна shared модель с phase token, без hard switch):**

```bash
cd ~/point_flow_match/PointFlowMatch
conda activate ./pfp-train-env
sbatch dexter/run_pointflowmatch_open_fridge_phase_conditioned.sbatch
```

**Phase‑Conditioned + gripper‑weighted loss:**

```bash
cd ~/point_flow_match/PointFlowMatch
conda activate ./pfp-train-env
sbatch dexter/run_pointflowmatch_open_fridge_gripper_weighted_phase_conditioned.sbatch
```

**Learned phase prediction (одна модель FMPolicy + `phase_head`, как в `pfp/policy/fm_policy.py`):**

Обучает `scripts/train.py` с `model=flow` (`FMPolicy`), `phase_conditioning=enabled`, `phase_prediction=enabled`. На инференсе фаза предсказывается из наблюдения; flow матчится с RLBench (текущая фаза на весь горизонт).

```bash
cd ~/point_flow_match/PointFlowMatch
conda activate ./pfp-train-env

sbatch dexter/run_pointflowmatch_open_fridge_phase_prediction.sbatch
```

Вариант с gripper-weighted loss:

```bash
sbatch dexter/run_pointflowmatch_open_fridge_gripper_weighted_phase_prediction.sbatch
```

Локально (без Slurm), из корня репо:

```bash
python dexter/train_pointflowmatch_open_fridge.py \
  --phase-conditioning enabled --phase-prediction enabled
```

**MeanFlow (новая модель, готовый sbatch):**

```bash
cd ~/point_flow_match/PointFlowMatch
conda activate ./pfp-train-env
sbatch dexter/run_pointflowmatch_open_fridge_meanflow.sbatch
```

**MeanFlow + Temporal Transformer (архитектурный эксперимент, готовые команды):**

```bash
cd ~/point_flow_match/PointFlowMatch
conda activate ./pfp-train-env

sbatch dexter/run_pointflowmatch_open_fridge_meanflow_transformer.sbatch
```

**Shortcut Flow (архитектурный эксперимент, готовый sbatch):**

```bash
cd ~/point_flow_match/PointFlowMatch
conda activate ./pfp-train-env

sbatch dexter/run_pointflowmatch_open_fridge_shortcut.sbatch
```

**StepConditionedMeanFlow (новая unified-модель, готовый sbatch):**

```bash
cd ~/point_flow_match/PointFlowMatch
conda activate ./pfp-train-env

sbatch dexter/run_pointflowmatch_open_fridge_step_conditioned_meanflow.sbatch
```

Для этого режима в `+experiment=pointflowmatch_step_conditioned_meanflow` включен
`checkpoint_schedule=final_1500_only`: сохраняется только финальный чекпоинт `ep1500`
(без промежуточных `ep300/600/900/1200`).

**Двухфазное обучение (pre, затем post)** — из корня репозитория, окружение `pfp-train-env` уже используется внутри `.sbatch` (как в `run_pointflowmatch_open_fridge.sbatch`):

```bash
cd ~/point_flow_match/PointFlowMatch
bash dexter/download_open_fridge_two_phase.sh   # если ещё нет train_pre_grasp / train_post_grasp

# Вариант A — две задачи подряд: post стартует только после успешного pre
sbatch dexter/run_open_fridge_pre_grasp.sbatch
sbatch dexter/run_open_fridge_post_grasp.sbatch
# то же самое: bash dexter/submit_open_fridge_post_after_pre.sh "$PRE"

# Вариант B — одна длинная задача: pre и post в одном job (14 суток walltime)
sbatch dexter/run_open_fridge_two_phase_chain.sbatch
```

Логи: `logs/pfm_open_fridge_pre_<JOB>.out`, `logs/pfm_open_fridge_post_<JOB>.out`, для цепочки — `logs/pfm_open_fridge_chain_<JOB>.out`. Чекпоинты: `ckpt/<run_name>/`; бэкап в `${HOME}/checkpoints/pointflowmatch_open_fridge_two_phase/` (переопределить: `export CKPT_BACKUP_DIR=...` перед `sbatch`).

Локальная виртуалка без conda в репо — старые скрипты с `.venv`: `two_layers_planning/sbatch/train_open_fridge_*_grasp.sbatch`. Подробности: **`two_layers_planning/README.md`**.

Проверить очередь и логи:

```bash
squeue -u <user>                     # статус задач
ls logs/                             # файлы логов
tail -f logs/pfm_open_fridge_<JOB>.out
tail -f logs/pfm_open_fridge_pre_<JOB>.out
```

Скрипт `run_pointflowmatch_open_fridge.sbatch`:
- активирует окружение `./pfp-train-env`,
- при отсутствии данных вызывает `dexter/download_dataset.sh`,
- запускает обучение:
  - `scripts/train.py task_name=open_fridge +experiment=pointflowmatch`,
- по окончании копирует чекпоинты из `ckpt/` в
  `${HOME}/checkpoints/pointflowmatch` (можно поменять путь в переменной
  `CKPT_BACKUP_DIR` в начале `.sbatch`).

Скрипт `run_pointflowmatch_open_fridge_gripper_weighted.sbatch` делает всё то же самое,
но использует эксперимент `+experiment=pointflowmatch_gripper_weighted`, где лосс по
времени усиливается в окрестности смены гриппера (см. раздел 8).

#### 4. Резюме команд (короткая шпаргалка)

```bash
# Один раз
cd ~/point_flow_match
git clone <URL_твоего_fork_PointFlowMatch> PointFlowMatch
git clone <URL_repo_diffusion_policy> diffusion_policy

cd PointFlowMatch
conda env create -f dexter/pfp_train_env.yml -p ./pfp-train-env
conda activate ./pfp-train-env
pip install -e ../diffusion_policy
pip install -e . --no-deps

# Для каждого нового запуска
cd ~/point_flow_match/PointFlowMatch
git pull
conda activate ./pfp-train-env
bash dexter/download_dataset.sh                 # baseline: train + valid
bash dexter/download_open_fridge_two_phase.sh   # pre + post zarr для двухфазного обучения
sbatch dexter/run_pointflowmatch_open_fridge.sbatch
# двухфазное (pre → post), см. §3 выше:
# PRE=$(sbatch --parsable dexter/run_open_fridge_pre_grasp.sbatch)
# sbatch --dependency=afterok:"$PRE" dexter/run_open_fridge_post_grasp.sbatch
```

#### 5. Где лежат веса и как скачать на свою машину

После обучения чекпоинты лежат в двух местах на Dexter:

1. **В репозитории:** `~/point_flow_match/PointFlowMatch/ckpt/<run_name>/`  
   Пример `run_name`: `1771602945-cautious-adder` (есть в логе обучения, строка `Run name: ...`).
2. **Бэкап (из sbatch):** `~/checkpoints/pointflowmatch/<run_name>/`  
   Путь задаётся переменной `CKPT_BACKUP_DIR` в `dexter/run_pointflowmatch_open_fridge.sbatch`.

Внутри папки `run_name`: `config.yaml`, файлы вида `ep500.pt`, `ep1000.pt`, `latest.pt` (или только `latest.pt`).

**Скачать на локальную машину** (подставь свой логин и `run_name`):

```bash
# из корня репо на Dexter
scp -r USER@Dexter-Host:~/point_flow_match/PointFlowMatch/ckpt/<run_name> ./ckpt/

# или из бэкапа
scp -r USER@Dexter-Host:~/checkpoints/pointflowmatch/<run_name> ./ckpt/
```

На локальной машине нужна папка `ckpt/` в корне клонированного PointFlowMatch; в неё и кладётся `<run_name>`.

**Валидация (100 эпизодов, accuracy):** запуск `scripts/validate_accuracy.py` — см. раздел 6.

#### 6. Валидация: 100 эпизодов, accuracy

Скрипт `scripts/validate_accuracy.py` запускает 100 эпизодов в симуляции и считает долю успешных (accuracy). Чекпоинт должен уже лежать в `ckpt/<run_name>/`.

**На Dexter** (после обучения, в headless):

```bash
cd ~/point_flow_match/PointFlowMatch
conda activate ./pfp-train-env
export PYTHONPATH=~/point_flow_match/diffusion_policy:${PYTHONPATH:-}

python scripts/validate_accuracy.py policy.ckpt_name=<run_name> env_runner.num_episodes=100
```

**Локально** (если чекпоинт скачан в `ckpt/<run_name>` и есть CoppeliaSim/RLBench):

```bash
conda activate pfp_env
python scripts/validate_accuracy.py policy.ckpt_name=<run_name> env_runner.num_episodes=100
```

В конце выводится строка вида: `Accuracy: 87/100 (87.0%)`.

#### 7. «No space left on device» во время одного запуска

Если место кончается **в середине одного** прогона обучения (не после нескольких запусков), чаще всего виноват **своп**: при нехватке RAM ядро пишет своп на диск (часто на тот же раздел, что и домашний каталог), и он может вырасти на десятки гигабайт.

Что сделать:

1. **Проверить, что забито:** в логе job в начале есть блок `Disk space before training` (`df -h`). Смотри, какой раздел заполняется к моменту падения (например, `df -h` на ноде или после падения).
2. **Проверить своп:** `free -h` — если `Swap used` большой и растёт во время обучения, снизь потребление RAM:
   - в sbatch добавь в вызов `train.py` переопределения: `dataloader.batch_size=64` и/или `dataloader.num_workers=4`;
   - так свопа будет меньше и диск не будет забиваться под своп-файл.
3. **Чекпоинты** теперь пишутся в одну папку в репо (`ckpt/<run_name>/`), а не в отдельную папку на каждый запуск; перед обучением старые `outputs/` удаляются. Если падает при сохранении чекпоинта — проверь, что на разделе с репо есть несколько гигабайт свободного места (нужно под новый файл до удаления старого).

#### 8. Эксперимент с весами вокруг смены гриппера

Есть альтернативный режим обучения, где лосс по времени усиливается в окрестности моментов, когда меняется состояние гриппера (0 → 1 или 1 → 0). Это позволяет модели «внимательнее» учиться фазам захвата/отпуска.

- В базовой модели (`conf/model/flow.yaml`) добавлены параметры:

  ```yaml
  use_gripper_motion_weights: false
  gripper_motion_lambda: 3.0
  gripper_motion_window: 2   # окно ±2 шага → 5 шагов вокруг смены
  ```

- Эксперимент `pointflowmatch_gripper_weighted` включает этот режим:

  ```yaml
  # conf/experiment/pointflowmatch_gripper_weighted.yaml
  # @package _global_
  defaults:
    - override /model: flow

  model:
    use_gripper_motion_weights: true
    gripper_motion_lambda: 3.0
    gripper_motion_window: 2
  ```

- Для запуска такого обучения на Dexter есть отдельный sbatch:

  ```bash
  cd ~/point_flow_match/PointFlowMatch
  conda activate ./pfp-train-env
  sbatch dexter/run_pointflowmatch_open_fridge_gripper_weighted.sbatch
  ```

  Этот скрипт запускает:

  ```bash
  python scripts/train.py \
      task_name=open_fridge \
      +experiment=pointflowmatch_gripper_weighted \
      dataloader.num_workers=8 \
      dataloader.batch_size=128 \
      log_wandb=False
  ```

Базовое обучение (`+experiment=pointflowmatch`) остаётся без временных весов (все шаги равны), а gripper-weighted эксперимент — это отдельный запуск, чекпоинты которого ложатся в свою папку `ckpt/<run_name>/`.

#### 9. Расписание чекпоинтов (300, 600, 900, 1200, 1500)

По умолчанию `conf/train.yaml` + `checkpoint_schedule=milestones_1500`:

- `save_each_n_epochs: 300`
- `save_num_checkpoints_to_keep: 5` — не удалять ранние milestone при сохранении поздних
- `checkpoint_milestones: []` — без дублирующих `milestone_ep*.pt` (экономия диска)

В `ckpt/<run_name>/` будут файлы вида `ep0300-*.pt`, `ep0600-*.pt`, …, `ep1500-*.pt` (плюс `latest-rank0.pt` для autoresume, если задан `run_name`).

**Валидация по эпохам** (локально, 100 эпизодов):

```bash
# baseline
bash bash/run_validate_milestone_sweep.sh <run_name> 100 \
  phase_conditioning=disabled phase_prediction=disabled

# learned phase
bash bash/run_validate_milestone_sweep.sh <run_name> 100 \
  phase_conditioning=enabled phase_prediction=enabled
```

Один эпизод вручную: `policy.ckpt_episode=ep0600`.

Старое поведение (каждые 100 ep, keep 3): `checkpoint_schedule=legacy_every_100`.

#### 10. Соответствие моделей и sbatch (шпаргалка)

| Цель | Slurm / команда | Hydra |
|------|-----------------|-------|
| Baseline PointFlowMatch | `run_pointflowmatch_open_fridge.sbatch` | `+experiment=pointflowmatch`, phase off |
| Oracle phase (GT в датасете) | `run_pointflowmatch_open_fridge_phase_conditioned.sbatch` | `phase_conditioning=enabled`, `phase_prediction=disabled` |
| **Learned phase (эта модель)** | `run_pointflowmatch_open_fridge_phase_prediction.sbatch` | `phase_conditioning=enabled`, `phase_prediction=enabled` |
| Learned phase + gripper weights | `run_pointflowmatch_open_fridge_gripper_weighted_phase_prediction.sbatch` | `+experiment=pointflowmatch_gripper_weighted` + phase flags |
| StepConditionedMeanFlow | `run_pointflowmatch_open_fridge_step_conditioned_meanflow.sbatch` | `+experiment=pointflowmatch_step_conditioned_meanflow` |

Строки `Baseline/Oracle/Learned phase/Learned phase + gripper weights` используют
**`conf/model/flow.yaml`** → **`pfp.policy.fm_policy.FMPolicy`**.
`StepConditionedMeanFlow` использует отдельные
**`conf/model/step_conditioned_meanflow.yaml`** → **`pfp.policy.step_conditioned_meanflow_policy.StepConditionedMeanFlowPolicy`**.

Перед длинным job скрипт `dexter/verify_training_setup.py` проверяет, что Hydra собирает именно `FMPolicy` и что `phase_head` создаётся при `phase_prediction=enabled`.

