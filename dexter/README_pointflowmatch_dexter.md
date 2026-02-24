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

#### 2. Загрузка датасета `open_fridge`

Из корня репозитория `PointFlowMatch`:

```bash
cd ~/point_flow_match/PointFlowMatch

# Если данных ещё нет:
bash dexter/download_dataset.sh
```

Скрипт:
- получает прямую ссылку на архив с Яндекс.Диска,
- скачивает `demos_open_fridge_sim.tar.gz` (~4.3 ГБ),
- распаковывает его в `demos/sim/open_fridge/`,
- удаляет архив.

После выполнения должны существовать директории:

```text
demos/sim/open_fridge/train
demos/sim/open_fridge/valid
```

#### 3. Запуск обучения через Slurm

Из корня `PointFlowMatch`. В sbatch-скрипте уже прописан `PYTHONPATH` на соседний `diffusion_policy`, при ручном запуске его нужно выставить самому (см. проверку выше).

На Dexter в sbatch используется системный conda: `source /opt/miniconda3/etc/profile.d/conda.sh` (как в примере из инструкции кластера).

```bash
cd ~/point_flow_match/PointFlowMatch
conda activate ./pfp-train-env

sbatch dexter/run_pointflowmatch_open_fridge.sbatch
```

Проверить очередь и логи:

```bash
squeue -u <user>                     # статус задач
ls logs/                             # файлы логов
tail -f logs/pfm_open_fridge_<JOB>.out
```

Скрипт `run_pointflowmatch_open_fridge.sbatch`:
- активирует окружение `./pfp-train-env`,
- при отсутствии данных вызывает `dexter/download_dataset.sh`,
- запускает обучение:
  - `scripts/train.py task_name=open_fridge +experiment=pointflowmatch`,
- по окончании копирует чекпоинты из `ckpt/` в
  `${HOME}/checkpoints/pointflowmatch` (можно поменять путь в переменной
  `CKPT_BACKUP_DIR` в начале `.sbatch`).

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
bash dexter/download_dataset.sh      # если датасета ещё нет
sbatch dexter/run_pointflowmatch_open_fridge.sbatch
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

