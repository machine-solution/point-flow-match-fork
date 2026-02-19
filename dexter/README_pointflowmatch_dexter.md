### Настройка PointFlowMatch на Dexter (DGX A100)

Этот файл описывает полный путь от пустой директории до запуска обучения `open_fridge` на Dexter.

#### 0. Подготовка директории и репозиториев

На Dexter:

```bash
ssh <user>@dexter

mkdir -p ~/point_flow_match
cd ~/point_flow_match

# Если репозитории ещё не склонированы:
git clone https://github.com/machine-solution/point-flow-match-fork.git PointFlowMatch
git clone https://github.com/real-stanford/diffusion_policy.git diffusion_policy

# Если уже клонировал раньше — просто обнови:
cd ~/point_flow_match/PointFlowMatch
git pull
cd ../diffusion_policy
git pull
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

Дальше ставим соседний репозиторий и сам проект. Обе команды запускай из корня `PointFlowMatch`, с уже активированным окружением (`conda activate ./pfp-train-env`). Путь `../diffusion_policy` как раз и есть соседняя папка рядом с `PointFlowMatch`.

```bash
cd ~/point_flow_match/PointFlowMatch
conda activate ./pfp-train-env

pip install -e ../diffusion_policy
pip install -e . --no-deps
```

Проверка, что всё на месте:

```bash
python -c "import diffusion_policy; import pfp; print('OK')"
```

Если выводится `OK`, окружение готово к обучению.

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

Из корня `PointFlowMatch`:

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

