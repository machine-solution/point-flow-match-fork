#!/bin/bash

# Record 10 *separate* episodes of the open_fridge task.
# Each episode is recorded by a separate Python process so that
# CoppeliaSim / RLBench resources are fully released between runs.
#
# Files are saved under recordings/ as:
#   recordings/open_fridge_seed<SEED>.json

set -e

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pfp_env

CKPT_NAME="1717446565-astute-stingray"
CKPT_PATH="ckpt/${CKPT_NAME}"

if [ ! -d "$CKPT_PATH" ]; then
    echo "Warning: Checkpoint not found at $CKPT_PATH"
    echo "Please download it from: http://pointflowmatch.cs.uni-freiburg.de/download/1717446565-astute-stingray.zip"
    echo "And extract it to the ckpt folder."
    exit 1
fi

# Base seed – must match cfg.seed in your eval config to be consistent,
# but мы можем задать его и здесь явно.
BASE_SEED=5678

echo "Recording 10 separate open_fridge episodes"
echo "Checkpoint: $CKPT_NAME"
echo "Base seed: $BASE_SEED"
echo

mkdir -p recordings

for i in $(seq 0 9); do
    # Немного «распределяем» seed, чтобы задачи отличались заметнее
    SEED=$((BASE_SEED + i * 17))
    OUT_FILE="recordings/open_fridge_seed${SEED}.json"

    # Пропускаем, если файл уже есть (можно добивать недостающие эпизоды)
    if [ -f "$OUT_FILE" ]; then
        echo "=== Episode ${i} (seed=${SEED}) -> ${OUT_FILE} already exists, skipping ==="
        echo
        continue
    fi

    echo "=== Episode ${i} (seed=${SEED}) -> ${OUT_FILE} ==="

    # Каждый запуск пишет ровно 1 эпизод (env_runner.num_episodes=1)
    # и внутри record_actions.py / RLBenchRunnerRecord:
    #   - set_seeds(SEED)
    #   - episode_seed = base_seed + 0 = SEED
    #
    # Используем xvfb-run, т.к. CoppeliaSim требует X даже в headless режиме.
    xvfb-run -a -s "-screen 0 1024x768x24" python scripts/record_actions.py \
        log_wandb=False \
        seed=$SEED \
        env_runner.env_config.vis=False \
        env_runner.env_config.headless=True \
        env_runner.num_episodes=1 \
        policy.ckpt_name=$CKPT_NAME \
        policy.ckpt_episode=ep1500 \
        +output_file=$OUT_FILE

    echo
done

echo "Done. Per-episode files are in recordings/open_fridge_seed<SEED>.json"

