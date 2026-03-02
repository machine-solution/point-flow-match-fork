#!/bin/bash
#
# Запуск валидации (инференс) и сохранение результата в файл.
# Запускать из корня репо: bash bash/run_validate_accuracy.sh [ckpt_name]
#
# Пример:
#   bash bash/run_validate_accuracy.sh
#   bash bash/run_validate_accuracy.sh open_fridge_0103_1500_resume
#
# RLBench требует CoppeliaSim 4.1.0 (не 4.10). Если не задан COPPELIASIM_ROOT,
# скрипт подставит CoppeliaSim_Edu_V4_1_0_Ubuntu20_04 из корня репо (если есть).

set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# CoppeliaSim 4.1 для RLBench (иначе Handle Panda does not exist)
if [ -z "${COPPELIASIM_ROOT}" ] && [ -d "$REPO_ROOT/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04" ]; then
    export COPPELIASIM_ROOT="$REPO_ROOT/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
    export LD_LIBRARY_PATH="${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH:-}"
    export QT_QPA_PLATFORM_PLUGIN_PATH="${COPPELIASIM_ROOT}"
fi

# Имя чекпоинта: первый аргумент или дефолт
CKPT_NAME="${1:-open_fridge_0103_1500_resume}"
NUM_EPISODES="${2:-300}"

# Conda env (подставь свой, если не pfp_env)
CONDA_ENV="${CONDA_ENV:-pfp_env}"

# Подгрузка conda, если ещё не в PATH (при запуске скрипта не интерактивно .bashrc не читается)
if ! command -v conda &>/dev/null; then
    for _conda in "$CONDA_BASE" "$CONDA_PREFIX" "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" "/opt/miniconda3"; do
        if [ -n "$_conda" ] && [ -f "${_conda}/etc/profile.d/conda.sh" ]; then
            source "${_conda}/etc/profile.d/conda.sh"
            break
        fi
    done
fi
if ! command -v conda &>/dev/null; then
    echo "ERROR: conda not found. Activate env manually and run: python scripts/validate_accuracy.py policy.ckpt_name=$CKPT_NAME env_runner.num_episodes=$NUM_EPISODES"
    exit 1
fi

# Соседний репо для diffusion_policy (если есть)
DIFFUSION_ABS="$(cd "$REPO_ROOT/../diffusion_policy" 2>/dev/null && pwd)"
if [ -n "$DIFFUSION_ABS" ]; then
    export PYTHONPATH="${DIFFUSION_ABS}:${PYTHONPATH:-}"
fi

# Куда писать результат
RESULTS_DIR="$REPO_ROOT/results"
mkdir -p "$RESULTS_DIR"
OUTPUT_FILE="$RESULTS_DIR/validate_accuracy_${CKPT_NAME}_$(date +%Y%m%d_%H%M%S).txt"

echo "Checkpoint: $CKPT_NAME | Episodes: $NUM_EPISODES | Output: $OUTPUT_FILE"
echo "---"

# Запуск через conda run, чтобы гарантированно использовать Python из env
conda run -n "$CONDA_ENV" python scripts/validate_accuracy.py \
    policy.ckpt_name="$CKPT_NAME" \
    env_runner.num_episodes="$NUM_EPISODES" \
    2>&1 | tee "$OUTPUT_FILE"

echo "---"
echo "Results saved to: $OUTPUT_FILE"
