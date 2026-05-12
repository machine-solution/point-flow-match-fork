#!/bin/bash
#
# Запуск валидации (инференс) и сохранение результата в файл.
# Запускать из корня репо: bash bash/run_validate_accuracy.sh [ckpt_name]
#
# Пример:
#   bash bash/run_validate_accuracy.sh
#   bash bash/run_validate_accuracy.sh open_fridge_0103_1500_resume
#
# RLBench: по умолчанию подставляется CoppeliaSim 4.1.0 из корня репо (см. блок COPPELIASIM_ROOT ниже).

set -e
set -o pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# RLBench / collect_demos ожидают CoppeliaSim 4.1.0. Если в окружении стоит 4.10+ (например
# ~/CoppeliaSim), PyRep часто падает с heap corruption ("free(): invalid next size").
# По умолчанию принудительно берём 4.1.0 из корня репо, если папка есть.
# Чтобы оставить свой COPPELIASIM_ROOT:  PFP_SKIP_REPO_COPPELIASIM=1 bash bash/run_validate_accuracy.sh ...
CS41="$REPO_ROOT/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
if [ -d "$CS41" ] && [ -z "${PFP_SKIP_REPO_COPPELIASIM:-}" ]; then
    if [ -n "${COPPELIASIM_ROOT:-}" ] && [ "${COPPELIASIM_ROOT}" != "$CS41" ]; then
        echo "[run_validate_accuracy] Overriding COPPELIASIM_ROOT (was: ${COPPELIASIM_ROOT}) -> ${CS41}"
    fi
    export COPPELIASIM_ROOT="$CS41"
    export LD_LIBRARY_PATH="${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH:-}"
    export QT_QPA_PLATFORM_PLUGIN_PATH="${COPPELIASIM_ROOT}"
elif [ -z "${COPPELIASIM_ROOT:-}" ] && [ -d "$CS41" ]; then
    export COPPELIASIM_ROOT="$CS41"
    export LD_LIBRARY_PATH="${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH:-}"
    export QT_QPA_PLATFORM_PLUGIN_PATH="${COPPELIASIM_ROOT}"
fi
if [ -n "${COPPELIASIM_ROOT:-}" ]; then
    echo "[run_validate_accuracy] COPPELIASIM_ROOT=${COPPELIASIM_ROOT}"
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

# Прогресс tqdm в консоли пропадает, если stdout не TTY (типично при `| tee`).
# util-linux `script` выдаёт дочернему процессу псевдо-TTY — tqdm рисуется в терминале,
# тот же поток дублируем в файл через tee.
export PYTHONUNBUFFERED=1

# НЕ писать «conda run ... env VAR=1 python»: conda воспринимает «env» как команду внутри env.
# PYTHONUNBUFFERED уже export выше — наследуется дочерним shell и conda run.
VAL_CMD="cd $(printf '%q' "$REPO_ROOT") && conda run --no-capture-output -n $(printf '%q' "$CONDA_ENV") python scripts/validate_accuracy.py policy.ckpt_name=$(printf '%q' "$CKPT_NAME") env_runner.num_episodes=$(printf '%q' "$NUM_EPISODES")"

if command -v script >/dev/null 2>&1 && script -V 2>&1 | grep -q util-linux; then
    # typescript в /dev/null — нужен только stdout в tee
    script -qec "$VAL_CMD" /dev/null 2>&1 | tee "$OUTPUT_FILE"
else
    echo "[run_validate_accuracy] WARNING: util-linux \`script\` not found; tqdm may not show a bar (output still goes to file)."
    eval "$VAL_CMD" 2>&1 | stdbuf -oL -eL tee "$OUTPUT_FILE"
fi

echo "---"
echo "Results saved to: $OUTPUT_FILE"
