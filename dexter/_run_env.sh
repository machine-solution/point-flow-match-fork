# Shared training environment for Dexter: Slurm (sbatch) or direct (bash dexter/run_local.sh …).
# Source after `set -e` from any dexter/*.sbatch in the repo root.

# Slurm if the job id is set; else local/direct server.
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  PFP_RUN_MODE="${PFP_RUN_MODE:-slurm}"
  REPO_ROOT="${SLURM_SUBMIT_DIR:-}"
else
  PFP_RUN_MODE="${PFP_RUN_MODE:-local}"
  _pfp_caller="${BASH_SOURCE[1]:-${BASH_SOURCE[0]}}"
  REPO_ROOT="$(cd "$(dirname "${_pfp_caller}")/.." && pwd)"
fi

if [[ -z "${REPO_ROOT}" || ! -d "${REPO_ROOT}" ]]; then
  echo "ERROR: could not resolve repo root (SLURM_SUBMIT_DIR / script path)" >&2
  exit 1
fi

export PFP_RUN_MODE REPO_ROOT
cd "${REPO_ROOT}"
mkdir -p logs

PFP_JOB_ID="${SLURM_JOB_ID:-local$(date +%s)}"
export PFP_JOB_ID

# Direct-server limits (admin policy); Slurm uses #SBATCH allocation instead.
PFP_MAX_CPU_THREADS="${PFP_MAX_CPU_THREADS:-16}"
PFP_MAX_GPUS="${PFP_MAX_GPUS:-2}"
PFP_MAX_RAM_GB="${PFP_MAX_RAM_GB:-64}"

_pfp_count_gpus() {
  local spec="${1:-}"
  if [[ -z "${spec}" ]]; then
    echo 1
    return
  fi
  local n=0
  local part
  IFS=',' read -ra parts <<< "${spec}"
  for part in "${parts[@]}"; do
    part="${part// /}"
    if [[ "${part}" =~ ^[0-9]+$ ]]; then
      n=$((n + 1))
    elif [[ "${part}" =~ ^[0-9]+-[0-9]+$ ]]; then
      local lo="${part%-*}" hi="${part#*-}"
      n=$((n + hi - lo + 1))
    fi
  done
  echo "${n}"
}

_pfp_apply_local_limits() {
  local gpu_count
  gpu_count="$(_pfp_count_gpus "${CUDA_VISIBLE_DEVICES:-}")"
  if (( gpu_count > PFP_MAX_GPUS )); then
    echo "ERROR: CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}' → ${gpu_count} GPU(s); max ${PFP_MAX_GPUS} on direct server." >&2
    echo "  Use e.g. CUDA_VISIBLE_DEVICES=0 or CUDA_VISIBLE_DEVICES=0,1" >&2
    exit 1
  fi

  local ncpu
  ncpu="$(nproc 2>/dev/null || echo 8)"
  if (( ncpu > PFP_MAX_CPU_THREADS )); then
    export OMP_NUM_THREADS="${PFP_MAX_CPU_THREADS}"
  else
    export OMP_NUM_THREADS="${ncpu}"
  fi
  export PFP_DATALOADER_WORKERS="${PFP_DATALOADER_WORKERS:-$(( OMP_NUM_THREADS > 8 ? 8 : OMP_NUM_THREADS ))}"
}

if [[ "${PFP_RUN_MODE}" == "slurm" ]]; then
  export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
  export PFP_DATALOADER_WORKERS="${PFP_DATALOADER_WORKERS:-8}"
else
  _pfp_apply_local_limits
fi

PFP_CONDA_SH="${PFP_CONDA_SH:-/opt/miniconda3/etc/profile.d/conda.sh}"
PFP_CONDA_ENV="${PFP_CONDA_ENV:-./pfp-train-env}"
if [[ -f "${PFP_CONDA_SH}" ]]; then
  # shellcheck source=/dev/null
  source "${PFP_CONDA_SH}"
  conda activate "${PFP_CONDA_ENV}"
else
  echo "WARNING: conda init not found at ${PFP_CONDA_SH}; using current shell env" >&2
fi

export PYTHONPATH="${REPO_ROOT}/../diffusion_policy:${PYTHONPATH:-}"
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"

echo "=== PFP run environment ==="
echo " mode              : ${PFP_RUN_MODE}"
echo " repo              : ${REPO_ROOT}"
echo " job_id            : ${PFP_JOB_ID}"
echo " host              : $(hostname)"
echo " OMP_NUM_THREADS   : ${OMP_NUM_THREADS}"
echo " CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-(default 0)}"
if [[ "${PFP_RUN_MODE}" == "local" ]]; then
  echo " limits (direct)   : GPU<=${PFP_MAX_GPUS}, CPU<=${PFP_MAX_CPU_THREADS}, RAM<=${PFP_MAX_RAM_GB}GB (RAM not enforced in shell)"
fi
echo " python            : $(command -v python 2>/dev/null || command -v python3 2>/dev/null || echo missing) ($({ python -V || python3 -V; } 2>&1 || true))"

echo "=== nvidia-smi (before training) ==="
nvidia-smi || true

echo "=== disk space ==="
df -h . /tmp 2>/dev/null || df -h .
