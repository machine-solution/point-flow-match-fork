#!/usr/bin/env bash
# Отправить post-grasp задачу после успешного завершения pre-grasp.
#
# Использование:
#   PRE_JOB=$(sbatch --parsable dexter/run_open_fridge_pre_grasp.sbatch)
#   bash dexter/submit_open_fridge_post_after_pre.sh "$PRE_JOB"
#
# или в одну строку:
#   bash dexter/submit_open_fridge_post_after_pre.sh "$(sbatch --parsable dexter/run_open_fridge_pre_grasp.sbatch)"
#
set -euo pipefail
PRE_JOB="${1:?usage: $0 <SLURM_JOB_ID_PRE>}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
POST_JOB="$(sbatch --parsable --dependency="afterok:${PRE_JOB}" dexter/run_open_fridge_post_grasp.sbatch)"
echo "Pre job:  ${PRE_JOB}"
echo "Post job: ${POST_JOB} (starts after ${PRE_JOB} succeeds)"
