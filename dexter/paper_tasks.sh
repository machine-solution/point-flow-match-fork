# Paper RLBench tasks (keep in sync with pfp/common/paper_tasks.py).
PFP_DEFAULT_TASK_NAME="open_fridge"
PFP_PAPER_TASKS=(
  unplug_charger
  close_door
  open_box
  open_fridge
  take_frame_off_hanger
  open_oven
  put_books_on_bookshelf
  take_shoes_out_of_box
)

pfp_is_paper_task() {
  local t="$1"
  local x
  for x in "${PFP_PAPER_TASKS[@]}"; do
    [[ "$x" == "$t" ]] && return 0
  done
  return 1
}

pfp_validate_task_name() {
  local t="${1:-${TASK_NAME:-}}"
  if [[ -z "$t" ]]; then
    echo "ERROR: task_name / TASK_NAME is empty" >&2
    return 1
  fi
  if ! pfp_is_paper_task "$t"; then
    echo "WARNING: '${t}' is not one of the 8 paper tasks; continuing anyway." >&2
  fi
}
