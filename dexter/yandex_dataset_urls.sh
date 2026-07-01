#!/usr/bin/env bash
# Yandex Disk public share URLs for paper task demo bundles (demos_<task>_sim.tar.gz).
# Keep in sync with dexter/download_dataset.sh and pfp/common/paper_tasks.py.

pfp_yandex_url_for_task() {
  local task="$1"
  case "$task" in
    unplug_charger) echo "https://disk.yandex.ru/d/U4icXHaV6LsAyg" ;;
    close_door) echo "https://disk.yandex.ru/d/_yhpQY_xSnZIBw" ;;
    open_box) echo "https://disk.yandex.ru/d/MNVtWFTXgsmyLw" ;;
    open_fridge) echo "https://disk.yandex.ru/d/Ssr_BffZItISOg" ;;
    take_frame_off_hanger) echo "https://disk.yandex.ru/d/SHSiu0K2YF8ixQ" ;;
    open_oven) echo "https://disk.yandex.ru/d/NrysegZdYg8OhQ" ;;
    put_books_on_bookshelf) echo "https://disk.yandex.ru/d/VwHJHr_9JVdp-Q" ;;
    take_shoes_out_of_box) echo "https://disk.yandex.ru/d/8mhtToS_WpgXEQ" ;;
    *) return 1 ;;
  esac
}

# Accept common short aliases.
pfp_normalize_task_name() {
  local t="$1"
  case "$t" in
    unplug) echo "unplug_charger" ;;
    put_books) echo "put_books_on_bookshelf" ;;
    take_shoes) echo "take_shoes_out_of_box" ;;
    take_frame) echo "take_frame_off_hanger" ;;
    *) echo "$t" ;;
  esac
}
