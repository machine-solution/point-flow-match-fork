#!/usr/bin/env bash
# Deprecated: use bash/create_demos_sim_bundles.sh (demos_<task>_sim.tar.gz at repo root).
exec bash "$(dirname "$0")/create_demos_sim_bundles.sh" "$@"
