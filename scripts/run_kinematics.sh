#!/usr/bin/env bash
# Stage 2: local articulation. Runs the PhysPony kinematic model on every
# sequence in <data_dir> (after run_trajectory.sh).
#
#   conda activate physpony
#   bash scripts/run_kinematics.sh <data_dir> [out_dir]
#
# Writes <seq>_<frame>__params.pkl per frame to out_dir (default outputs/kinematics).
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../configs/paths.sh"

DATA_DIR="$(realpath "${1:?usage: run_kinematics.sh <data_dir> [out_dir]}")"
OUT_DIR="$(realpath -m "${2:-$PHYSPONY_ROOT/outputs/kinematics}")"

cd "$PHYSPONY_MODEL"
export PYTHONPATH="$PHYSPONY_MODEL"
python run.py --config-name test_physpony_horse \
    checkpoint_path="$PHYSPONY_KINEMATIC_CKPT" \
    dataset.test_data_dir="$DATA_DIR" \
    test_result_dir="$OUT_DIR"
