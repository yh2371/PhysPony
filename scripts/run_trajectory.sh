#!/usr/bin/env bash
# Stage 1: global trajectory. FastVGGT camera poses + UniDepthV2 metric scale.
#
#   conda activate physpony
#   bash scripts/run_trajectory.sh <data_dir>
#
# <data_dir> holds one folder per sequence with {frame}_rgb.jpg, _mask.png and
# _box.txt. Writes {frame}_scaled_rotation.npy (metric camera-to-world) and
# median_scale.txt next to the frames.
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../configs/paths.sh"

DATA_DIR="$(realpath "${1:?usage: run_trajectory.sh <data_dir>}")"
WORK_DIR="${WORK_DIR:-$PHYSPONY_ROOT/outputs/trajectory}"
mkdir -p "$WORK_DIR"

cd "$PHYSPONY_FASTVGGT"
export PYTHONPATH="$PHYSPONY_FASTVGGT"
python eval/batch_windowed_inference.py --data_dir "$DATA_DIR" --out_dir "$WORK_DIR" \
    --window_sizes 20 --overlaps 5 --smoothing_factors 0.1 --alignment_methods icp
python eval/parse_pony_val_trajectories.py --data_dir "$WORK_DIR" --out_dir "$DATA_DIR" --image_dir "$DATA_DIR"

cd "$PHYSPONY_UNIDEPTH"
export PYTHONPATH="$PHYSPONY_UNIDEPTH"
python unidepth_pony.py "$DATA_DIR" --device cuda:0
python align_depth.py "$DATA_DIR" --slam_suffix _metric_depth.npy --pred_suffix _vggt_depth.npy --method hybrid
