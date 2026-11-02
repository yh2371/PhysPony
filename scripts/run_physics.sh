#!/usr/bin/env bash
# Stage 3: physics-based refinement. Finetunes the mocap-pretrained tracking
# policy on each reconstructed sequence and saves the simulated motion.
#
#   conda activate physpony-sim
#   bash scripts/run_physics.sh <data_dir> [seq ...]     # default: every sequence
#
# Reads outputs/kinematics (KINEMATICS_DIR) and the metric scale in <data_dir>.
# Writes OUT_DIR/<seq>_states.npz (default outputs/physics).
#
# Options (environment variables):
#   HOURS=7        training budget per sequence (on one L40)
#   ZERO_SHOT=1    only roll out the pretrained policy, no finetuning
#   DECIMATE=4     for 120 Hz PFERD captures
set -o pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../configs/paths.sh"

DATA_DIR="$(realpath "${1:?usage: run_physics.sh <data_dir> [seq ...]}")"; shift
KINEMATICS_DIR="$(realpath "${KINEMATICS_DIR:-$PHYSPONY_ROOT/outputs/kinematics}")"
OUT_DIR="$(realpath -m "${OUT_DIR:-$PHYSPONY_ROOT/outputs/physics}")"
HOURS=${HOURS:-7}
WARMUP_HOURS=${WARMUP_HOURS:-1}
CURRICULUM_MODE=${CURRICULUM_MODE:-continuous}     # or: steps
START_SAMPLES=${START_SAMPLES:-0}                  # resume a run part-way
NUM_ENVS=${NUM_ENVS:-4096}
SAMPLES_PER_HOUR=57278464

cd "$PHYSPONY_SIM"
export PYTHONPATH="$PHYSPONY_SIM"
export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}"
export WANDB_MODE=${WANDB_MODE:-offline}

python prepare_refinement_inputs.py --params_dir "$KINEMATICS_DIR" --data_dir "$DATA_DIR" \
    --out_dir "$OUT_DIR" --decimate "${DECIMATE:-1}" ${1:+--seqs "$@"} \
    || echo "[physics] some sequences could not be prepared (see above)"
[ "$#" -gt 0 ] || set -- $(ls "$OUT_DIR"/envs/env_*.yaml 2>/dev/null | sed 's|.*/env_||; s|\.yaml$||')

for seq in "$@"; do
  TEST_ENV="$OUT_DIR/envs/env_${seq}.yaml"
  MOTION="$OUT_DIR/motions/${seq}.pkl"
  [ -f "$TEST_ENV" ] && [ -f "$MOTION" ] || { echo "[physics] $seq: no inputs, skipped"; continue; }
  model="$MIMICKIT_UNI_CKPT"

  if [ -z "${ZERO_SHOT:-}" ]; then
    mkdir -p "$OUT_DIR/$seq"
    env="$OUT_DIR/$seq/train_env.yaml"
    python - "$OUT_DIR/envs_rsi/env_${seq}.yaml" "$env" "$CURRICULUM_MODE" \
        "$((WARMUP_HOURS * SAMPLES_PER_HOUR))" "$(((HOURS - WARMUP_HOURS) * SAMPLES_PER_HOUR))" \
        "$START_SAMPLES" <<'PY'
import sys
src, dst, mode, warmup, anneal, start = sys.argv[1:7]
block = ["", "    pose_termination_curriculum:",
         f"        warmup_samples: {warmup}", f"        anneal_samples: {anneal}", f"        start_samples: {start}"]
if mode == "steps":
    block += ["        steps:"] + [f"            - [{d}, {p}]" for d, p in
              [(3.0, 5e-5), (2.0, 1e-4), (1.5, 1.5e-4), (1.0, 2e-4), (0.75, 2.5e-4), (0.5, 3e-4)]]
else:
    block += ["        start_dist: 3.0", "        end_dist: 0.5",
              "        start_power_coefficient: 0.00005", "        end_power_coefficient: 0.00030"]
text = open(src).read()
anchor = "    power_coefficient: 0.00005\n"
assert text.count(anchor) == 1, f"unexpected env layout: {src}"
open(dst, "w").write(text.replace(anchor, anchor + "\n".join(block) + "\n"))
PY
    echo "[physics] $seq: refining for ${HOURS} h"
    python mimickit/run.py --mode train --num_envs "$NUM_ENVS" \
        --max_samples $((HOURS * SAMPLES_PER_HOUR)) \
        --env_config "$env" \
        --agent_config data/agents/deepmimic_horse_ppo_agent_2.yaml \
        --model_file "$model" \
        --motion_file "$MOTION" \
        --out_model_file "$OUT_DIR/$seq/model.pt" \
        --log_file "$OUT_DIR/$seq/log.txt" \
        --visualize false 2>&1 | tail -5
    model="$OUT_DIR/$seq/model.pt"
  fi

  python mimickit/run.py --mode test --num_envs 1 \
      --arg_file args/deepmimic_horse_ppo_34.txt \
      --env_config "$TEST_ENV" \
      --model_file "$model" \
      --motion_file "$MOTION" \
      --test_episodes 1 \
      --save_states_path "$OUT_DIR/${seq}_states.npz" \
      --visualize false 2>&1 | grep -E "Mean Episode Length|Success Rate"
done
