#!/usr/bin/env bash
# Paths used by the PhysPony scripts. Export any of these before sourcing to override.

export PHYSPONY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export PHYSPONY_FASTVGGT="${PHYSPONY_FASTVGGT:-$PHYSPONY_ROOT/perception/fastvggt}"
export PHYSPONY_UNIDEPTH="${PHYSPONY_UNIDEPTH:-$PHYSPONY_ROOT/perception/unidepth}"
export PHYSPONY_MODEL="${PHYSPONY_MODEL:-$PHYSPONY_ROOT/model}"
export PHYSPONY_SIM="${PHYSPONY_SIM:-$PHYSPONY_ROOT/sim}"
export PHYSPONY_DATA="${PHYSPONY_DATA:-$PHYSPONY_ROOT/data/magicpony}"

export FASTVGGT_CKPT="${FASTVGGT_CKPT:-$PHYSPONY_ROOT/checkpoints/fastvggt/model_tracker_fixed_e20.pt}"
export UNIDEPTH_HF_ID="${UNIDEPTH_HF_ID:-lpiccinelli/unidepth-v2-vitl14}"
export DESSIE_CKPT="${DESSIE_CKPT:-$PHYSPONY_ROOT/checkpoints/dessie/best.ckpt}"
export PHYSPONY_KINEMATIC_CKPT="${PHYSPONY_KINEMATIC_CKPT:-$PHYSPONY_ROOT/checkpoints/physpony/kinematic.pth}"
export MIMICKIT_UNI_CKPT="${MIMICKIT_UNI_CKPT:-$PHYSPONY_ROOT/checkpoints/physpony/tracking_policy.pt}"
