"""Central, portable path resolution for the PhysPony model (Ponymation) package.

All paths resolve relative to this file's location so the repo is relocatable,
with optional environment-variable overrides. No absolute paths are baked in.

Layout assumed:
    <REPO_ROOT>/model/model/paths.py   (this file)
    <REPO_ROOT>/model/model/geometry/SMAL/...
    <REPO_ROOT>/checkpoints/{dessie,ponymation,fastvggt}/...
    <REPO_ROOT>/data/magicpony/...      (symlink)
"""
import os

# model/model  (the importable `model` package directory)
MODEL_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
# model  (contains run.py and the package)
MODEL_DIR = os.path.dirname(MODEL_PKG_DIR)
# repo root (PhysPony)
REPO_ROOT = os.environ.get("PHYSPONY_ROOT", os.path.dirname(MODEL_DIR))

# --- geometry / SMAL assets (bundled inside the package) ---
GEOMETRY_DIR = os.path.join(MODEL_PKG_DIR, "geometry")
SMAL_DIR = os.environ.get("PHYSPONY_SMAL_DIR", os.path.join(GEOMETRY_DIR, "SMAL"))
SMAL_MODEL_PKL = os.path.join(SMAL_DIR, "my_smpl_0000_horse_new_skeleton_horse.pkl")
UVMAP_OBJ = os.path.join(SMAL_DIR, "uvmap", "uvmap_from_TextPaper", "TexPaper_uv_mask.obj")
TPOSE_NEW_JOINTS = os.path.join(GEOMETRY_DIR, "tpose_new_joints.npy")
TPOSE_JOINTS = os.path.join(GEOMETRY_DIR, "tpose_joints.npy")

# --- checkpoints (symlinked under the repo) ---
CHECKPOINTS_DIR = os.environ.get("PHYSPONY_CHECKPOINTS", os.path.join(REPO_ROOT, "checkpoints"))
DESSIE_CKPT = os.environ.get("DESSIE_CKPT", os.path.join(CHECKPOINTS_DIR, "dessie", "best.ckpt"))
PONYMATION_CKPT_DIR = os.environ.get(
    "PONYMATION_CKPT_DIR",
    os.path.join(CHECKPOINTS_DIR, "ponymation",
                 "pretrained_horse_stage2_rot6d_withrendering_smoothing_extra_global"),
)

# --- data (symlink to the shared magicpony dataset) ---
DATA_ROOT = os.environ.get("PHYSPONY_DATA", os.path.join(REPO_ROOT, "data", "magicpony"))
