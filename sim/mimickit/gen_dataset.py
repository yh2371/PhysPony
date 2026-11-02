import os
import yaml
from random import shuffle
random_seed = 42
import numpy as np
np.random.seed(random_seed)
import shutil
# === Configuration ===
motion_dir = "data/motions/horse_chunked_ID4_partial_new"   # directory containing .pkl files
output_train_file = "data/datasets/dataset_horse_motions_ID4_partial_new_train.yaml"      # output file name
output_val_file = "data/datasets/dataset_horse_motions_ID4_partial_new_val.yaml"      # output file name
default_weight = 1.0              # default weight

# === Collect .pkl files ===
pkl_files = sorted(
    [f for f in os.listdir(motion_dir) if f.endswith(".pkl")]
)

# === Build data structure ===
motions = [{"file": f"{motion_dir}/{f}", "weight": default_weight} for f in pkl_files]
np.random.shuffle(motions)

train_motions = motions[:int(len(motions) * 0.8)]
val_motions = motions[int(len(motions) * 0.8):]

os.makedirs("data/motions/horse_chunked_ID4_partial_new_train", exist_ok=True)
os.makedirs("data/motions/horse_chunked_ID4_partial_new_val", exist_ok=True)
for motion in train_motions:
    shutil.copy(motion["file"], "data/motions/horse_chunked_ID4_partial_new_train")
for motion in val_motions:
    shutil.copy(motion["file"], "data/motions/horse_chunked_ID4_partial_new_val")
# === Save to YAML ===
with open(output_train_file, "w") as f:
    yaml.dump({"motions": train_motions}, f, sort_keys=False)
with open(output_val_file, "w") as f:
    yaml.dump({"motions": val_motions}, f, sort_keys=False)

print(f"✅ Saved {len(train_motions)} motions to {output_train_file} and {len(val_motions)} motions to {output_val_file}")
