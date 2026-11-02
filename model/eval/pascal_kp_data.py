import json
import os

# ACSM/animals3d canonical 16-keypoint order, per
# Dessie: src/staths/constants.py::QUAD_JOINT_NAMES
# (adapted from https://github.com/statho/animals3d/blob/main/acsm/constants.py)
ACSM16_NAMES = [
    'L_eye', 'R_eye', 'L_ear', 'R_ear', 'Nose', 'Throat', 'Tail', 'Withers',
    'L_F_elbow', 'R_F_elbow', 'L_B_elbow', 'R_B_elbow',
    'L_F_paw', 'R_F_paw', 'L_B_paw', 'R_B_paw',
]

# maps ACSM16 index -> LASSIE14 name (from kp_transfer.KP_INDICES); drops
# Throat (idx 5) and Withers (idx 7), which LASSIE14 has no equivalent for.
ACSM16_TO_LASSIE14 = {
    0: 'leye', 1: 'reye', 2: 'lear', 3: 'rear', 4: 'muzzle', 6: 'tail',
    8: 'lfuleg', 9: 'rfuleg', 10: 'lbuleg', 11: 'rbuleg',
    12: 'lflleg', 13: 'rflleg', 14: 'lblleg', 15: 'rblleg',
}


def conform_acsm16_to_lassie14(kp_16):
    """kp_16: list/array of 16 [x,y,vis] rows in ACSM16_NAMES order.
    Returns a list of 14 [x,y,vis] rows in kp_transfer.KP_NAMES order."""
    try:
        from .kp_transfer import KP_INDICES
    except ImportError:
        from kp_transfer import KP_INDICES
    out = [None] * len(KP_INDICES)
    for acsm_idx, lassie_name in ACSM16_TO_LASSIE14.items():
        out[KP_INDICES[lassie_name]] = list(kp_16[acsm_idx])
    assert all(row is not None for row in out)
    return out


def load_coco_pascal_nomask(jsons_path, images_dir=None, pascal_only=True):
    """Loads Dessie's /coco_pascal_nomask.jsons.
    Each entry: {'img_id', 'image_path', 'kp': [[x,y,vis]*16]} in ACSM16 order.

    Returns {img_id: {'image_path': resolved path, 'kp16': [[x,y,vis]*16]}}
    filtered to pascal-sourced entries only (drops the coco-sourced subset)
    if pascal_only=True.
    """
    with open(jsons_path) as f:
        data = json.load(f)
    if images_dir is None:
        images_dir = os.path.join(os.path.dirname(os.path.dirname(jsons_path)), 'images')
    out = {}
    for entry in data:
        if pascal_only and 'pascal' not in entry['image_path'].lower():
            continue
        img_id = str(entry['img_id'])
        basename = os.path.basename(entry['image_path'])
        out[img_id] = {
            'image_path': os.path.join(images_dir, basename),
            'basename': basename,
            'kp16': entry['kp'],
        }
    return out
