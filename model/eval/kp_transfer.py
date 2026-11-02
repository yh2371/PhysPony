# find_nearest_vertex and the 14-keypoint name/order below are ported
# verbatim from Dessie's evalkp_utils/data_utils.py, itself adapted from
# https://github.com/google/lassie/blob/main/utils/data_utils.py
# (Apache License 2.0, Copyright 2023 Google LLC).
import torch

# keypoint name -> index in the 14-length GT keypoint array
KP_INDICES = {
    'leye': 0,
    'reye': 1,
    'lear': 2,
    'rear': 3,
    'muzzle': 4,
    'tail': 5,
    'rfuleg': 6,
    'rbuleg': 7,
    'lfuleg': 8,
    'lbuleg': 9,
    'rflleg': 10,
    'rblleg': 11,
    'lflleg': 12,
    'lblleg': 13,
}

# keypoint index -> human-readable name, in KP_INDICES order
KP_NAMES = [name for name, _ in sorted(KP_INDICES.items(), key=lambda kv: kv[1])]


def find_nearest_vertex(keypoints, verts_2d, verts_vis):
    ''' keypoints: N*2; verts_2d: V*2; verts_vis: V -- returns LongTensor[N] of matched vertex indices '''
    vert_indices = torch.arange(verts_2d.shape[0]).long()
    outputs = torch.zeros(keypoints.shape[0]).long()
    for i in range(keypoints.shape[0]):
        dist = ((verts_2d[verts_vis > 0] - keypoints[i, :2]) ** 2).sum(1)
        outputs[i] = vert_indices[verts_vis > 0][dist.argmin()]
    return outputs


def kp_transfer_pck(kps_list, verts_2d_list, verts_vis_list, img_size=256, thresholds=(0.05, 0.10)):
    """Cross-instance keypoint-transfer PCK, matching Dessie's evaluate_kptransfer.py exactly.

    kps_list:       list of [N,3] (x, y, vis) GT keypoint tensors, one per image
    verts_2d_list:  list of [V,2] projected-mesh-vertex tensors, one per image
    verts_vis_list: list of [V] visibility tensors (>0 == visible), one per image

    Returns a dict with pck@each threshold, num_pairs, and a per-keypoint pck@0.10 breakdown.
    """
    num_imgs = len(kps_list)
    assert num_imgs == len(verts_2d_list) == len(verts_vis_list)
    num_kp = kps_list[0].shape[0]

    # find_nearest_vertex only depends on image i1 -- precompute once per image
    # instead of once per pair (numerically identical to Dessie's approach, faster).
    v_matched_per_image = [
        find_nearest_vertex(kps_list[i], verts_2d_list[i], verts_vis_list[i])
        for i in range(num_imgs)
    ]

    pck = {t: 0.0 for t in thresholds}
    per_kp_correct = {t: torch.zeros(num_kp) for t in thresholds}
    per_kp_valid = torch.zeros(num_kp)
    num_pairs = 0

    for i1 in range(num_imgs):
        kps1 = kps_list[i1]
        v_matched = v_matched_per_image[i1]
        for i2 in range(num_imgs):
            if i1 == i2:
                continue
            kps2 = kps_list[i2]
            verts2 = verts_2d_list[i2]

            kps_trans = verts2[v_matched]
            valid = (kps1[:, 2] > 0) & (kps2[:, 2] > 0)
            if valid.sum() == 0:
                continue

            kps_trans_norm = kps_trans[:, :2] / img_size
            kps2_norm = kps2[:, :2] / img_size
            dist = ((kps_trans_norm - kps2_norm) ** 2).sum(1).sqrt()

            valid_f = valid.float()
            for t in thresholds:
                correct = (dist <= t).float() * valid_f
                pck[t] += correct.sum() / valid_f.sum()
                per_kp_correct[t] += correct
            per_kp_valid += valid_f
            num_pairs += 1

    if num_pairs == 0:
        raise RuntimeError("No valid keypoint-transfer pairs found (num_pairs == 0)")

    result = {f"pck{int(t*100):02d}": float(pck[t] / num_pairs) for t in thresholds}
    result["num_pairs"] = num_pairs
    result["num_images"] = num_imgs
    for t in thresholds:
        key = f"per_kp_pck{int(t*100):02d}"
        safe_valid = per_kp_valid.clamp(min=1)
        result[key] = {
            KP_NAMES[i]: float(per_kp_correct[t][i] / safe_valid[i])
            for i in range(num_kp)
        }
    return result
