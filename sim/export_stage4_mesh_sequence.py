#!/usr/bin/env python
"""
Export a stage-4 (MimicKit) rollout as a textured OBJ/MTL/PNG sequence,
loadable in Blender: one mesh per frame for the after-sim (tracked, physics
rollout) trajectory and, if present, the before-sim (reference/kinematic)
trajectory.

Since MimicKit itself has no appearance model, the "skin" is a static texture
atlas reused from a stage-3 (Ponymation) mesh export (see
model/model/models/AnimalModel.export_textured_mesh_batch, enabled via
`+model.export_mesh=true`), applied through the same SMAL UV atlas
(model.paths.UVMAP_OBJ) the stage-3 renderer bakes into. That atlas has more
UV vertices than mesh vertices (real seam splitting), which
mimickit.util.visualize_smal_states.export_mesh_obj can't represent (it
assumes one UV per vertex) - so this script writes OBJs directly via
pytorch3d.io.save_obj instead, same as render_2.save_textured_obj in stage 3.
Run this in an env with pytorch3d (e.g. `3danimals`), not `isaac`.

Usage:
    python export_stage4_mesh_sequence.py \
        --states_npz outputs/stage4_sim/0000034_states.npz \
        --texture_png outputs/stage3_ponymation/test_results_0000034_mesh/0000034_0015765__kd.png \
        --out_dir outputs/stage4_sim/mesh_sequence
"""
import argparse
import os
import sys

import numpy as np
import torch
import imageio.v2 as imageio
from pytorch3d.io import load_obj, save_obj

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mimickit.util.visualize_smal_states import (
    load_smal_states,
    generate_smal_mesh,
    kin2smal,   # states .npz are in KinChar joint order; generate_smal_mesh indexes SMAL
)


def _default_uvmap_obj():
    """Reuse the same UV atlas the stage-3 renderer bakes textures with."""
    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model")
    sys.path.insert(0, model_dir)
    from model.paths import UVMAP_OBJ
    return UVMAP_OBJ


def _export_sequence(verts_seq, faces, VT, FT, texture_map, out_dir, prefix):
    for t in range(verts_seq.shape[0]):
        out_path = os.path.join(out_dir, f"{prefix}_{t:04d}.obj")
        save_obj(
            f=out_path,
            verts=torch.from_numpy(verts_seq[t]).float(),
            faces=faces,
            verts_uvs=VT,
            faces_uvs=FT,
            texture_map=texture_map,
            decimal_places=6,
        )
    return verts_seq.shape[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--states_npz", required=True, help="Path to a MimicKit test-mode states .npz (TestStateCollector.save output).")
    parser.add_argument("--texture_png", default=None, help="Static texture atlas PNG to apply to every frame (e.g. a stage-3 *_kd.png). If omitted, a flat gray texture is used.")
    parser.add_argument("--out_dir", required=True, help="Directory to write the OBJ/MTL/PNG sequence to.")
    parser.add_argument("--smal_model_path", default=None, help="Path to the SMAL .pkl model (default: sim/mimickit/SMAL/my_smpl_0000_horse_new_skeleton_horse.pkl).")
    parser.add_argument("--uvmap_obj_path", default=None, help="Path to the UV-map OBJ (default: same TexPaper_uv_mask.obj the stage-3 renderer uses).")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no_reference", action="store_true", help="Skip exporting the before-sim (kinematic reference) sequence.")
    args = parser.parse_args()

    uvmap_obj_path = args.uvmap_obj_path or _default_uvmap_obj()
    os.makedirs(args.out_dir, exist_ok=True)

    uv_data = load_obj(uvmap_obj_path)
    VT = uv_data[-1].verts_uvs
    FT = uv_data[1].textures_idx

    if args.texture_png is not None:
        tex_map = torch.from_numpy(imageio.imread(args.texture_png)[..., :3].astype(np.float32) / 255.0)
    else:
        tex_map = torch.full((4, 4, 3), 0.6)

    states = load_smal_states(args.states_npz)
    tracked = states["tracked"]
    reference = states.get("reference", None)

    tracked_verts, _, smal_model = generate_smal_mesh(
        kin2smal(tracked["poses_body"]), tracked["trans"], tracked["poses_root"],
        smal_model_path=args.smal_model_path, device=args.device,
    )
    faces = smal_model.faces.cpu().long()
    assert faces.shape[0] == FT.shape[0], (
        f"Face count mismatch between SMAL model ({faces.shape[0]}) and UV atlas ({FT.shape[0]}); "
        f"they must be the same SMAL topology to reuse the stage-3 texture atlas."
    )

    n = _export_sequence(tracked_verts, faces, VT, FT, tex_map, args.out_dir, "tracked")
    print(f"Wrote {n} tracked (after-sim) frames to {args.out_dir}")

    if reference is not None and not args.no_reference:
        ref_verts, _, _ = generate_smal_mesh(
            kin2smal(reference["poses_body"]), reference["trans"], reference["poses_root"],
            smal_model_path=args.smal_model_path, device=args.device,
        )
        n_ref = _export_sequence(ref_verts, faces, VT, FT, tex_map, args.out_dir, "reference")
        print(f"Wrote {n_ref} reference (before-sim) frames to {args.out_dir}")
    elif reference is None:
        print("No reference states found in states_npz; skipping before-sim export.")


if __name__ == "__main__":
    main()
