#!/usr/bin/env python
"""
Serve textured OBJ mesh sequences (stage-4 MimicKit tracked/reference, and
optionally GT mocap lifted through the same SMAL model) as a scrubbable,
playable, toggle-able 3D view in the browser via viser. Works headlessly over
an SSH port-forward - no X11/Blender needed on the Slurm node.

Env: conda activate physpony-viser (pip install viser trimesh)

Usage (typical - stage-4 export_stage4_mesh_sequence.py output dir has both):
    python scripts/viser_mesh_sequence_viewer.py \
        --sim_dir outputs/stage4_sim/mesh_sequence --sim_pattern "tracked_*.obj" \
        --kin_dir outputs/stage4_sim/mesh_sequence --kin_pattern "reference_*.obj" \
        --gt_dir outputs/stage4_sim/mesh_sequence_gt --gt_pattern "gt_*.obj"

Any of --sim_dir/--kin_dir/--gt_dir may be omitted; that track (and its
visibility toggle) is simply not created. At least one must be given.

Then forward the port and open http://localhost:<port> locally, e.g.:
    ssh -L 8080:localhost:8080 <cluster-host>
"""
import argparse
import glob
import os
import re
import time

import trimesh
import viser

# name -> (display label, default color tint used only as a fallback if a
# frame has no baked texture, default visible)
_TRACK_SPECS = {
    "sim": ("Sim output (after physics)", (80, 140, 255)),
    "kin": ("Kinematic pred (before physics)", (255, 160, 60)),
    "gt": ("GT mocap", (60, 220, 100)),
}


def _natural_key(path):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", os.path.basename(path))]


def _load_track(mesh_dir, pattern):
    paths = sorted(glob.glob(os.path.join(mesh_dir, pattern)), key=_natural_key)
    if not paths:
        raise SystemExit(f"No files matching {pattern!r} found in {mesh_dir}")
    meshes = [trimesh.load(p, process=False) for p in paths]
    return paths, meshes


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sim_dir", default=None, help="Directory with the after-sim (physics rollout) OBJ sequence.")
    parser.add_argument("--sim_pattern", default="tracked_*.obj")
    parser.add_argument("--kin_dir", default=None, help="Directory with the before-sim (kinematic reference) OBJ sequence.")
    parser.add_argument("--kin_pattern", default="reference_*.obj")
    parser.add_argument("--gt_dir", default=None, help="Directory with a GT-mocap-derived OBJ sequence, if available.")
    parser.add_argument("--gt_pattern", default="gt_*.obj")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--fps", type=float, default=15.0, help="Default autoplay speed.")
    parser.add_argument("--floor_size", type=float, default=200.0, help="Side length of the grey floor plane (large enough to read as infinite).")
    args = parser.parse_args()

    dirs = {"sim": args.sim_dir, "kin": args.kin_dir, "gt": args.gt_dir}
    patterns = {"sim": args.sim_pattern, "kin": args.kin_pattern, "gt": args.gt_pattern}
    tracks = {}
    for key, mesh_dir in dirs.items():
        if mesh_dir is None:
            continue
        label, color = _TRACK_SPECS[key]
        paths, meshes = _load_track(mesh_dir, patterns[key])
        tracks[key] = {"label": label, "color": color, "paths": paths, "meshes": meshes}
        print(f"[{key}] loaded {len(meshes)} frames from {mesh_dir}")

    if not tracks:
        raise SystemExit("At least one of --sim_dir / --kin_dir / --gt_dir must be given.")

    num_frames = max(len(t["meshes"]) for t in tracks.values())

    server = viser.ViserServer(port=args.port)
    try:
        share_url = server.request_share_url()
        print("Public URL:", share_url)
    except Exception as e:
        print(f"(no public share URL: {e})")

    # Z-up (matches MimicKit/IsaacGym's UP_AXIS_Z) is viser's default already;
    # set explicitly so this doesn't silently break if the default changes.
    server.scene.set_up_direction("+z")
    server.scene.add_grid(
        "/ground", width=args.floor_size, height=args.floor_size,
        cell_color=(170, 170, 170), section_color=(140, 140, 140),
        plane_color=(120, 120, 120), plane_opacity=1.0,
        infinite_grid=True, fade_distance=max(args.floor_size, 50.0),
    )

    handles = {}
    checkboxes = {}
    for key, t in tracks.items():
        checkboxes[key] = server.gui.add_checkbox(f"Show: {t['label']}", initial_value=True)
        handles[key] = server.scene.add_mesh_trimesh(f"/{key}/frame", t["meshes"][0])

    frame_slider = server.gui.add_slider("Frame", min=0, max=num_frames - 1, step=1, initial_value=0)
    playing_checkbox = server.gui.add_checkbox("Play", initial_value=False)
    fps_slider = server.gui.add_slider("FPS", min=1, max=60, step=1, initial_value=args.fps)
    frame_label = server.gui.add_text("File(s)", initial_value="", disabled=True)

    def _track_index(key, idx):
        """Map the shared frame index onto a (possibly shorter/longer) track."""
        n = len(tracks[key]["meshes"])
        if n == num_frames:
            return idx
        return round(idx * (n - 1) / max(num_frames - 1, 1))

    def set_frame(idx):
        idx = int(idx) % num_frames
        labels = []
        for key, t in tracks.items():
            j = _track_index(key, idx)
            new_handle = server.scene.add_mesh_trimesh(f"/{key}/frame", t["meshes"][j])
            new_handle.visible = checkboxes[key].value
            handles[key].remove()  # add-then-remove avoids a blank-frame flicker
            handles[key] = new_handle
            labels.append(f"{key}: {os.path.basename(t['paths'][j])}")
        frame_label.value = " | ".join(labels)

    @frame_slider.on_update
    def _(_):
        set_frame(frame_slider.value)

    for key in tracks:
        @checkboxes[key].on_update
        def _(_, key=key):
            handles[key].visible = checkboxes[key].value

    set_frame(0)
    print(f"Viser server running at http://localhost:{args.port}  ({num_frames} frames, tracks: {list(tracks)})")
    while True:
        if playing_checkbox.value:
            next_idx = (frame_slider.value + 1) % num_frames
            frame_slider.value = next_idx  # updates the GUI display
            set_frame(next_idx)  # always advance the meshes directly - don't rely on
                                  # the slider's on_update firing (e.g. if value is
                                  # unchanged at the wrap-around 0 -> 0 edge case)
        time.sleep(1.0 / max(fps_slider.value, 1))


if __name__ == "__main__":
    main()
