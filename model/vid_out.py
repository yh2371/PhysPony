import os
import cv2
import re

def natural_key(string):
    """Sort strings with numbers in natural order (e.g., 1,2,10)."""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string)]

def frames_to_video(
    folder,
    output_path="output.mp4",
    fps=30
):
    # Collect all *_rgb.jpg files
    frames = [f for f in os.listdir(folder) if f.endswith("_image_pred.png")]
    if not frames:
        raise ValueError("No files ending with _rgb.jpg found.")

    # Sort frames in natural numeric order
    frames.sort(key=natural_key)

    # Read first frame to get size
    first_frame_path = os.path.join(folder, frames[0])
    frame0 = cv2.imread(first_frame_path)
    h, w, _ = frame0.shape

    # Create video writer (H.264)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    print(f"Writing video to: {output_path}")
    for fname in frames:
        img = cv2.imread(os.path.join(folder, fname))
        if img is None:
            print(f"Warning: could not read {fname}, skipping.")
            continue
        out.write(img)

    out.release()
    print("Done.")

# -------------------------
# Example usage:
# -------------------------
if __name__ == "__main__":
    folder = os.environ.get(
        "PHYSPONY_VID_FOLDER",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "data", "magicpony", "horse_combined", "val", "0000000"),
    )    # contains *_rgb.jpg files
    output = "reconstruction_rgb_pred.mp4"

    frames_to_video(folder, output_path=output, fps=30)
