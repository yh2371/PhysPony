import enum
import pickle

class LoopMode(enum.Enum):
    CLAMP = 0
    WRAP = 1

def load_motion(file):
    with open(file, "rb") as filestream:
        in_dict = pickle.load(filestream)
        try:
            loop_mode_val = int(in_dict["loop_mode"])
            fps = in_dict["fps"]
            frames = in_dict["frames"]
            
            loop_mode = LoopMode(loop_mode_val)

            motion_data = Motion(loop_mode=loop_mode,
                                fps=fps,
                                frames=frames)
        except:
            motion_data = in_dict
            fps = motion_data.fps
    motion_data.fps = min(30, fps)
    motion_data.loop_mode = LoopMode(0)
    print(motion_data.fps, motion_data.loop_mode, motion_data.frames.shape)
    return motion_data

class Motion():
    def __init__(self, loop_mode, fps, frames):
        self.loop_mode = loop_mode
        self.fps = fps
        self.frames = frames
        return

    def save(self, out_file):
        with open(out_file, "wb") as out_f:
            out_dict = {
                "loop_mode": self.loop_mode.value,
                "fps": self.fps,
                "frames": self.frames
            }
            pickle.dump(out_dict, out_f)
        return

    def get_length(self):
        num_frames = self.frames.shape[0]
        motion_len = float(num_frames - 1) / self.fps
        return motion_len