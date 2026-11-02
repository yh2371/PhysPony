import pickle
from anim.motion import LoopMode, Motion

def load_motion(file):
    with open(file, "rb") as filestream:
        in_dict = pickle.load(filestream)
        try:
            loop_mode_val = in_dict["loop_mode"]
            fps = in_dict["fps"]
            frames = in_dict["frames"]
            
            loop_mode = LoopMode(loop_mode_val)

            motion_data = Motion(loop_mode=loop_mode,
                                fps=fps,
                                frames=frames)
        except:
            motion_data = in_dict
    return motion_data


data = load_motion("data/motions/horse_chunked_ID4_full/20201129_ID_4_0002_0.pkl")
print(data.frames.shape, data.fps, data.loop_mode)