#!/usr/bin/env python3
"""
Shared camera-axis fixup for the model/ glue scripts that read a *_params.pkl
"rotation" (4x4 camera-to-world transform) and need it in the render/world
convention this codebase's downstream consumers (SMAL skinning, MimicKit
lifting) expect. Consolidates an identical 9-line block that was
copy-pasted across process_kinematic_data.py, process_kinematic_data_smal.py,
and batch_extract_and_plot_global.py.

Not shared with sim/process_kinematic_results.py, which keeps its own copy
since sim/ runs in a separate, incompatible conda env from model/.

model/visualize_global_2.py intentionally uses a different fixup (its own
"align to z-up" experimental variant) and is NOT a consumer of this helper.
"""

import numpy as np


def fix_camera_axes(M: np.ndarray) -> np.ndarray:
    """Apply the two fixed axis-remap transforms to a 4x4 camera-to-world M."""
    transform_temp = np.eye(4, dtype=np.float64)
    transform_temp[:3, :3] = np.array(
        [[-1, 0, 0],
         [0, -1, 0],
         [0, 0, 1]], dtype=np.float64)
    transform_temp2 = np.eye(4, dtype=np.float64)
    transform_temp2[:3, :3] = np.array(
        [[1, 0, 0],
         [0, 0, -1],
         [0, 1, 0]], dtype=np.float64)
    return transform_temp2 @ M @ transform_temp
