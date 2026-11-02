#!/usr/bin/env python3
"""Per-sequence metric scale (median_scale.txt from align_depth.py) for the
predicted root translation."""

import argparse
from pathlib import Path
from typing import Optional


def add_trans_scale_args(parser: argparse.ArgumentParser) -> None:
    """Register --trans_scale/--median_scale_file on an argparse parser."""
    parser.add_argument(
        "--trans_scale",
        type=float,
        default=None,
        help="Per-sequence metric scale to apply to the network's own "
             "predicted root translation before rotating into world frame "
             "(world = R_c2w @ (trans_scale * local_trans) + T_c2w). Pass "
             "the sequence's own median_scale from align_depth.py. If not "
             "given, falls back to --median_scale_file if set, else 1.0 "
             "(no scaling).",
    )
    parser.add_argument(
        "--median_scale_file",
        type=Path,
        default=None,
        help="Path to a median_scale.txt written by align_depth.py; read "
             "as --trans_scale if --trans_scale isn't given explicitly.",
    )


def resolve_trans_scale(
    trans_scale: Optional[float],
    median_scale_file: Optional[Path],
) -> float:
    """Resolve the effective trans_scale from explicit value / file / default."""
    if trans_scale is not None:
        return float(trans_scale)
    if median_scale_file is not None:
        return float(Path(median_scale_file).read_text().strip())
    return 1.0
