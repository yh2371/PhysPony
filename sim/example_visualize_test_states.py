#!/usr/bin/env python
"""
Example script for visualizing test states saved in SMAL format.

Usage:
    python example_visualize_test_states.py <npz_file> [options]

Options:
    --output-dir DIR     Directory to save outputs (default: output/visualizations)
    --no-reference       Don't show reference skeleton
    --fps FPS            Frames per second for animation (default: 30)
    --texture-dir DIR    Directory containing texture PNG files for OBJ export
    --uvmap-obj PATH     Path to UV map OBJ file for loading UV coordinates
"""
import sys
import os
import argparse

# Add mimickit to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mimickit.util.visualize_smal_states import (
    visualize_skeleton_3d,
    export_skeleton_obj,
    load_smal_states
)


def main():
    parser = argparse.ArgumentParser(description='Visualize SMAL format test states')
    parser.add_argument('npz_file', type=str, help='Path to .npz file with SMAL states')
    parser.add_argument('--output-dir', type=str, default='output/visualizations',
                       help='Directory to save outputs')
    parser.add_argument('--no-reference', action='store_true',
                       help='Don\'t show reference skeleton')
    parser.add_argument('--fps', type=int, default=30,
                       help='Frames per second for animation')
    parser.add_argument('--no-animation', action='store_true',
                       help='Skip animation, only export OBJ files')
    parser.add_argument('--no-obj', action='store_true',
                       help='Skip OBJ export, only create animation')
    parser.add_argument('--texture-dir', type=str, default=None,
                       help='Directory containing texture PNG files for OBJ export. '
                            'Texture files should be named like: frame_tracked_0000.png')
    parser.add_argument('--uvmap-obj', type=str, default=None,
                       help='Path to UV map OBJ file for loading UV coordinates')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.npz_file):
        print(f"Error: File not found: {args.npz_file}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and check data
    print(f"Loading states from {args.npz_file}...")
    states = load_smal_states(args.npz_file)
    tracked = states['tracked']
    reference = states.get('reference', None)
    
    print(f"Loaded {tracked['trans'].shape[0]} frames")
    print(f"Number of body joints: {tracked['poses_body'].shape[1]}")
    if reference is not None:
        print("Reference states found")
    else:
        print("No reference states found")
    
    # Create animation
    if not args.no_animation:
        print("\nCreating 3D animation...")
        anim_output = os.path.join(args.output_dir, 'skeleton_animation.gif')
        try:
            visualize_skeleton_3d(
                args.npz_file,
                output_path=anim_output,
                fps=args.fps,
                show_reference=not args.no_reference and reference is not None,
                use_mesh=True  # Use SMAL model for mesh generation
            )
            print(f"Animation saved to {anim_output}")
        except Exception as e:
            print(f"Error creating animation: {e}")
            import traceback
            traceback.print_exc()
    
    # Export OBJ files
    if not args.no_obj:
        print("\nExporting OBJ files...")
        obj_output_dir = os.path.join(args.output_dir, 'obj_files')
        try:
            export_skeleton_obj(
                args.npz_file,
                obj_output_dir,
                prefix='frame',
                show_reference=not args.no_reference and reference is not None,
                use_mesh=True,  # Use SMAL model for mesh generation
                texture_dir=args.texture_dir,
                uvmap_obj_path=args.uvmap_obj
            )
            print(f"OBJ files exported to {obj_output_dir}")
            if args.texture_dir:
                print(f"Textures applied from: {args.texture_dir}")
        except Exception as e:
            print(f"Error exporting OBJ files: {e}")
            import traceback
            traceback.print_exc()
    
    print("\nDone!")


if __name__ == "__main__":
    main()

